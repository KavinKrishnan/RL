# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish a live Megatron layout through ModelExpress `main`'s reshard seam.

`mx_megatron_helpers.collect_megatron_publish_set` classifies each native
Megatron parameter; MX `main` accepts that classification through
``refit.reshard.megatron_aliases.build_hf_aliases``, which turns native storage
into HF-canonical shard records without copying, and publishes it through
``publish_registered_shard_table``. This module is the translation between the
two, plus the name resolution that neither side owns.

It exists because MX `main` deliberately stops short of it. From
``modelexpress/refit/README.md``:

    This is an adapter contract, not a complete quick start. The repository does
    not currently include the RL framework lifecycle hooks or a general trainer
    publisher that derives ownership from arbitrary training backends.

The predecessor of this path was ``MxV2TrainingPublisher``, which is not on MX
`main` and is not coming: the incremental PRs behind it were closed unmerged and
the surface self-describes as a prototype shim pending server-side RPCs, which
landed as the Redis refit control plane. So the classification helper stays and
only the publish call changes.

The two sides use the same role vocabulary (``qkv_column``,
``gated_mlp_column``, ``expert_column``, ...) and the same extras keys
(``head_dim``, ``num_heads_local``, ``num_kv_heads_local``, ``gated_mlp_order``),
so no renaming happens here.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Iterator

from modelexpress.refit.reshard.megatron_aliases import (
    MegatronAliasInput,
    build_hf_aliases,
)
from modelexpress.refit.reshard.megatron_publisher import (
    publish_registered_shard_table,
)

from nemo_rl.distributed.mx_megatron_helpers import (
    MegatronRoleSpec,
    infer_megatron_tp_shard_geometry,
)

# build_hf_aliases treats any placement that is not exactly "SHARD" as
# replicated, so the replicated spelling is ours to choose and only has to be
# stable.
PLACEMENT_SHARD = "SHARD"
PLACEMENT_REPLICATE = "REPLICATE"

# HF MoE parameters carry the expert index as a path component. Megatron's
# grouped layout carries it as a leaf suffix (``linear_fc1.weight0``), which
# `canonicalize_grouped_expert_name` has already rewritten to the global id by
# the time we see it.
_HF_EXPERT_INDEX = re.compile(r"(?<=\.experts\.)\d+(?=\.)")

HfNameResolver = Callable[[str, dict[str, str]], "tuple[str, ...]"]


class UnmappedMegatronTensor(KeyError):
    """A published parameter has no HF target names.

    Fails the publish rather than skipping the tensor. A skipped source is not
    visibly broken: the receiver simply never sees those bytes, its coverage
    check reports a shortfall far from the cause, and with the coverage floor off
    it reports nothing at all.
    """


def make_bridge_resolver(
    name_map: dict[str, Iterable[str]],
    *,
    strict: bool = True,
) -> HfNameResolver:
    """Resolve a published Megatron name to HF target names via a Bridge map.

    ``name_map`` is what ``AutoBridge.get_conversion_tasks`` yields, keyed by
    Megatron parameter name. Grouped-expert names need two steps of care:

    * the key may carry the **EP-local** expert index, because the Bridge
      inspects one rank's module tree, while the published name carries the
      **global** index that `canonicalize_grouped_expert_name` substituted. So a
      miss on the global name is retried against the local one.
    * the HF names that come back then describe the local expert, and the global
      index has to be substituted into them, or every EP rank publishes over
      rank 0's experts.
    """
    resolved: dict[str, tuple[str, ...]] = {
        key: tuple(value) for key, value in name_map.items()
    }

    def resolve(name: str, extras: dict[str, str]) -> tuple[str, ...]:
        hit = resolved.get(name)
        if hit is not None:
            return hit

        global_id = extras.get("expert_id")
        local_id = extras.get("local_expert_id")
        if global_id is not None and local_id is not None:
            local_name = _swap_expert_leaf(name, global_id, local_id)
            hit = resolved.get(local_name)
            if hit is not None:
                # The map describes local expert `local_id`; this rank owns
                # global `global_id`.
                return tuple(
                    _HF_EXPERT_INDEX.sub(str(global_id), hf_name) for hf_name in hit
                )

        if strict:
            raise UnmappedMegatronTensor(
                f"{name!r} has no HF target names. Nearest keys: "
                f"{_nearest_keys(name, resolved)}"
            )
        return ()

    return resolve


def _swap_expert_leaf(name: str, from_id: str, to_id: str) -> str:
    """Rewrite a trailing ``weight<from_id>`` to ``weight<to_id>``."""
    parent, separator, leaf = name.rpartition(".")
    for prefix in ("weight", "bias", "scale"):
        if leaf == f"{prefix}{from_id}":
            swapped = f"{prefix}{to_id}"
            return f"{parent}{separator}{swapped}" if separator else swapped
    return name


def _nearest_keys(name: str, resolved: dict[str, tuple[str, ...]], limit: int = 3):
    """A few same-suffix keys, so an unmapped name is diagnosable from the log."""
    leaf = name.rpartition(".")[2]
    near = [key for key in resolved if key.rpartition(".")[2] == leaf]
    return sorted(near)[:limit] or sorted(resolved)[:limit]


def build_megatron_alias_inputs(
    publish_set: Iterable[tuple[str, Any, MegatronRoleSpec, dict[str, str]]],
    *,
    resolve_hf_names: HfNameResolver,
    tp_size: int,
    tp_rank: int,
    expert_tp_size: int | None = None,
    expert_tp_rank: int | None = None,
) -> Iterator[MegatronAliasInput]:
    """Translate a classified publish set into MX alias inputs.

    Geometry comes from `infer_megatron_tp_shard_geometry`, which returns None
    for a tensor this rank holds whole. That covers two different situations
    that MX describes the same way:

    * genuinely replicated tensors (norms, router gates);
    * expert tensors under EP with expert-TP of 1. Megatron's grouped layout
      gives each expert its own parameter, so EP partitions *names*, not an
      axis, and each rank owns its experts entire. Fan-in across EP ranks then
      happens by name in the rendezvous merge, with no shard arithmetic.

    Non-expert tensors under EP are byte-identical across every EP rank, so the
    fleet publishes DP replicas of them. That is intended: the receiver's merge
    deduplicates by geometry and reads each from one owner.
    """
    for name, tensor, spec, extras in publish_set:
        hf_names = resolve_hf_names(name, extras)
        if not hf_names:
            continue

        geometry = infer_megatron_tp_shard_geometry(
            local_shape=tuple(int(dim) for dim in tensor.shape),
            role=spec.role,
            tp_size=tp_size,
            tp_rank=tp_rank,
            expert_tp_size=expert_tp_size,
            expert_tp_rank=expert_tp_rank,
            descriptor_extras=spec.descriptor_extras,
        )

        if geometry is None:
            yield MegatronAliasInput(
                name=name,
                tensor=tensor,
                role=spec.role,
                hf_names=tuple(hf_names),
                global_shape=tuple(int(dim) for dim in tensor.shape),
                placement_kind=PLACEMENT_REPLICATE,
                shard_axis=None,
                local_shard_range=None,
                extras=dict(extras),
            )
            continue

        yield MegatronAliasInput(
            name=name,
            tensor=tensor,
            role=spec.role,
            hf_names=tuple(hf_names),
            global_shape=tuple(geometry.global_shape),
            placement_kind=PLACEMENT_SHARD,
            shard_axis=int(geometry.shard_axis),
            local_shard_range=tuple(geometry.local_shard_range),
            extras=dict(extras),
        )


def publish_megatron_hf_aliases(
    *,
    manager: Any,
    rendezvous: Any,
    items: list[MegatronAliasInput],
    metadata_endpoint: str,
) -> tuple[str, list]:
    """Alias a registered Megatron layout as HF shards and publish it.

    Returns the source id and the published table, the latter so a caller can
    report byte and shard counts without rebuilding it.

    The caller owns both the NIXL manager and the rendezvous. Publishing starts
    the source's READY heartbeat, and only the rendezvous owner can stop it, so a
    rendezvous created in here would leave that thread alive with no handle and
    the source would go stale only at interpreter exit.
    """
    if not items:
        raise ValueError("no alias inputs to publish")

    published = build_hf_aliases(items, agent_name=str(manager.agent_name))
    source_id = publish_registered_shard_table(
        manager=manager,
        rendezvous=rendezvous,
        published=published,
        metadata_endpoint=metadata_endpoint,
    )
    return source_id, published


def published_byte_count(published: Iterable[Any]) -> int:
    """Bytes described by a published table, counting each shard once."""
    total = 0
    for tensor in published:
        for shard in tensor.shards:
            count = 1
            for dim in shard.shape:
                count *= int(dim)
            total += count * int(tensor.elsize)
    return total


__all__ = [
    "PLACEMENT_REPLICATE",
    "PLACEMENT_SHARD",
    "UnmappedMegatronTensor",
    "build_megatron_alias_inputs",
    "make_bridge_resolver",
    "publish_megatron_hf_aliases",
    "published_byte_count",
]
