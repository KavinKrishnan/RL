#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EP Megatron publisher for the reshard refit benchmark, on ModelExpress `main`.

Same job as ``ep_publisher.py`` — load the real Qwen3-30B-A3B MoE in Megatron-Core
at EP=N, one rank per GPU, and hold its native shards registered so a TP-k vLLM
receiver can plan, pull and reshard — but through MX `main`'s seam instead of
``MxV2TrainingPublisher``, which is not on `main` and is not coming (its PRs were
closed unmerged; see AugustMegeValidation/03_V2_SURFACE_IS_SUPERSEDED).

What changes relative to ``ep_publisher.py``:

* ``MxV2TrainingPublisher`` + ``add_tensor`` + ``publish(version=)`` becomes
  ``build_hf_aliases`` + ``publish_registered_shard_table``, via
  ``nemo_rl.distributed.mx_reshard_publisher``.
* Names published are **HF-canonical**, not native Megatron. `main` aliases
  native storage into HF names on the publisher side, so the receiver plans
  directly against the names vLLM's loader uses and there is no sidecar shape
  registry and no receiver-side translation.
* We own the NIXL registration and the rendezvous. Registration is one arena per
  rank rather than one per tensor, and the rendezvous is closed on shutdown so
  the source is marked stale instead of lingering until exit.

Launch (torchrun; EP == world size), after the MX server is reachable:
  EP=4, single node, 4 GPUs:
    MODEL_EXPRESS_URL=... torchrun --nproc_per_node=4 ep_publisher_reshard.py
  EP=8 over two 4-GPU nodes, rank0 node:
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
      --master_addr=<r0-ip> --master_port=29500 ep_publisher_reshard.py
"""
from __future__ import annotations

import os
import socket
import time
from collections import Counter

import torch
import torch.distributed as dist

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
EP = WORLD  # expert-parallel over the whole world (TP=PP=1)
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-30B-A3B-Instruct-2507")
MX_URL = os.environ.get(
    "MODEL_EXPRESS_URL", "modelexpress-server.kavin.svc.cluster.local:8001"
)
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "7200")) + LOCAL_RANK
READY_FILE = os.environ.get("READY_FILE")
STOP_FILE = os.environ.get("STOP_FILE")
# How long to hold the registered arenas after publishing. Zero or negative means
# hold until STOP_FILE appears, which is what a multi-run campaign needs: a finite
# hold silently ends the run mid-campaign, and because the ranks then exit within
# seconds of each other, the ones still alive see the rendezvous TCPStore drop and
# SIGABRT. The result reads as a publisher crash in the log while the actual cause
# was the timer expiring, and the receiver it abandoned simply hangs in handshake.
HOLD_S = int(os.environ.get("HOLD_S", "0"))

_NVIS = torch.cuda.device_count()
DEV_ID = LOCAL_RANK % _NVIS
torch.cuda.set_device(DEV_ID)

dist.init_process_group(backend="nccl", world_size=WORLD, rank=RANK)
from megatron.core import parallel_state  # noqa: E402

parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=EP,
)
ep_rank = parallel_state.get_expert_model_parallel_rank()
print(
    f"[pub r{RANK}] EP={EP} ep_rank={ep_rank} host={socket.gethostname()} "
    f"gpu={DEV_ID} (local_rank={LOCAL_RANK})",
    flush=True,
)

from modelexpress.client import MxClient  # noqa: E402
from modelexpress.nixl_transfer import NixlTransferManager  # noqa: E402
from modelexpress.refit.reshard.rendezvous import MxReshardRendezvous  # noqa: E402

# The NIXL agent is created BEFORE the model loads, and the order is load-bearing.
#
# Bringing it up after the Megatron weight load leaves UCX with no InfiniBand and
# no CUDA component: it reports `UCX_IB_GID_INDEX` and `UCX_CUDA_COPY_DMABUF` as
# "unused environment variables", warns that `mlx5_0:1..mlx5_3:1` "are not
# available" while offering only the tcp devices, and then, because UCX_TLS=^tcp
# excludes tcp, fails agent construction outright with
# "no active messages transport ... self/memory, sysv/memory, posix/memory" ->
# NIXL_ERR_BACKEND. Constructing the agent first, with everything else identical,
# loads both components and the same env vars are consumed. The devices are fine
# either way -- ibv_devinfo shows all four mlx5 ports ACTIVE before and after.
#
# Bringing the transport up first is also the better failure ordering: a fabric
# problem now costs seconds instead of surfacing after a 40 s model load, where it
# reads as a publisher fault.
client = MxClient(MX_URL)
manager = NixlTransferManager(
    agent_name=f"{socket.gethostname()}-ep{EP}-pub-r{ep_rank}",
    device_id=DEV_ID,
    listen_port=LISTEN_PORT,
)
# Constructing the manager does not create the NIXL agent; register_tensors
# raises "NIXL agent not initialized" without this. It also starts the listen
# thread the receiver's P2P handshake connects back to, so it must happen before
# the endpoint is advertised in the rendezvous blob.
manager.initialize()
print(f"[pub r{RANK}] NIXL agent up on device {DEV_ID}, port {LISTEN_PORT}", flush=True)

from megatron.bridge import AutoBridge  # noqa: E402

t0 = time.perf_counter()
bridge = AutoBridge.from_hf_pretrained(MODEL_ID, trust_remote_code=True)
provider = bridge.to_megatron_provider(load_weights=True)
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.expert_model_parallel_size = EP
provider.expert_tensor_parallel_size = 1
provider.bf16 = True
provider.gradient_accumulation_fusion = False
provider.sequence_parallel = False
provider.finalize()
model_list = provider.provide_distributed_model(wrap_with_ddp=False)
model = model_list[0] if isinstance(model_list, list) else model_list
print(f"[pub r{RANK}] model loaded EP={EP} in {time.perf_counter()-t0:.1f}s", flush=True)

from nemo_rl.distributed.mx_megatron_helpers import (  # noqa: E402
    collect_megatron_publish_set,
)
from nemo_rl.distributed.mx_reshard_publisher import (  # noqa: E402
    build_megatron_alias_inputs,
    make_bridge_resolver,
    publish_megatron_hf_aliases,
    published_byte_count,
)

# --- HF name map, from the Bridge's conversion tasks -------------------------
tasks = bridge.get_conversion_tasks([model])
name_map: dict[str, list[str]] = {}
for task in tasks:
    m_name = task.global_param_name or task.param_name
    hf_attr = getattr(task.mapping, "hf_param", None)
    if isinstance(hf_attr, str):
        hf_names = [hf_attr]
    elif isinstance(hf_attr, dict):
        # A fused QKV source maps to three HF tensors and must stay in q,k,v
        # order: build_hf_aliases assigns hf_names[0..2] to the Q, K and V row
        # bands of the fused parent, so dict order would silently transpose them.
        hf_names = (
            [hf_attr["q"], hf_attr["k"], hf_attr["v"]]
            if set(hf_attr.keys()) == {"q", "k", "v"}
            else list(hf_attr.values())
        )
    else:
        continue
    name_map[m_name] = hf_names

tcfg = getattr(bridge, "transformer_config", None) or provider
num_heads = getattr(tcfg, "num_attention_heads", None)
kv_groups = getattr(tcfg, "num_query_groups", None) or num_heads
hidden = getattr(tcfg, "hidden_size", None)
kv_channels = getattr(tcfg, "kv_channels", None) or (
    hidden // num_heads if num_heads else None
)
num_experts_total = getattr(tcfg, "num_moe_experts", None) or getattr(
    tcfg, "num_experts", None
)
num_local_experts = (int(num_experts_total) // EP) if num_experts_total else None
print(
    f"[pub r{RANK}] num_experts_total={num_experts_total} ep={EP} "
    f"num_local_experts={num_local_experts}",
    flush=True,
)

# --- Collect, pack into one registered arena, alias, publish -----------------
collected = list(
    collect_megatron_publish_set(
        model,
        tp_size=1,
        pp_size=1,
        pp_rank=0,
        ep_size=EP,
        ep_rank=ep_rank,
        tp_rank=0,
        num_local_experts=num_local_experts,
        num_attention_heads=num_heads,
        num_kv_heads=kv_groups,
        head_dim=kv_channels,
        target_dtype=torch.bfloat16,
    )
)

# One NIXL registration per rank instead of thousands. The aliases must describe
# the packed views, so pack first and rebuild the tuples around the copies.
ALIGN = 256
offsets: list[tuple[str, int, int]] = []
cursor = 0
for name, tensor, _spec, _extras in collected:
    cursor = (cursor + ALIGN - 1) // ALIGN * ALIGN
    nbytes = tensor.numel() * tensor.element_size()
    offsets.append((name, cursor, nbytes))
    cursor += nbytes
arena = torch.empty(cursor, dtype=torch.uint8, device=f"cuda:{DEV_ID}")
packed: list[tuple[str, torch.Tensor, object, dict]] = []
for (name, tensor, spec, extras), (_n, offset, nbytes) in zip(collected, offsets):
    view = arena.narrow(0, offset, nbytes).view(tensor.dtype).view(tensor.shape)
    view.copy_(tensor)
    packed.append((name, view, spec, extras))
del collected
torch.cuda.synchronize()
print(
    f"[pub r{RANK}] packed {len(packed)} tensors into a {cursor} byte arena",
    flush=True,
)

# Registering the packed views rather than the arena tensor: register_tensors
# builds the per-name descriptors the transport matches on, and under
# MX_POOL_REG=1 it collapses them to the one allocation the arena occupies, so
# this is a single ibv_reg_mr either way. Publishing happens against
# manager.nixl_metadata, which this call populates.
manager.register_tensors({name: view for name, view, _spec, _extras in packed})

resolver = make_bridge_resolver(name_map)
items = list(
    build_megatron_alias_inputs(
        packed,
        resolve_hf_names=resolver,
        tp_size=1,
        tp_rank=0,
        expert_tp_size=1,
        expert_tp_rank=0,
    )
)
roles: Counter = Counter(item.role for item in items)
print(f"[pub r{RANK}] {len(items)} alias inputs; roles={dict(roles)}", flush=True)

rendezvous = MxReshardRendezvous(
    client,
    role="trainer",
    rank=ep_rank,
    model_name=MODEL_ID,
    worker_id=f"trainer-ep{EP}-r{ep_rank}",
)
metadata_endpoint = f"{os.environ.get('POD_IP', socket.gethostbyname(socket.gethostname()))}:{LISTEN_PORT}"

try:
    source_id, published = publish_megatron_hf_aliases(
        manager=manager,
        rendezvous=rendezvous,
        items=items,
        metadata_endpoint=metadata_endpoint,
    )
    print(
        f"[pub r{RANK}] published ep_rank={ep_rank} sid={source_id} "
        f"tensors={len(published)} bytes={published_byte_count(published)} "
        f"endpoint={metadata_endpoint} READY",
        flush=True,
    )
    if READY_FILE:
        # Per-rank path. Every rank publishes its own source, so one shared file
        # would report whichever rank wrote last and a launcher waiting on it
        # would start the receiver while other ranks are still loading -- which
        # surfaces as a rendezvous quorum timeout, not as a missing publisher.
        with open(f"{READY_FILE}.r{ep_rank}", "w") as handle:
            handle.write(f"{source_id}\n")

    # No collective after publishing: ranks publish independently and the
    # receiver discovers them all from the MX server, so a barrier here only
    # risks tearing ranks down while they must stay alive holding their agents.
    deadline = None if HOLD_S <= 0 else time.time() + HOLD_S
    while deadline is None or time.time() < deadline:
        if STOP_FILE and os.path.exists(STOP_FILE):
            print(f"[pub r{RANK}] stop file seen", flush=True)
            break
        time.sleep(2)
    if deadline is not None and time.time() >= deadline:
        print(
            f"[pub r{RANK}] HOLD_S={HOLD_S}s elapsed; exiting while a receiver may "
            f"still need these sources",
            flush=True,
        )
finally:
    rendezvous.close()
    print(f"[pub r{RANK}] rendezvous closed, source marked stale", flush=True)
    # Tear the process group down before exiting. Ranks leave within seconds of each
    # other, and a rank that is still in its shutdown path treats the rendezvous
    # store disappearing as a fatal error and aborts, which buries the real reason
    # for the exit under pages of SIGABRT frames on every other rank.
    if dist.is_initialized():
        dist.destroy_process_group()
