#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check that an MX metadata server can carry MX `main`'s reshard rendezvous.

Runs the three RPCs the reshard path depends on -- ``publish_metadata``,
``list_sources``, ``get_metadata`` -- as a publish-then-discover round trip, and
checks the shard table survives the round trip intact. CPU only, seconds, no
model and no GPU.

Worth its own script because the alternative way to discover a server mismatch is
a 20-minute Megatron load followed by a rendezvous timeout, which reads as a
publisher fault. The servers in this namespace predate MX `main`: the deployment
carrying the earliest reshard work did not implement ``ListSources`` at all, and a
publisher against that server simply never becomes discoverable.

Usage (inside any pod with the MX client):
  python3 mx_server_handshake_check.py --mx-server host:8001 [--model NAME]
"""
from __future__ import annotations

import argparse
import socket
import sys
import uuid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mx-server", required=True)
    parser.add_argument("--model", default="handshake-check")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    from modelexpress.client import MxClient
    from modelexpress.refit.reshard.rendezvous import (
        MxReshardRendezvous,
        PublishedShard,
        PublishedTensor,
        wrap_rendezvous_blob,
    )

    # A unique model name per invocation. Sharing one would let a previous run's
    # READY source satisfy discovery, so a server that cannot publish at all
    # would still look healthy.
    model = f"{args.model}-{uuid.uuid4().hex[:8]}"
    client = MxClient(server_url=args.mx_server)

    # Two shards of one tensor, so the check also covers the nested shard list
    # rather than only the outer record.
    tensors = [
        PublishedTensor(
            name="model.layers.0.self_attn.q_proj.weight",
            dtype="torch.bfloat16",
            elsize=2,
            full_shape=(4096, 2048),
            shards=[
                PublishedShard(
                    agent_name="handshake-agent",
                    device_id=0,
                    addr=0xDEADBEEF,
                    shard_offset=(0, 0),
                    shape=(2048, 2048),
                    digest="abc123",
                ),
                PublishedShard(
                    agent_name="handshake-agent",
                    device_id=0,
                    addr=0xDEADBEEF + 2048 * 2048 * 2,
                    shard_offset=(2048, 0),
                    shape=(2048, 2048),
                    digest="def456",
                ),
            ],
        )
    ]
    blob = wrap_rendezvous_blob(
        agent_metadata=b"not-a-real-nixl-agent",
        agent_name="handshake-agent",
        metadata_endpoint=f"{socket.gethostname()}:7200",
        tensors=tensors,
        publisher_step=7,
    )

    pub = MxReshardRendezvous(
        client, role="trainer", rank=0, model_name=model, worker_id="handshake-pub"
    )
    source_id = pub.publish(blob)
    print(f"  publish_metadata OK: source_id={source_id}")

    try:
        sub = MxReshardRendezvous(
            client, role="inference", rank=0, model_name=model, worker_id="handshake-sub"
        )
        payloads = sub.discover_trainers(1, timeout=args.timeout, poll_interval=1.0)
        print(f"  list_sources + get_metadata OK: discovered {len(payloads)} trainer(s)")

        got = payloads[0]
        # The shard table is what the receiver plans against, so a server that
        # truncates or re-encodes nixl_metadata would produce a plan for the wrong
        # geometry rather than an error.
        if got.agent_name != "handshake-agent":
            print(f"  FAIL agent_name round trip: {got.agent_name!r}")
            return 2
        if len(got.tensors) != 1 or got.tensors[0].name != tensors[0].name:
            print(f"  FAIL shard table round trip: {got.tensors}")
            return 3
        rt = got.tensors[0]
        if tuple(rt.full_shape) != (4096, 2048) or len(rt.shards) != 2:
            print(f"  FAIL geometry round trip: shape={rt.full_shape} shards={len(rt.shards)}")
            return 4
        if [s.digest for s in rt.shards] != ["abc123", "def456"]:
            print(f"  FAIL digest round trip: {[s.digest for s in rt.shards]}")
            return 4
        if tuple(rt.shards[1].shard_offset) != (2048, 0):
            print(f"  FAIL shard offset round trip: {rt.shards[1].shard_offset}")
            return 4
        if got.publisher_step != 7:
            print(f"  FAIL publisher_step round trip: {got.publisher_step}")
            return 5

        print(f"  shard table intact: step={got.publisher_step} endpoint={got.metadata_endpoint}")
        print("RESULT: PASS - server carries MX main's reshard rendezvous")
        return 0
    except TimeoutError as exc:
        # The likely reading of a timeout here is an old server, since publish
        # already succeeded and nothing else is in the path.
        print(f"  FAIL discover_trainers: {exc}")
        print("RESULT: FAIL - server accepted publish but never made it discoverable")
        return 6
    finally:
        pub.close()


if __name__ == "__main__":
    sys.exit(main())
