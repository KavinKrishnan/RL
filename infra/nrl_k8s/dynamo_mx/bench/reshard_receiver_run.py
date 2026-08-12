#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inference-side runner for the reshard-refit E2E benchmark.

Loads Qwen3-30B in vLLM (TP=k), builds a VllmReshardReceiver per rank, and
measures a real inter-node RDMA refit: discover -> P2P handshake -> no-gather
RDMA pull -> install. Splits per-cycle transfer vs install, and times BOTH
installers (PWAL, meaning reshard-refit without MDL, and MDL) on the same pulled
buffers so the only difference is the install seam.

Correctness: corrupt the live model, run one refit, confirm greedy generation
recovers vs the pre-corruption baseline.

Usage (receiver pod, k GPUs), after the publisher is READY:
  MX_POOL_REG=1 python3 reshard_receiver_run.py \
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 --tp 2 \
      --mx-server modelexpress-server.kavin.svc.cluster.local:8001 \
      --warm-cycles 10 --out /mnt/rl-workspace/kavink/reshard_e2e/tp2.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

PROMPTS = ["The capital of France is", "def add(a, b):\n    return"]
_BASE_PORT = 7300


def w_build(
    worker,
    rendezvous_name,
    mx_server,
    num_trainers,
    capture_layout,
    capture_cache_path,
):
    import torch

    from modelexpress.engines.vllm.refit.receiver import VllmReshardReceiver

    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        rank = int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001
        rank = 0
    dev = next(worker.model_runner.model.parameters()).device
    vc = getattr(worker, "vllm_config", None) or getattr(
        worker.model_runner, "vllm_config", None
    )
    # MX main's receiver takes no installer/capture kwargs: the install path is
    # fixed and MDL is selected per refit through MX_LOAD_MODE (see w_refit_timed).
    recv = VllmReshardReceiver(
        model=worker.model_runner.model,
        vllm_config=vc,
        model_config=vc.model_config,
        model_name=rendezvous_name,
        mx_server=mx_server,
        agent_name=f"infer-{rank}",
        local_rank=dev.index,
        global_rank=rank,
        num_trainer_sources=num_trainers,
        device=dev,
        listen_port=_BASE_PORT + rank,
    )
    worker._recv = recv
    # snapshot reference (post-load correct params) on CPU
    worker._ref = {
        n: p.detach().to("cpu", copy=True)
        for n, p in worker.model_runner.model.named_parameters()
    }
    return {"rank": rank, "device": str(dev)}


_LOAD_MODE = {"pwal": "stock", "mdl": "direct"}


def w_refit_timed(worker, n, installer_name):
    """Run one install arm and split wire, install, transformation and E2E.

    The per-stage split comes from MX's own RefitTimingRecorder rather than from
    wrapping ``_install``: main records the whole vocabulary (wire_transfer,
    installation, transformation, ...) plus unattributed time, which is what the
    benchmarking rules ask us to report and is finer than a wrapper can see.
    """
    import torch

    from modelexpress.refit.timing import RefitTimingRecorder, use_refit_timing

    if installer_name not in _LOAD_MODE:
        raise ValueError(
            f"unknown install arm {installer_name!r}; expected one of {sorted(_LOAD_MODE)}"
        )
    os.environ["MX_LOAD_MODE"] = _LOAD_MODE[installer_name]

    recv = worker._recv
    step = int(getattr(worker, "_refit_step", 0))
    if recv._plan is None:
        recv.update_weights(step)  # cold prepare + first transfer/install
        step += 1
    recv.update_weights(step)  # unreported warm-up for this arm
    step += 1

    totals = []
    records = []
    for _ in range(n):
        rec = RefitTimingRecorder(
            backend="reshard-nixl", version=step, rank=recv._global_rank
        )
        t0 = time.perf_counter()
        with use_refit_timing(rec):
            recv.update_weights(step)
            torch.cuda.synchronize()
        totals.append((time.perf_counter() - t0) * 1e3)
        rec.finish()
        records.append(rec.as_dict())
        step += 1
    worker._refit_step = step

    def stage(name):
        return [r["stages"][name]["duration_ms"] for r in records]

    bytes_planned = recv._plan.bytes_planned() if recv._plan is not None else 0
    return {
        "e2e_ms": totals,
        "install_ms": stage("installation"),
        "quantization_ms": stage("transformation"),
        "transfer_ms": stage("wire_transfer"),
        "unattributed_ms": [r["unattributed_ms"] for r in records],
        "bytes_planned": bytes_planned,
        "selected_modes": [_LOAD_MODE[installer_name]] * len(records),
        "stage_records": records,
    }


def w_corrupt(worker):
    import torch

    with torch.no_grad():
        for p in worker.model_runner.model.parameters():
            try:
                p.data.view(torch.uint8).random_(0, 256)
            except Exception:  # noqa: BLE001
                p.data.normal_(0.0, 0.5)
    torch.cuda.synchronize()
    return {"ok": True}


def w_refit_once(worker):
    worker._recv.update_weights(9999)
    return {"ok": True}


def w_param_equality(worker, rtol, atol):
    """Compare live params against the pre-refit snapshot, per parameter.

    Phase 1 wants parameter equality and not just generation agreement: a
    mis-mapped slice can leave tokens intact while a minority of params hold the
    wrong bytes. worker._ref is the post-load reference captured in w_setup.
    """
    import torch

    worst_name, worst_err, mismatched = None, 0.0, []
    with torch.no_grad():
        live = dict(worker.model_runner.model.named_parameters())
        for name, ref in worker._ref.items():
            got = live.get(name)
            if got is None:
                mismatched.append({"name": name, "reason": "missing"})
                continue
            cur = got.detach().to("cpu", copy=False).to(ref.dtype)
            if cur.shape != ref.shape:
                mismatched.append(
                    {
                        "name": name,
                        "reason": "shape",
                        "ref": list(ref.shape),
                        "got": list(cur.shape),
                    }
                )
                continue
            err = float((cur.float() - ref.float()).abs().max())
            if err > worst_err:
                worst_name, worst_err = name, err
            if not torch.allclose(cur.float(), ref.float(), rtol=rtol, atol=atol):
                mismatched.append({"name": name, "reason": "value", "max_abs_err": err})
    return {
        "params_compared": len(worker._ref),
        "max_abs_err": worst_err,
        "max_abs_err_param": worst_name,
        "num_mismatched": len(mismatched),
        # Bounded so a wholesale failure cannot flood the record.
        "mismatched": mismatched[:20],
    }


def _gen(llm):
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=24)
    return [tuple(o.outputs[0].token_ids) for o in llm.generate(PROMPTS, sp, use_tqdm=False)]


def _agree(a, b):
    tot = ok = 0
    for x, y in zip(a, b):
        m = min(len(x), len(y))
        tot += max(len(x), len(y))
        ok += sum(1 for i in range(m) if x[i] == y[i])
    return ok / tot if tot else 0.0


def _crit(rank_lists, key):
    per = [r.get(key) or [] for r in rank_lists]
    if not any(per):
        return None
    n = min(len(x) for x in per)
    crit = [max(per[r][i] for r in range(len(per))) for i in range(n)]
    s = sorted(crit)
    return {
        "n": len(s),
        "min_ms": min(s),
        "median_ms": statistics.median(s),
        "p95_ms": s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))],
        "max_ms": max(s),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="vLLM model to load (bf16 id or fp8 dir)")
    ap.add_argument("--rendezvous-name", required=True, help="shared model_name matching the publisher")
    ap.add_argument("--dtype", choices=["bf16", "fp8"], default="bf16")
    ap.add_argument("--source-dtype", choices=["bf16", "fp8"], default="bf16")
    ap.add_argument(
        "--quantization",
        default=None,
        help="vLLM online quantization method (for example 'fp8')",
    )
    ap.add_argument(
        "--capture-layout",
        choices=["load_time", "runtime", "live", "cache"],
        default="load_time",
    )
    ap.add_argument(
        "--capture-cache-path",
        default=None,
        help="per-rank pickle path template; may contain {rank}",
    )
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--enable-expert-parallel", action="store_true")
    ap.add_argument("--num-trainers", type=int, default=8)
    ap.add_argument(
        "--installers",
        default="pwal,quantizing_mdl",
        help="comma-separated installer arms",
    )
    ap.add_argument("--mx-server", required=True)
    ap.add_argument("--warm-cycles", type=int, default=10)
    # bf16 refit is a byte copy, so the default is exact equality; loosen only
    # for a quantized destination where PWAL re-derives scales.
    ap.add_argument("--equality-rtol", type=float, default=0.0)
    ap.add_argument("--equality-atol", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    from vllm import LLM

    is_moe = "A3B" in args.model or "moe" in args.model.lower()
    kw = dict(
        model=args.model,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=float(os.environ.get("MX_GPU_MEM_UTIL", "0.45")),
        max_model_len=1024,
        trust_remote_code=True,
    )
    if args.dtype == "bf16":
        kw["dtype"] = "bfloat16"
    if args.quantization:
        kw["quantization"] = args.quantization
    if is_moe:
        kw["moe_backend"] = "triton"
        if args.enable_expert_parallel:
            kw["enable_expert_parallel"] = True
    print(f"[recv] loading {args.model} tp={args.tp}", flush=True)
    llm = LLM(**kw)

    rec = {
        "run_id": f"reshard-e2e-tp{args.tp}-{int(time.time())}",
        "model": args.model,
        "rendezvous_name": args.rendezvous_name,
        "source_dtype": args.source_dtype,
        "target_dtype": args.dtype,
        "quantization_method": args.quantization,
        "trainer_topology": f"FSDP{args.num_trainers}+EP{args.num_trainers}",
        "trainer_gpus": args.num_trainers,
        "inference_topology": (
            f"TP{args.tp}+EP{args.tp}"
            if args.enable_expert_parallel
            else f"TP{args.tp}+EP1"
        ),
        "generator_gpus": args.tp,
        "tp": args.tp,
        "warm_cycles": args.warm_cycles,
        "transport": "NIXL/RDMA inter-node",
    }
    base = _gen(llm)
    llm.collective_rpc(
        w_build,
        args=(
            args.rendezvous_name,
            args.mx_server,
            args.num_trainers,
            args.capture_layout,
            args.capture_cache_path,
        ),
    )
    rec["arms"] = {}
    last_arm = None
    for installer_name in [
        item.strip() for item in args.installers.split(",") if item.strip()
    ]:
        refit = llm.collective_rpc(
            w_refit_timed, args=(args.warm_cycles, installer_name)
        )
        transfer = _crit(refit, "transfer_ms")
        install = _crit(refit, "install_ms")
        quantization = _crit(refit, "quantization_ms")
        e2e = _crit(refit, "e2e_ms")
        bytes_per_rank = [r["bytes_planned"] for r in refit]
        total_bytes = sum(bytes_per_rank)
        aggregate_gbps = None
        if transfer and transfer["median_ms"] > 0:
            aggregate_gbps = (
                total_bytes * 8.0 / (transfer["median_ms"] / 1e3) / 1e9
            )
        modes = sorted(
            {
                mode
                for rank_result in refit
                for mode in rank_result.get("selected_modes", [])
            }
        )
        rec["arms"][installer_name] = {
            "selected_modes": modes,
            "bytes_planned_per_rank": bytes_per_rank,
            "aggregate_wire_gbps": aggregate_gbps,
            "transfer": transfer,
            "install": install,
            "quantization": quantization,
            "e2e": e2e,
        }
        last_arm = installer_name

    # Correctness gates the final selected arm after an actual transfer/install.
    llm.collective_rpc(w_corrupt, args=())
    corrupt_tok = _gen(llm)
    llm.collective_rpc(w_refit_once, args=())
    post = _gen(llm)
    equality = llm.collective_rpc(
        w_param_equality, args=(args.equality_rtol, args.equality_atol)
    )
    params_equal = all(r["num_mismatched"] == 0 for r in equality)
    rec["correctness"] = {
        "installer": last_arm,
        "corruption_detected": _agree(base, corrupt_tok) < 0.999,
        "recovery_tokens": _agree(base, post),
        "params_equal": params_equal,
        "param_equality_per_rank": equality,
    }
    rec["result"] = (
        "PASS"
        if rec["correctness"]["corruption_detected"]
        and rec["correctness"]["recovery_tokens"] >= 0.999
        and params_equal
        else "FAIL"
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2), flush=True)
    print(f"[out] {args.out} RESULT={rec['result']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
