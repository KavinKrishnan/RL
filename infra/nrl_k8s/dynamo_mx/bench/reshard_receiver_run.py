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
            # update_weights returns its own stage timings and byte economics.
            # That return value is the only instrument that covers this path: the
            # reshard receiver does not write into the ambient RefitTimingRecorder
            # (only the vLLM MDL installer does), so reading the recorder's
            # wire_transfer/installation stages reports a confident 0.0 ms for
            # every stage of a refit that plainly took seconds. The recorder stays
            # in scope because the installer leg does use it.
            metrics = recv.update_weights(step)
            torch.cuda.synchronize()
        totals.append((time.perf_counter() - t0) * 1e3)
        rec.finish()
        records.append({"mx": metrics, "recorder": rec.as_dict()})
        step += 1
    worker._refit_step = step

    def ms(metrics, *names):
        return sum(float(metrics.get(n, 0.0)) for n in names) * 1e3

    def per_cycle(fn):
        return [fn(r["mx"]) for r in records]

    # The wire leg is reported under different keys depending on whether the fused
    # single-transfer path ran, so sum whichever are present instead of naming one.
    def wire_ms(metrics):
        return (
            sum(
                float(v)
                for k, v in metrics.items()
                if k.startswith("wire_") and k.endswith("_s")
            )
            * 1e3
        )

    # Only duration keys end in _s; counts such as reslice_copies do not, so this
    # stays a time sum. The stages are sequential on this path, so summing them is
    # valid -- revisit if any leg is ever overlapped.
    def accounted_ms(metrics):
        return sum(float(v) for k, v in metrics.items() if k.endswith("_s")) * 1e3

    e2e = totals
    acc = per_cycle(accounted_ms)
    bytes_planned = recv._plan.bytes_planned() if recv._plan is not None else 0
    # What this rank's parameters actually occupy. This is the honest denominator for
    # amplification: MX's extra_wire_bytes counts only the duplication it attributes
    # to replicated offers, so wire minus extra is not "the bytes the model needed" --
    # a full-pulled source that is sliced locally moves bytes that are discarded and
    # are not counted as extra. Measuring the engine's own footprint separates the two.
    engine_param_bytes = sum(
        p.numel() * p.element_size()
        for p in worker.model_runner.model.parameters()
    )
    return {
        "e2e_ms": e2e,
        "transfer_ms": per_cycle(wire_ms),
        # Re-slicing and dtype conversion are the transformation leg here; MX keeps
        # them separate, and both are reported separately below as well.
        "quantization_ms": per_cycle(lambda m: ms(m, "reslice_s", "convert_s")),
        "reslice_ms": per_cycle(lambda m: ms(m, "reslice_s")),
        "convert_ms": per_cycle(lambda m: ms(m, "convert_s")),
        "install_ms": per_cycle(lambda m: ms(m, "install_s")),
        "accounted_ms": acc,
        "unattributed_ms": [t - a for t, a in zip(e2e, acc)],
        "attribution_pct": [100.0 * a / t if t else 0.0 for t, a in zip(e2e, acc)],
        "bytes_planned": bytes_planned,
        "engine_param_bytes": engine_param_bytes,
        "bytes_received": per_cycle(lambda m: float(m.get("bytes_received", 0))),
        "extra_wire_bytes": per_cycle(lambda m: float(m.get("extra_wire_bytes", 0))),
        "wire_gbps": [
            (b * 8.0 / (w / 1e3) / 1e9) if w > 0 else 0.0
            for b, w in zip(
                per_cycle(lambda m: float(m.get("bytes_received", 0))),
                per_cycle(wire_ms),
            )
        ],
        "fallback": per_cycle(lambda m: float(m.get("fallback", 0))),
        "converts": per_cycle(lambda m: float(m.get("converts", 0))),
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


def _crit_gbps(rank_lists, key):
    """Fleet-critical throughput, which is the slowest rank rather than the fastest.

    _crit takes the per-cycle max across ranks because for a duration the fleet waits
    on the slowest rank. Reusing it for a rate inverts the meaning: the max Gbps is
    the *best* rank, and a refit is not finished until the worst one is. So take the
    min across ranks per cycle, and label the fields as rates, not milliseconds.
    """
    per = [r.get(key) or [] for r in rank_lists]
    if not any(per):
        return None
    n = min(len(x) for x in per)
    crit = [min(per[r][i] for r in range(len(per))) for i in range(n)]
    s = sorted(crit)
    return {
        "n": len(s),
        "min_gbps": min(s),
        "median_gbps": statistics.median(s),
        "max_gbps": max(s),
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
        # Must be keys of _LOAD_MODE. The previous default named an arm
        # ("quantizing_mdl") that _LOAD_MODE never had, so any run using the default
        # aborted after completing its cycles -- late enough to have already paid for
        # the transfers and thrown away the timings.
        default="pwal,mdl",
        help=f"comma-separated installer arms, from {sorted(_LOAD_MODE)}",
    )
    ap.add_argument(
        "--trainer-topology",
        default=None,
        help="publisher arm as run, e.g. 'Megatron-EP4'; recorded verbatim",
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

    # Validate arm names here rather than inside the worker. The worker-side check
    # only runs once a cycle starts, so a typo costs a full model load and a real
    # transfer before it is reported, and reports it as a worker crash.
    requested = [a for a in (s.strip() for s in args.installers.split(",")) if a]
    unknown = [a for a in requested if a not in _LOAD_MODE]
    if unknown:
        raise SystemExit(
            f"unknown installer arm(s) {unknown}; expected from {sorted(_LOAD_MODE)}"
        )

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
        # Recorded verbatim from the launcher: the receiver cannot see which trainer
        # backend published, and the old hardcoded "FSDP{n}+EP{n}" mislabelled every
        # Megatron run as FSDP -- exactly the arm confusion the reporting rules forbid.
        "trainer_topology": args.trainer_topology or f"EP{args.num_trainers}",
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
        modes = sorted(
            {
                mode
                for rank_result in refit
                for mode in rank_result.get("selected_modes", [])
            }
        )

        def med(values):
            ordered = sorted(values)
            return ordered[len(ordered) // 2] if ordered else 0.0

        # Byte economics per rank, measured rather than predicted. wire minus extra
        # is the payload the model actually needed; extra is the duplication the
        # published plan carries.
        wire_bytes = [med(r.get("bytes_received", [0])) for r in refit]
        extra_bytes = [med(r.get("extra_wire_bytes", [0])) for r in refit]
        attribution = [med(r.get("attribution_pct", [0])) for r in refit]
        # An UPPER BOUND on fleet rate, not a measurement, and named so. Dividing the
        # sum of both ranks' bytes by a median wire duration silently assumes the ranks
        # transferred in the same wall-clock window. They do not: the ranks are driven
        # by separate collective_rpc calls and their wire legs stagger. Reported as a
        # measurement this produced 2463 Gbps on a node whose fabric is 1600 Gbps,
        # which reads as a transport fault and is really just this arithmetic. Per-rank
        # rates below are sound -- each divides one rank's bytes by that rank's own
        # duration -- and the honest fleet number needs timestamps we do not collect.
        aggregate_upper_bound_gbps = None
        if transfer and transfer["median_ms"] > 0:
            aggregate_upper_bound_gbps = (
                sum(wire_bytes) * 8.0 / (transfer["median_ms"] / 1e3) / 1e9
            )
        rec["arms"][installer_name] = {
            "selected_modes": modes,
            "bytes_planned_per_rank": bytes_per_rank,
            "wire_bytes_per_rank": wire_bytes,
            # MX's own duplication figure: bytes attributed to replicated offers.
            "extra_wire_bytes_per_rank": extra_bytes,
            "wire_minus_reported_extra_per_rank": [
                w - e for w, e in zip(wire_bytes, extra_bytes)
            ],
            # What the rank's parameters occupy, which is the floor a perfect refit
            # would move. Amplification is measured against this rather than against
            # wire minus extra, because full-pulled sources move bytes that are
            # sliced away and are not counted in extra_wire_bytes.
            "engine_param_bytes_per_rank": [
                r.get("engine_param_bytes", 0) for r in refit
            ],
            "amplification_vs_reported_extra_pct": [
                100.0 * e / (w - e) if (w - e) > 0 else 0.0
                for w, e in zip(wire_bytes, extra_bytes)
            ],
            "amplification_vs_engine_pct": [
                100.0 * (w - n) / n if n > 0 else 0.0
                for w, n in zip(
                    wire_bytes, [r.get("engine_param_bytes", 0) for r in refit]
                )
            ],
            "aggregate_wire_gbps_upper_bound": aggregate_upper_bound_gbps,
            "node_fabric_gbps": 1600.0,
            "per_rank_wire_gbps": _crit_gbps(refit, "wire_gbps"),
            "transfer": transfer,
            "install": install,
            "quantization": quantization,
            "reslice": _crit(refit, "reslice_ms"),
            "convert": _crit(refit, "convert_ms"),
            "e2e": e2e,
            "accounted": _crit(refit, "accounted_ms"),
            "unattributed": _crit(refit, "unattributed_ms"),
            # The reporting rules only accept a breakdown at >=95% attribution or
            # <=100 ms unattributed, so the row carries its own admissibility.
            "attribution_pct_per_rank": attribution,
            "attribution_ok": all(
                a >= 95.0 for a in attribution
            ) or all(
                u <= 100.0
                for r in refit
                for u in r.get("unattributed_ms", [])
            ),
            "fallback_total": sum(
                sum(r.get("fallback", [])) for r in refit
            ),
            "converts_total": sum(sum(r.get("converts", [])) for r in refit),
            # MX's own per-cycle metrics, verbatim and per rank. The aggregates above
            # are derived from these, and the byte-accounting categories MX reports
            # (full_pull_sources, unbounded_sources, descriptor_savings, segments)
            # are not all represented in them. Without this the JSON cannot answer a
            # question that was not anticipated when the aggregates were chosen, and
            # the run has to be repeated to ask it.
            "mx_metrics_per_rank": [
                [record["mx"] for record in r.get("stage_records", [])]
                for r in refit
            ],
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
