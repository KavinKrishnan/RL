#!/bin/bash
# Functional smoke for ModelExpress reshard refit on one 2-GPU node:
# Megatron TP1 trainer (1 GPU) -> vLLM TP1 rollout (1 GPU), two GRPO steps.
#
# Requires the mcore + vllm extras, ModelExpress/NIXL, a reachable MX metadata
# server, and a 2-GPU allocation:
#   MX_SERVER_URL=host:8001 \
#     uv run --extra mcore --extra vllm \
#     bash tests/functional/grpo_model_express_refit.sh

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
EXP_NAME=$(basename "$0" .sh)
EXP_DIR="${SCRIPT_DIR}/${EXP_NAME}"
LOG_DIR="${EXP_DIR}/logs"
JSON_METRICS="${EXP_DIR}/metrics.json"
RUN_LOG="${EXP_DIR}/run.log"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export MX_SERVER_URL="${MX_SERVER_URL:-modelexpress-server.kavin.svc.cluster.local:8001}"
# The metric gates below are proxies: they detect a refit broken badly enough to
# move the logprobs. This asks the direct question instead, and is what makes the
# first-refit assertion at the bottom possible.
export MX_REFIT_VERIFY=1

rm -rf "${EXP_DIR}"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"
uv run coverage run -a \
    --data-file="${PROJECT_ROOT}/tests/.coverage" \
    --source="${PROJECT_ROOT}/nemo_rl" \
    "${PROJECT_ROOT}/examples/run_grpo.py" \
    --config "${PROJECT_ROOT}/examples/configs/grpo_math_1B_megatron.yaml" \
    policy.model_name=Qwen/Qwen3-0.6B \
    grpo.num_prompts_per_step=4 \
    grpo.num_generations_per_prompt=8 \
    policy.train_global_batch_size=16 \
    policy.train_micro_batch_size=1 \
    policy.logprob_batch_size=1 \
    policy.max_total_sequence_length=512 \
    policy.megatron_cfg.enabled=true \
    policy.megatron_cfg.tensor_model_parallel_size=1 \
    policy.megatron_cfg.pipeline_model_parallel_size=1 \
    policy.dtensor_cfg.enabled=false \
    policy.generation.backend=vllm \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.colocated.resources.gpus_per_node=1 \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    policy.generation.vllm_cfg.async_engine=true \
    ++policy.generation.refit_transport=model_express \
    ++policy.generation.refit_cfg.model_express.server_url="${MX_SERVER_URL}" \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=2 \
    grpo.max_num_steps=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir="${LOG_DIR}" \
    logger.wandb_enabled=false \
    checkpointing.enabled=false \
    "$@" \
    2>&1 | tee "${RUN_LOG}"

uv run tests/json_dump_tb_logs.py "${LOG_DIR}" --output_path "${JSON_METRICS}"

# A broken refit corrupts generation weights and makes the train/gen
# importance-sampling ratio explode.
#
# The grad_norm check keeps the ratio check from going vacuous. GRPO's
# leave-one-out baseline gives zero advantage to any prompt group whose rewards
# are all equal, so a run where the model never gets a mixed group trains
# nothing, the second refit re-sends bit-identical weights, and the ratio check
# passes for a model a no-op would also satisfy. 4 prompts x 8 generations over
# two steps is sized so that outcome is rare; at the observed ~1/8 solve rate
# for this model it is on the order of 1e-4.
#
# The KL-family check uses js_divergence_error rather than gen_kl_error. Both
# measure trainer-vs-rollout logprob disagreement, but that disagreement has a
# non-zero floor set by Megatron-vs-vLLM kernel differences, and the floor grows
# with model size. Measured with a refit that provably changed no parameter, the
# gen_kl_error floor is 8.7e-4 to 1.3e-3 on Qwen3-30B-A3B, so an absolute
# gen_kl_error < 1e-3 gate is unpassable there no matter how correct the refit is.
# js_divergence_error is bounded and symmetric, and one bound holds across every
# scale measured: 1.6e-4 here on 0.6B, 1.3e-4 on 4B dense, 5.0e-4 on 30B MoE.
uv run tests/check_metrics.py "${JSON_METRICS}" \
    'max(data["train/token_mult_prob_error"]) < 1.05' \
    'max(data["train/js_divergence_error"]) < 1e-3' \
    'max(data["train/grad_norm"]) > 0'

# vLLM loads from the HF checkpoint and the trainer converts from that same
# checkpoint, so before the first optimizer step the two already agree. The
# first refit must therefore change nothing; anything it does change is transport
# or conversion error. Absence of records is a failure, not a pass -- that is the
# state a silently disabled check produces.
uv run python - "${RUN_LOG}" <<'PY'
import json
import sys

records = [
    json.loads(line.split("MX_REFIT_VERIFY ", 1)[1])
    for line in open(sys.argv[1], errors="replace")
    if "MX_REFIT_VERIFY {" in line
]
first = [r for r in records if r["first_refit"]]
if not first:
    sys.exit("MX_REFIT_VERIFY produced no first-refit records; the check did not run")
bad = [r for r in first if r["params_changed"]]
if bad:
    sys.exit(
        f"first refit changed {bad[0]['params_changed']} of "
        f"{bad[0]['params_compared']} parameters, e.g. {bad[0]['changed_sample']}"
    )
print(f"MX_REFIT_VERIFY: {len(first)} rank(s) clean on the first refit")
PY
