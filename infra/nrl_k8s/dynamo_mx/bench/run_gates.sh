#!/usr/bin/env bash
# Drive gates 3 to 5 of the MX-main validation plan on the GCP GB200 cluster.
#
# Each gate is a subcommand so a failure can be re-run without repeating the ones
# before it, and so the long ones can be started and left. The publisher pods hold a
# loaded Megatron model and a registered NIXL arena, which takes minutes to rebuild,
# so publish and refit are deliberately separate steps rather than one script.
#
#   ./run_gates.sh pods            # apply the pub-1 and staging-receiver pods
#   ./run_gates.sh stage           # copy the harness into every pod
#   ./run_gates.sh publish8        # launch the EP8 publisher across both pub pods
#   ./run_gates.sh gate3           # EP8 -> TP2 correctness on the main image
#   ./run_gates.sh gate4           # EP8 -> TP2 timing, 10 measured cycles, main image
#   ./run_gates.sh gate5 <arm>     # staging image: baseline|cache|batch|spread
#   ./run_gates.sh collect         # pull every result JSON into the evidence dir
#
# Requires a live Teleport session; run `tsh kube login dynamo-gcp-dev-02` first.
set -euo pipefail

CTX=nv-prd-dgxc.teleport.sh-dynamo-gcp-dev-02
NS=kavin
HARNESS=/home/kavink/Work/Github/RL/rl-mx-harness
BENCH=$HARNESS/infra/nrl_k8s/dynamo_mx/bench
EVIDENCE=/home/kavink/Work/MX_RL_DESIGN_HOME/AugustMegeValidation/evidence
OUTDIR=/mnt/rl-workspace/kavink/mxmain_validation
MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
MXSERVER=miles-modelexpress-server.kavin.svc.cluster.local:8001

PUB0=mxmain-mega-pub-0
PUB1=mxmain-mega-pub-1
RECV_MAIN=mxmain-reshard-recv
RECV_STAGING=mxstaging-reshard-recv

k() { kubectl --context=$CTX -n $NS "$@"; }

RECVPY=/opt/dynamo/venv/bin/python

# The publisher needs the venv that can import megatron.bridge AND modelexpress,
# which is the Megatron Ray actor's, not the driver's. The actor venv is named after
# the worker class, so it is long and version-dependent; probe for it rather than
# hardcode it, because the failure mode of guessing is an ImportError several minutes
# into a model load.
pubpy() {
  local pod=$1
  k exec "$pod" -- bash -lc '
    for py in /opt/ray_venvs/*MegatronPolicyWorker/bin/python3 \
              /opt/ray_venvs/*MegatronPolicyWorker/bin/python \
              /opt/nemo_rl_venv/bin/python /usr/bin/python3; do
      [ -x "$py" ] || continue
      if "$py" -c "import megatron.bridge, modelexpress" >/dev/null 2>&1; then
        echo "$py"; exit 0
      fi
    done
    echo "no venv with both megatron.bridge and modelexpress" >&2; exit 1' | tr -d '\r'
}

need_pod() {
  local pod=$1
  local phase
  phase=$(k get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null || true)
  if [[ "$phase" != "Running" ]]; then
    echo "pod $pod is '$phase', not Running" >&2
    exit 1
  fi
}

cmd_pods() {
  k apply -f "$BENCH/configs/mxmain_megatron_pub1.gb200.yaml"
  k apply -f "$BENCH/configs/mxstaging_reshard_recv.gb200.yaml"
  echo "waiting for pods to be Running"
  k wait --for=condition=Ready pod/$PUB1 --timeout=900s
  k wait --for=condition=Ready pod/$RECV_STAGING --timeout=900s
  k get pods -o wide
}

cmd_stage() {
  # The adapter and its helpers live in the trainer image's nemo_rl package, so they
  # are copied over the installed module rather than into /tmp: the publisher imports
  # nemo_rl.distributed.mx_reshard_publisher by package path.
  local pkg py
  for pod in $PUB0 $PUB1; do
    need_pod "$pod"
    py=$(pubpy "$pod")
    echo "$pod publisher interpreter: $py"
    pkg=$(k exec "$pod" -- "$py" -c 'import nemo_rl,os;print(os.path.dirname(nemo_rl.__file__))' | tr -d '\r')
    k cp "$HARNESS/nemo_rl/distributed/mx_reshard_publisher.py" "$pod:$pkg/distributed/mx_reshard_publisher.py"
    k cp "$HARNESS/nemo_rl/distributed/mx_megatron_helpers.py" "$pod:$pkg/distributed/mx_megatron_helpers.py"
    k cp "$BENCH/ep_publisher_reshard.py" "$pod:/tmp/ep_publisher_reshard.py"
    echo "staged $pod ($pkg)"
  done
  for pod in $RECV_MAIN $RECV_STAGING; do
    need_pod "$pod"
    k cp "$BENCH/reshard_receiver_run.py" "$pod:/tmp/reshard_receiver_run.py"
    echo "staged $pod"
  done
}

cmd_publish8() {
  need_pod $PUB0
  need_pod $PUB1
  local master
  master=$(k get pod $PUB0 -o jsonpath='{.status.podIP}')
  echo "EP8 publisher, master_addr=$master"
  # Rank 0's node must be up before the other joins, but torchrun's rendezvous
  # tolerates either order, so both are launched without a sleep between them.
  local node=0 py
  for pod in $PUB0 $PUB1; do
    py=$(pubpy "$pod")
    k exec "$pod" -- bash -lc "
      rm -f /tmp/pub.ready.r* /tmp/pub.stop /tmp/pub.log
      cd /tmp
      nohup env MODEL_EXPRESS_URL=$MXSERVER READY_FILE=/tmp/pub.ready \
        $py -m torch.distributed.run \
          --nnodes=2 --node_rank=$node --nproc_per_node=4 \
          --master_addr=$master --master_port=29500 \
          /tmp/ep_publisher_reshard.py > /tmp/pub.log 2>&1 &
      echo 'launched node_rank=$node'"
    node=$((node + 1))
  done
  echo "publishing; watch with: $0 publog"
}

cmd_publog() {
  for pod in $PUB0 $PUB1; do
    echo "===== $pod"
    k exec "$pod" -- bash -lc 'tail -25 /tmp/pub.log; echo "--- ready files:"; ls -1 /tmp/pub.ready.r* 2>/dev/null || echo none'
  done
}

# Gate 3 and 4 differ only in cycle count and which checks gate the row, so they
# share one runner. num_trainers is 8 because the receiver plans against 8 sources.
run_refit() {
  local pod=$1 py=$2 out=$3 cycles=$4
  shift 4
  need_pod "$pod"
  # The log is named after the output, not reused as /tmp/recv.log. The stage records
  # MX emits at WARNING are only in the log, so overwriting it discards the raw
  # evidence for the previous gate -- which is exactly what happened to gate 3.
  local log="/tmp/recv.${out%.json}.log"
  k exec "$pod" -- bash -lc "
    nohup env MX_POOL_REG=1 MX_RESHARD_MAX_GBPS=800 MX_REFIT_STAGE_RECORD=1 $* \
      $py /tmp/reshard_receiver_run.py \
      --model $MODEL --rendezvous-name $MODEL \
      --tp 2 --num-trainers 8 --installers pwal,mdl \
      --trainer-topology Megatron-EP8 \
      --mx-server $MXSERVER \
      --warm-cycles $cycles \
      --out $OUTDIR/$out > $log 2>&1 &
    echo 'refit launched -> $out (log $log)'"
}

cmd_gate3() { run_refit $RECV_MAIN $RECVPY gate3_ep8_tp2_correctness.json 3; }
cmd_gate4() { run_refit $RECV_MAIN $RECVPY gate4_ep8_tp2_timing.json 10; }

cmd_gate5() {
  local arm=${1:?arm required: baseline|cache|batch|spread}
  # Every row names its own gates. baseline is the closest this image gets to main:
  # c2's dedup is unconditional, so it cannot be switched off and this row is
  # "main plus dedup", not main. See 06_GATE2 §3.1.
  local env_args out
  case "$arm" in
    baseline) env_args="MX_RESHARD_CACHE_DESCRIPTORS=0 MX_RESHARD_BATCH_INSTALL=0 MX_RESHARD_SPREAD_SOURCES=0" ;;
    cache)    env_args="MX_RESHARD_CACHE_DESCRIPTORS=1 MX_RESHARD_BATCH_INSTALL=0 MX_RESHARD_SPREAD_SOURCES=0" ;;
    batch)    env_args="MX_RESHARD_CACHE_DESCRIPTORS=1 MX_RESHARD_BATCH_INSTALL=1 MX_RESHARD_SPREAD_SOURCES=0" ;;
    spread)   env_args="MX_RESHARD_CACHE_DESCRIPTORS=1 MX_RESHARD_BATCH_INSTALL=1 MX_RESHARD_SPREAD_SOURCES=1" ;;
    *) echo "unknown arm $arm" >&2; exit 1 ;;
  esac
  out="gate5_ep8_tp2_$arm.json"
  run_refit $RECV_STAGING $RECVPY "$out" 10 "$env_args"
}

cmd_recvlog() {
  local pod=${1:-$RECV_MAIN}
  k exec "$pod" -- bash -lc 'tail -40 $(ls -t /tmp/recv.*.log 2>/dev/null | head -1)'
}

cmd_collect() {
  mkdir -p "$EVIDENCE"
  # Both receivers write to the same shared PVC path, so list from each: a gate 5
  # result exists only on the staging receiver's view of it if the main one is gone.
  for pod in $RECV_MAIN $RECV_STAGING; do
    for f in $(k exec "$pod" -- bash -lc "ls -1 $OUTDIR/*.json 2>/dev/null" | tr -d '\r'); do
      k cp "$pod:$f" "$EVIDENCE/$(basename "$f")" 2>/dev/null || true
    done
  done
  # Raw logs travel with the JSON: a row without its log cannot be re-read later, and
  # the MX_REFIT_STAGE records are only in the log.
  for pod in $PUB0 $PUB1; do
    k exec "$pod" -- bash -lc 'cat /tmp/pub.log' > "$EVIDENCE/${pod}.pub.log" 2>/dev/null || true
  done
  for pod in $RECV_MAIN $RECV_STAGING; do
    for log in $(k exec "$pod" -- bash -lc 'ls -1 /tmp/recv.*.log 2>/dev/null' | tr -d '\r'); do
      k exec "$pod" -- bash -lc "cat $log" > "$EVIDENCE/${pod}.$(basename "$log")" 2>/dev/null || true
    done
  done
  ls -la "$EVIDENCE"
  python3 "$EVIDENCE/report.py" "$EVIDENCE"/*.json
}

sub=${1:?usage: $0 pods|stage|publish8|publog|gate3|gate4|gate5 <arm>|recvlog [pod]|collect}
shift || true
"cmd_$sub" "$@"
