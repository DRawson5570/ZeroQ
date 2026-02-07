#!/usr/bin/env bash
set -euo pipefail

# Detached multinode launcher for true multi-node QLoRA training (ZeRO-Q).
# Starts rank0 on MASTER_HOST and rank1 on WORKER_HOST in a fully detached state.

MASTER_HOST=${MASTER_HOST:-poweredge2}
WORKER_HOST=${WORKER_HOST:-poweredge3}
MASTER_ADDR=${MASTER_ADDR:-10.0.10.2}
MASTER_PORT=${MASTER_PORT:-29501}

MASTER_NPROC=${MASTER_NPROC:-5}
WORKER_NPROC=${WORKER_NPROC:-2}

MASTER_IFNAME=${MASTER_IFNAME:-bond0}
WORKER_IFNAME=${WORKER_IFNAME:-bond0}

CONDA_PREFIX_BASE=${CONDA_PREFIX_BASE:-/home/drawson/anaconda3}
ENV_NAME=${ENV_NAME:-m40_env}
TORCHRUN_BIN=${TORCHRUN_BIN:-${CONDA_PREFIX_BASE}/envs/${ENV_NAME}/bin/torchrun}

REMOTE_DIR=${REMOTE_DIR:-/home/drawson/Phoenix}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-Coder-32B-Instruct}
DATA_PATH=${DATA_PATH:-/home/drawson/phoenix_training/phoenix_grok.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/home/drawson/phoenix_training/zeroq_qwen32b_lora}

SEQ_LEN=${SEQ_LEN:-128}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-2e-4}
MAX_STEPS=${MAX_STEPS:-2}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}

RUN_ID=${RUN_ID:-port${MASTER_PORT}}

RANK0_LOG=${RANK0_LOG:-/tmp/zeroq_train32b_${RUN_ID}_rank0.log}
RANK1_LOG=${RANK1_LOG:-/tmp/zeroq_train32b_${RUN_ID}_rank1.log}
RANK0_PID=${RANK0_PID:-/tmp/zeroq_train32b_${RUN_ID}_rank0.pid}
RANK1_PID=${RANK1_PID:-/tmp/zeroq_train32b_${RUN_ID}_rank1.pid}

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=10
  -o ServerAliveCountMax=3
  -T
)

remote_launch() {
  local host=$1
  local rank=$2
  local ifname=$3
  local nproc=$4
  local pidfile=$5
  local logfile=$6

  ssh "${SSH_OPTS[@]}" "$host" bash -s -- \
    "$REMOTE_DIR" \
    "$TORCHRUN_BIN" \
    "$MASTER_ADDR" \
    "$MASTER_PORT" \
    "$rank" \
    "$ifname" \
    "$nproc" \
    "$pidfile" \
    "$logfile" \
    "$MODEL_ID" \
    "$DATA_PATH" \
    "$OUTPUT_DIR" \
    "$SEQ_LEN" \
    "$BATCH_SIZE" \
    "$GRAD_ACCUM" \
    "$LR" \
    "$MAX_STEPS" \
    "$LORA_R" \
    "$LORA_ALPHA" \
    "$LORA_DROPOUT" <<'REMOTE'
set -euo pipefail

remote_dir="$1"
torchrun_bin="$2"
master_addr="$3"
master_port="$4"
rank="$5"
ifname="$6"
nproc="$7"
pidfile="$8"
logfile="$9"
model_id="${10}"
data_path="${11}"
output_dir="${12}"
seq_len="${13}"
batch_size="${14}"
grad_accum="${15}"
lr="${16}"
max_steps="${17}"
lora_r="${18}"
lora_alpha="${19}"
lora_dropout="${20}"

cd "$remote_dir"
rm -f "$pidfile" "$logfile"

nohup env \
  PYTHONUNBUFFERED=1 \
  NCCL_SOCKET_IFNAME="$ifname" \
  GLOO_SOCKET_IFNAME="$ifname" \
  NCCL_IB_DISABLE=1 \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$torchrun_bin" \
    --nproc_per_node="$nproc" \
    --nnodes=2 \
    --node_rank="$rank" \
    --master_addr="$master_addr" \
    --master_port="$master_port" \
    ZeroQ/examples/train_32b_maxwell.py \
      --model "$model_id" \
      --data "$data_path" \
      --output_dir "$output_dir" \
      --seq_len "$seq_len" \
      --batch_size "$batch_size" \
      --grad_accum "$grad_accum" \
      --lr "$lr" \
      --max_steps "$max_steps" \
      --lora_r "$lora_r" \
      --lora_alpha "$lora_alpha" \
      --lora_dropout "$lora_dropout" \
  > "$logfile" 2>&1 < /dev/null &

echo $! > "$pidfile"
echo "started host=$(hostname) rank=$rank pid=$(cat "$pidfile") log=$logfile"
REMOTE
}

echo "Launching detached ZeRO-Q multi-node training:" 
echo "  master: $MASTER_HOST (rank0) addr=$MASTER_ADDR port=$MASTER_PORT if=$MASTER_IFNAME"
echo "  worker: $WORKER_HOST (rank1) if=$WORKER_IFNAME"
echo "  nproc:  master=$MASTER_NPROC worker=$WORKER_NPROC (world_size=$((MASTER_NPROC + WORKER_NPROC)))"

echo "[1/2] Launching rank0 on $MASTER_HOST..."
remote_launch "$MASTER_HOST" 0 "$MASTER_IFNAME" "$MASTER_NPROC" "$RANK0_PID" "$RANK0_LOG"

echo "[2/2] Launching rank1 on $WORKER_HOST..."
remote_launch "$WORKER_HOST" 1 "$WORKER_IFNAME" "$WORKER_NPROC" "$RANK1_PID" "$RANK1_LOG"

echo "Done. Logs: $RANK0_LOG / $RANK1_LOG"
