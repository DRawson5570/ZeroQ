#!/usr/bin/env bash
set -euo pipefail

# Detached multinode launcher for ZeRO-Q smoke test.
# Starts rank0 on MASTER_HOST and rank1 on WORKER_HOST in a fully detached state
# (remote process survives SSH disconnect).

# Default to the two GPU nodes.
# poweredge2: 5x Tesla M40
# poweredge3: 2x Tesla M40
MASTER_HOST=${MASTER_HOST:-poweredge2}
WORKER_HOST=${WORKER_HOST:-poweredge3}
# IMPORTANT: use a single rendezvous address that is reachable from ALL nodes.
# With the current Phoenix backbone, 10.0.10.2 (poweredge2 bond0) is routable from poweredge3 via poweredge1.
MASTER_ADDR=${MASTER_ADDR:-10.0.10.2}
MASTER_PORT=${MASTER_PORT:-29501}

# Processes per node (use all GPUs by default: 5 on poweredge2, 2 on poweredge3)
MASTER_NPROC=${MASTER_NPROC:-5}
WORKER_NPROC=${WORKER_NPROC:-2}

# Network interfaces for NCCL/Gloo (override if needed)
# Default to the backbone bonds.
MASTER_IFNAME=${MASTER_IFNAME:-bond0}
WORKER_IFNAME=${WORKER_IFNAME:-bond0}

# Conda env path (avoids `conda activate` to prevent shell/job-control issues)
CONDA_PREFIX_BASE=${CONDA_PREFIX_BASE:-/home/drawson/anaconda3}
ENV_NAME=${ENV_NAME:-m40_env}
TORCHRUN_BIN=${TORCHRUN_BIN:-${CONDA_PREFIX_BASE}/envs/${ENV_NAME}/bin/torchrun}

REMOTE_DIR=${REMOTE_DIR:-/home/drawson/Phoenix}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-1.5B}
QUANT=${QUANT:-q4}
STEPS=${STEPS:-2}
SEQ_LEN=${SEQ_LEN:-256}
COMPUTE_DTYPE=${COMPUTE_DTYPE:-fp32}

VERIFY_SHARD_LAYERS=${VERIFY_SHARD_LAYERS:-2}
VERIFY_SHARD_ELEMS=${VERIFY_SHARD_ELEMS:-4096}

RUN_ID=${RUN_ID:-port${MASTER_PORT}}

RANK0_LOG=${RANK0_LOG:-/tmp/zeroq_multinode_${RUN_ID}_rank0.log}
RANK1_LOG=${RANK1_LOG:-/tmp/zeroq_multinode_${RUN_ID}_rank1.log}
RANK0_PID=${RANK0_PID:-/tmp/zeroq_multinode_${RUN_ID}_rank0.pid}
RANK1_PID=${RANK1_PID:-/tmp/zeroq_multinode_${RUN_ID}_rank1.pid}

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

  # Use nohup + stdin from /dev/null so it fully detaches.
  # Avoid pkill patterns; rely on pidfiles for cleanup.
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
    "$QUANT" \
    "$STEPS" \
    "$SEQ_LEN" \
    "$COMPUTE_DTYPE" \
    "$VERIFY_SHARD_LAYERS" \
    "$VERIFY_SHARD_ELEMS" <<'REMOTE'
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
quant="${11}"
steps="${12}"
seq_len="${13}"
compute_dtype="${14}"
verify_layers="${15}"
verify_elems="${16}"

cd "$remote_dir"
rm -f "$pidfile" "$logfile"

nohup env \
  PYTHONUNBUFFERED=1 \
  NCCL_SOCKET_IFNAME="$ifname" \
  GLOO_SOCKET_IFNAME="$ifname" \
  NCCL_IB_DISABLE=1 \
  NCCL_P2P_DISABLE=0 \
  NCCL_SHM_DISABLE=0 \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$torchrun_bin" \
    --nproc_per_node="$nproc" \
    --nnodes=2 \
    --node_rank="$rank" \
    --master_addr="$master_addr" \
    --master_port="$master_port" \
    ZeroQ/tests/test_multinode.py \
      --model-id "$model_id" \
      --quant "$quant" \
      --steps "$steps" \
      --seq-len "$seq_len" \
      --compute-dtype "$compute_dtype" \
      --verify-shard-layers "$verify_layers" \
      --verify-shard-elems "$verify_elems" \
  > "$logfile" 2>&1 < /dev/null &

echo $! > "$pidfile"
echo "started host=$(hostname) rank=$rank pid=$(cat "$pidfile") log=$logfile"
REMOTE
}

echo "Launching detached multi-node run:"
echo "  master: $MASTER_HOST (rank0) addr=$MASTER_ADDR port=$MASTER_PORT if=$MASTER_IFNAME"
echo "  worker: $WORKER_HOST (rank1) if=$WORKER_IFNAME"
echo "  nproc:  master=$MASTER_NPROC worker=$WORKER_NPROC (total world_size=$((MASTER_NPROC + WORKER_NPROC)))"

if [[ "$MASTER_ADDR" == 192.168.50.* ]]; then
  echo "WARNING: MASTER_ADDR is on the management LAN ($MASTER_ADDR)." >&2
  echo "         Consider using 10.0.10.1 to force rendezvous over the backbone." >&2
fi

echo "[1/2] Launching rank0 on $MASTER_HOST..."
remote_launch "$MASTER_HOST" 0 "$MASTER_IFNAME" "$MASTER_NPROC" "$RANK0_PID" "$RANK0_LOG"

echo "[2/2] Launching rank1 on $WORKER_HOST..."
remote_launch "$WORKER_HOST" 1 "$WORKER_IFNAME" "$WORKER_NPROC" "$RANK1_PID" "$RANK1_LOG"

echo "Done. Use: ZeroQ/scripts/status_multinode_detached.sh"
