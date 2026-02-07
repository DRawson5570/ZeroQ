#!/bin/bash
# Multi-node ZeRO-Q launch script with CUDA timeout fixes
# The "CUDA error: the launch timed out" comes from CUDA driver, not NCCL

# === CRITICAL CUDA FIXES ===
# Disable synchronous CUDA calls (allows async kernel waiting)
export CUDA_LAUNCH_BLOCKING=0
# More CUDA connections for parallel ops
export CUDA_DEVICE_MAX_CONNECTIONS=32
# Disable CUDA memory caching issues
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

# === NCCL NETWORK CONFIG ===
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_P2P_DISABLE=1
# Force all communication through sockets, no GPU direct
export NCCL_SHM_DISABLE=1
export NCCL_P2P_LEVEL=0
export NCCL_NET=Socket

# === NCCL TIMEOUT CONFIG (Extended for routed network) ===
# Main NCCL timeout - 1 hour in milliseconds
export NCCL_TIMEOUT=3600000
export NCCL_ASYNC_ERROR_HANDLING=1
# Socket-level timeouts
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=8

# === PYTORCH/TORCH DISTRIBUTED TIMEOUT CONFIG ===
# PyTorch NCCL watchdog heartbeat - 30 minutes (default is 8)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
# Disable the monitoring thread that kills on stuck watchdog
export TORCH_NCCL_ENABLE_MONITORING=0
# Don't abort on async NCCL errors - let it retry
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
# Don't rethrow CUDA errors from watchdog (let them complete)
export TORCH_NCCL_RETHROW_CUDA_ERRORS=0
# Blocking wait mode - wait synchronously (more stable for slow networks)
export TORCH_NCCL_BLOCKING_WAIT=1
# Extended dump timeout
export TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC=60000

# === GLOO FALLBACK (if NCCL fails, use TCP) ===
export GLOO_SOCKET_IFNAME=bond0

# === DEBUG ===
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Print config
echo "=== Multi-Node ZeRO-Q Config ==="
echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"
echo "TORCH_NCCL_BLOCKING_WAIT=$TORCH_NCCL_BLOCKING_WAIT"
echo "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=$TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"
echo "TORCH_NCCL_ENABLE_MONITORING=$TORCH_NCCL_ENABLE_MONITORING"
echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "==============================="

# Get node config from args or defaults
MASTER_ADDR=${1:-10.0.10.2}  # PE2's bond0
MASTER_PORT=${2:-29500}
NNODES=${3:-2}
NODE_RANK=${4:-0}
NPROC=${5:-3}

echo "Node config: master=$MASTER_ADDR:$MASTER_PORT, nnodes=$NNODES, rank=$NODE_RANK, nproc=$NPROC"

# Activate conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate m40_env

cd ~/Phoenix/ZeroQ/examples

# Run with torchrun
torchrun \
    --nproc_per_node=$NPROC \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_zeroq_real.py
