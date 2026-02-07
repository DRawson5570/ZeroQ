#!/bin/bash
# Multi-node ZeRO-Q Training Launcher
# Experiment 3: Real ZeRO-Q across PE2 (3 GPUs) + PE3 (2 GPUs) = 5 GPUs

# Configuration
MASTER_ADDR="10.0.10.2"  # PE2 bond0 IP
MASTER_PORT="29500"
NNODES=2
WORLD_SIZE=5  # 3 + 2 GPUs

echo "=============================================="
echo "EXPERIMENT 3: Multi-Node Real ZeRO-Q"
echo "=============================================="
echo "Master: PE2 ($MASTER_ADDR)"
echo "Nodes: PE2 (3 GPUs) + PE3 (2 GPUs)"
echo "Total: $WORLD_SIZE GPUs"
echo "=============================================="

# Kill any existing training processes
echo "Cleaning up old processes..."
ssh poweredge2 'pkill -f train_zeroq 2>/dev/null' &
ssh poweredge3 'pkill -f train_zeroq 2>/dev/null' &
wait
sleep 2

# Launch PE2 (master node, 3 GPUs)
echo "Launching PE2 (master, 3 GPUs)..."
ssh poweredge2 "cd ~/Phoenix/ZeroQ/examples && \
    export PATH=~/anaconda3/envs/m40_env/bin:\$PATH && \
    export NCCL_SOCKET_IFNAME=bond0 && \
    export NCCL_IB_DISABLE=1 && \
    export NCCL_DEBUG=WARN && \
    export NCCL_TIMEOUT=1800 && \
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 && \
    nohup ~/anaconda3/envs/m40_env/bin/torchrun \
        --nproc_per_node=3 \
        --nnodes=$NNODES \
        --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_zeroq_real.py \
        > ~/zeroq_multinode.log 2>&1 &"

sleep 3

# Launch PE3 (worker node, 2 GPUs)
echo "Launching PE3 (worker, 2 GPUs)..."
ssh poweredge3 "cd ~/Phoenix/ZeroQ/examples && \
    export PATH=~/anaconda3/envs/m40_env/bin:\$PATH && \
    export NCCL_SOCKET_IFNAME=bond0 && \
    export NCCL_IB_DISABLE=1 && \
    export NCCL_DEBUG=WARN && \
    export NCCL_TIMEOUT=1800 && \
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 && \
    nohup ~/anaconda3/envs/m40_env/bin/torchrun \
        --nproc_per_node=2 \
        --nnodes=$NNODES \
        --node_rank=1 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_zeroq_real.py \
        > ~/zeroq_multinode.log 2>&1 &"

echo ""
echo "Multi-node training launched!"
echo "Monitor with:"
echo "  ssh poweredge2 'tail -f ~/zeroq_multinode.log'"
echo "  ssh poweredge3 'tail -f ~/zeroq_multinode.log'"
echo ""
echo "GPU status:"
echo "  ssh poweredge2 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv'"
echo "  ssh poweredge3 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv'"
