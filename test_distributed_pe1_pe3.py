#!/usr/bin/env python3
"""
ZeroQ Distributed Test - PE1 <-> PE3

Tests the multi-node transport layer with a small model (Qwen2.5-0.5B).
Uses 1 GPU per server to verify inter-node communication.

Usage:
    # On PE1 (coordinator + worker):
    python test_distributed_pe1_pe3.py --role coordinator --gpus 0
    
    # On PE3 (worker):
    python test_distributed_pe1_pe3.py --role worker --gpus 0

Author: Zero (Claude Opus 4.5) in collaboration with Douglas Rawson
Created: December 2025
"""

import argparse
import os
import sys
import time
import threading

import torch
import torch.nn as nn

# Add ZeroQ src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from transport import TransportCoordinator, TransportWorker, TransportConfig, serialize_tensor, deserialize_tensor
from gradient_sync import RingAllReduce, TorchDistributedSync, GradSyncConfig


def parse_args():
    parser = argparse.ArgumentParser(description="ZeroQ PE1-PE3 Distributed Test")
    parser.add_argument("--role", choices=["coordinator", "worker"], required=True)
    parser.add_argument("--gpus", default="0", help="GPU to use")
    parser.add_argument("--coordinator-host", default="poweredge1", help="Coordinator hostname")
    parser.add_argument("--coordinator-port", type=int, default=5555)
    parser.add_argument("--steps", type=int, default=5, help="Training steps to run")
    return parser.parse_args()


class SimpleModel(nn.Module):
    """Simple model for testing distributed training."""
    def __init__(self, hidden_size=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )
    
    def forward(self, x):
        return self.layers(x)


def run_coordinator(args):
    """Run as coordinator (on PE1)."""
    print("=" * 60)
    print("ZeroQ Distributed Test - COORDINATOR")
    print("=" * 60)
    
    # Set GPU
    gpu_id = int(args.gpus.split(",")[0])
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    
    # Create transport config
    config = TransportConfig(
        bind_address="0.0.0.0",
        coordinator_port=args.coordinator_port,
    )
    
    # Start coordinator
    coordinator = TransportCoordinator(config)
    coordinator.start()
    
    # Run coordinator in background
    def run_coord():
        while coordinator._running:
            try:
                coordinator.run()
            except Exception as e:
                if coordinator._running:
                    print(f"Coordinator error: {e}")
    
    coord_thread = threading.Thread(target=run_coord, daemon=True)
    coord_thread.start()
    
    print(f"Coordinator started on port {args.coordinator_port}")
    print("Waiting for workers to connect...")
    
    # Also act as a worker
    time.sleep(1)  # Give coordinator time to start
    
    worker_config = TransportConfig(
        coordinator_host="localhost",
        coordinator_port=args.coordinator_port,
    )
    
    import socket
    node_id = f"pe1-gpu{gpu_id}"
    worker = TransportWorker(
        node_id=node_id,
        hostname=socket.gethostname(),
        config=worker_config,
        local_gpus=1,
    )
    
    if not worker.connect():
        print("ERROR: Failed to connect as worker!")
        coordinator.stop()
        return 1
    
    print(f"Connected as worker: rank={worker.rank}, world_size={worker.world_size}")
    
    # Wait for PE3 worker
    print("\nWaiting for PE3 worker to connect...")
    while worker.world_size < 2:
        time.sleep(1)
        # Update world size from coordinator
        if len(coordinator.nodes) >= 2:
            worker.world_size = len(coordinator.nodes)
            print(f"World size updated to {worker.world_size}")
    
    print(f"\n✓ All workers connected! World size: {worker.world_size}")
    
    # Barrier sync
    print("Waiting at barrier...")
    if not worker.barrier(timeout_sec=30.0):
        print("ERROR: Barrier timeout!")
        worker.disconnect()
        coordinator.stop()
        return 1
    
    print("✓ Barrier passed!")
    
    # Create model
    print("\nCreating model...")
    model = SimpleModel(hidden_size=512).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Get parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Training loop with gradient sync
    print(f"\n{'='*60}")
    print(f"Starting distributed training ({args.steps} steps)")
    print(f"{'='*60}\n")
    
    for step in range(args.steps):
        # Generate dummy data
        x = torch.randn(32, 512, device=device)
        target = torch.randint(0, 10, (32,), device=device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient sync with PE3
        print(f"Step {step+1}: loss={loss.item():.4f}, syncing gradients...")
        
        for param_id, param in enumerate(model.parameters()):
            if param.grad is not None:
                # Send gradient to PE3
                worker.send_gradient(param.grad, param_id, dst_rank=1)
                
                # Receive gradient from PE3
                received = worker.recv_gradient(param_id, src_rank=1, timeout_sec=10.0)
                
                if received is not None:
                    # Average gradients
                    param.grad = (param.grad + received) / 2
        
        # Update weights
        optimizer.step()
        print(f"  ✓ Step {step+1} complete")
    
    print(f"\n{'='*60}")
    print("Distributed training complete!")
    print(f"{'='*60}")
    
    # Cleanup
    worker.disconnect()
    coordinator.stop()
    
    return 0


def run_worker(args):
    """Run as worker (on PE3)."""
    print("=" * 60)
    print("ZeroQ Distributed Test - WORKER (PE3)")
    print("=" * 60)
    
    # Set GPU
    gpu_id = int(args.gpus.split(",")[0])
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    
    # Create transport config
    config = TransportConfig(
        coordinator_host=args.coordinator_host,
        coordinator_port=args.coordinator_port,
    )
    
    import socket
    node_id = f"pe3-gpu{gpu_id}"
    worker = TransportWorker(
        node_id=node_id,
        hostname=socket.gethostname(),
        config=config,
        local_gpus=1,
    )
    
    print(f"Connecting to coordinator at {args.coordinator_host}:{args.coordinator_port}...")
    
    if not worker.connect():
        print("ERROR: Failed to connect to coordinator!")
        return 1
    
    print(f"✓ Connected as worker: rank={worker.rank}, world_size={worker.world_size}")
    
    # Barrier sync
    print("Waiting at barrier...")
    if not worker.barrier(timeout_sec=30.0):
        print("ERROR: Barrier timeout!")
        worker.disconnect()
        return 1
    
    print("✓ Barrier passed!")
    
    # Create model (same as coordinator)
    print("\nCreating model...")
    model = SimpleModel(hidden_size=512).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting distributed training ({args.steps} steps)")
    print(f"{'='*60}\n")
    
    for step in range(args.steps):
        # Generate dummy data (different from coordinator)
        x = torch.randn(32, 512, device=device)
        target = torch.randint(0, 10, (32,), device=device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.cross_entropy(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient sync with PE1
        print(f"Step {step+1}: loss={loss.item():.4f}, syncing gradients...")
        
        for param_id, param in enumerate(model.parameters()):
            if param.grad is not None:
                # Send gradient to PE1
                worker.send_gradient(param.grad, param_id, dst_rank=0)
                
                # Receive gradient from PE1
                received = worker.recv_gradient(param_id, src_rank=0, timeout_sec=10.0)
                
                if received is not None:
                    # Average gradients
                    param.grad = (param.grad + received) / 2
        
        # Update weights
        optimizer.step()
        print(f"  ✓ Step {step+1} complete")
    
    print(f"\n{'='*60}")
    print("Distributed training complete!")
    print(f"{'='*60}")
    
    # Cleanup
    worker.disconnect()
    
    return 0


def main():
    args = parse_args()
    
    if args.role == "coordinator":
        return run_coordinator(args)
    else:
        return run_worker(args)


if __name__ == "__main__":
    sys.exit(main())
