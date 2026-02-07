#!/usr/bin/env python3
"""
ZeroQ Distributed Training Runner

Entry point for multi-node distributed training with ZeroQ.
Handles node coordination, process spawning, and training orchestration.

Usage:
    # Start coordinator (on master node):
    python run_distributed.py --role coordinator --host pe1
    
    # Start workers (on each node):
    python run_distributed.py --role worker --coordinator-host pe1 --gpus 0,1,2,3
    
    # All-in-one for single node:
    python run_distributed.py --role both --gpus 0,1,2,3

Author: Zero (Claude Opus 4.5) in collaboration with Douglas Rawson
Created: December 2025
"""

import argparse
import os
import signal
import socket
import sys
import threading
import time
from typing import Optional, List

import torch
import torch.multiprocessing as mp

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import ZeroQConfig, MultiNodeConfig, MAXWELL_CONFIG, PHOENIX_CLUSTER_CONFIG
from src.transport import TransportCoordinator, TransportWorker, TransportConfig
from src.gradient_sync import create_gradient_sync, GradSyncConfig


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ZeroQ Distributed Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start coordinator on master node:
    python run_distributed.py --role coordinator --host pe1 --port 5555
    
    # Start worker on compute node:
    python run_distributed.py --role worker --coordinator pe1:5555 --gpus 0,1,2,3
    
    # Run training with config file:
    python run_distributed.py --role worker --config config.json --script train.py
        """
    )
    
    # Role selection
    parser.add_argument(
        "--role",
        choices=["coordinator", "worker", "both"],
        default="worker",
        help="Node role: coordinator (master), worker, or both"
    )
    
    # Network configuration
    parser.add_argument(
        "--host",
        default=socket.gethostname(),
        help="This node's hostname (default: auto-detect)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Coordinator port (default: 5555)"
    )
    parser.add_argument(
        "--coordinator",
        default="localhost:5555",
        help="Coordinator address as host:port (default: localhost:5555)"
    )
    
    # GPU configuration
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPU IDs to use (default: 0)"
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=None,
        help="Number of GPUs per node (auto-detect if not specified)"
    )
    
    # Training script
    parser.add_argument(
        "--script",
        default=None,
        help="Training script to run (optional)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config JSON file"
    )
    
    # Distributed settings
    parser.add_argument(
        "--backend",
        choices=["nccl", "gloo", "ring"],
        default="nccl",
        help="Communication backend (default: nccl)"
    )
    parser.add_argument(
        "--gradient-compression",
        action="store_true",
        help="Enable gradient compression"
    )
    parser.add_argument(
        "--hierarchical-sync",
        action="store_true",
        help="Use hierarchical all-reduce (intra-node NCCL + inter-node ring)"
    )
    
    # Debug options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without starting"
    )
    
    return parser.parse_args()


def get_gpu_list(gpu_str: str) -> List[int]:
    """Parse GPU string into list of GPU IDs."""
    if not gpu_str:
        return [0]
    return [int(g.strip()) for g in gpu_str.split(",")]


def get_coordinator_address(addr_str: str) -> tuple:
    """Parse coordinator address string."""
    if ":" in addr_str:
        host, port = addr_str.split(":")
        return host, int(port)
    return addr_str, 5555


def run_coordinator(args):
    """Run the coordinator process."""
    print(f"[Coordinator] Starting on {args.host}:{args.port}")
    
    config = TransportConfig(
        bind_address="0.0.0.0",
        coordinator_port=args.port,
    )
    
    coordinator = TransportCoordinator(config)
    coordinator.start()
    
    # Handle shutdown signals
    shutdown_event = threading.Event()
    
    def signal_handler(signum, frame):
        print("\n[Coordinator] Shutting down...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run coordinator loop
    try:
        while not shutdown_event.is_set():
            coordinator.run()  # This has internal timeout
    finally:
        coordinator.stop()
    
    print("[Coordinator] Stopped")


def run_worker(args, local_rank: int = 0, local_size: int = 1):
    """Run a worker process."""
    gpu_list = get_gpu_list(args.gpus)
    gpu_id = gpu_list[local_rank] if local_rank < len(gpu_list) else local_rank
    
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    
    # Parse coordinator address
    coord_host, coord_port = get_coordinator_address(args.coordinator)
    
    # Generate unique node ID
    node_id = f"{args.host}-gpu{gpu_id}-{os.getpid()}"
    
    print(f"[Worker {node_id}] Starting on device {device}")
    print(f"[Worker {node_id}] Connecting to coordinator at {coord_host}:{coord_port}")
    
    # Create transport config
    config = TransportConfig(
        coordinator_host=coord_host,
        coordinator_port=coord_port,
    )
    
    # Create and connect worker
    worker = TransportWorker(
        node_id=node_id,
        hostname=args.host,
        config=config,
        local_gpus=local_size,
    )
    
    if not worker.connect():
        print(f"[Worker {node_id}] Failed to connect to coordinator!")
        return 1
    
    print(f"[Worker {node_id}] Registered as rank {worker.rank} (world_size={worker.world_size})")
    
    # Wait for all workers to join
    print(f"[Worker {node_id}] Waiting for barrier...")
    if worker.barrier(timeout_sec=120.0):
        print(f"[Worker {node_id}] Barrier passed - all workers ready!")
    else:
        print(f"[Worker {node_id}] Barrier timeout!")
        worker.disconnect()
        return 1
    
    # Create gradient sync backend
    grad_sync = create_gradient_sync(
        mode="ring" if args.backend == "ring" else "nccl",
        transport=worker,
        config=GradSyncConfig(
            compress=args.gradient_compression,
        ),
    )
    
    # If a training script was specified, run it
    if args.script:
        # Import and run training script
        import importlib.util
        spec = importlib.util.spec_from_file_location("training", args.script)
        training_module = importlib.util.module_from_spec(spec)
        
        # Inject distributed context
        training_module.RANK = worker.rank
        training_module.WORLD_SIZE = worker.world_size
        training_module.LOCAL_RANK = local_rank
        training_module.DEVICE = device
        training_module.GRAD_SYNC = grad_sync
        training_module.TRANSPORT = worker
        
        spec.loader.exec_module(training_module)
        
        # Call main if it exists
        if hasattr(training_module, "main"):
            training_module.main()
    else:
        # Demo mode - show distributed setup is working
        print(f"\n[Worker {node_id}] Distributed setup complete!")
        print(f"  Rank: {worker.rank}/{worker.world_size}")
        print(f"  Device: {device}")
        print(f"  Peers: {list(worker.peers.keys())}")
        print(f"\nTo run training, use --script <training_script.py>")
        
        # Keep alive for testing
        print("\nPress Ctrl+C to exit...")
        try:
            while True:
                time.sleep(1.0)
                if not worker.barrier(timeout_sec=5.0):
                    print("[Worker] Lost connection to cluster")
                    break
        except KeyboardInterrupt:
            pass
    
    # Cleanup
    worker.disconnect()
    print(f"[Worker {node_id}] Shutdown complete")
    return 0


def spawn_local_workers(args):
    """Spawn multiple worker processes for multi-GPU on single node."""
    gpu_list = get_gpu_list(args.gpus)
    num_gpus = len(gpu_list)
    
    print(f"Spawning {num_gpus} workers for GPUs: {gpu_list}")
    
    # Use torch.multiprocessing for proper CUDA handling
    mp.set_start_method("spawn", force=True)
    
    processes = []
    for local_rank in range(num_gpus):
        p = mp.Process(
            target=run_worker,
            args=(args, local_rank, num_gpus),
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes
    for p in processes:
        p.join()


def run_both(args):
    """Run both coordinator and worker on same node."""
    # Start coordinator in background thread
    coord_thread = threading.Thread(target=run_coordinator, args=(args,), daemon=True)
    coord_thread.start()
    
    # Give coordinator time to start
    time.sleep(1.0)
    
    # Set coordinator to localhost
    args.coordinator = f"localhost:{args.port}"
    
    # Run workers
    spawn_local_workers(args)


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        print("Configuration:")
        print(f"  Role: {args.role}")
        print(f"  Host: {args.host}")
        print(f"  Port: {args.port}")
        print(f"  Coordinator: {args.coordinator}")
        print(f"  GPUs: {args.gpus}")
        print(f"  Backend: {args.backend}")
        print()
    
    if args.dry_run:
        print("Dry run - configuration valid")
        return 0
    
    # Verify CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
    else:
        print("Warning: CUDA not available, running on CPU")
    
    # Run based on role
    if args.role == "coordinator":
        run_coordinator(args)
    elif args.role == "worker":
        gpu_list = get_gpu_list(args.gpus)
        if len(gpu_list) > 1:
            spawn_local_workers(args)
        else:
            run_worker(args)
    elif args.role == "both":
        run_both(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
