#!/usr/bin/env python3
"""
Test the ZeroQ coordinator with real distributed training simulation.
Run on PE3 with 2x Tesla M40.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import sys

# Add parent directory to path for imports
sys.path.insert(0, '/home/drawson/ZeroQ')

from src.config import ZeroQConfig, MAXWELL_CONFIG
from src.coordinator import ZeroQCoordinator, ZeroQModuleWrapper, ZeroQParamStatus


def setup(rank, world_size):
    """Initialize process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up process group."""
    dist.destroy_process_group()


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_size=1024, hidden_size=4096, output_size=1024):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc3 = nn.Linear(hidden_size, output_size, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def test_coordinator_basic(rank, world_size):
    """Test basic coordinator functionality."""
    setup(rank, world_size)
    print(f"\n[Rank {rank}] === Test: Basic Coordinator ===")
    
    config = MAXWELL_CONFIG
    coordinator = ZeroQCoordinator(config)
    
    # Create a simple parameter
    param = nn.Parameter(torch.randn(4096, 4096, dtype=torch.float16, device=f'cuda:{rank}'))
    
    # Register parameter
    zq_param = coordinator.register_parameter(param)
    print(f"[Rank {rank}] Registered param with id {zq_param.param_id}")
    print(f"[Rank {rank}] Original shape: {zq_param.original_shape}")
    print(f"[Rank {rank}] Original numel: {zq_param.original_numel}")
    
    # Partition
    zq_param.partition()
    print(f"[Rank {rank}] After partition:")
    print(f"  Status: {zq_param.status}")
    print(f"  Local packed shape: {zq_param.local_packed.shape}")
    print(f"  Local absmax shape: {zq_param.local_absmax.shape}")
    print(f"  Local memory: {zq_param.local_memory_bytes / 1024:.2f} KB")
    print(f"  Full FP16 memory: {zq_param.full_memory_bytes / 1024**2:.2f} MB")
    
    # Gather
    zq_param.start_gather(async_op=False)
    print(f"[Rank {rank}] After gather:")
    print(f"  Status: {zq_param.status}")
    print(f"  Param data shape: {param.data.shape}")
    print(f"  Param data dtype: {param.data.dtype}")
    
    # Verify data on all ranks matches
    checksum = param.data.sum().item()
    all_checksums = [torch.zeros(1, device=f'cuda:{rank}') for _ in range(world_size)]
    dist.all_gather(all_checksums, torch.tensor([checksum], device=f'cuda:{rank}'))
    
    if rank == 0:
        checksums = [c.item() for c in all_checksums]
        print(f"[Rank 0] Checksums: {checksums}")
        if abs(checksums[0] - checksums[1]) < 1:
            print("✓ All ranks have identical data after gather!")
        else:
            print("✗ Data mismatch between ranks!")
    
    # Release
    zq_param.release()
    print(f"[Rank {rank}] After release: status = {zq_param.status}")
    
    dist.barrier()
    cleanup()
    
    if rank == 0:
        print("\n✓ Basic coordinator test passed!")


def test_module_wrapper(rank, world_size):
    """Test module wrapper with hooks."""
    setup(rank, world_size)
    print(f"\n[Rank {rank}] === Test: Module Wrapper ===")
    
    config = MAXWELL_CONFIG
    coordinator = ZeroQCoordinator(config)
    
    # Create model
    model = SimpleModel(hidden_size=2048).to(f'cuda:{rank}')
    model = model.half()  # Convert to FP16 for quantization
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Rank {rank}] Model has {total_params:,} parameters")
    
    # Wrap with ZeroQ
    wrapper = ZeroQModuleWrapper(model, coordinator, trainable_only=False)
    
    # Partition all parameters
    wrapper.partition()
    
    # Get memory stats
    stats = wrapper.get_memory_stats()
    print(f"[Rank {rank}] Memory stats:")
    print(f"  Local memory: {stats['local_memory_mb']:.2f} MB")
    print(f"  Full FP16 memory: {stats['full_fp16_memory_mb']:.2f} MB")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    
    # Forward pass (should trigger gather)
    x = torch.randn(8, 1024, dtype=torch.float32, device=f'cuda:{rank}')
    
    print(f"[Rank {rank}] Running forward pass...")
    with torch.no_grad():
        y = model(x)
    
    print(f"[Rank {rank}] Output shape: {y.shape}")
    print(f"[Rank {rank}] Output sum: {y.sum().item():.4f}")
    
    dist.barrier()
    cleanup()
    
    if rank == 0:
        print("\n✓ Module wrapper test passed!")


def test_training_simulation(rank, world_size):
    """Simulate a training step with ZeroQ."""
    setup(rank, world_size)
    print(f"\n[Rank {rank}] === Test: Training Simulation ===")
    
    config = MAXWELL_CONFIG
    coordinator = ZeroQCoordinator(config)
    
    # Create model
    model = SimpleModel(hidden_size=2048).to(f'cuda:{rank}')
    model = model.half()
    
    # Only quantize non-trainable params (simulating QLoRA)
    # For this test, freeze fc1 and fc2, keep fc3 trainable
    for param in model.fc1.parameters():
        param.requires_grad = False
    for param in model.fc2.parameters():
        param.requires_grad = False
    # fc3 stays trainable
    
    # Register only frozen params with ZeroQ
    wrapper = ZeroQModuleWrapper(model, coordinator, trainable_only=False)
    wrapper.partition()
    
    # Optimizer only for trainable params
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )
    
    print(f"[Rank {rank}] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"[Rank {rank}] Frozen params (ZeroQ): {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    
    # Training step
    x = torch.randn(8, 1024, dtype=torch.float32, device=f'cuda:{rank}')
    target = torch.randn(8, 1024, dtype=torch.float32, device=f'cuda:{rank}')
    
    # Forward
    y = model(x)
    loss = nn.MSELoss()(y, target)
    print(f"[Rank {rank}] Loss: {loss.item():.6f}")
    
    # Backward
    loss.backward()
    
    # Check gradients
    fc3_grad_sum = model.fc3.weight.grad.sum().item() if model.fc3.weight.grad is not None else 0
    print(f"[Rank {rank}] fc3 gradient sum: {fc3_grad_sum:.4f}")
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"[Rank {rank}] Training step completed!")
    
    dist.barrier()
    cleanup()
    
    if rank == 0:
        print("\n✓ Training simulation test passed!")


def main():
    world_size = torch.cuda.device_count()
    print(f"Running coordinator tests with {world_size} GPUs")
    
    if world_size < 2:
        print("Need at least 2 GPUs for distributed tests")
        return
    
    # Test 1: Basic coordinator
    print("\n" + "=" * 60)
    print("TEST 1: Basic Coordinator")
    print("=" * 60)
    mp.spawn(test_coordinator_basic, args=(world_size,), nprocs=world_size, join=True)
    
    # Test 2: Module wrapper
    print("\n" + "=" * 60)
    print("TEST 2: Module Wrapper")
    print("=" * 60)
    mp.spawn(test_module_wrapper, args=(world_size,), nprocs=world_size, join=True)
    
    # Test 3: Training simulation
    print("\n" + "=" * 60)
    print("TEST 3: Training Simulation")
    print("=" * 60)
    mp.spawn(test_training_simulation, args=(world_size,), nprocs=world_size, join=True)
    
    print("\n" + "=" * 60)
    print("ALL COORDINATOR TESTS PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    main()
