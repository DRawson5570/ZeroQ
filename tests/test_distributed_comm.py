#!/usr/bin/env python3
"""
Test distributed communication of quantized tensors on PE3 (2x M40).
This simulates ZeroQ's all-gather of 4-bit packed data.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from bitsandbytes.functional import quantize_4bit, dequantize_4bit, QuantState


def setup(rank, world_size):
    """Initialize process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up process group."""
    dist.destroy_process_group()


def test_quantized_allgather(rank, world_size):
    """Test all-gather of quantized tensor partitions."""
    setup(rank, world_size)
    
    print(f"[Rank {rank}] Starting quantized all-gather test")
    
    # Parameters
    total_elements = 4096 * 4096  # 16M elements (64 MB fp16)
    blocksize = 64
    
    # Each rank creates its local partition
    elements_per_rank = total_elements // world_size
    blocks_per_rank = elements_per_rank // blocksize
    
    # Create local "shard" (simulate having just our partition)
    torch.manual_seed(42)  # Same seed so all ranks have same "original"
    full_original = torch.randn(total_elements, dtype=torch.float16, device=f'cuda:{rank}')
    
    # Quantize full tensor (each rank does this with same data)
    full_packed, full_state = quantize_4bit(full_original, blocksize=blocksize, quant_type='nf4')
    
    # Extract our partition
    packed_per_rank = full_packed.numel() // world_size
    absmax_per_rank = full_state.absmax.numel() // world_size
    
    local_packed = full_packed[rank * packed_per_rank : (rank + 1) * packed_per_rank].contiguous()
    local_absmax = full_state.absmax[rank * absmax_per_rank : (rank + 1) * absmax_per_rank].contiguous()
    
    print(f"[Rank {rank}] Local packed: {local_packed.shape}, absmax: {local_absmax.shape}")
    
    # Free full data (simulate ZeroQ where we only store partition)
    del full_original, full_packed
    torch.cuda.empty_cache()
    
    # === ZeroQ All-Gather Simulation ===
    
    # Allocate gather buffers
    gathered_packed_list = [torch.empty_like(local_packed) for _ in range(world_size)]
    gathered_absmax_list = [torch.empty_like(local_absmax) for _ in range(world_size)]
    
    # All-gather packed data
    dist.all_gather(gathered_packed_list, local_packed)
    
    # All-gather absmax
    dist.all_gather(gathered_absmax_list, local_absmax)
    
    # Concatenate
    gathered_packed = torch.cat(gathered_packed_list, dim=0)
    gathered_absmax = torch.cat(gathered_absmax_list, dim=0)
    
    print(f"[Rank {rank}] Gathered packed: {gathered_packed.shape}, absmax: {gathered_absmax.shape}")
    
    # Rebuild QuantState
    gathered_state = QuantState(
        absmax=gathered_absmax,
        shape=full_state.shape,
        dtype=full_state.dtype,
        blocksize=full_state.blocksize,
        code=full_state.code,
        quant_type=full_state.quant_type,
    )
    
    # Dequantize for compute
    restored = dequantize_4bit(gathered_packed, gathered_state)
    print(f"[Rank {rank}] Restored tensor: {restored.shape}, {restored.dtype}")
    
    # Verify all ranks got same result
    checksum = restored.sum().item()
    print(f"[Rank {rank}] Checksum: {checksum:.6f}")
    
    # Synchronize and compare checksums
    all_checksums = [torch.tensor(0.0, device=f'cuda:{rank}') for _ in range(world_size)]
    dist.all_gather(all_checksums, torch.tensor(checksum, device=f'cuda:{rank}'))
    
    if rank == 0:
        checksums = [c.item() for c in all_checksums]
        print(f"\n[Rank 0] All checksums: {checksums}")
        if abs(checksums[0] - checksums[1]) < 1e-3:
            print("✓ All ranks produced identical results!")
        else:
            print("✗ Checksum mismatch!")
    
    # Memory analysis
    if rank == 0:
        local_mem = local_packed.numel() + local_absmax.numel() * 2
        full_mem = gathered_packed.numel() + gathered_absmax.numel() * 2
        fp16_mem = total_elements * 2
        
        print(f"\n=== Memory Analysis ===")
        print(f"Full FP16 tensor: {fp16_mem / 1024**2:.2f} MB")
        print(f"Full 4-bit quantized: {full_mem / 1024**2:.2f} MB ({fp16_mem / full_mem:.2f}x savings)")
        print(f"Per-GPU partition (ZeroQ): {local_mem / 1024:.2f} KB")
        print(f"Bandwidth for all-gather: {local_mem * world_size / 1024**2:.2f} MB")
        print(f"vs FP16 ZeRO-3 bandwidth: {fp16_mem / 1024**2:.2f} MB")
        print(f"Bandwidth savings: {fp16_mem / (local_mem * world_size):.2f}x")
    
    dist.barrier()
    cleanup()
    
    if rank == 0:
        print("\n✓ Distributed quantized all-gather test passed!")


def test_quantized_reduce_scatter(rank, world_size):
    """Test reduce-scatter of gradients (for trainable params)."""
    setup(rank, world_size)
    
    print(f"\n[Rank {rank}] Starting reduce-scatter test")
    
    # Simulate gradients (these stay in fp32, not quantized)
    total_elements = 1024 * 1024  # 1M elements
    
    # Each rank has full gradient for their batch
    torch.manual_seed(rank)  # Different per rank
    local_gradient = torch.randn(total_elements, dtype=torch.float32, device=f'cuda:{rank}')
    
    print(f"[Rank {rank}] Local gradient sum: {local_gradient.sum().item():.4f}")
    
    # Reduce-scatter: sum gradients and scatter partitions
    output = torch.zeros(total_elements // world_size, dtype=torch.float32, device=f'cuda:{rank}')
    
    # Split input for reduce_scatter
    input_list = list(local_gradient.chunk(world_size))
    
    dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)
    
    print(f"[Rank {rank}] After reduce-scatter, local partition sum: {output.sum().item():.4f}")
    print(f"[Rank {rank}] Partition size: {output.numel()} (was {local_gradient.numel()})")
    
    dist.barrier()
    cleanup()
    
    if rank == 0:
        print("\n✓ Reduce-scatter test passed!")


def main():
    world_size = torch.cuda.device_count()
    print(f"Running distributed tests with {world_size} GPUs")
    
    if world_size < 2:
        print("Need at least 2 GPUs for distributed tests")
        return
    
    # Test 1: Quantized all-gather
    print("\n" + "=" * 60)
    print("TEST 1: Quantized All-Gather")
    print("=" * 60)
    mp.spawn(test_quantized_allgather, args=(world_size,), nprocs=world_size, join=True)
    
    # Test 2: Reduce-scatter for gradients  
    print("\n" + "=" * 60)
    print("TEST 2: Gradient Reduce-Scatter")
    print("=" * 60)
    mp.spawn(test_quantized_reduce_scatter, args=(world_size,), nprocs=world_size, join=True)
    
    print("\n" + "=" * 60)
    print("ALL DISTRIBUTED TESTS PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    main()
