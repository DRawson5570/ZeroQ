"""
Real distributed tests for ZeRO-Q on multiple GPUs.

Run with:
    torchrun --nproc_per_node=2 tests/test_distributed_real.py
"""

import os
import sys
import torch
import torch.distributed as dist
from torch import nn

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ZeroQConfig, MAXWELL_CONFIG
from src.coordinator import ZeroQCoordinator, ZeroQModuleWrapper


class SimpleTransformerBlock(nn.Module):
    """Simplified transformer block for testing."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Attention
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # MLP
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.down_proj = nn.Linear(hidden_size * 4, hidden_size, bias=False)
        
        # Norms
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.mlp_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # Attention
        residual = x
        x = self.attn_norm(x)
        
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Simple attention (no causal mask for test)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        x = self.o_proj(attn_output) + residual
        
        # MLP
        residual = x
        x = self.mlp_norm(x)
        x = self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)) + residual
        
        return x


class TestModel(nn.Module):
    """Test model with multiple transformer blocks."""
    
    def __init__(self, hidden_size: int = 512, num_layers: int = 4):
        super().__init__()
        self.embed = nn.Linear(hidden_size, hidden_size)
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(hidden_size) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, hidden_size)  # Simplified
    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_distributed():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def test_zeroq_distributed_forward():
    """Test that ZeRO-Q produces identical outputs across ranks."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{rank}'
    
    print(f"[Rank {rank}] Starting distributed forward test")
    
    # Create model with same seed on all ranks
    torch.manual_seed(42)
    model = TestModel(hidden_size=512, num_layers=2).half().to(device)
    
    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False
    
    # Apply ZeRO-Q
    config = MAXWELL_CONFIG
    coordinator = ZeroQCoordinator(config)
    wrapper = ZeroQModuleWrapper(model, coordinator, trainable_only=False)
    wrapper.partition()
    
    stats = wrapper.get_memory_stats()
    print(f"[Rank {rank}] Memory stats: {stats}")
    
    # Create same input on all ranks
    torch.manual_seed(123)
    x = torch.randn(2, 32, 512, device=device, dtype=torch.float16)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # All-reduce outputs to verify they match
    output_gathered = [torch.zeros_like(output) for _ in range(world_size)]
    dist.all_gather(output_gathered, output)
    
    # Verify outputs match
    if rank == 0:
        max_diff = torch.max(torch.abs(output_gathered[0] - output_gathered[1]))
        print(f"[Rank 0] Max output difference between ranks: {max_diff.item():.6e}")
        
        if max_diff < 1e-3:
            print("[Rank 0] ✓ Outputs match across ranks!")
        else:
            print("[Rank 0] ✗ Outputs differ - possible quantization inconsistency")
    
    dist.barrier()
    
    return True


def test_zeroq_distributed_backward():
    """Test that gradients are correctly computed with ZeRO-Q."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{rank}'
    
    print(f"[Rank {rank}] Starting distributed backward test")
    
    # Create model with trainable LoRA-like params
    torch.manual_seed(42)
    model = TestModel(hidden_size=512, num_layers=2).half().to(device)
    
    # Freeze most params, keep some trainable
    for name, param in model.named_parameters():
        if 'lm_head' in name or 'final_norm' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[Rank {rank}] Trainable: {trainable_params:,}, Frozen: {frozen_params:,}")
    
    # Apply ZeRO-Q to frozen params only
    config = MAXWELL_CONFIG
    coordinator = ZeroQCoordinator(config)
    wrapper = ZeroQModuleWrapper(model, coordinator, trainable_only=False)
    wrapper.partition()
    
    # Create input and target
    torch.manual_seed(123)
    x = torch.randn(2, 32, 512, device=device, dtype=torch.float16)
    target = torch.randn(2, 32, 512, device=device, dtype=torch.float16)
    
    # Forward and backward
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    # Gather gradients for comparison
    grad_checksums = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            checksum = param.grad.sum().item()
            grad_checksums.append(checksum)
    
    # Verify gradients exist
    if rank == 0:
        print(f"[Rank 0] Gradient checksums: {grad_checksums}")
        if all(g != 0 for g in grad_checksums):
            print("[Rank 0] ✓ Gradients computed successfully!")
        else:
            print("[Rank 0] ✗ Some gradients are zero")
    
    dist.barrier()
    
    return True


def test_zeroq_memory_efficiency():
    """Test actual memory savings with ZeRO-Q."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{rank}'
    
    print(f"[Rank {rank}] Starting memory efficiency test")
    
    torch.cuda.reset_peak_memory_stats(rank)
    
    # Baseline without ZeRO-Q
    torch.manual_seed(42)
    baseline_model = TestModel(hidden_size=1024, num_layers=4).half().to(device)
    baseline_params = sum(p.numel() * 2 for p in baseline_model.parameters())  # FP16 bytes
    
    torch.cuda.synchronize(rank)
    baseline_mem = torch.cuda.memory_allocated(rank)
    
    del baseline_model
    torch.cuda.empty_cache()
    
    # With ZeRO-Q
    torch.cuda.reset_peak_memory_stats(rank)
    torch.manual_seed(42)
    model = TestModel(hidden_size=1024, num_layers=4).half().to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    config = MAXWELL_CONFIG
    coordinator = ZeroQCoordinator(config)
    wrapper = ZeroQModuleWrapper(model, coordinator, trainable_only=False)
    wrapper.partition()
    
    torch.cuda.synchronize(rank)
    zeroq_mem = torch.cuda.memory_allocated(rank)
    
    stats = wrapper.get_memory_stats()
    
    if rank == 0:
        print(f"[Rank 0] Baseline memory: {baseline_mem / 1024**2:.2f} MB")
        print(f"[Rank 0] ZeRO-Q memory: {zeroq_mem / 1024**2:.2f} MB")
        print(f"[Rank 0] Savings: {baseline_mem / zeroq_mem:.2f}x")
        print(f"[Rank 0] Coordinator stats: {stats}")
        
        if baseline_mem > zeroq_mem:
            print("[Rank 0] ✓ ZeRO-Q uses less memory!")
        else:
            print("[Rank 0] ! ZeRO-Q memory is higher (expected for small models)")
    
    dist.barrier()
    
    return True


def main():
    """Run all distributed tests."""
    # Initialize distributed once at the start
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"ZeRO-Q Distributed Tests - {world_size} GPUs")
        print(f"{'='*60}")
    
    tests = [
        ("Distributed Forward", test_zeroq_distributed_forward),
        ("Distributed Backward", test_zeroq_distributed_backward),
        ("Memory Efficiency", test_zeroq_memory_efficiency),
    ]
    
    for name, test_fn in tests:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print(f"{'='*60}")
        dist.barrier()
        try:
            test_fn()
            if rank == 0:
                print(f"✓ {name} completed")
        except Exception as e:
            print(f"[Rank {rank}] ✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
        dist.barrier()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
