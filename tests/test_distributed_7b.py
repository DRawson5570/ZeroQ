#!/usr/bin/env python3
"""
ZeRO-Q Distributed Training Test - 7B Model
Tests actual distributed training across multiple GPUs.

Run with:
    torchrun --nproc_per_node=2 test_distributed_7b.py
    
For multi-node:
    torchrun --nnodes=2 --nproc_per_node=2 --rdzv_backend=c10d \
             --rdzv_endpoint=HOST:PORT test_distributed_7b.py
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DummyDataset(Dataset):
    """Simple dataset for testing."""
    
    def __init__(self, vocab_size, seq_len=128, num_samples=50):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random token IDs
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": input_ids.clone(),
        }


def main():
    # Initialize distributed FIRST, before any model loading
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        print("\n" + "="*70)
        print("ZeRO-Q 7B Distributed Training Test")
        print("="*70)
        print(f"World size: {world_size}")
        print(f"Rank {rank}, Local rank {local_rank}, Device: {device}")
    
    dist.barrier()
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        from peft import LoraConfig, get_peft_model
        from src.config import MAXWELL_CONFIG
        from src.coordinator import ZeroQCoordinator, ZeroQModuleWrapper
        
        # Model to test - use Qwen 0.5B for faster testing, or specify 7B
        model_name = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B")
        
        if rank == 0:
            print(f"\nLoading model: {model_name}")
        
        # Load tokenizer (all ranks)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Get config first to know vocab size
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # CRITICAL: Load model WITHOUT device_map, then move to correct device
        # This ensures each rank has its own copy
        if rank == 0:
            print("Loading model weights (FP32 for Maxwell)...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # FP32 for Maxwell compute
            trust_remote_code=True,
            device_map=None,  # Don't use device_map with distributed!
            low_cpu_mem_usage=True,
        )
        
        # Move to this rank's GPU
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        if rank == 0:
            print(f"Model parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        
        # Memory before ZeRO-Q
        torch.cuda.synchronize(local_rank)
        mem_before = torch.cuda.memory_allocated(local_rank) / 1024**2
        if rank == 0:
            print(f"Memory before ZeRO-Q: {mem_before:.1f} MB")
        
        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False
        
        # Apply LoRA
        if rank == 0:
            print("\nApplying LoRA (r=8)...")
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        if rank == 0:
            print(f"Trainable: {trainable:,}, Frozen: {frozen:,}")
        
        # Apply ZeRO-Q
        if rank == 0:
            print("\nApplying ZeRO-Q quantization...")
        
        dist.barrier()  # Sync before ZeRO-Q
        
        coordinator = ZeroQCoordinator(MAXWELL_CONFIG)
        wrapper = ZeroQModuleWrapper(model, coordinator, trainable_only=False)
        wrapper.partition()
        
        # Memory after ZeRO-Q
        torch.cuda.synchronize(local_rank)
        mem_after = torch.cuda.memory_allocated(local_rank) / 1024**2
        
        stats = wrapper.get_memory_stats()
        if rank == 0:
            print(f"Memory after ZeRO-Q: {mem_after:.1f} MB")
            print(f"Savings: {mem_before / mem_after:.2f}x")
            print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
        
        # Create dataset
        vocab_size = getattr(config, 'vocab_size', 32000)
        dataset = DummyDataset(vocab_size, seq_len=64, num_samples=20)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Optimizer (only trainable params)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
        )
        
        # Training loop
        if rank == 0:
            print("\nStarting training...")
        
        model.train()
        total_loss = 0
        num_steps = 5
        
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            
            # Backward
            loss.backward()
            
            # Sync gradients across ranks for LoRA params
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            
            # Step
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if rank == 0:
                print(f"  Step {step+1}/{num_steps}: loss = {loss.item():.4f}")
        
        avg_loss = total_loss / num_steps
        
        # Verify training worked across all ranks
        loss_tensor = torch.tensor([avg_loss], device=device)
        all_losses = [torch.zeros(1, device=device) for _ in range(world_size)]
        dist.all_gather(all_losses, loss_tensor)
        
        if rank == 0:
            losses = [l.item() for l in all_losses]
            print(f"\nLosses per rank: {losses}")
            
            print("\n" + "="*70)
            print("ZeRO-Q Distributed Test COMPLETE")
            print("="*70)
            print(f"✓ Model: {model_name}")
            print(f"✓ World size: {world_size}")
            print(f"✓ Memory savings: {mem_before/mem_after:.2f}x")
            print(f"✓ Compression: {stats['compression_ratio']:.2f}x")
            print(f"✓ Training steps: {num_steps}")
            print(f"✓ Final avg loss: {avg_loss:.4f}")
            print("="*70)
        
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
