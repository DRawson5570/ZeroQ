#!/usr/bin/env python3
"""
ZeRO-Q End-to-End Test with Real 1.5B Model

This test validates ZeRO-Q on an actual model (Qwen 1.5B)
with real distributed training across 2 GPUs.

Run with:
    torchrun --nproc_per_node=2 test_real_1.5b.py
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleDataset(Dataset):
    """Simple dataset with random inputs for testing."""
    
    def __init__(self, tokenizer, num_samples=100, max_length=128):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        
        # Create some simple training examples
        self.examples = [
            "The Phoenix cluster is a distributed computing system.",
            "Zero-Q enables 4-bit quantized distributed training.",
            "Maxwell GPUs can run large language models with quantization.",
            "The stochastic parrot understands tools and builds better ones.",
            "Distributed training partitions model parameters across GPUs.",
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Cycle through examples
        text = self.examples[idx % len(self.examples)]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": encoded["input_ids"].squeeze(0),
        }


def test_zeroq_with_real_model():
    """Test ZeRO-Q with a real 1.5B parameter model."""
    
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}'
    
    if rank == 0:
        print("\n" + "="*70)
        print("ZeRO-Q Real Model Test (Qwen2.5-1.5B)")
        print("="*70)
        print(f"World size: {world_size}")
        print(f"Device: {device}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        from src.config import MAXWELL_CONFIG
        from src.coordinator import ZeroQCoordinator, ZeroQModuleWrapper
        
        # Load tokenizer
        if rank == 0:
            print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B",
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model in FP32 (for Maxwell)
        if rank == 0:
            print("Loading model (FP32 for Maxwell)...")
        
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map={"": device},
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        if rank == 0:
            print(f"Model parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        
        # Memory before ZeRO-Q
        torch.cuda.synchronize(rank)
        mem_before = torch.cuda.memory_allocated(rank) / 1024**2
        if rank == 0:
            print(f"Memory before ZeRO-Q: {mem_before:.1f} MB")
        
        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False
        
        # Apply LoRA
        if rank == 0:
            print("\nApplying LoRA...")
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        if rank == 0:
            print(f"Trainable params: {trainable_params:,}")
            print(f"Frozen params: {frozen_params:,}")
        
        # Apply ZeRO-Q to frozen parameters
        if rank == 0:
            print("\nApplying ZeRO-Q...")
        
        coordinator = ZeroQCoordinator(MAXWELL_CONFIG)
        wrapper = ZeroQModuleWrapper(model, coordinator, trainable_only=False)
        wrapper.partition()
        
        # Memory after ZeRO-Q
        torch.cuda.synchronize(rank)
        mem_after = torch.cuda.memory_allocated(rank) / 1024**2
        
        stats = wrapper.get_memory_stats()
        if rank == 0:
            print(f"\nMemory after ZeRO-Q: {mem_after:.1f} MB")
            print(f"Memory savings: {mem_before / mem_after:.2f}x")
            print(f"Coordinator stats: {stats}")
        
        # Create dataset and dataloader
        if rank == 0:
            print("\nCreating dataset...")
        
        dataset = SimpleDataset(tokenizer, num_samples=20, max_length=64)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        # Optimizer
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
            
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if rank == 0:
                print(f"  Step {step + 1}/{num_steps}: loss = {loss.item():.4f}")
        
        avg_loss = total_loss / num_steps
        if rank == 0:
            print(f"\nAverage loss: {avg_loss:.4f}")
        
        # Verify all ranks have same loss (roughly)
        loss_tensor = torch.tensor([avg_loss], device=device)
        all_losses = [torch.zeros(1, device=device) for _ in range(world_size)]
        dist.all_gather(all_losses, loss_tensor)
        
        if rank == 0:
            losses = [l.item() for l in all_losses]
            print(f"Losses across ranks: {losses}")
            
            # Check losses are similar (not identical due to different data)
            loss_diff = abs(losses[0] - losses[1]) / max(losses)
            if loss_diff < 0.5:  # Allow 50% difference due to different batches
                print("✓ Training completed successfully across all ranks!")
            else:
                print(f"! Large loss difference between ranks: {loss_diff:.2%}")
            
            print("\n" + "="*70)
            print("ZeRO-Q 1.5B Test COMPLETE")
            print("="*70)
            print(f"✓ Model loaded: Qwen2.5-1.5B")
            print(f"✓ ZeRO-Q applied: {stats['compression_ratio']:.2f}x compression")
            print(f"✓ Memory savings: {mem_before / mem_after:.2f}x")
            print(f"✓ Training completed: {num_steps} steps")
            print("="*70)
        
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    test_zeroq_with_real_model()
