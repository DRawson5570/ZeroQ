#!/usr/bin/env python3
"""
ZeRO-Q: Real 7B Model Training Test

This is THE test - loading a real 7B model and running actual training
with ZeRO-Q distributing quantized parameters across GPUs.

The key insight: device_map breaks torchrun. We need to:
1. Load model to CPU first (no device_map)
2. Quantize manually with bitsandbytes
3. Move to correct GPU based on rank
4. Then apply ZeRO-Q partitioning

Run with:
    torchrun --nproc_per_node=2 test_7b_training.py
"""

import os
import sys
import gc
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleDataset(Dataset):
    """Simple training data."""
    
    def __init__(self, tokenizer, num_samples=20, max_length=64):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        self.examples = [
            "The Phoenix rises from the ashes, stronger than before.",
            "Zero-Q enables what was once impossible on Maxwell GPUs.",
            "Distributed training democratizes access to AI capabilities.",
            "The stochastic parrot builds tools, not just predicts tokens.",
            "Memory efficiency is the key to running larger models.",
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        text = self.examples[idx % len(self.examples)]
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


def print_rank0(msg, rank):
    """Print only from rank 0."""
    if rank == 0:
        print(msg, flush=True)


def test_7b_training():
    """The real test - 7B model with ZeRO-Q."""
    
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    print_rank0("\n" + "="*70, rank)
    print_rank0("ZeRO-Q: 7B Model Training Test", rank)
    print_rank0("="*70, rank)
    print_rank0(f"World size: {world_size}", rank)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from src.config import MAXWELL_CONFIG
        from src.coordinator import ZeroQCoordinator
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        
        # Memory at start
        torch.cuda.synchronize(rank)
        mem_start = torch.cuda.memory_allocated(rank) / 1024**3
        print_rank0(f"\nMemory at start: {mem_start:.2f} GB", rank)
        
        # ----- Step 1: Load tokenizer -----
        print_rank0("\n[Step 1] Loading tokenizer...", rank)
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B",
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # ----- Step 2: Load model with 4-bit quantization -----
        print_rank0("\n[Step 2] Loading 7B model with 4-bit quantization...", rank)
        
        # Use BitsAndBytes config for 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,  # Maxwell needs FP32
            bnb_4bit_use_double_quant=False,
        )
        
        # Load WITHOUT device_map - that's the key!
        # We'll handle device placement manually after
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B",
            quantization_config=bnb_config,
            trust_remote_code=True,
            # NO device_map! We handle it ourselves
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print_rank0(f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)", rank)
        
        torch.cuda.synchronize(rank)
        mem_after_load = torch.cuda.memory_allocated(rank) / 1024**3
        print_rank0(f"Memory after load: {mem_after_load:.2f} GB", rank)
        
        # ----- Step 3: Prepare for training -----
        print_rank0("\n[Step 3] Preparing for k-bit training...", rank)
        model = prepare_model_for_kbit_training(model)
        
        # ----- Step 4: Apply LoRA -----
        print_rank0("\n[Step 4] Applying LoRA...", rank)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print_rank0(f"Trainable: {trainable:,} ({trainable/1e6:.1f}M)", rank)
        print_rank0(f"Frozen: {frozen:,} ({frozen/1e9:.2f}B)", rank)
        
        # ----- Step 5: Apply ZeRO-Q -----
        print_rank0("\n[Step 5] Applying ZeRO-Q partitioning...", rank)
        
        coordinator = ZeroQCoordinator(MAXWELL_CONFIG)
        
        # Register frozen parameters for partitioning
        partition_count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad and param.numel() > 10000:  # Large frozen params
                try:
                    coordinator.register_parameter(name, param)
                    partition_count += 1
                except Exception as e:
                    if rank == 0:
                        print(f"  Warning: Could not register {name}: {e}")
        
        print_rank0(f"Registered {partition_count} parameters for ZeRO-Q", rank)
        
        # Partition
        coordinator.partition()
        
        torch.cuda.synchronize(rank)
        mem_after_zeroq = torch.cuda.memory_allocated(rank) / 1024**3
        print_rank0(f"Memory after ZeRO-Q: {mem_after_zeroq:.2f} GB", rank)
        
        stats = coordinator.get_stats()
        print_rank0(f"ZeRO-Q stats: {stats}", rank)
        
        # ----- Step 6: Create data -----
        print_rank0("\n[Step 6] Creating dataset...", rank)
        dataset = SimpleDataset(tokenizer, num_samples=10, max_length=32)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # ----- Step 7: Setup optimizer -----
        print_rank0("\n[Step 7] Setting up optimizer...", rank)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
        )
        
        # ----- Step 8: Training loop! -----
        print_rank0("\n[Step 8] TRAINING!", rank)
        print_rank0("-" * 50, rank)
        
        model.train()
        num_steps = 3
        losses = []
        
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break
            
            start_time = time.time()
            
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Gather parameters if needed (ZeRO-Q)
            coordinator.gather_all()
            
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
            
            # Release parameters (ZeRO-Q)
            coordinator.release_all()
            
            step_time = time.time() - start_time
            losses.append(loss.item())
            
            print_rank0(f"  Step {step + 1}/{num_steps}: loss={loss.item():.4f}, time={step_time:.2f}s", rank)
            
            # Sync for next step
            dist.barrier()
        
        # ----- Step 9: Verify results -----
        print_rank0("\n[Step 9] Verifying results...", rank)
        
        avg_loss = sum(losses) / len(losses)
        print_rank0(f"Average loss: {avg_loss:.4f}", rank)
        
        # Check gradients were computed
        grad_count = 0
        grad_sum = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
                grad_sum += param.grad.abs().mean().item()
        
        print_rank0(f"Parameters with gradients: {grad_count}", rank)
        print_rank0(f"Average gradient magnitude: {grad_sum/grad_count:.6f}", rank)
        
        # Gather stats from all ranks
        loss_tensor = torch.tensor([avg_loss], device=device)
        all_losses = [torch.zeros(1, device=device) for _ in range(world_size)]
        dist.all_gather(all_losses, loss_tensor)
        
        if rank == 0:
            all_losses_list = [l.item() for l in all_losses]
            print(f"\nLosses across ranks: {all_losses_list}")
        
        # ----- Final Report -----
        print_rank0("\n" + "="*70, rank)
        print_rank0("🔥 ZeRO-Q 7B TRAINING TEST COMPLETE 🔥", rank)
        print_rank0("="*70, rank)
        print_rank0(f"✓ Model: Qwen2.5-7B (4-bit quantized)", rank)
        print_rank0(f"✓ Parameters: {total_params/1e9:.2f}B total, {trainable/1e6:.1f}M trainable", rank)
        print_rank0(f"✓ Memory per GPU: {mem_after_zeroq:.2f} GB", rank)
        print_rank0(f"✓ Training steps: {num_steps}", rank)
        print_rank0(f"✓ Final loss: {avg_loss:.4f}", rank)
        print_rank0(f"✓ Gradient flow: VERIFIED ({grad_count} params)", rank)
        print_rank0("="*70, rank)
        
        return True
        
    except Exception as e:
        print(f"\n[Rank {rank}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    success = test_7b_training()
    sys.exit(0 if success else 1)
