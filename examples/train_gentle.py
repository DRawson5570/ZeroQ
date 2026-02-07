#!/usr/bin/env python3
"""
Gentle Multi-GPU Training - Experiment 1
Simple DDP-style gradient sync with throttling for stability.

NO ZeRO-Q weight partitioning - just proves distributed training works.
Designed to be gentle on hardware (power/thermal management).

Usage:
    # Single node (PE3 with 2 GPUs)
    torchrun --nproc_per_node=2 train_gentle.py
    
    # Multi-node when PE2 is back
    # PE2: torchrun --nproc_per_node=3 --nnodes=2 --node_rank=0 --master_addr=10.0.10.2 --master_port=29500 train_gentle.py
    # PE3: torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=10.0.10.2 --master_port=29500 train_gentle.py
"""

import os
import sys
import json
import time
import gc
from typing import Optional
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [Rank %(rank)s] %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - THE GAS PEDAL
# =============================================================================

# Model settings
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
TRAINING_DATA = os.path.expanduser("~/phoenix_training/phoenix_grok.jsonl")
OUTPUT_DIR = os.path.expanduser("~/phoenix_training/gentle_3b")

# Training settings
MAX_LENGTH = 128          # Short sequences = faster steps
BATCH_SIZE = 1            # Minimal batch
GRADIENT_ACCUMULATION = 4 # Effective batch = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
MAX_STEPS = 50            # Limit steps for testing

# LoRA settings
LORA_R = 8
LORA_ALPHA = 16

# =============================================================================
# THROTTLING CONFIG - BE GENTLE ON HARDWARE
# =============================================================================

THROTTLE_ENABLED = True           # Master switch for all throttling
STEP_DELAY_SEC = 0.5              # Pause between steps (lets GPUs cool/stabilize)
SYNC_DELAY_SEC = 0.1              # Pause before/after gradient sync
BARRIER_BEFORE_FORWARD = True     # Sync all ranks before forward pass
BARRIER_AFTER_BACKWARD = True     # Sync all ranks after backward pass
CUDA_SYNC_AFTER_STEP = True       # Force CUDA sync after each step
EMPTY_CACHE_EVERY_N_STEPS = 5     # Clear CUDA cache periodically
LOG_MEMORY_EVERY_N_STEPS = 10     # Log GPU memory usage

# Power management
GPU_POWER_LIMIT_WATTS = 200       # Limit GPU power (None = no limit, 200 = 80% of 250W TDP)


class LoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return msg, {**kwargs, 'extra': {'rank': self.extra.get('rank', '?')}}


class PhoenixDataset(Dataset):
    """Simple dataset for Phoenix training data."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
                self.examples.append(text)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0)
        }


def setup_distributed(log):
    """Initialize distributed training with extended timeouts."""
    timeout = timedelta(minutes=30)  # Long timeout for slow GPUs
    dist.init_process_group(backend='nccl', timeout=timeout)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    torch.cuda.set_device(local_rank)
    
    log.info(f"Initialized: rank {rank}/{world_size}, local_rank {local_rank}")
    return rank, world_size, local_rank


def set_gpu_power_limit(local_rank: int, watts: Optional[int], log):
    """Set GPU power limit to reduce power spikes."""
    if watts is None:
        return
    
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '-i', str(local_rank), '-pl', str(watts)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            log.info(f"Set GPU {local_rank} power limit to {watts}W")
        else:
            log.warning(f"Could not set power limit: {result.stderr}")
    except Exception as e:
        log.warning(f"Power limit setting failed: {e}")


def throttle_delay(seconds: float, reason: str = ""):
    """Sleep for throttling, only if enabled."""
    if THROTTLE_ENABLED and seconds > 0:
        time.sleep(seconds)


def sync_barrier(enabled: bool, name: str = ""):
    """Synchronization barrier with optional delay."""
    if enabled:
        throttle_delay(SYNC_DELAY_SEC, f"pre-barrier-{name}")
        dist.barrier()
        throttle_delay(SYNC_DELAY_SEC, f"post-barrier-{name}")


def load_model_and_tokenizer(local_rank: int, log):
    """Load quantized model with 4-bit bitsandbytes."""
    log.info(f"Loading {MODEL_NAME} with 4-bit quantization...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,  # FP32 for Maxwell compatibility
        bnb_4bit_use_double_quant=False
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={'': local_rank},
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model, tokenizer


def log_memory(local_rank: int, step: int, log):
    """Log GPU memory usage."""
    allocated = torch.cuda.memory_allocated(local_rank) / 1024**3
    reserved = torch.cuda.memory_reserved(local_rank) / 1024**3
    log.info(f"Step {step} - GPU {local_rank} memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def train_gentle(model, train_loader, optimizer, rank, world_size, local_rank, log):
    """Gentle training loop with throttling and synchronization."""
    model.train()
    total_loss = 0
    step = 0
    
    log.info(f"Starting gentle training - throttle={'ON' if THROTTLE_ENABLED else 'OFF'}")
    log.info(f"Settings: step_delay={STEP_DELAY_SEC}s, barriers={BARRIER_BEFORE_FORWARD}/{BARRIER_AFTER_BACKWARD}")
    
    for epoch in range(NUM_EPOCHS):
        log.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(train_loader):
            if step >= MAX_STEPS:
                log.info(f"Reached MAX_STEPS={MAX_STEPS}, stopping")
                break
            
            # === BARRIER: Sync before forward ===
            sync_barrier(BARRIER_BEFORE_FORWARD, "forward")
            
            # Move batch to GPU
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / GRADIENT_ACCUMULATION
            
            # Backward pass
            loss.backward()
            
            # === BARRIER: Sync after backward ===
            sync_barrier(BARRIER_AFTER_BACKWARD, "backward")
            
            # Gradient accumulation step
            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                # Synchronize gradients across all ranks
                throttle_delay(SYNC_DELAY_SEC, "pre-allreduce")
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
                throttle_delay(SYNC_DELAY_SEC, "post-allreduce")
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Force CUDA sync
                if CUDA_SYNC_AFTER_STEP:
                    torch.cuda.synchronize()
                
                step += 1
                avg_loss = loss.item() * GRADIENT_ACCUMULATION
                total_loss += avg_loss
                
                # Logging
                if rank == 0:
                    log.info(f"Step {step}: loss = {avg_loss:.4f}")
                
                # Periodic memory logging
                if step % LOG_MEMORY_EVERY_N_STEPS == 0:
                    log_memory(local_rank, step, log)
                
                # Periodic cache clearing
                if step % EMPTY_CACHE_EVERY_N_STEPS == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                
                # === THROTTLE: Pause between steps ===
                throttle_delay(STEP_DELAY_SEC, "inter-step")
        
        if step >= MAX_STEPS:
            break
    
    log.info(f"Training complete. {step} steps, avg loss: {total_loss / max(step, 1):.4f}")
    return total_loss / max(step, 1)


def main():
    """Main entry point."""
    # Basic logger until we have rank
    log = LoggerAdapter(logger, {'rank': '?'})
    
    # Initialize distributed
    rank, world_size, local_rank = setup_distributed(log)
    log = LoggerAdapter(logger, {'rank': rank})
    
    if rank == 0:
        log.info("=" * 60)
        log.info("GENTLE DISTRIBUTED TRAINING - Experiment 1")
        log.info(f"Model: {MODEL_NAME}")
        log.info(f"World size: {world_size} GPUs")
        log.info(f"Throttling: {'ENABLED' if THROTTLE_ENABLED else 'DISABLED'}")
        log.info("=" * 60)
    
    # Set GPU power limit (requires sudo or nvidia-smi permissions)
    # Commented out by default - uncomment if you have permissions
    # set_gpu_power_limit(local_rank, GPU_POWER_LIMIT_WATTS, log)
    
    # Barrier to ensure all ranks ready
    dist.barrier()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(local_rank, log)
    
    # Barrier after model load
    dist.barrier()
    
    # Create dataset
    dataset = PhoenixDataset(TRAINING_DATA, tokenizer, MAX_LENGTH)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    
    if rank == 0:
        log.info(f"Dataset: {len(dataset)} examples")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE
    )
    
    # Final barrier before training
    dist.barrier()
    
    # Train!
    final_loss = train_gentle(model, train_loader, optimizer, rank, world_size, local_rank, log)
    
    # Save (rank 0 only)
    if rank == 0:
        log.info(f"Final loss: {final_loss:.4f}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        log.info(f"Saved to {OUTPUT_DIR}")
    
    # Cleanup
    dist.barrier()
    dist.destroy_process_group()
    
    if rank == 0:
        log.info("Done!")


if __name__ == "__main__":
    main()
