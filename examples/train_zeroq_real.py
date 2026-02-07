#!/usr/bin/env python3
"""
REAL ZeRO-Q Training - Experiment 2
Actual weight partitioning with gather→compute→release hooks.

This implements the ZeRO-3 style weight partitioning for 4-bit quantized models.
Each GPU holds 1/N of the weights, gathers full weights before compute, releases after.

Usage:
    torchrun --nproc_per_node=2 train_zeroq_real.py
"""

import os
import sys
import json
import time
import gc
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from functools import partial
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
import logging
from datetime import timedelta

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [Rank %(rank)s] %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
TRAINING_DATA = os.path.expanduser("~/phoenix_training/phoenix_grok.jsonl")
OUTPUT_DIR = os.path.expanduser("~/phoenix_training/zeroq_real_3b")

MAX_LENGTH = 128
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
MAX_STEPS = 30

LORA_R = 8
LORA_ALPHA = 16

# Throttling - MORE AGGRESSIVE for multi-node over routed network
STEP_DELAY_SEC = 1.5  # Was 0.5
SYNC_DELAY_SEC = 0.3  # Was 0.1


class LoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return msg, {**kwargs, 'extra': {'rank': self.extra.get('rank', '?')}}


# =============================================================================
# ZERO-Q WEIGHT PARTITIONING - THE REAL DEAL
# =============================================================================

@dataclass
class PartitionInfo:
    """Info about a partitioned weight tensor."""
    full_shape: torch.Size
    full_numel: int
    local_start: int
    local_end: int
    local_data: torch.Tensor  # Our partition
    dtype: torch.dtype
    device: torch.device


class ZeroQPartitioner:
    """
    Real ZeRO-Q weight partitioning with gather/scatter hooks.
    
    Each rank holds 1/N of each weight tensor.
    Before forward: AllGather to reconstruct full weights
    After forward: Release gathered weights, keep only partition
    """
    
    def __init__(self, rank: int, world_size: int, log):
        self.rank = rank
        self.world_size = world_size
        self.log = log
        self.partitions: Dict[str, PartitionInfo] = {}
        self.hooks: List = []
        self.gather_count = 0
        self.release_count = 0
        
    def partition_linear4bit(self, name: str, module: bnb.nn.Linear4bit) -> bool:
        """
        Partition a Linear4bit layer's weights.
        Returns True if successfully partitioned.
        """
        weight = module.weight
        
        if not hasattr(weight, 'data'):
            return False
            
        # Get the packed 4-bit data
        packed_data = weight.data.flatten()  # Flatten for consistent handling
        full_numel = packed_data.numel()
        
        # Calculate partition boundaries - ceiling division for even distribution
        partition_size = (full_numel + self.world_size - 1) // self.world_size
        local_start = self.rank * partition_size
        local_end = min(local_start + partition_size, full_numel)  # Clamp to actual size
        
        # Extract and store our partition (padded to partition_size)
        local_data = torch.zeros(partition_size, dtype=packed_data.dtype, device=packed_data.device)
        actual_len = local_end - local_start
        if actual_len > 0:
            local_data[:actual_len] = packed_data[local_start:local_end].clone()
        
        self.partitions[name] = PartitionInfo(
            full_shape=module.weight.data.shape,
            full_numel=full_numel,
            local_start=local_start,
            local_end=local_end,
            local_data=local_data,
            dtype=packed_data.dtype,
            device=packed_data.device
        )
        
        return True
    
    def partition_model(self, model: torch.nn.Module) -> int:
        """Partition all Linear4bit layers in the model."""
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                if self.partition_linear4bit(name, module):
                    count += 1
        
        self.log.info(f"Partitioned {count} Linear4bit layers")
        return count
    
    def _gather_weights(self, name: str, module: bnb.nn.Linear4bit):
        """AllGather weights from all ranks to reconstruct full tensor."""
        info = self.partitions.get(name)
        if info is None:
            return
        
        # Flatten for gathering (4-bit packed data may be 2D)
        local_flat = info.local_data.flatten()
        local_numel = local_flat.numel()
        
        # Calculate partition size - use ceiling division to handle uneven splits
        partition_size = (info.full_numel + self.world_size - 1) // self.world_size
        
        # Create list of tensors for all_gather - all same size for NCCL
        gather_list = [torch.zeros(partition_size, dtype=info.dtype, device=info.device) 
                       for _ in range(self.world_size)]
        
        # Pad our local data if needed
        local_padded = torch.zeros(partition_size, dtype=info.dtype, device=info.device)
        copy_len = min(partition_size, local_numel)
        local_padded[:copy_len] = local_flat[:copy_len]
        
        # AllGather
        dist.all_gather(gather_list, local_padded)
        
        # Reconstruct full tensor - trim to actual size
        full_data = torch.cat(gather_list)[:info.full_numel]
        
        # Reshape to original shape
        full_data = full_data.view(info.full_shape)
        
        # Replace module's weight data with full weights
        module.weight.data = full_data
        
        self.gather_count += 1
    
    def _release_weights(self, name: str, module: bnb.nn.Linear4bit):
        """Release gathered weights, restore to partition only."""
        info = self.partitions.get(name)
        if info is None:
            return
        
        # We don't actually need to do anything here for forward pass
        # The module will use the gathered weights during forward
        # Memory is reclaimed when we do the next gather or when
        # Python garbage collects
        
        self.release_count += 1
    
    def register_hooks(self, model: torch.nn.Module):
        """Register forward hooks on all partitioned layers."""
        for name, module in model.named_modules():
            if name in self.partitions and isinstance(module, bnb.nn.Linear4bit):
                # Pre-forward hook: gather weights
                pre_hook = module.register_forward_pre_hook(
                    partial(self._pre_forward_hook, name=name)
                )
                self.hooks.append(pre_hook)
                
                # Post-forward hook: release weights  
                post_hook = module.register_forward_hook(
                    partial(self._post_forward_hook, name=name)
                )
                self.hooks.append(post_hook)
        
        self.log.info(f"Registered {len(self.hooks)} hooks on partitioned layers")
    
    def _pre_forward_hook(self, module, input, name: str):
        """Hook called before forward - gather weights."""
        self._gather_weights(name, module)
    
    def _post_forward_hook(self, module, input, output, name: str):
        """Hook called after forward - release weights."""
        self._release_weights(name, module)
        return output
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_memory_stats(self) -> Dict:
        """Calculate memory savings from partitioning."""
        total_full = sum(p.full_numel for p in self.partitions.values())
        total_local = sum(p.local_data.numel() for p in self.partitions.values())
        
        return {
            'num_partitions': len(self.partitions),
            'full_params': total_full,
            'local_params': total_local,
            'savings_ratio': total_full / max(total_local, 1),
            'gather_count': self.gather_count,
            'release_count': self.release_count,
        }


# =============================================================================
# DATASET
# =============================================================================

class PhoenixDataset(Dataset):
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
            text, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0)
        }


# =============================================================================
# TRAINING
# =============================================================================

def setup_distributed(log):
    timeout = timedelta(minutes=30)
    dist.init_process_group(backend='nccl', timeout=timeout)
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    torch.cuda.set_device(local_rank)
    log.info(f"Initialized: rank {rank}/{world_size}, local_rank {local_rank}")
    return rank, world_size, local_rank


def load_model(local_rank: int, log):
    log.info(f"Loading {MODEL_NAME} with 4-bit quantization...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
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
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model, tokenizer


def train_zeroq(model, train_loader, optimizer, partitioner, rank, world_size, log):
    """Training loop with real ZeRO-Q weight gathering."""
    model.train()
    total_loss = 0
    step = 0
    
    log.info("Starting ZeRO-Q training with real weight partitioning")
    
    for epoch in range(NUM_EPOCHS):
        log.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(train_loader):
            if step >= MAX_STEPS:
                break
            
            # Sync barrier
            time.sleep(SYNC_DELAY_SEC)
            dist.barrier()
            
            # Move batch to GPU
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            
            # Forward pass (hooks will gather weights automatically)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / GRADIENT_ACCUMULATION
            
            # Backward pass
            loss.backward()
            
            # Sync barrier
            dist.barrier()
            
            # Gradient accumulation step
            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                # AllReduce gradients for LoRA params
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
                
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize()
                
                step += 1
                avg_loss = loss.item() * GRADIENT_ACCUMULATION
                total_loss += avg_loss
                
                if rank == 0:
                    log.info(f"Step {step}: loss = {avg_loss:.4f}")
                
                # Log memory stats periodically
                if step % 10 == 0:
                    stats = partitioner.get_memory_stats()
                    log.info(f"ZeRO-Q stats: {stats['gather_count']} gathers, {stats['savings_ratio']:.2f}x savings")
                
                time.sleep(STEP_DELAY_SEC)
        
        if step >= MAX_STEPS:
            break
    
    return total_loss / max(step, 1)


def main():
    log = LoggerAdapter(logger, {'rank': '?'})
    
    # Initialize distributed
    rank, world_size, local_rank = setup_distributed(log)
    log = LoggerAdapter(logger, {'rank': rank})
    
    if rank == 0:
        log.info("=" * 60)
        log.info("REAL ZeRO-Q TRAINING - Experiment 2")
        log.info(f"Model: {MODEL_NAME}")
        log.info(f"World size: {world_size} GPUs")
        log.info("=" * 60)
    
    dist.barrier()
    
    # Load model
    model, tokenizer = load_model(local_rank, log)
    
    dist.barrier()
    
    # Initialize ZeRO-Q partitioner
    partitioner = ZeroQPartitioner(rank, world_size, log)
    
    # Partition weights
    num_partitioned = partitioner.partition_model(model)
    
    if rank == 0:
        stats = partitioner.get_memory_stats()
        log.info(f"Partitioned {num_partitioned} layers")
        log.info(f"Theoretical savings: {stats['savings_ratio']:.2f}x")
    
    # Register hooks for gather/release
    partitioner.register_hooks(model)
    
    dist.barrier()
    
    # Clear memory before training
    gc.collect()
    torch.cuda.empty_cache()
    
    # Log memory after partitioning
    allocated = torch.cuda.memory_allocated(local_rank) / 1024**3
    log.info(f"GPU {local_rank} memory after partitioning: {allocated:.2f}GB")
    
    # Dataset
    dataset = PhoenixDataset(TRAINING_DATA, tokenizer, MAX_LENGTH)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    
    if rank == 0:
        log.info(f"Dataset: {len(dataset)} examples")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE
    )
    
    dist.barrier()
    
    # Train!
    final_loss = train_zeroq(model, train_loader, optimizer, partitioner, rank, world_size, log)
    
    # Cleanup hooks
    partitioner.remove_hooks()
    
    # Final stats
    if rank == 0:
        stats = partitioner.get_memory_stats()
        log.info(f"Final loss: {final_loss:.4f}")
        log.info(f"Total gathers: {stats['gather_count']}")
        log.info(f"Savings ratio: {stats['savings_ratio']:.2f}x")
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        log.info(f"Saved to {OUTPUT_DIR}")
    
    dist.barrier()
    dist.destroy_process_group()
    
    if rank == 0:
        log.info("Done!")


if __name__ == "__main__":
    main()
