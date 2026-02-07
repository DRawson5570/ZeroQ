#!/usr/bin/env python3
"""
ZeRO-Q Distributed Training on Phoenix Cluster
Real multi-node 4-bit quantized training with gradient sync

Usage (from PE2):
    # Single node, multi-GPU
    torchrun --nproc_per_node=3 train_distributed_zeroq.py
    
    # Multi-node (PE2 as master, PE3 as worker)
    # On PE2:
    torchrun --nproc_per_node=3 --nnodes=2 --node_rank=0 \
             --master_addr=192.168.1.69 --master_port=29500 \
             train_distributed_zeroq.py
    # On PE3:
    torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 \
             --master_addr=192.168.1.69 --master_port=29500 \
             train_distributed_zeroq.py
"""

import os
import sys
import json
import gc
from typing import Dict, Any
from dataclasses import dataclass
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

# === Configuration ===
MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"  # Start with 3B for testing
TRAINING_DATA = os.path.expanduser("~/phoenix_training/phoenix_grok.jsonl")
OUTPUT_DIR = os.path.expanduser("~/phoenix_training/zeroq_grok_3b")
MAX_LENGTH = 128  # Reduced from 256 to avoid GPU timeout
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
LORA_R = 8
LORA_ALPHA = 16
DISABLE_GRADIENT_CHECKPOINTING = True  # Faster but uses more memory


@dataclass
class PartitionedParam:
    """Stores partition info for a quantized parameter."""
    name: str
    local_data: torch.Tensor
    local_absmax: torch.Tensor
    data_size: int
    absmax_size: int


class BnbZeroQCoordinator:
    """
    Coordinates partitioning of bitsandbytes Params4bit weights.
    Works with already-quantized models from HuggingFace.
    """
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.partitions: Dict[str, PartitionedParam] = {}
        self.original_memory = 0
        self.partitioned_memory = 0
        
    def partition_model(self, model: torch.nn.Module, free_non_local: bool = True) -> int:
        """Partition all Params4bit weights."""
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                weight = module.weight
                if hasattr(weight, 'data') and hasattr(weight, 'quant_state'):
                    self._partition_param(name, weight, free_non_local)
                    count += 1
        
        if free_non_local:
            gc.collect()
            torch.cuda.empty_cache()
            
        return count
    
    def _partition_param(self, name: str, param, free_non_local: bool):
        """Partition a single Params4bit parameter."""
        packed_data = param.data
        quant_state = param.quant_state
        
        # Handle different bitsandbytes versions
        # Older versions (0.39.x) use list, newer use tensor
        if isinstance(quant_state, list):
            # Old format: quant_state is [absmax, shape, dtype, blocksize, ...]
            absmax = quant_state[0] if len(quant_state) > 0 else None
            if absmax is None:
                logger.warning(f"Skipping {name}: no absmax in quant_state")
                return
        elif hasattr(quant_state, 'absmax'):
            absmax = quant_state.absmax
        else:
            logger.warning(f"Skipping {name}: unknown quant_state format")
            return
        
        self.original_memory += packed_data.numel() + absmax.numel() * 4
        
        # Partition packed data
        data_size = packed_data.numel()
        data_per_rank = data_size // self.world_size
        data_start = self.rank * data_per_rank
        data_end = data_start + data_per_rank if self.rank < self.world_size - 1 else data_size
        
        # Partition absmax
        absmax_size = absmax.numel()
        absmax_per_rank = absmax_size // self.world_size
        absmax_start = self.rank * absmax_per_rank
        absmax_end = absmax_start + absmax_per_rank if self.rank < self.world_size - 1 else absmax_size
        
        # Clone local partitions
        local_data = packed_data.data[data_start:data_end].clone()
        local_absmax = absmax[absmax_start:absmax_end].clone()
        
        self.partitioned_memory += local_data.numel() + local_absmax.numel() * 4
        
        self.partitions[name] = PartitionedParam(
            name=name,
            local_data=local_data,
            local_absmax=local_absmax,
            data_size=data_size,
            absmax_size=absmax_size,
        )

    def synchronize_gradients(self, model: torch.nn.Module):
        """All-reduce gradients across ranks for LoRA parameters."""
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        
    def get_stats(self) -> Dict[str, Any]:
        return {
            'num_partitions': len(self.partitions),
            'original_memory_mb': self.original_memory / 1024**2,
            'partitioned_memory_mb': self.partitioned_memory / 1024**2,
            'memory_savings': self.original_memory / max(self.partitioned_memory, 1),
        }


class PhoenixDataset(Dataset):
    """Dataset for Phoenix grokking training."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        logger.info(f"Loading training data from {data_path}")
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Format: instruction -> output
                text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
                self.examples.append(text)
        
        logger.info(f"Loaded {len(self.examples)} training examples")
    
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


def setup_distributed():
    """Initialize distributed training."""
    import datetime
    # CRITICAL: Use very long timeout for M40 GPUs over network
    # Default is 30 min, we need much more for slow GPUs
    timeout = datetime.timedelta(minutes=60)
    dist.init_process_group(backend='nccl', timeout=timeout)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    torch.cuda.set_device(local_rank)
    
    logger.info(f"Rank {rank}/{world_size} initialized on GPU {local_rank}")
    return rank, world_size, local_rank


def load_model_and_tokenizer(local_rank: int):
    """Load model with 4-bit quantization."""
    logger.info(f"Loading {MODEL_NAME} with 4-bit quantization...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,  # FP32 for Maxwell
        bnb_4bit_use_double_quant=False
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={'': local_rank},
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # Gradient checkpointing - disable for faster steps (avoids GPU timeout)
    if not DISABLE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing ENABLED (slower but less memory)")
    else:
        logger.info("Gradient checkpointing DISABLED (faster steps)")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model, tokenizer


def train_with_zeroq(model, train_loader, optimizer, coordinator, rank, world_size):
    """Training loop with ZeRO-Q coordination."""
    model.train()
    total_loss = 0
    step = 0
    
    for epoch in range(NUM_EPOCHS):
        logger.info(f"[Rank {rank}] Starting epoch {epoch + 1}/{NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(train_loader):
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
            
            # Step optimizer every GRADIENT_ACCUMULATION steps
            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                # Synchronize gradients across ranks
                coordinator.synchronize_gradients(model)
                
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                
                # Log progress
                avg_loss = loss.item() * GRADIENT_ACCUMULATION
                total_loss += avg_loss
                
                if rank == 0:
                    logger.info(f"Step {step}: loss = {avg_loss:.4f}")
        
        # End of epoch
        if rank == 0:
            logger.info(f"Epoch {epoch + 1} complete. Avg loss: {total_loss / max(step, 1):.4f}")
    
    return total_loss / max(step, 1)


def main():
    """Main training function."""
    # Initialize distributed
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        logger.info("=" * 60)
        logger.info("ZeRO-Q Distributed Training")
        logger.info(f"Model: {MODEL_NAME}")
        logger.info(f"World size: {world_size} GPUs")
        logger.info("=" * 60)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(local_rank)
    
    # Initialize ZeRO-Q coordinator
    coordinator = BnbZeroQCoordinator(rank=rank, world_size=world_size)
    
    if rank == 0:
        logger.info("Partitioning model weights with ZeRO-Q...")
    
    num_partitioned = coordinator.partition_model(model)
    stats = coordinator.get_stats()
    
    if rank == 0:
        logger.info(f"ZeRO-Q: {num_partitioned} layers partitioned")
        logger.info(f"Memory: {stats['original_memory_mb']:.1f}MB -> {stats['partitioned_memory_mb']:.1f}MB ({stats['memory_savings']:.2f}x savings)")
    
    # Create dataset and dataloader
    dataset = PhoenixDataset(TRAINING_DATA, tokenizer, MAX_LENGTH)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    
    # Optimizer (only for LoRA params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE
    )
    
    # Train!
    if rank == 0:
        logger.info("Starting training...")
    
    final_loss = train_with_zeroq(model, train_loader, optimizer, coordinator, rank, world_size)
    
    # Save model (rank 0 only)
    if rank == 0:
        logger.info(f"Training complete! Final loss: {final_loss:.4f}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        logger.info(f"Model saved to {OUTPUT_DIR}")
    
    # Cleanup
    dist.destroy_process_group()
    
    if rank == 0:
        logger.info("Done!")


if __name__ == "__main__":
    main()
