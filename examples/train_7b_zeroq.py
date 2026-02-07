#!/usr/bin/env python3
"""
Example: Train a 7B model with ZeRO-Q on 2x Tesla M40

This demonstrates the full ZeRO-Q workflow:
1. Load model with 4-bit quantization
2. Apply LoRA for parameter-efficient training
3. Partition quantized weights across GPUs with ZeRO-Q
4. Train with standard HuggingFace Trainer

Requirements:
    pip install transformers peft bitsandbytes accelerate
    
Hardware:
    2+ GPUs with 12GB+ VRAM each (tested on Tesla M40)

Usage:
    torchrun --nproc_per_node=2 train_7b_zeroq.py
"""

import os
import sys
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional

# Add ZeRO-Q to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType
from datasets import Dataset

from src.config import MAXWELL_CONFIG, ZeroQConfig
from src.integration import ZeroQTrainer, apply_lora_with_zeroq


@dataclass
class TrainConfig:
    """Training configuration."""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "./zeroq_7b_output"
    
    # LoRA settings
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: tuple = ("q_proj", "v_proj")
    lora_dropout: float = 0.05
    
    # Training settings
    num_train_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    max_seq_length: int = 256
    
    # Memory settings (per GPU)
    max_memory_gb: float = 10.0


def create_dummy_dataset(tokenizer, num_samples: int = 100) -> Dataset:
    """Create a dummy dataset for testing."""
    
    examples = []
    for i in range(num_samples):
        text = f"Question: What is {i} + {i}?\nAnswer: {i + i}"
        examples.append({"text": text})
    
    dataset = Dataset.from_list(examples)
    
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        )
    
    return dataset.map(tokenize, remove_columns=["text"])


def main():
    config = TrainConfig()
    
    # Initialize distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    
    print(f"[Rank {local_rank}/{world_size}] Starting ZeRO-Q training example")
    
    # BitsAndBytes 4-bit config (works on M40!)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,  # FP32 for Maxwell
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    print(f"[Rank {local_rank}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 4-bit quantization
    print(f"[Rank {local_rank}] Loading model with 4-bit quantization...")
    
    # Calculate max memory per GPU
    max_memory = {
        i: f"{config.max_memory_gb}GiB" 
        for i in range(torch.cuda.device_count())
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    
    print(f"[Rank {local_rank}] Model loaded. Memory per GPU:")
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1024**3
        print(f"  GPU {i}: {mem:.2f} GB")
    
    # LoRA config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA and ZeRO-Q
    print(f"[Rank {local_rank}] Applying LoRA and ZeRO-Q...")
    zeroq_config = MAXWELL_CONFIG
    model = apply_lora_with_zeroq(model, lora_config, zeroq_config)
    
    # Print memory stats
    if hasattr(model, '_zeroq_wrapper'):
        stats = model._zeroq_wrapper.get_memory_stats()
        print(f"[Rank {local_rank}] ZeRO-Q stats:")
        print(f"  Local memory: {stats['local_memory_mb']:.2f} MB")
        print(f"  Full FP16 memory: {stats['full_fp16_memory_mb']:.2f} MB")
        print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    
    # Create dataset
    print(f"[Rank {local_rank}] Creating dataset...")
    train_dataset = create_dummy_dataset(tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=False,  # Maxwell doesn't support FP16
        bf16=False,
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="none",
    )
    
    # Create trainer
    trainer = ZeroQTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        zeroq_config=zeroq_config,
    )
    
    # Train!
    print(f"[Rank {local_rank}] Starting training...")
    trainer.train()
    
    print(f"[Rank {local_rank}] Training complete!")
    
    # Save LoRA adapter (not the full model)
    if local_rank == 0:
        model.save_pretrained(config.output_dir)
        print(f"Saved LoRA adapter to {config.output_dir}")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
