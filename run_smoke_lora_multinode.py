#!/usr/bin/env python3
"""ZeroQ multi-node LoRA smoke training.

Goals:
- 2 nodes (pe1 + pe3), 1 GPU each (use torchrun --nproc_per_node=1).
- Tiny synthetic dataset (no external files required).
- Support --quant flag: 4bit (ZeRO-Q shard+gather), 8bit (bnb int8 load), none.

Examples (run on both nodes with different --node_rank):
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=pe1 --master_port=29500 run_smoke_lora_multinode.py --quant 4bit
  torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=pe1 --master_port=29500 run_smoke_lora_multinode.py --quant 4bit

Notes:
- quant=4bit uses ZeRO-Q's own bitsandbytes.functional quantize_4bit partitioning to "spread" base weights across ranks.
- We do manual gradient all-reduce on trainable (LoRA) params to avoid DDP assumptions about partitioned base params.
"""

from __future__ import annotations

import argparse
import os
import random
import socket
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)


@dataclass
class SmokeConfig:
    model: str
    quant: str
    steps: int
    seq_len: int
    batch_size: int
    lr: float
    seed: int
    cache_dir: Optional[str]


class TinyTextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, seq_len: int, num_samples: int, seed: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_samples = num_samples
        rng = random.Random(seed)

        base = [
            "Phoenix is a distributed compute cluster.",
            "ZeRO-Q shards parameters across GPUs.",
            "LoRA adapts large models efficiently.",
            "Quantization enables larger models on limited VRAM.",
            "Test sample for smoke training.",
        ]
        self.texts = [rng.choice(base) + f" sample={i}" for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _rank0_print(rank: int, msg: str):
    if rank == 0:
        print(msg, flush=True)


def _init_distributed() -> tuple[int, int, int, torch.device]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, world_size, local_rank, device


def _manual_allreduce_grads(model: torch.nn.Module, world_size: int):
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


def _load_model_and_tokenizer(cfg: SmokeConfig, local_rank: int, device: torch.device, rank: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True, cache_dir=cfg.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = cfg.quant.lower()

    if quant == "none":
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir=cfg.cache_dir,
        ).to(device)
        return model, tokenizer

    if quant == "8bit":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError("transformers BitsAndBytesConfig not available; install transformers>=4.30") from e

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map={"": local_rank},
            cache_dir=cfg.cache_dir,
        )
        return model, tokenizer

    if quant == "4bit":
        # Load in fp16, then ZeRO-Q partitions to q4 shards across ranks.
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir=cfg.cache_dir,
        ).to(device)
        return model, tokenizer

    raise ValueError(f"Unknown --quant: {cfg.quant}")


def _apply_lora_and_zeroq(cfg: SmokeConfig, model: torch.nn.Module, rank: int):
    from peft import LoraConfig, get_peft_model

    # Freeze everything, then LoRA will mark its params trainable
    for p in model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if cfg.quant.lower() == "4bit":
        # ZeRO-Q partitions only frozen weights (see config.frozen_only).
        from src.config import MAXWELL_CONFIG
        from src.coordinator import ZeroQCoordinator, ZeroQModuleWrapper

        coordinator = ZeroQCoordinator(MAXWELL_CONFIG)
        wrapper = ZeroQModuleWrapper(model, coordinator, trainable_only=False)

        _rank0_print(rank, "[ZeroQ] Partitioning base weights into q4 shards...")
        wrapper.partition()

        stats = wrapper.get_memory_stats()
        _rank0_print(rank, f"[ZeroQ] compression_ratio={stats['compression_ratio']:.2f}x, num_params={stats['num_params']}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _rank0_print(rank, f"Trainable (LoRA) params: {trainable:,}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B"),
        help="HuggingFace model id (default: Qwen/Qwen2.5-1.5B)",
    )
    parser.add_argument(
        "--quant",
        default=os.environ.get("QUANT", "4bit"),
        choices=["4bit", "8bit", "none"],
        help="Quantization mode: 4bit (ZeRO-Q shard), 8bit (bnb int8 load), none",
    )
    parser.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", "2")))
    parser.add_argument("--seq-len", type=int, default=int(os.environ.get("SEQ_LEN", "256")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH", "1")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-4")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "1234")))
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_CACHE_DIR"),
        help="Optional HF cache_dir to keep both nodes consistent.",
    )

    args = parser.parse_args()
    cfg = SmokeConfig(
        model=args.model,
        quant=args.quant,
        steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )

    rank, world_size, local_rank, device = _init_distributed()
    hostname = socket.gethostname()

    dist.barrier()
    print(f"[Rank {rank}] host={hostname} local_rank={local_rank} device={device} quant={cfg.quant} model={cfg.model}", flush=True)
    dist.barrier()

    torch.manual_seed(cfg.seed)

    _rank0_print(rank, "\n=== ZeroQ LoRA Smoke (multinode) ===")

    # Load
    t0 = time.time()
    model, tokenizer = _load_model_and_tokenizer(cfg, local_rank, device, rank)
    _rank0_print(rank, f"Loaded model+tokenizer in {time.time() - t0:.1f}s")

    # Apply LoRA (+ ZeRO-Q sharding if quant=4bit)
    model = _apply_lora_and_zeroq(cfg, model, rank)

    # Dataset
    dataset = TinyTextDataset(tokenizer, seq_len=cfg.seq_len, num_samples=max(cfg.steps * cfg.batch_size, 8), seed=cfg.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    # Optimizer (LoRA params only)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    model.train()

    losses = []
    it = iter(loader)
    for step in range(cfg.steps):
        batch = next(it)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optim.zero_grad(set_to_none=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        # Data-parallel sync for trainable LoRA params.
        _manual_allreduce_grads(model, world_size)

        optim.step()

        losses.append(loss.detach().float())
        if rank == 0:
            print(f"step {step+1}/{cfg.steps} loss={loss.item():.4f}", flush=True)

    # Report
    avg = torch.stack(losses).mean()
    avg_all = avg.clone()
    dist.all_reduce(avg_all, op=dist.ReduceOp.SUM)
    avg_all /= world_size

    _rank0_print(rank, f"avg_loss_local={avg.item():.4f} avg_loss_world={avg_all.item():.4f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
