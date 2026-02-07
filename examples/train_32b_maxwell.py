#!/usr/bin/env python3
"""ZeRO-Q multi-node QLoRA training (Maxwell / Tesla M40).

This example is intentionally minimal and is designed to:
- Initialize NCCL distributed via torchrun (multi-node supported)
- Instantiate a large HF model without materializing full weights
- Stream weights from safetensors shards and partition them with ZeRO-Q
- Attach LoRA adapters (trainable) and run a short training smoke

Typical multi-node usage (PE2 master, PE3 worker):

  # PE2
  torchrun --nproc_per_node=5 --nnodes=2 --node_rank=0 \
    --master_addr=10.0.10.2 --master_port=29501 \
    train_32b_maxwell.py --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --data ~/phoenix_training/phoenix_grok.jsonl --max_steps 2

  # PE3
  torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 \
    --master_addr=10.0.10.2 --master_port=29501 \
    train_32b_maxwell.py --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --data ~/phoenix_training/phoenix_grok.jsonl --max_steps 2
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import math

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from accelerate import init_empty_weights
except Exception as e:  # pragma: no cover
    raise ImportError("accelerate is required for this example") from e

try:
    from safetensors.torch import safe_open
except Exception as e:  # pragma: no cover
    raise ImportError("safetensors is required for streamed loading") from e

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None

try:
    from peft import LoraConfig, get_peft_model
except Exception as e:  # pragma: no cover
    raise ImportError("peft is required for LoRA") from e

# Ensure `ZeroQ/src` is importable when running this file directly.
ZEROQ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ZEROQ_ROOT))

from src.config import MAXWELL_CONFIG
from src.coordinator import ZeroQCoordinator, ZeroQModuleWrapper


logger = logging.getLogger("zeroq.train_32b_maxwell")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")


class JsonlInstructionDataset(Dataset):
    def __init__(self, path: str, tokenizer, seq_len: int):
        self._tokenizer = tokenizer
        self._seq_len = seq_len
        self._examples: list[dict[str, str]] = []
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                inst = str(item.get("instruction", "")).strip()
                inp = str(item.get("input", "")).strip()
                out = str(item.get("output", "")).strip()

                if tokenizer.eos_token and out and not out.endswith(tokenizer.eos_token):
                    out = out + tokenizer.eos_token

                if inp:
                    prompt = "\n\n".join(
                        [
                            "### Instruction:\n" + inst,
                            "### Input:\n" + inp,
                            "### Response:\n",
                        ]
                    )
                else:
                    prompt = "\n\n".join(
                        [
                            "### Instruction:\n" + inst,
                            "### Response:\n",
                        ]
                    )

                self._examples.append({"prompt": prompt, "output": out})

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx: int):
        ex = self._examples[idx]
        prompt = ex["prompt"]
        output_text = ex["output"]

        # Tokenize prompt and output separately so we can *always* reserve response tokens.
        # This avoids edge cases where truncation consumes the entire sequence and yields
        # an all-ignored label tensor (which can produce NaN loss in some models).
        prompt_ids = self._tokenizer(
            prompt,
            truncation=False,
            padding=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        out_ids = self._tokenizer(
            output_text,
            truncation=False,
            padding=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        # Ensure there's at least one output token.
        if int(out_ids.numel()) == 0:
            eos_id = self._tokenizer.eos_token_id
            if eos_id is None:
                eos_id = 0
            out_ids = torch.tensor([int(eos_id)], dtype=torch.long)

        min_out = 1
        max_prompt_len = max(int(self._seq_len) - min_out, 0)
        if int(prompt_ids.numel()) > max_prompt_len:
            prompt_ids = prompt_ids[:max_prompt_len]

        remaining = int(self._seq_len) - int(prompt_ids.numel())
        if remaining <= 0:
            # Degenerate, but keep a 1-token sequence to avoid empty tensors.
            input_ids = prompt_ids[:1]
        else:
            out_ids = out_ids[:remaining]
            input_ids = torch.cat([prompt_ids, out_ids], dim=0)

        attn = torch.ones_like(input_ids, dtype=torch.long)
        prompt_len = int(prompt_ids.numel())

        labels = input_ids.clone()
        if prompt_len > 0:
            labels[:prompt_len] = -100

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def collate_causal_lm_batch(tokenizer, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids_list = [x["input_ids"] for x in batch]
    attn_list = [x["attention_mask"] for x in batch]
    labels_list = [x["labels"] for x in batch]

    pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attn_list, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def setup_distributed() -> tuple[int, int, int]:
    timeout = datetime.timedelta(minutes=60)
    dist.init_process_group(backend="nccl", timeout=timeout)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def barrier_cuda(local_rank: int):
    if dist.is_initialized():
        dist.barrier(device_ids=[local_rank])


def resolve_model_dir(
    model: str,
    cache_dir: Optional[str],
    rank: int,
    local_rank: int,
) -> str:
    # If it's a local directory, use it.
    if os.path.isdir(model):
        return model

    if snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub is not available and model is not a local directory"
        )

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    # Each node has its own local filesystem/cache. Download once per node
    # (LOCAL_RANK==0), then have all ranks resolve from the local cache.
    if local_rank == 0:
        logger.info(f"Downloading model snapshot on node leader (rank={rank}, local_rank={local_rank})...")
        model_dir = snapshot_download(
            repo_id=model,
            local_files_only=False,
            **kwargs,
        )
        logger.info(f"Snapshot ready: {model_dir}")
    else:
        model_dir = ""

    barrier_cuda(local_rank)

    if local_rank != 0:
        logger.info(f"Resolving model snapshot from local cache (rank={rank}, local_rank={local_rank})...")
        model_dir = snapshot_download(
            repo_id=model,
            local_files_only=True,
            **kwargs,
        )
        logger.info(f"Snapshot resolved: {model_dir}")

    barrier_cuda(local_rank)
    return model_dir


def iter_safetensors_files(model_dir: str) -> Iterable[str]:
    p = Path(model_dir)
    files = sorted(str(x) for x in p.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors found in {model_dir}")
    return files


def materialize_lora_params_if_meta(model: torch.nn.Module, device: torch.device) -> dict[str, int]:
    # PEFT may create LoRA params on meta when used with init_empty_weights.
    # Materialize them on-device with standard init (A ~ N(0, 0.02), B = 0).
    # NOTE: You cannot do `param.data = cuda_tensor` when `param` is a meta tensor;
    # autograd enforces type-compatibility for set_data(). Replace the Parameter.
    to_materialize: list[tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.device.type == "meta":
            to_materialize.append((name, param))

    stats: dict[str, int] = {"total": 0, "lora_a": 0, "lora_b": 0, "other": 0}
    for name, param in to_materialize:
        shape = tuple(param.shape)
        dtype = param.dtype if param.dtype != torch.float32 else torch.float32
        lname = name.lower()
        is_lora_a = ("lora_a" in lname) and lname.endswith("weight")
        is_lora_b = ("lora_b" in lname) and lname.endswith("weight")
        stats["total"] += int(param.numel())
        with torch.no_grad():
            if is_lora_a:
                stats["lora_a"] += int(param.numel())
                new_data = torch.empty(shape, device=device, dtype=dtype)
                torch.nn.init.normal_(new_data, mean=0.0, std=0.02)
            elif is_lora_b:
                stats["lora_b"] += int(param.numel())
                new_data = torch.zeros(shape, device=device, dtype=dtype)
            else:
                stats["other"] += int(param.numel())
                # Default to zeros for any other trainable meta params (conservative).
                new_data = torch.zeros(shape, device=device, dtype=dtype)

        new_param = torch.nn.Parameter(new_data, requires_grad=True)
        if "." in name:
            parent_path, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_path)
            setattr(parent, attr, new_param)
        else:
            setattr(model, name, new_param)

    return stats


def _max_abs_named_params(model: torch.nn.Module, pred) -> float:
    m = 0.0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not pred(n, p):
                continue
            try:
                if p is None or int(p.numel()) == 0:
                    continue
                v = float(p.detach().abs().max().item())
                if v > m:
                    m = v
            except Exception:
                continue
    return m


def ensure_lora_nonzero_init(model: torch.nn.Module) -> bool:
    """Return True if a defensive reinit was applied.

    If all trainable params are exactly zero (common when LoRA A/B were materialized incorrectly),
    reinitialize LoRA A with N(0,0.02) and LoRA B with zeros.
    """
    trainable_max = _max_abs_named_params(model, lambda _n, p: bool(getattr(p, "requires_grad", False)))
    if trainable_max != 0.0:
        return False

    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad or int(p.numel()) == 0:
                continue
            lname = name.lower()
            if ("lora_a" in lname) and lname.endswith("weight"):
                torch.nn.init.normal_(p.data, mean=0.0, std=0.02)
            elif ("lora_b" in lname) and lname.endswith("weight"):
                p.data.zero_()
    return True


def build_model_skeleton(model_dir_or_id: str) -> tuple[torch.nn.Module, any]:
    config = AutoConfig.from_pretrained(model_dir_or_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir_or_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with init_empty_weights():
        base_model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

    return base_model, tokenizer


def apply_lora(base_model: torch.nn.Module, r: int, alpha: int, dropout: float):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(base_model, lora_config)


def freeze_base_unfreeze_lora(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if "lora_" in name.lower():
            p.requires_grad = True


def cast_trainable_params_to_fp32(model: torch.nn.Module) -> None:
    # On Maxwell (M40), fp16 grads can underflow to exact zeros. Keep LoRA/trainable params in fp32.
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.dtype != torch.float32:
                p.data = p.data.to(dtype=torch.float32)


def ensure_rotary_buffers_on_device(
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int,
    dtype: torch.dtype,
):
    """Ensure Qwen2 rotary embedding caches exist on the local CUDA device.

    Transformers 4.37's `Qwen2RotaryEmbedding.forward()` assumes `cos_cached`
    and `sin_cached` are already built and only refreshes them when
    `seq_len > max_seq_len_cached`. Under `init_empty_weights()`, these caches
    can be created on CPU (or buffers can be meta), and clearing them without
    also resetting `max_seq_len_cached` will crash with `NoneType`.

    We avoid calling `model.to(device)` (base weights are ZeRO-Q/meta) and
    instead rebuild only the tiny RoPE buffers per-rank.
    """

    for module in model.modules():
        if not (hasattr(module, "_set_cos_sin_cache") and hasattr(module, "dim") and hasattr(module, "base")):
            continue
        buffers = getattr(module, "_buffers", None)
        if buffers is None or "inv_freq" not in buffers:
            continue

        # Recreate `inv_freq` on the correct device even if the existing buffer
        # is on CPU or meta.
        with torch.no_grad():
            dim = int(getattr(module, "dim"))
            base = float(getattr(module, "base"))
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
            module._buffers["inv_freq"] = inv_freq

            # Build small caches (seq_len of this training run) on CUDA.
            module._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)


def partition_from_checkpoint(
    coordinator: ZeroQCoordinator,
    base_model: torch.nn.Module,
    model_dir: str,
    device: torch.device,
    rank: int,
):
    name_to_param: Dict[str, torch.nn.Parameter] = dict(base_model.named_parameters())
    name_to_buffer: Dict[str, torch.Tensor] = dict(base_model.named_buffers())

    total = 0
    missing = 0

    for st_file in iter_safetensors_files(model_dir):
        if rank == 0:
            logger.info(f"Loading shard: {st_file}")

        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                buf = name_to_buffer.get(key)
                if buf is not None:
                    with torch.no_grad():
                        buf.data = f.get_tensor(key).to(device, non_blocking=False)
                    continue

                param = name_to_param.get(key)
                if param is None:
                    continue

                zq_param = coordinator.get_param_for_tensor(param)
                if zq_param is None:
                    continue

                # Only partition frozen/base params (trainable LoRA stays local)
                if param.requires_grad:
                    continue

                weight = f.get_tensor(key)
                # Keep tensors small on CPU; move into CUDA only for quant/partition.
                zq_param.partition_from_full_precision(weight)
                total += 1

        gc.collect()
        torch.cuda.empty_cache()

    # Quick sanity: report any registered params still without partitions.
    for _, param in base_model.named_parameters():
        zq_param = coordinator.get_param_for_tensor(param)
        if zq_param is None:
            continue
        if zq_param.local_packed is None or zq_param.local_absmax is None:
            missing += 1

    if rank == 0:
        logger.info(f"Partitioned tensors: {total}")
        if missing:
            logger.warning(f"Missing partitions for {missing} tensors")


def sync_trainable_grads(model: torch.nn.Module):
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def save_lora(model: torch.nn.Module, output_dir: str, rank: int):
    if rank != 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    # PEFT save_pretrained writes only adapters.
    model.save_pretrained(output_dir)


def snapshot_trainable_params_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                out[name] = p.detach().float().cpu().clone()
    return out


def restore_trainable_params_from_cpu(model: torch.nn.Module, state: dict[str, torch.Tensor]):
    with torch.no_grad():
        name_to_param = dict(model.named_parameters())
        for name, cpu_tensor in state.items():
            p = name_to_param.get(name)
            if p is None:
                continue
            if not p.requires_grad:
                continue
            p.copy_(cpu_tensor.to(device=p.device, dtype=p.dtype))


def global_mean_scalar(x: float, device: torch.device, world_size: int) -> float:
    t = torch.tensor(float(x), device=device, dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t / float(world_size)
    return float(t.item())


def global_sum_scalar(x: float, device: torch.device) -> float:
    t = torch.tensor(float(x), device=device, dtype=torch.float32)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def all_ranks_finite_loss(loss: torch.Tensor, device: torch.device) -> bool:
    """Return True iff every rank has a finite loss for this microbatch.

    Must be called by all ranks in the same order to avoid desync.
    """

    local_ok = torch.isfinite(loss).to(dtype=torch.int32, device=device)
    if dist.is_initialized():
        dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
    return int(local_ok.item()) == 1


def all_ranks_finite_grads(trainable_params: list[torch.nn.Parameter], device: torch.device) -> bool:
    """Return True iff every rank has finite grads for all trainable params."""

    local_ok = torch.tensor(1, device=device, dtype=torch.int32)
    for p in trainable_params:
        g = p.grad
        if g is None:
            continue
        if not torch.isfinite(g).all():
            local_ok.fill_(0)
            break
    if dist.is_initialized():
        dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
    return int(local_ok.item()) == 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_dir", default=os.path.expanduser("~/phoenix_training/zeroq_qwen32b_lora"))
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=2,
        help="Number of optimizer steps (not micro-batches). Will loop over the dataset as needed.",
    )
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument(
        "--dataloader_workers",
        type=int,
        default=int(os.environ.get("ZEROQ_DATALOADER_WORKERS", "0")),
        help="DataLoader workers per rank. Increase to speed up tokenization (env ZEROQ_DATALOADER_WORKERS).",
    )

    # Optional early stopping + best restore (cost control)
    parser.add_argument(
        "--early_stop_loss",
        type=float,
        default=None,
        help="If set, stop training once loss is <= threshold for `early_stop_patience` steps (after min steps).",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=3,
        help="Consecutive optimizer steps below threshold required to early-stop.",
    )
    parser.add_argument(
        "--early_stop_min_steps",
        type=int,
        default=0,
        help="Minimum optimizer steps before early stopping is allowed.",
    )
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    try:
        device = torch.device(f"cuda:{local_rank}")

        if rank == 0:
            logger.info("=" * 70)
            logger.info("ZeRO-Q QLoRA (Maxwell) multi-node")
            logger.info(f"World size: {world_size}")
            logger.info(f"Model: {args.model}")
            logger.info("=" * 70)

        model_dir = resolve_model_dir(args.model, args.cache_dir, rank=rank, local_rank=local_rank)

        # 1) Build model skeleton without weights
        if rank == 0:
            logger.info("Building model skeleton (meta weights)...")
        base_model, tokenizer = build_model_skeleton(model_dir)
        if rank == 0:
            logger.info("Skeleton built.")

        # Gradient checkpointing is critical for 11GB GPUs. Enable by default;
        # opt-out with: GRAD_CHECKPOINTING=0
        enable_gc = os.environ.get("GRAD_CHECKPOINTING", "1").strip().lower() not in (
            "0",
            "false",
            "no",
        )
        if enable_gc and hasattr(base_model, "gradient_checkpointing_enable"):
            # PyTorch warns (and will error in 2.5+) if `use_reentrant` isn't explicit.
            # Transformers' API has changed over time, so try the modern kwargs first.
            try:
                base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                try:
                    base_model.gradient_checkpointing_enable(use_reentrant=False)
                except TypeError:
                    base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "config"):
            base_model.config.use_cache = False

        # 2) Apply LoRA (trainable adapters)
        if rank == 0:
            logger.info("Applying LoRA...")
        model = apply_lora(base_model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        if rank == 0:
            logger.info("LoRA applied.")

        # 3) Freeze base weights; keep LoRA trainable
        freeze_base_unfreeze_lora(model)

        # PEFT + checkpointing: ensure embeddings produce grads.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        # 4) Materialize trainable (LoRA) params if PEFT created them on meta
        meta_stats = materialize_lora_params_if_meta(model, device=device)

        # Keep LoRA params in fp32 to avoid zero-grad underflow on older GPUs.
        cast_trainable_params_to_fp32(model)

        # Defensive: if LoRA ended up all-zeros, re-init so gradients can flow.
        reinit_applied = ensure_lora_nonzero_init(model)

        if rank == 0:
            lora_a_max = _max_abs_named_params(model, lambda n, p: ("lora_a" in n.lower()) and n.lower().endswith("weight"))
            lora_b_max = _max_abs_named_params(model, lambda n, p: ("lora_b" in n.lower()) and n.lower().endswith("weight"))
            logger.info(f"LoRA meta materialized elems: {meta_stats} reinit_applied={reinit_applied} lora_A_maxabs={lora_a_max:.2e} lora_B_maxabs={lora_b_max:.2e}")

        if rank == 0:
            trainable_dtypes: dict[str, int] = {}
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                trainable_dtypes[str(p.dtype)] = trainable_dtypes.get(str(p.dtype), 0) + p.numel()
            logger.info(f"Trainable dtypes: {trainable_dtypes}")

        # Ensure Qwen2 rotary embedding buffers/caches live on CUDA.
        ensure_rotary_buffers_on_device(model, device=device, seq_len=args.seq_len, dtype=torch.float32)

        # 5) Create coordinator + wrapper (register frozen/base params + install hooks)
        if rank == 0:
            logger.info("Registering parameters with ZeRO-Q...")
        config = MAXWELL_CONFIG
        coordinator = ZeroQCoordinator(config)
        _ = ZeroQModuleWrapper(model, coordinator, trainable_only=False)
        if rank == 0:
            logger.info("Registration complete.")

        # 6) Stream-load checkpoint tensors and partition them across ranks
        barrier_cuda(local_rank)
        if rank == 0:
            logger.info("Partitioning from safetensors shards...")
        partition_from_checkpoint(
            coordinator=coordinator,
            base_model=base_model,
            model_dir=model_dir,
            device=device,
            rank=rank,
        )
        if rank == 0:
            logger.info("Partitioning complete.")
        barrier_cuda(local_rank)

        if rank == 0:
            stats = coordinator.get_memory_stats()
            logger.info(
                f"ZeRO-Q stats: local={stats['local_memory_mb']:.1f}MB "
                f"full_fp16={stats['full_fp16_memory_mb']:.1f}MB "
                f"ratio={stats['compression_ratio']:.2f}x"
            )

        # 7) Data
        dataset = JsonlInstructionDataset(args.data, tokenizer, seq_len=args.seq_len)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=int(args.dataloader_workers),
            pin_memory=True,
            collate_fn=lambda b: collate_causal_lm_batch(tokenizer, b),
        )

        # 8) Optimizer (trainable LoRA only)
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters found (LoRA may not have been applied or was left frozen).")
        optimizer = torch.optim.AdamW(trainable, lr=args.lr)

        if rank == 0:
            trainable_params = sum(int(p.numel()) for p in trainable)
            logger.info(f"Dataset: {len(dataset)} examples")
            logger.info(f"Trainable (LoRA) params: {trainable_params:,}")

        model.train()

        micro_step = 0
        opt_step = 0
        optimizer.zero_grad(set_to_none=True)

        accum_loss_sum = 0.0
        accum_resp_tokens = 0
        accum_seq_len_sum = 0
        good_micro_in_accum = 0
        step_t0 = torch.cuda.Event(enable_timing=True)
        step_t1 = torch.cuda.Event(enable_timing=True)
        step_t0.record()

        best_loss = math.inf
        best_step = 0
        best_state: Optional[dict[str, torch.Tensor]] = None
        below_count = 0
        early_stopped = False

        epoch = 0
        while opt_step < args.max_steps:
            sampler.set_epoch(epoch)
            for batch in loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                outputs = model(**batch)
                # IMPORTANT: all ranks must agree whether this microbatch is usable.
                # Otherwise ranks can diverge and deadlock inside NCCL collectives.
                all_finite = all_ranks_finite_loss(outputs.loss, device=device)
                if not all_finite:
                    if rank == 0:
                        logger.warning(
                            f"Non-finite loss encountered (at least one rank) at micro_step={micro_step}; "
                            "resetting grad accumulation and continuing."
                        )
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss_sum = 0.0
                    accum_resp_tokens = 0
                    accum_seq_len_sum = 0
                    good_micro_in_accum = 0
                    step_t0.record()
                    micro_step += 1
                    continue

                # Note: `outputs.loss` is mean over non-ignored labels.
                loss = outputs.loss / args.grad_accum
                loss.backward()

                accum_loss_sum += float(outputs.loss.item())
                # Count only response tokens (labels != -100) for a meaningful throughput metric.
                accum_resp_tokens += int((batch["labels"] != -100).sum().item())
                accum_seq_len_sum += int(batch["attention_mask"].sum().item())
                good_micro_in_accum += 1

                # If any rank produced non-finite grads during accumulation, reset uniformly.
                if not all_ranks_finite_grads(trainable, device=device):
                    if rank == 0:
                        logger.warning(
                            f"Non-finite gradients encountered (at least one rank) at micro_step={micro_step}; "
                            "resetting grad accumulation and continuing."
                        )
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss_sum = 0.0
                    accum_resp_tokens = 0
                    accum_seq_len_sum = 0
                    good_micro_in_accum = 0
                    step_t0.record()
                    micro_step += 1
                    continue

                if good_micro_in_accum >= int(args.grad_accum):
                    # Pre-sync diagnostics (rank0 only): if grads are non-zero here but zero after sync,
                    # the sync path is suspect; if they're already zero here, learning signal is missing.
                    pre_grad_norm = None
                    pre_max_abs_grad = None
                    pre_gp = None
                    sample_before = None
                    if rank == 0:
                        pre_sq = 0.0
                        pre_with_grad = 0
                        pre_total = 0
                        pre_max_abs = 0.0
                        for p in trainable:
                            pre_total += 1
                            if p.grad is None:
                                continue
                            pre_with_grad += 1
                            g = p.grad.detach()
                            pre_sq += float(g.float().pow(2).sum().item())
                            try:
                                ga = float(g.abs().max().item())
                                if ga > pre_max_abs:
                                    pre_max_abs = ga
                            except Exception:
                                pass
                        pre_grad_norm = math.sqrt(pre_sq) if pre_sq > 0.0 else 0.0
                        pre_max_abs_grad = pre_max_abs
                        pre_gp = f"{int(pre_with_grad)}/{int(pre_total)}"

                        # Tiny update sanity-check: sample a few elements from a few tensors.
                        sample = []
                        max_tensors = 4
                        max_elems = 1024
                        for p in trainable[:max_tensors]:
                            flat = p.detach().view(-1)
                            if int(flat.numel()) == 0:
                                continue
                            sample.append(flat[: min(int(flat.numel()), max_elems)].float().cpu().clone())
                        sample_before = sample

                    sync_trainable_grads(model)

                    # Grad norm (rank0 only; after grad sync so it's representative).
                    grad_norm = None
                    grad_params_with_grad = None
                    grad_total_params = None
                    max_abs_grad = None
                    if rank == 0:
                        sq = 0.0
                        with_grad = 0
                        total = 0
                        max_abs = 0.0
                        for p in trainable:
                            total += 1
                            if p.grad is None:
                                continue
                            with_grad += 1
                            g = p.grad.detach()
                            sq += float(g.float().pow(2).sum().item())
                            try:
                                ga = float(g.detach().abs().max().item())
                                if ga > max_abs:
                                    max_abs = ga
                            except Exception:
                                pass
                        grad_norm = math.sqrt(sq) if sq > 0.0 else 0.0
                        grad_params_with_grad = with_grad
                        grad_total_params = total
                        max_abs_grad = max_abs

                    optimizer.step()

                    # Post-step update sanity-check (rank0 only).
                    sample_max_abs_update = None
                    if rank == 0 and sample_before is not None:
                        sample = []
                        max_tensors = min(4, len(trainable))
                        max_elems = 1024
                        for p in trainable[:max_tensors]:
                            flat = p.detach().view(-1)
                            if int(flat.numel()) == 0:
                                continue
                            sample.append(flat[: min(int(flat.numel()), max_elems)].float().cpu())
                        try:
                            deltas = []
                            for a, b in zip(sample_before, sample):
                                deltas.append(float((b - a).abs().max().item()))
                            sample_max_abs_update = max(deltas) if deltas else 0.0
                        except Exception:
                            sample_max_abs_update = None

                    optimizer.zero_grad(set_to_none=True)
                    opt_step += 1

                    # Compute a global mean loss (across ranks) for consistent best/stop decisions.
                    step_loss = accum_loss_sum / float(args.grad_accum)
                    step_loss = global_mean_scalar(step_loss, device=device, world_size=world_size)

                    # Throughput diagnostics (global tokens/sec and avg seq len).
                    tokens_global = global_sum_scalar(float(accum_resp_tokens), device=device)
                    seqlen_global = global_sum_scalar(float(accum_seq_len_sum), device=device)

                    step_t1.record()
                    torch.cuda.synchronize(device)
                    elapsed_ms = float(step_t0.elapsed_time(step_t1))
                    elapsed_s = max(elapsed_ms / 1000.0, 1e-6)
                    toks_per_s = tokens_global / elapsed_s
                    avg_seq_len = seqlen_global / float(world_size) / float(args.batch_size) / float(args.grad_accum)

                    # Reset accumulators for next optimizer step.
                    accum_loss_sum = 0.0
                    accum_resp_tokens = 0
                    accum_seq_len_sum = 0
                    good_micro_in_accum = 0
                    step_t0.record()

                    if rank == 0:
                        if step_loss < best_loss:
                            best_loss = step_loss
                            best_step = opt_step
                            best_state = snapshot_trainable_params_cpu(model)

                        lr = float(optimizer.param_groups[0].get("lr", args.lr))
                        if grad_norm is None:
                            logger.info(
                                f"opt_step={opt_step} loss={step_loss:.4f} best={best_loss:.4f}@{best_step} lr={lr:.2e} toks/s={toks_per_s:.1f} avg_seq={avg_seq_len:.1f}"
                            )
                        else:
                            gp = "?"
                            if grad_params_with_grad is not None and grad_total_params is not None:
                                gp = f"{int(grad_params_with_grad)}/{int(grad_total_params)}"
                            mag = "?"
                            if max_abs_grad is not None:
                                mag = f"{float(max_abs_grad):.2e}"
                            pre_bits = ""
                            if pre_grad_norm is not None and pre_max_abs_grad is not None and pre_gp is not None:
                                pre_bits = f" pre_grad_norm={float(pre_grad_norm):.2e} pre_max_abs_grad={float(pre_max_abs_grad):.2e} pre_grad_params={pre_gp}"
                            upd_bits = ""
                            if sample_max_abs_update is not None:
                                upd_bits = f" sample_max_abs_update={float(sample_max_abs_update):.2e}"
                            logger.info(
                                f"opt_step={opt_step} loss={step_loss:.4f} best={best_loss:.4f}@{best_step} lr={lr:.2e} grad_norm={grad_norm:.2e} max_abs_grad={mag} grad_params={gp}{pre_bits}{upd_bits} toks/s={toks_per_s:.1f} avg_seq={avg_seq_len:.1f}"
                            )

                    # Early stop logic (rank0 decides; broadcast to all ranks).
                    stop_tensor = torch.tensor(0, device=device, dtype=torch.int32)
                    if args.early_stop_loss is not None and opt_step >= int(args.early_stop_min_steps):
                        if step_loss <= float(args.early_stop_loss):
                            below_count += 1
                        else:
                            below_count = 0
                        if rank == 0 and below_count >= int(args.early_stop_patience):
                            stop_tensor.fill_(1)
                    if dist.is_initialized():
                        dist.broadcast(stop_tensor, src=0)
                    if int(stop_tensor.item()) == 1:
                        early_stopped = True
                        if rank == 0:
                            logger.info(
                                f"EARLY STOP triggered at opt_step={opt_step} "
                                f"(loss={step_loss:.4f} <= {float(args.early_stop_loss):.4f} "
                                f"for {int(args.early_stop_patience)} steps; min_steps={int(args.early_stop_min_steps)})"
                            )
                        break

                    if opt_step >= args.max_steps:
                        break

                micro_step += 1

            if early_stopped:
                break

            epoch += 1

        barrier_cuda(local_rank)

        # Restore best LoRA weights on rank 0 before saving.
        if rank == 0 and best_state is not None and best_step > 0:
            restore_trainable_params_from_cpu(model, best_state)

        save_lora(model, args.output_dir, rank)
        barrier_cuda(local_rank)

        # Always emit a completion line per-rank so worker logs show clean exit.
        logger.info(f"RANK {rank} COMPLETE")
        barrier_cuda(local_rank)

        if rank == 0:
            logger.info("TRAINING COMPLETE")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
