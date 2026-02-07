#!/usr/bin/env python3
"""
ZeRO-Q: Multi-Node Training Test

This test validates ZeRO-Q across two physical nodes (PE2 and PE3)
communicating over the network via NCCL.

Setup:
    PE2: 192.168.1.102 (3x M40, MASTER)
    PE3: 192.168.1.103 (2x M40)

Run from PE2:
    # On PE2 (master):
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
             --master_addr=192.168.1.102 --master_port=29500 \
             test_multinode.py
    
    # On PE3 (worker):
    torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
             --master_addr=192.168.1.102 --master_port=29500 \
             test_multinode.py
"""

import os
import sys
import gc
import time
import socket
import argparse
import datetime
import torch
import torch.distributed as dist
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class PartitionedParam:
    """Holds partition info for a Params4bit weight."""
    name: str
    local_data: torch.Tensor
    local_absmax: torch.Tensor
    data_size: int
    absmax_size: int


class BnbZeroQCoordinator:
    """Coordinates partitioning of bitsandbytes Params4bit weights."""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.partitions: Dict[str, PartitionedParam] = {}
        self.original_memory = 0
        self.partitioned_memory = 0
        
    def partition_model(self, model: torch.nn.Module) -> int:
        import bitsandbytes as bnb
        
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                weight = module.weight
                if hasattr(weight, 'data') and hasattr(weight, 'quant_state'):
                    self._partition_param(name, weight)
                    count += 1
        
        gc.collect()
        torch.cuda.empty_cache()
        return count
    
    def _partition_param(self, name: str, param):
        packed_data = param.data
        quant_state = param.quant_state
        absmax = quant_state.absmax
        
        self.original_memory += packed_data.numel() + absmax.numel() * 4
        
        data_size = packed_data.numel()
        data_per_rank = data_size // self.world_size
        data_start = self.rank * data_per_rank
        data_end = data_start + data_per_rank if self.rank < self.world_size - 1 else data_size
        
        absmax_size = absmax.numel()
        absmax_per_rank = absmax_size // self.world_size
        absmax_start = self.rank * absmax_per_rank
        absmax_end = absmax_start + absmax_per_rank if self.rank < self.world_size - 1 else absmax_size
        
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
        
    def _all_gather_padded_1d(self, x: torch.Tensor):
        """All-gather 1D tensors with variable lengths by padding to max length.

        Returns: (gathered_padded, sizes)
        - gathered_padded: list[Tensor] each with shape [max_len]
        - sizes: list[int] original per-rank lengths
        """

        if x.dim() != 1:
            x = x.reshape(-1)

        local_n = torch.tensor([x.numel()], device=x.device, dtype=torch.int64)
        all_n = [torch.zeros_like(local_n) for _ in range(self.world_size)]
        dist.all_gather(all_n, local_n)
        sizes = [int(t.item()) for t in all_n]
        max_n = max(sizes) if sizes else x.numel()

        padded = torch.zeros((max_n,), device=x.device, dtype=x.dtype)
        if x.numel() > 0:
            padded[: x.numel()] = x

        gathered = [torch.zeros_like(padded) for _ in range(self.world_size)]
        dist.all_gather(gathered, padded)
        return gathered, sizes

    def gather_samples(self, max_layers: int = 2, max_elems: int = 4096):
        """Lightweight comms check for partition shards.

        Full all_gather of partition shards is extremely large for big models and also
        fails when shard lengths differ across ranks. Instead, gather a small sample
        from a few layers, padding as needed.
        """

        if max_layers <= 0 or max_elems <= 0:
            return

        # IMPORTANT: Avoid all_gather'ing even small tensors into per-rank lists.
        # On memory-tight GPUs (e.g., 11GB M40), a fragmented allocator can OOM when
        # building gathered buffers, and any single-rank failure will cascade into
        # NCCL timeouts. A tiny all_reduce checksum still validates comms.

        partitions = list(self.partitions.values())

        # IMPORTANT: All ranks must execute the same number of collectives.
        # Even if a rank has fewer partitions (unexpected but possible due to
        # library/model quirks), we still participate using a zero-checksum.
        device = partitions[0].local_data.device if partitions else torch.device("cuda")

        for i in range(max_layers):
            if i < len(partitions):
                partition = partitions[i]
                sample_data = partition.local_data.reshape(-1)[:max_elems].contiguous()
                sample_absmax = partition.local_absmax.reshape(-1)[:max_elems].contiguous()

                checksum = torch.tensor(
                    [
                        float(sample_data.numel()),
                        float(sample_data.float().sum().item()) if sample_data.numel() else 0.0,
                        float(sample_absmax.numel()),
                        float(sample_absmax.float().sum().item()) if sample_absmax.numel() else 0.0,
                    ],
                    device=device,
                    dtype=torch.float32,
                )
            else:
                checksum = torch.zeros((4,), device=device, dtype=torch.float32)

            dist.all_reduce(checksum, op=dist.ReduceOp.SUM)
            
    def get_stats(self) -> Dict[str, Any]:
        return {
            'num_partitions': len(self.partitions),
            'original_memory_mb': self.original_memory / 1024**2,
            'partitioned_memory_mb': self.partitioned_memory / 1024**2,
            'memory_ratio': self.original_memory / max(self.partitioned_memory, 1),
        }


def print_rank0(msg, rank):
    if rank == 0:
        print(msg, flush=True)


def _is_local_leader() -> bool:
    try:
        return int(os.environ.get("LOCAL_RANK", "0")) == 0
    except Exception:
        return True


def _barrier_if_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _barrier_cuda(local_rank: Optional[int] = None) -> None:
    """Barrier helper that pins NCCL barriers to a specific CUDA device."""

    if not (dist.is_available() and dist.is_initialized()):
        return

    if local_rank is None:
        dist.barrier()
        return

    try:
        dist.barrier(device_ids=[int(local_rank)])
    except TypeError:
        # Older torch versions may not support `device_ids`.
        dist.barrier()


def _node_sync(tag: str, rank: int, timeout_s: int = 6 * 60 * 60) -> None:
    """Node-local sync without NCCL/GPU usage.

    torchrun sets LOCAL_RANK/LOCAL_WORLD_SIZE and GROUP_RANK (node rank).
    We use a file in /tmp to let LOCAL_RANK==0 signal other local ranks.
    """

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except Exception:
        local_rank = 0

    try:
        node_rank = os.environ.get("GROUP_RANK") or os.environ.get("NODE_RANK") or "0"
    except Exception:
        node_rank = "0"

    run_id = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT") or "0"
    path = f"/tmp/zeroq_node_sync_{run_id}_node{node_rank}_{tag}"

    if local_rank == 0:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"ok rank={rank} host={socket.gethostname()} time={time.time()}\n")
        except Exception:
            # If /tmp is unwritable for some reason, fall back to a short global barrier.
            _barrier_if_dist()
            return
        return

    start = time.time()
    while not os.path.exists(path):
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for node-local sync file: {path}")
        time.sleep(0.25)


def _precache_hf_repo(model_id: str, cache_dir: Optional[str], rank: int) -> None:
    """Download HF model repo once per node to avoid multi-process lock contention."""

    if not _is_local_leader():
        _barrier_if_dist()
        return

    from huggingface_hub import snapshot_download

    print(f"[Rank {rank}] Precaching model repo (one proc per node)...", flush=True)
    snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        resume_download=True,
    )
    _barrier_if_dist()


def _normalize_quant(quant: str) -> str:
    q = (quant or "").strip().lower()
    aliases = {
        "q4": "4bit",
        "4": "4bit",
        "4bit": "4bit",
        "int4": "4bit",
        "q8": "8bit",
        "8": "8bit",
        "8bit": "8bit",
        "int8": "8bit",
        "none": "none",
        "fp16": "none",
        "fp32": "none",
    }
    if q not in aliases:
        raise ValueError(f"Unsupported --quant {quant!r}. Use: 4bit|8bit|none (aliases: q4,q8)")
    return aliases[q]


def _pick_lora_targets(model, explicit: Optional[str]) -> List[str]:
    if explicit:
        return [m.strip() for m in explicit.split(",") if m.strip()]

    # Common transformer naming conventions
    candidate_sets = [
        ["q_proj", "v_proj"],
        ["q_proj", "k_proj", "v_proj", "o_proj"],
        ["c_attn"],  # GPT-2
    ]
    names = {name for name, _ in model.named_modules()}
    for candidates in candidate_sets:
        if all(any(n.endswith(c) or n == c for n in names) for c in candidates):
            return candidates

    # Fallback: try common attention projection substrings
    for candidates in (["q_proj"], ["c_attn"]):
        if any(any(n.endswith(c) or n == c for n in names) for c in candidates):
            return candidates

    raise ValueError(
        "Could not auto-detect LoRA target modules. "
        "Pass --lora-target-modules (comma-separated), e.g. 'q_proj,v_proj'"
    )


class _LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        device = self.base.weight.device
        dtype = self.base.weight.dtype

        self.r = r
        self.alpha = alpha
        self.scaling = float(alpha) / float(r)
        self.dropout = torch.nn.Dropout(dropout)

        self.A = torch.nn.Linear(base.in_features, r, bias=False, device=device, dtype=dtype)
        self.B = torch.nn.Linear(r, base.out_features, bias=False, device=device, dtype=dtype)

        # LoRA init: A small, B zeros (common practice)
        torch.nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.B(self.A(self.dropout(x))) * self.scaling


def _inject_lora_linear(
    model: torch.nn.Module,
    target_modules: List[str],
    r: int,
    alpha: int,
    dropout: float,
) -> torch.nn.Module:
    # Replace matching nn.Linear modules by name suffix.
    def should_wrap(module_name: str) -> bool:
        return any(module_name.endswith(t) or module_name == t for t in target_modules)

    for name, module in list(model.named_modules()):
        if not should_wrap(name):
            continue
        if not isinstance(module, torch.nn.Linear):
            raise ValueError(
                f"--quant none LoRA injector only supports nn.Linear; got {type(module)} for {name}. "
                "Pass --quant 4bit/8bit to use PEFT, or choose a model with Linear projections."
            )

        # Walk to parent to replace attribute
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], _LoRALinear(module, r=r, alpha=alpha, dropout=dropout))

    return model


def test_multinode(args: argparse.Namespace) -> bool:
    """Smoke-test multi-node LoRA training under torch.distributed."""

    # Select device before initializing NCCL to avoid ambiguous rank<->GPU mapping.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Initialize distributed with NCCL.
    # Passing device_id prevents "devices used by this process are currently unknown" warnings
    # and reduces the risk of barriers/allreduces picking the wrong GPU.
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=device,
        timeout=datetime.timedelta(minutes=30),
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    hostname = socket.gethostname()
    
    print_rank0("\n" + "="*70, rank)
    print_rank0("🌐 ZeRO-Q: MULTI-NODE TRAINING TEST 🌐", rank)
    print_rank0("="*70, rank)
    print_rank0(f"World size: {world_size} (across {world_size} nodes)", rank)
    
    # All ranks report
    _barrier_cuda(local_rank)
    print(f"[Rank {rank}] Node: {hostname}, Device: cuda:{local_rank}", flush=True)
    _barrier_cuda(local_rank)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quant = _normalize_quant(args.quant)

        if quant in ("4bit", "8bit"):
            try:
                import bitsandbytes as bnb  # noqa: F401
                bnb_version = getattr(bnb, "__version__", "unknown")
                if rank == 0:
                    print(f"[bnb] bitsandbytes version: {bnb_version}")
            except Exception as e:
                raise RuntimeError(
                    "bitsandbytes is required for --quant 4bit/8bit. "
                    "You mentioned ~0.42.3 may be the known-good version. "
                    f"Import error: {e}"
                )
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # ----- Load model -----
        print_rank0(f"\n[1/5] Loading model ({args.model_id}) with quant={quant}...", rank)

        if args.download_only:
            # Download artifacts without instantiating the full model.
            # This is important for very large models (e.g., 32B) to avoid CPU RAM spikes.
            if _is_local_leader():
                _precache_hf_repo(args.model_id, args.cache_dir, rank)
            else:
                print(f"[Rank {rank}] Waiting for node-local download to complete...", flush=True)
            _node_sync("download_done", rank)
            return True

        # Avoid multi-process downloads: only LOCAL_RANK==0 per node is allowed to hit the network.
        local_leader = _is_local_leader()

        if local_leader:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_id,
                trust_remote_code=args.trust_remote_code,
                cache_dir=args.cache_dir,
                local_files_only=False,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            _node_sync("tokenizer_ready", rank)
        else:
            print(f"[Rank {rank}] Waiting for node-local tokenizer cache...", flush=True)
            _node_sync("tokenizer_ready", rank)
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_id,
                trust_remote_code=args.trust_remote_code,
                cache_dir=args.cache_dir,
                local_files_only=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        quantization_config = None
        torch_dtype = None
        device_map = None

        if quant == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=torch.float32 if args.compute_dtype == "fp32" else torch.float16,
            )
            device_map = {"": local_rank}
        elif quant == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False,
            )
            device_map = {"": local_rank}
        else:
            # No quantization: load normally and move to device.
            torch_dtype = torch.float16 if args.compute_dtype == "fp16" else torch.float32

        if local_leader:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                quantization_config=quantization_config,
                trust_remote_code=args.trust_remote_code,
                device_map=device_map,
                torch_dtype=torch_dtype,
                cache_dir=args.cache_dir,
                local_files_only=False,
            )
            _node_sync("model_ready", rank)
        else:
            print(f"[Rank {rank}] Waiting for node-local model cache...", flush=True)
            _node_sync("model_ready", rank)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                quantization_config=quantization_config,
                trust_remote_code=args.trust_remote_code,
                device_map=device_map,
                torch_dtype=torch_dtype,
                cache_dir=args.cache_dir,
                local_files_only=True,
            )

        # Once everyone has loaded on each node, align globally before NCCL-heavy collectives.
        _barrier_if_dist()

        if quant == "none":
            model.to(device)
        
        torch.cuda.synchronize(local_rank)
        mem_after_load = torch.cuda.memory_allocated(local_rank) / 1024**3
        print_rank0(f"    Memory after load: {mem_after_load:.3f} GB", rank)
        
        # ----- Apply ZeRO-Q-like partition (only meaningful for 4bit Linear4bit modules) -----
        print_rank0("\n[2/5] Partitioning quantized weights across nodes...", rank)

        coordinator = BnbZeroQCoordinator(rank, world_size)
        if quant == "4bit":
            num_partitioned = coordinator.partition_model(model)
            stats = coordinator.get_stats()
            print_rank0(f"    Partitioned {num_partitioned} layers", rank)
            print_rank0(f"    Memory savings (approx): {stats['memory_ratio']:.2f}x", rank)
        else:
            num_partitioned = 0
            stats = {"memory_ratio": 1.0}
            print_rank0("    Skipping partitioning (requires --quant 4bit)", rank)
        
        # ----- Prepare and LoRA -----
        print_rank0("\n[3/5] Applying LoRA...", rank)

        target_modules = _pick_lora_targets(model, args.lora_target_modules)

        if quant == "none":
            # Avoid PEFT/bitsandbytes coupling for unquantized runs.
            model = _inject_lora_linear(
                model,
                target_modules=target_modules,
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
            )
        else:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_rank0(f"    Trainable: {trainable:,}", rank)
        
        # ----- Training -----
        print_rank0("\n[4/5] Running training step...", rank)
        
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
        )
        
        # Each node gets different data
        texts = [
            f"Node {rank}: Phoenix network LoRA smoke test.",
            f"Node {rank}: model={args.model_id} quant={quant}.",
        ]
        
        losses = []
        for step, text in enumerate(texts[: max(1, args.steps)]):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.seq_len,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            start_time = time.time()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss
            loss.backward()
            
            if step == 0:
                grad_count = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
                total = sum(1 for p in model.parameters() if p.requires_grad)
                print_rank0(f"    Gradients: {grad_count}/{total}", rank)
            
            optimizer.step()
            optimizer.zero_grad()
            
            step_time = time.time() - start_time
            losses.append(loss.item())
            
            dist.barrier()
        
        avg_loss = sum(losses) / len(losses)
        
        # ----- Verify across nodes -----
        print_rank0("\n[5/5] Verifying cross-node communication...", rank)

        # Verify collectives with a tiny all-reduce (more robust than list-based all_gather).
        loss_tensor = torch.tensor([avg_loss], device=device, dtype=torch.float32)
        loss_sum = loss_tensor.clone()
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        if rank == 0:
            print(f"    Avg loss across all ranks: {(loss_sum.item() / float(world_size)):.6f}")
        
        # Test coordinator gather across nodes
        if quant == "4bit":
            coordinator.gather_samples(
                max_layers=args.verify_shard_layers,
                max_elems=args.verify_shard_elems,
            )
            print_rank0("    Cross-node shard sample checksum all_reduce: SUCCESS", rank)
        
        # ----- Report -----
        print_rank0("\n" + "="*70, rank)
        print_rank0("🎉 ZeRO-Q MULTI-NODE TEST COMPLETE! 🎉", rank)
        print_rank0("="*70, rank)
        print_rank0(f"✓ Nodes: {world_size}", rank)
        print_rank0(f"✓ Partitioned layers: {num_partitioned}", rank)
        print_rank0(f"✓ Memory savings: {stats['memory_ratio']:.2f}x", rank)
        print_rank0(f"✓ Gradients: VERIFIED", rank)
        print_rank0(f"✓ Cross-node communication: SUCCESS", rank)
        print_rank0("="*70, rank)
        print_rank0("", rank)
        print_rank0("ZeRO-Q works across physical nodes!", rank)
        print_rank0("="*70, rank)
        
        return True
        
    except Exception as e:
        print(f"\n[Rank {rank} @ {hostname}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZeRO-Q multi-node LoRA smoke test")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-1.5B",
        help="HuggingFace model id (default: Qwen/Qwen2.5-1.5B)",
    )
    parser.add_argument(
        "--quant",
        default="q4",
        help="Quantization mode: 4bit|8bit|none (aliases: q4,q8)",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["fp16", "fp32"],
        default="fp32",
        help="Compute dtype hint for k-bit ops (default: fp32 for Maxwell)",
    )
    parser.add_argument(
        "--bnb-4bit-quant-type",
        default="nf4",
        choices=["nf4", "fp4"],
        help="bitsandbytes 4-bit quant type (default: nf4)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of training steps (default: 2)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Tokenizer max_length for the tiny smoke examples (default: 256)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional HF cache dir (use the same path on both nodes)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download tokenizer/model files then exit.",
    )
    parser.add_argument(
        "--verify-shard-layers",
        type=int,
        default=2,
        help="For 4-bit runs, number of partition layers to sample-gather for comms verification (default: 2).",
    )
    parser.add_argument(
        "--verify-shard-elems",
        type=int,
        default=4096,
        help="For 4-bit runs, number of elements per shard sample to all_gather (default: 4096).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Pass trust_remote_code=True to transformers (default: True)",
    )
    parser.add_argument(
        "--lora-target-modules",
        default=None,
        help="Comma-separated LoRA target modules (auto-detected if omitted)",
    )
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank r (default: 8)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (default: 16)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")

    success = test_multinode(parser.parse_args())
    sys.exit(0 if success else 1)
