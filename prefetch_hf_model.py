#!/usr/bin/env python3
"""Prefetch a HuggingFace model snapshot into a local directory.

This is optional, but helps ensure both nodes have the model before starting
multinode training.

Example:
  python prefetch_hf_model.py --model Qwen/Qwen2.5-1.5B --local-dir /mnt/hf

Then run training with:
  HF_CACHE_DIR=/mnt/hf python run_smoke_lora_multinode.py ...
"""

from __future__ import annotations

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-1.5B")
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Directory to store snapshots (will be created).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision/tag.",
    )

    args = parser.parse_args()
    os.makedirs(args.local_dir, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise SystemExit(
            "huggingface_hub is required (usually comes with transformers). "
            "Install with: pip install huggingface_hub"
        ) from e

    path = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        cache_dir=args.local_dir,
        local_dir=None,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"Downloaded: {args.model}")
    print(f"Cache dir: {args.local_dir}")
    print(f"Snapshot: {path}")


if __name__ == "__main__":
    main()
