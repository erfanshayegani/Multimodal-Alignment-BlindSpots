"""Argument helpers shared by the scripts in ``scripts/``."""

from __future__ import annotations

import argparse
import os

from .models import ADAPTERS, SETTINGS

ALL_SETTINGS = [
    "no_img_no_swap", "no_img_cross_role",
    "img_in_no_swap", "img_in_cross_role",
    "img_end_no_swap", "img_end_cross_role",
    "img_outside_no_swap", "img_outside_cross_role",
]


def add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--model", required=True, choices=sorted(ADAPTERS), help="Model family.")
    g.add_argument("--model-id", default=None, help="HF id or local path (defaults to the paper's checkpoint).")
    g.add_argument("--adapter", default=None, help="Optional LoRA adapter directory (e.g. from train_sft.py).")
    g.add_argument("--attn-implementation", default=None, help="e.g. flash_attention_2, sdpa, eager (auto by default).")
    g.add_argument("--gpu", default=None, help="Sets CUDA_VISIBLE_DEVICES before loading.")


def add_generation_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("generation")
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--max-new-tokens", type=int, default=256)
    g.add_argument("--greedy", action="store_true", help="Greedy decoding (the paper samples at default temperature).")
    g.add_argument("--system-prompt", action="store_true", help="Keep the model's default system prompt (Qwen).")
    g.add_argument("--seed", type=int, default=0)


def add_settings_arg(p: argparse.ArgumentParser, default: list[str] | None = None) -> None:
    p.add_argument(
        "--settings", nargs="+", default=default or ALL_SETTINGS, choices=sorted(SETTINGS),
        help="Structural perturbation settings (A/B/C/newB/newC are aliases).",
    )


def apply_gpu(args) -> None:
    if getattr(args, "gpu", None) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


def gen_kwargs(args) -> dict:
    return dict(batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, do_sample=not args.greedy)
