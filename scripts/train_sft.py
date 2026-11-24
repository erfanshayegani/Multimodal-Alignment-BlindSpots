#!/usr/bin/env python
"""Adversarial SFT mitigation (QLoRA) under structural perturbations.

Notebook counterpart (training half of the ``notebooks/SFT Fresh/<MODEL>/SFT *.ipynb``).

# Stage 1: text-only role swap on harmful examples
python scripts/train_sft.py --model qwen --recipe stage1 --output-dir checkpoints/qwen_stage1

# Stage 2: all six image settings, continuing from stage 1, random image pool
python scripts/train_sft.py --model qwen --recipe mix --init-from checkpoints/qwen_stage1 \
    --image-dir dataset/images --output-dir checkpoints/qwen_mix

# Generalization test: train on a single setting only
python scripts/train_sft.py --model qwen --settings img_end_cross_role --image-dir dataset/images \
    --output-dir checkpoints/qwen_newC

Then evaluate with:  python scripts/run_inference.py --model qwen --adapter checkpoints/qwen_mix ...
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindspots import sft  # noqa: E402
from blindspots.cli import apply_gpu  # noqa: E402
from blindspots.data import (  # noqa: E402
    FIREARMS_IMAGE, SFT_HARMFUL_PATH, SFT_HARMLESS_PATH, load_image, load_image_dir, load_sft_dataset,
)
from blindspots.models import ADAPTERS, SETTINGS  # noqa: E402
from blindspots.utils import hf_login, set_seed, setup_logging  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    p.add_argument("--init-from", default=None,
                   help="Base model id/path to start from. A stage-1 adapter dir works here too "
                        "(transformers loads base + adapter). Defaults to the original checkpoint.")
    p.add_argument("--recipe", choices=list(sft.RECIPES), default=None, help="Preset list of settings.")
    p.add_argument("--settings", nargs="+", choices=sorted(SETTINGS), default=None, help="Explicit settings (overrides --recipe).")
    p.add_argument("--swap-harmful-only", action="store_true",
                   help="Leave harmless examples unperturbed (default for --recipe stage1).")
    img = p.add_mutually_exclusive_group()
    img.add_argument("--image", default=str(FIREARMS_IMAGE), help="Single image used for every example.")
    img.add_argument("--image-dir", default=None, help="Sample a random image per example from this directory.")
    p.add_argument("--harmful-data", default=str(SFT_HARMFUL_PATH))
    p.add_argument("--harmless-data", default=str(SFT_HARMLESS_PATH))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=-1, help="Stop after N optimizer steps (for smoke tests).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=None, help="Default: 8 (4 for phi).")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--early-stopping-patience", type=int, default=3)
    p.add_argument("--report-to", default="none", help="'wandb' to log (set WANDB_API_KEY / WANDB_PROJECT).")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=888)
    p.add_argument("--gpu", default=None)
    args = p.parse_args()

    if args.settings is None and args.recipe is None:
        p.error("give --recipe or --settings")
    settings = args.settings or sft.RECIPES[args.recipe]
    swap_harmful_only = args.swap_harmful_only or args.recipe == "stage1"

    apply_gpu(args)
    setup_logging()
    set_seed(args.seed)
    hf_login()

    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import EarlyStoppingCallback
    from datasets import Dataset
    from trl import SFTTrainer

    adapter, model = ADAPTERS[args.model].load_for_training(args.init_from)

    data = load_sft_dataset(args.harmful_data, args.harmless_data)
    rng = random.Random(args.seed)
    train_texts = sft.render_split(data["train"], adapter, settings, swap_harmful_only)
    eval_texts = sft.render_split(data["test"], adapter, settings, swap_harmful_only)
    rng.shuffle(train_texts)
    rng.shuffle(eval_texts)
    print(f"{len(train_texts)} train / {len(eval_texts)} eval examples. Settings: {settings}")
    print("Example:\n", train_texts[0][:500])

    uses_image = any(SETTINGS[s].uses_image for s in settings)
    collator = sft.ResponseOnlyCollator(
        adapter,
        image=load_image(args.image) if uses_image and not args.image_dir else None,
        image_pool=load_image_dir(args.image_dir) if uses_image and args.image_dir else None,
        seed=args.seed,
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, sft.lora_config(adapter, r=args.lora_r, alpha=args.lora_alpha))
    model.print_trainable_parameters()

    config = sft.sft_config(
        args.output_dir, seed=args.seed, epochs=args.epochs, lr=args.lr,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size or (4 if args.model == "phi" else 8),
        grad_accum=args.grad_accum, report_to=args.report_to, run_name=args.run_name,
        tf32=args.model != "phi", max_steps=args.max_steps,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_dict({"text": train_texts}),
        eval_dataset=Dataset.from_dict({"text": eval_texts}),
        processing_class=adapter.tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
