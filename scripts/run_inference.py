#!/usr/bin/env python
"""Core experiment: generate responses under structural perturbations and score refusals.

Notebook counterpart (notebooks in ``notebooks/Inference``) (No Image / Firearms / Flower) and the testing
half of the ``notebooks/SFT Fresh`` notebooks (pass ``--adapter`` to evaluate a fine-tuned model).

Examples
--------
# Paper setting: AdvBench, firearms image, all 8 settings
python scripts/run_inference.py --model qwen --datasets advbench

# Flower image (image content barely matters)
python scripts/run_inference.py --model llava --image dataset/flower.jpg --tag flower

# Evaluate an SFT adapter on harmful + harmless sets
python scripts/run_inference.py --model qwen --adapter checkpoints/qwen_mix \
    --datasets advbench harmbench alpaca --tag sft_mix
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindspots import metrics  # noqa: E402
from blindspots.cli import add_generation_args, add_model_args, add_settings_arg, apply_gpu, gen_kwargs  # noqa: E402
from blindspots.data import FIREARMS_IMAGE, load_image, load_instructions  # noqa: E402
from blindspots.models import SETTINGS, load_model  # noqa: E402
from blindspots.utils import RESULTS_DIR, hf_login, save_json, set_seed, setup_logging  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    add_generation_args(p)
    add_settings_arg(p)
    p.add_argument("--datasets", nargs="+", default=["advbench"],
                   help="advbench / harmbench / alpaca / xstest, or paths to JSON instruction files.")
    p.add_argument("--image", default=str(FIREARMS_IMAGE), help="Image used for image settings.")
    p.add_argument("--limit", type=int, default=None, help="Only use the first N instructions (for quick tests).")
    p.add_argument("--tag", default="inference", help="Results go to results/<tag>/<model>/<dataset>/.")
    p.add_argument("--output-dir", default=None, help="Override the results root directory.")
    args = p.parse_args()

    apply_gpu(args)
    setup_logging()
    set_seed(args.seed)
    hf_login()

    adapter = load_model(args.model, args.model_id, args.adapter, args.attn_implementation)
    image = load_image(args.image)
    root = Path(args.output_dir) if args.output_dir else RESULTS_DIR / args.tag

    for dataset in args.datasets:
        instructions = load_instructions(dataset)[: args.limit]
        out_dir = root / args.model / Path(dataset).stem
        summary = {}
        for name in args.settings:
            setting = SETTINGS[name]
            suffix = "_sys" if args.system_prompt else ""
            records = adapter.run(instructions, setting, image, system_prompt=args.system_prompt, **gen_kwargs(args))
            save_json(out_dir / f"{setting.name}{suffix}.json", records)
            summary[setting.name + suffix] = metrics.report(records, f"{dataset} | {setting.name}{suffix}", ("jade", "extended"))
        save_json(out_dir / "summary.json", summary)
        print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
