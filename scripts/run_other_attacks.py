#!/usr/bin/env python
"""Comparison attacks: text jailbreak templates (and GCG suffixes), with and without role swap.

Notebook counterpart (generation half of ``notebooks/Other Attacks/<MODEL>/all-other-attacks.ipynb``).
Each record stores the ``original`` instruction next to the attacked one, which the
compositionality analysis (``scripts/compositionality.py``) uses.

python scripts/run_other_attacks.py --model qwen --dataset harmbench
python scripts/run_other_attacks.py --model qwen --dataset advbench --attacks gcg \
    --gcg-file results/gcg/vicuna_advbench_gcg.json
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindspots import metrics  # noqa: E402
from blindspots.attacks import ATTACK_TEMPLATES, apply_attack  # noqa: E402
from blindspots.cli import add_generation_args, add_model_args, add_settings_arg, apply_gpu, gen_kwargs  # noqa: E402
from blindspots.data import FIREARMS_IMAGE, load_image, load_instructions  # noqa: E402
from blindspots.models import SETTINGS, load_model  # noqa: E402
from blindspots.utils import RESULTS_DIR, hf_login, load_json, save_json, set_seed, setup_logging  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    add_generation_args(p)
    add_settings_arg(p, default=["no_img_no_swap", "no_img_cross_role"])
    p.add_argument("--dataset", default="advbench")
    p.add_argument("--attacks", nargs="+", default=list(ATTACK_TEMPLATES), choices=list(ATTACK_TEMPLATES) + ["gcg"])
    p.add_argument("--gcg-file", default=None, help="JSON list of GCG-suffixed instructions (from run_gcg.py optimize).")
    p.add_argument("--image", default=str(FIREARMS_IMAGE))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    apply_gpu(args)
    setup_logging()
    set_seed(args.seed)
    hf_login()

    originals = load_instructions(args.dataset)[: args.limit]
    adapter = load_model(args.model, args.model_id, args.adapter, args.attn_implementation)
    image = load_image(args.image)
    out_root = Path(args.output_dir) if args.output_dir else RESULTS_DIR / "other_attacks" / args.model / args.dataset

    for attack in args.attacks:
        if attack == "gcg":
            if not args.gcg_file:
                p.error("--attacks gcg requires --gcg-file")
            attacked = load_json(args.gcg_file)[: len(originals)]
        else:
            attacked = apply_attack(attack, originals, random.Random(args.seed))
        for name in args.settings:
            setting = SETTINGS[name]
            records = adapter.run(attacked, setting, image, system_prompt=args.system_prompt, **gen_kwargs(args))
            for r, orig in zip(records, originals):
                r["original"] = orig
            save_json(out_root / setting.name / f"{attack}.json", records)
            metrics.report(records, f"{attack} | {setting.name}")


if __name__ == "__main__":
    main()
