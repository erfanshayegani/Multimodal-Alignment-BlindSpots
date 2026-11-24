#!/usr/bin/env python
"""Compositionality check: does adding role swap to a text jailbreak push further against refusal?

Notebook counterpart (``notebooks/Other Attacks/<MODEL>/compositionality_check.ipynb``). For each attack:

* ``x``   = mean(successful attack prompts, plain)        - mean(their original instructions, plain)
* ``a+x`` = mean(successful attack prompts, cross role)   - mean(their original instructions, plain)

and reports the mean cosine similarity of each with ``-refusal`` over layers 11-14.

python scripts/compositionality.py --model qwen \
    --attack-dir results/other_attacks/qwen/advbench \
    --refusal results/vectors/qwen/advbench/refusal.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from blindspots.cli import add_model_args, apply_gpu  # noqa: E402
from blindspots.metrics import is_jailbroken  # noqa: E402
from blindspots.models import Setting, load_model  # noqa: E402
from blindspots.utils import hf_login, load_json, save_json, set_seed, setup_logging  # noqa: E402
from blindspots.vectors import mean_similarity_to_refusal  # noqa: E402

PLAIN, SWAP = Setting("none", False), Setting("none", True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    p.add_argument("--attack-dir", required=True, help="run_other_attacks.py output dir (has no_img_no_swap/ and no_img_cross_role/).")
    p.add_argument("--refusal", required=True, help="refusal.npy from extract_vectors.py")
    p.add_argument("--layers", type=int, nargs=2, default=[11, 15], metavar=("START", "STOP"),
                   help="Layer range (0-indexed, half-open) to average similarity over.")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    apply_gpu(args)
    setup_logging()
    set_seed(0)
    hf_login()

    adapter = load_model(args.model, args.model_id, args.adapter, args.attn_implementation)
    refusal = np.load(args.refusal)
    attack_dir = Path(args.attack_dir)
    out = Path(args.output_dir or attack_dir / "vectors")
    out.mkdir(parents=True, exist_ok=True)
    layers = range(*args.layers)

    table = {}
    for variant, setting, folder in (("x", PLAIN, "no_img_no_swap"), ("a+x", SWAP, "no_img_cross_role")):
        for f in sorted((attack_dir / folder).glob("*.json")):
            records = [r for r in load_json(f) if is_jailbroken(r["response"])]
            if not records:
                print(f"[skip] {variant} {f.stem}: no successful prompts")
                continue
            attacked = adapter.last_token_activations([r["instruction"] for r in records], setting, desc=f"{variant} {f.stem}")
            original = adapter.last_token_activations([r["original"] for r in records], PLAIN, desc=f"{f.stem} originals")
            vec = attacked.mean(0) - original.mean(0)
            np.save(out / f"{f.stem}{'_CrossRole' if variant == 'a+x' else ''}.npy", vec)
            sim = mean_similarity_to_refusal(refusal, vec, layers)
            table.setdefault(f.stem, {})[variant] = {"similarity_to_-refusal": sim, "n_successful": len(records)}

    print(f"\n{'attack':<22}{'x':>10}{'a+x':>10}")
    for attack, row in table.items():
        fmt = lambda k: f"{row[k]['similarity_to_-refusal']:.3f}" if k in row else "-"  # noqa: E731
        print(f"{attack:<22}{fmt('x'):>10}{fmt('a+x'):>10}")
    save_json(out / "compositionality.json", table)


if __name__ == "__main__":
    main()
