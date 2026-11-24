#!/usr/bin/env python
"""Layer-wise cosine similarity of each perturbation direction with the (negated) refusal direction.

Notebook counterpart (``notebooks/Visualization/generic vector visualization.ipynb``). Input is the folder written by
``scripts/extract_vectors.py``.

python scripts/plot_similarity.py results/vectors/qwen/advbench --output figures/qwen_cosine_refusal.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from blindspots.plotting import plot_similarity_curves  # noqa: E402
from blindspots.vectors import layerwise_cosine  # noqa: E402

LABELS = {
    "A": "swap", "img_in_B": "img pos", "img_in_C": "img pos_swap",
    "img_outside_B": "img out", "img_outside_C": "img out_swap",
    "img_end_B": "img end", "img_end_C": "img end_swap",
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("vectors_dir")
    p.add_argument("--target", default="refusal", help="Vector to compare against (negated if 'refusal').")
    p.add_argument("--no-random", action="store_true", help="Omit the random-vector baseline.")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    d = Path(args.vectors_dir)
    target = np.load(d / f"{args.target}.npy")
    if args.target == "refusal":
        target = -target
    curves = {}
    for stem, label in LABELS.items():
        if (d / f"{stem}.npy").exists():
            curves[label] = layerwise_cosine(np.load(d / f"{stem}.npy"), target)
    if not args.no_random:
        curves["random"] = layerwise_cosine(np.random.default_rng(0).random(target.shape), target)
    if args.target == "refusal":
        curves["refusal"] = layerwise_cosine(target, target)
    plot_similarity_curves(curves, args.output)
    for name, c in curves.items():
        print(f"{name:<14} mean cos = {c.mean():+.3f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
