#!/usr/bin/env python
"""Layer-wise PCA of last-token activations: harmful vs harmless vs each perturbation.

Notebook counterpart (``notebooks/Visualization/FAPlot.ipynb``). Input is the ``activations/`` folder written by
``scripts/extract_vectors.py --save-activations``.

python scripts/plot_pca.py results/vectors/qwen/advbench/activations --output figures/qwen_pca.pdf
python scripts/plot_pca.py results/vectors/qwen/advbench/activations --layer 15 --output figures/qwen_layer16.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from blindspots.plotting import layerwise_pca, plot_pca_grid, plot_pca_single_layer  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("activations_dir")
    p.add_argument("--sets", nargs="+", default=None,
                   help="Which perturbation activation files to include (stem names). Default: all.")
    p.add_argument("--layer", type=int, default=None, help="0-indexed layer for a single-layer figure; omit for the full grid.")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    d = Path(args.activations_dir)
    harmless, harmful = np.load(d / "harmless_plain.npy"), np.load(d / "harmful_plain.npy")
    names = args.sets or sorted(f.stem for f in d.glob("*.npy") if f.stem not in ("harmless_plain", "harmful_plain"))
    others = {n: np.load(d / f"{n}.npy") for n in names}
    projected = layerwise_pca(harmless, harmful, others)
    if args.layer is None:
        plot_pca_grid(projected, args.output)
    else:
        plot_pca_single_layer(projected, args.layer, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
