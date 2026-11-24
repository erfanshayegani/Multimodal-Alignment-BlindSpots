#!/usr/bin/env python
"""Score existing result files (any JSON list of ``{"response": ...}`` records) for refusals.

python scripts/evaluate_refusal.py results/inference/qwen/advbench/*.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindspots import metrics  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--prefix-sets", nargs="+", default=["jade", "extended"], choices=list(metrics.PREFIX_SETS))
    args = p.parse_args()
    for f in args.files:
        if Path(f).name == "summary.json":
            continue
        metrics.report(f, Path(f).stem, args.prefix_sets)


if __name__ == "__main__":
    main()
