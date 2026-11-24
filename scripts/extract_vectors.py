#!/usr/bin/env python
"""Compute the refusal direction and the perturbation directions (A, B, C, ...) from activations.

Notebook counterpart (activation sections of ``notebooks/Other Attacks/<MODEL>/all-other-attacks.ipynb``)
and produces the ``.npy`` files that the Visualization notebooks load.

Directions (all ``(num_layers, hidden_dim)``, last prompt token):

* ``refusal``          mean(harmful, plain) - mean(alpaca, plain)
* ``A``                successful *no-image cross-role* prompts: mean(swapped) - mean(plain)
* ``img_<pos>_B``      successful *image-at-<pos>, no swap* prompts: mean(perturbed) - mean(plain, no image)
* ``img_<pos>_C``      successful *image-at-<pos>, cross role*:     mean(perturbed) - mean(plain, no image)

"Successful" = the perturbed prompt was jailbroken in the ``run_inference.py`` results.
Paper names: B = img_outside_B, C = img_outside_C, newB = img_end_B, newC = img_end_C.

python scripts/extract_vectors.py --model qwen --responses-dir results/inference/qwen/advbench --save-activations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from blindspots.cli import ALL_SETTINGS, add_model_args, apply_gpu  # noqa: E402
from blindspots.data import FIREARMS_IMAGE, load_alpaca, load_image, load_instructions  # noqa: E402
from blindspots.metrics import is_jailbroken  # noqa: E402
from blindspots.models import SETTINGS, Setting, load_model  # noqa: E402
from blindspots.utils import RESULTS_DIR, hf_login, load_json, set_seed, setup_logging  # noqa: E402

PLAIN = Setting("none", False)


def successful_instructions(path: Path) -> list[str]:
    return [r["instruction"] for r in load_json(path) if is_jailbroken(r["response"])]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    p.add_argument("--dataset", default="advbench", help="Harmful set used for the refusal direction.")
    p.add_argument("--responses-dir", required=True, help="Directory with run_inference.py outputs for this model/dataset.")
    p.add_argument("--image", default=str(FIREARMS_IMAGE))
    p.add_argument("--positions", nargs="+", default=["in", "end", "outside"], choices=["in", "end", "outside"])
    p.add_argument("--save-activations", action="store_true",
                   help="Also save full activation sets (for scripts/plot_pca.py).")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    apply_gpu(args)
    setup_logging()
    set_seed(0)
    hf_login()

    adapter = load_model(args.model, args.model_id, args.adapter, args.attn_implementation)
    image = load_image(args.image)
    responses = Path(args.responses_dir)
    out = Path(args.output_dir or RESULTS_DIR / "vectors" / args.model / args.dataset)
    out.mkdir(parents=True, exist_ok=True)

    def acts(instructions, setting, desc):
        return adapter.last_token_activations(instructions, setting, image if setting.uses_image else None, desc=desc)

    harmful = load_instructions(args.dataset)
    harmful_acts = acts(harmful, PLAIN, "harmful")
    harmless_acts = acts(load_alpaca(), PLAIN, "harmless")
    np.save(out / "refusal.npy", harmful_acts.mean(0) - harmless_acts.mean(0))

    # vector name -> perturbed setting; every vector is compared with the same prompts, plain.
    jobs = {"A": "no_img_cross_role"}
    for pos in args.positions:
        jobs[f"img_{pos}_B"] = f"img_{pos}_no_swap"
        jobs[f"img_{pos}_C"] = f"img_{pos}_cross_role"

    for vec_name, setting_name in jobs.items():
        path = responses / f"{SETTINGS[setting_name].name}.json"
        if not path.exists():
            print(f"[skip] {vec_name}: {path} not found")
            continue
        success = successful_instructions(path)
        print(f"{vec_name}: {len(success)} successful prompts in {path.name}")
        if not success:
            continue
        perturbed = acts(success, SETTINGS[setting_name], vec_name)
        baseline = acts(success, PLAIN, f"{vec_name} baseline")
        np.save(out / f"{vec_name}.npy", perturbed.mean(0) - baseline.mean(0))

    if args.save_activations:
        act_dir = out / "activations"
        act_dir.mkdir(exist_ok=True)
        np.save(act_dir / "harmful_plain.npy", harmful_acts)
        np.save(act_dir / "harmless_plain.npy", harmless_acts)
        for name in ALL_SETTINGS:
            if name == "no_img_no_swap":
                continue
            np.save(act_dir / f"{name}.npy", acts(harmful, SETTINGS[name], name))

    print(f"Saved vectors to {out}")


if __name__ == "__main__":
    main()
