"""Dataset loaders. All instruction loaders return a plain ``list[str]``."""

from __future__ import annotations

import os
import random
from pathlib import Path

import pandas as pd
from PIL import Image

from .utils import DATASET_DIR, load_json

ADVBENCH_PATH = DATASET_DIR / "advbench_instructions.json"
ALPACA_PATH = DATASET_DIR / "alpaca_instructions.json"
HARMBENCH_PATH = DATASET_DIR / "harmbench_behaviors_text_all.csv"
XSTEST_PATH = DATASET_DIR / "xstest_prompts.csv"
SFT_HARMFUL_PATH = DATASET_DIR / "sft" / "circuit_breakers_train.json"
SFT_HARMLESS_PATH = DATASET_DIR / "sft" / "alpaca_harmless_train.json"
IMAGES_DIR = DATASET_DIR / "images"
FIREARMS_IMAGE = DATASET_DIR / "firearms.jpg"
FLOWER_IMAGE = DATASET_DIR / "flower.jpg"


def load_advbench(source: str = "local") -> list[str]:
    """AdvBench harmful behaviors (520). ``source="hf"`` pulls ``walledai/AdvBench`` instead."""
    if source == "hf":
        from datasets import load_dataset

        return [row["prompt"] for row in load_dataset("walledai/AdvBench", split="train")]
    return [row["instruction"] for row in load_json(ADVBENCH_PATH)]


def load_alpaca() -> list[str]:
    """520 harmless Alpaca instructions (evaluation / refusal-direction baseline)."""
    return [row["instruction"] for row in load_json(ALPACA_PATH)]


def load_harmbench(n: int = 200) -> list[str]:
    """First ``n`` HarmBench behaviors (the first 200 are the 'standard' split)."""
    return pd.read_csv(HARMBENCH_PATH)["Behavior"][:n].tolist()


def load_xstest_safe() -> list[str]:
    """Safe XSTest prompts used to measure over-refusal (definitions/privacy types excluded)."""
    df = pd.read_csv(XSTEST_PATH)
    skip_types = {"definitions", "privacy_public", "privacy_fictional"}
    return [p for p, label, t in zip(df["prompt"], df["label"], df["type"]) if label != "unsafe" and t not in skip_types]


DATASETS = {
    "advbench": load_advbench,
    "alpaca": load_alpaca,
    "harmbench": load_harmbench,
    "xstest": load_xstest_safe,
}


def load_instructions(name_or_path: str) -> list[str]:
    """Load a named dataset (advbench/alpaca/harmbench/xstest) or a JSON file.

    A JSON file may be a list of strings, or a list of dicts with an ``instruction`` key.
    """
    if name_or_path.lower() in DATASETS:
        return DATASETS[name_or_path.lower()]()
    data = load_json(name_or_path)
    return [d if isinstance(d, str) else d["instruction"] for d in data]


def load_image(path: str | os.PathLike | None) -> Image.Image | None:
    if path is None or str(path).lower() == "none":
        return None
    return Image.open(path).convert("RGB")


def load_image_dir(directory: str | os.PathLike = IMAGES_DIR, size: tuple[int, int] = (600, 600)) -> list[Image.Image]:
    """Load every image in a directory, resized (used as a random image pool during SFT)."""
    paths = sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return [Image.open(p).convert("RGB").resize(size) for p in paths]


def load_sft_dataset(
    harmful_path: str | os.PathLike = SFT_HARMFUL_PATH,
    harmless_path: str | os.PathLike = SFT_HARMLESS_PATH,
    test_size: float = 0.2,
    shuffle_seed: int = 42,
):
    """Harmful (circuit-breaker prompts + Llama-3 refusals, label 1) + harmless Alpaca (label 0).

    Returns a stratified ``DatasetDict`` with ``train`` / ``test`` splits and columns
    ``prompt``, ``response``, ``label``.
    """
    from datasets import Dataset, concatenate_datasets

    harmful = load_json(harmful_path)
    harmless = load_json(harmless_path)
    ds_harmful = Dataset.from_dict(
        {
            "prompt": [d["prompt"] for d in harmful],
            "response": [d["llama3_output"] for d in harmful],
            "label": [1] * len(harmful),
        }
    )
    ds_harmless = Dataset.from_dict(
        {
            "prompt": [d["instruction"] for d in harmless],
            "response": [d["output"] for d in harmless],
            "label": [0] * len(harmless),
        }
    )
    merged = concatenate_datasets([ds_harmful, ds_harmless]).shuffle(seed=shuffle_seed)
    merged = merged.class_encode_column("label")
    return merged.train_test_split(test_size=test_size, stratify_by_column="label")


def sample_images(pool: list[Image.Image], k: int, rng: random.Random | None = None) -> list[Image.Image]:
    rng = rng or random
    return rng.sample(pool, k) if k <= len(pool) else [rng.choice(pool) for _ in range(k)]
