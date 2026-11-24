"""Small shared helpers: paths, JSON I/O, seeding, Hugging Face login, GPU memory."""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
RESULTS_DIR = REPO_ROOT / "results"

logger = logging.getLogger("blindspots")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    # transformers is very chatty at INFO/WARNING level during generation.
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except ImportError:
        pass


def set_seed(seed: int) -> None:
    from transformers import set_seed as hf_set_seed

    hf_set_seed(seed)


def save_json(path: str | os.PathLike, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def hf_login() -> None:
    """Log in to the Hugging Face Hub if ``HF_TOKEN`` is set (needed for gated models).

    Tokens are never hard-coded: export ``HF_TOKEN`` in your shell or put it in a ``.env``
    file that you source yourself.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.info("HF_TOKEN not set; skipping Hugging Face login (fine for non-gated models).")
        return
    from huggingface_hub import login

    login(token=token)


def free_gpu_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info(
            "GPU memory allocated: %.2f GB | reserved: %.2f GB",
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.memory_reserved() / 1024**3,
        )
