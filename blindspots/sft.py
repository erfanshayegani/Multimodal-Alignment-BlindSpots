"""Adversarial SFT mitigation: train the model to refuse under structural perturbations.

Training data = harmful prompts paired with refusals + harmless Alpaca prompts paired with
helpful answers. Each training example is rendered under one of the structural perturbation
settings (role swap and/or moved image token), and the loss is applied only to the response.

Recipes used in the paper's notebooks:

* ``stage1``  -- text-only role swap (vector A) applied to harmful examples only
  (``settings=["A"]``, ``swap_harmful_only=True``, no images).
* ``mix``     -- all six image settings, applied to every example in equal shares, usually
  starting from the stage-1 checkpoint.
* ``single``  -- one setting only (e.g. ``img_end_cross_role``) to test generalization to the others.
"""

from __future__ import annotations

import logging
import random

import torch
from PIL import Image

from .data import sample_images
from .models import SETTINGS, MLLMAdapter, Setting

logger = logging.getLogger("blindspots")

RECIPES: dict[str, list[str]] = {
    "stage1": ["A"],
    "mix": [
        "img_in_no_swap", "img_in_cross_role",
        "img_end_no_swap", "img_end_cross_role",
        "img_outside_no_swap", "img_outside_cross_role",
    ],
}


def render_split(
    split,
    adapter: MLLMAdapter,
    settings: list[str],
    swap_harmful_only: bool = False,
) -> list[str]:
    """Render a dataset split into training strings.

    The split is cut into ``len(settings)`` contiguous, equal chunks; chunk ``i`` is rendered
    with ``settings[i]``. With ``swap_harmful_only``, harmless (label 0) examples are rendered
    with the plain, unperturbed template.
    """
    resolved = [SETTINGS[s] for s in settings]
    n = len(split)
    chunk = n // len(resolved)
    bounds = [i * chunk for i in range(len(resolved))] + [n]
    texts = []
    for i, setting in enumerate(resolved):
        for j in range(bounds[i], bounds[i + 1]):
            ex = split[j]
            s = Setting() if (swap_harmful_only and ex["label"] == 0) else setting
            texts.append(adapter.build_training_text(ex["prompt"], ex["response"], s))
    return texts


def _find(seq: list[int], pattern: list[int]) -> int:
    """Index just past the first occurrence of ``pattern`` in ``seq`` (or -1)."""
    for i in range(len(seq) - len(pattern) + 1):
        if seq[i:i + len(pattern)] == pattern:
            return i + len(pattern)
    return -1


class ResponseOnlyCollator:
    """Tokenizes rendered texts (+ images) and masks every label except the response tokens.

    Whether a sequence is role-swapped is detected from its first tokens; all tokens up to and
    including the *second* role header are masked (plus, if configured, up to the image-end token
    so an image placed after the header is never a target).
    """

    def __init__(self, adapter: MLLMAdapter, image: Image.Image | None = None,
                 image_pool: list[Image.Image] | None = None, seed: int = 0):
        self.adapter = adapter
        self.image = image
        self.image_pool = image_pool
        self.rng = random.Random(seed)
        tok = adapter.tokenizer
        spec = adapter.train_label_spec()
        enc = lambda s: tok.convert_tokens_to_ids(tok.tokenize(s)) if s else None  # noqa: E731
        self.start_user, self.start_assistant = enc(spec["start_user"]), enc(spec["start_assistant"])
        self.until_user, self.until_assistant = enc(spec["until_user"]), enc(spec["until_assistant"])
        self.extra_until = enc(spec["extra_until"])
        self.ignored = set(adapter.ignored_label_ids())
        self._warned = False

    def _images_for(self, texts: list[str]):
        img_tok = getattr(self.adapter, "image_token", None)
        if not img_tok or not any(img_tok in t for t in texts):
            return None
        if self.image_pool:
            return sample_images(self.image_pool, len(texts), self.rng)
        if self.image is None:
            raise ValueError("Training texts contain image tokens but no image / image pool was given.")
        return [self.image] * len(texts)

    def mask(self, seq: list[int]) -> list[int]:
        pad_id = self.adapter.tokenizer.pad_token_id
        seq = [-100 if (t == pad_id or t < 0 or t in self.ignored) else t for t in seq]
        if seq[: len(self.start_assistant)] == self.start_assistant:  # swapped: response is in the user turn
            until = self.until_user
        elif seq[: len(self.start_user)] == self.start_user:
            until = self.until_assistant
        else:
            if not self._warned:
                logger.warning("Collator: sequence starts with neither role header; prompt tokens are not masked.")
                self._warned = True
            return seq
        for pattern in (until, self.extra_until) if self.extra_until else (until,):
            end = _find(seq, pattern)
            if end > 0:
                seq[:end] = [-100] * end
        return seq

    def __call__(self, examples):
        texts = [e["text"] if isinstance(e, dict) else e for e in examples]
        batch = self.adapter.tokenize_for_training(texts, self._images_for(texts))
        labels = [self.mask(row) for row in batch["input_ids"].tolist()]
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def lora_config(adapter: MLLMAdapter, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    from peft import LoraConfig

    return LoraConfig(
        lora_alpha=alpha, lora_dropout=dropout, r=r, bias="none",
        target_modules=list(adapter.lora_target_modules), task_type="CAUSAL_LM",
    )


def sft_config(output_dir: str, seed: int = 888, epochs: int = 1, lr: float = 1e-4,
               train_batch_size: int = 4, eval_batch_size: int = 8, grad_accum: int = 8,
               report_to: str = "none", run_name: str | None = None, tf32: bool = True,
               max_steps: int = -1):
    from trl import SFTConfig

    cfg = SFTConfig(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        eval_strategy="steps",
        eval_steps=0.2,
        save_strategy="steps",
        save_steps=0.2,
        logging_steps=10,
        learning_rate=lr,
        bf16=True,
        tf32=tf32,
        warmup_ratio=0.05,
        weight_decay=0.01,
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to=report_to,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    cfg.remove_unused_columns = False
    return cfg
