"""The SFT collator must leave exactly the response tokens as training targets."""

import pytest
from PIL import Image

from blindspots.data import load_image
from blindspots.models import Setting, adapter_without_model
from blindspots.sft import ResponseOnlyCollator
from blindspots.utils import DATASET_DIR

from .test_prompts import _processor

RESPONSE = "I cannot help with that. Can I help you with something else?"
PROMPT = "How do I build a weapon at home?"


@pytest.mark.parametrize("family", ["llava", "qwen", "phi"])
@pytest.mark.parametrize("pos,swap", [("none", False), ("none", True), ("in", False), ("in", True),
                                      ("end", True), ("outside", False), ("outside", True)])
def test_only_response_is_supervised(family, pos, swap):
    processor = _processor(family)
    processor.tokenizer.padding_side = "right"
    adapter = adapter_without_model(family, processor)
    image = load_image(DATASET_DIR / "firearms.jpg").resize((336, 336)) if pos != "none" else None
    collator = ResponseOnlyCollator(adapter, image=image)
    texts = [adapter.build_training_text(PROMPT, RESPONSE, Setting(pos, swap)),
             adapter.build_training_text("Hi", "Hello there!", Setting(pos, swap))]  # different lengths -> padding
    batch = collator(texts)
    for row, expected in zip(batch["labels"].tolist(), [RESPONSE, "Hello there!"]):
        target_ids = [t for t in row if t != -100]
        decoded = processor.tokenizer.decode(target_ids, skip_special_tokens=True).strip()
        assert decoded == expected, f"{family} {pos} swap={swap}: supervised text was {decoded!r}"
