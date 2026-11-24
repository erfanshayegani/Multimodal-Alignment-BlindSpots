"""Model adapters for the three MLLM families studied in the paper.

Each adapter knows its model's chat-template strings and implements the two structural
perturbations from the paper:

* **Role swap ("cross role")** -- exchange the user and assistant role tags in the formatted
  prompt, so the instruction sits in the *assistant* turn and generation starts in the *user* turn.
* **Image-token placement** -- where the image placeholder goes:

  ==========  ===============================================================
  ``none``    text only, no image
  ``in``      default position (inside the user turn, before the text)
  ``end``     end of the user turn, right before the assistant header
  ``outside`` after the assistant header, i.e. outside the user turn
  ==========  ===============================================================

The vector names used in the paper map onto these settings as follows:
``A`` = (none, swap), ``B`` = (outside, no swap), ``C`` = (outside, swap),
``newB`` = (end, no swap), ``newC`` = (end, swap).
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

logger = logging.getLogger("blindspots")

IMAGE_POSITIONS = ("none", "in", "end", "outside")
_SWAP_SENTINEL = "<<__ROLE_SWAP__>>"


@dataclass(frozen=True)
class Setting:
    """A structural perturbation: image placement + whether roles are swapped."""

    image_position: str = "none"
    swap: bool = False

    def __post_init__(self):
        if self.image_position not in IMAGE_POSITIONS:
            raise ValueError(f"image_position must be one of {IMAGE_POSITIONS}, got {self.image_position!r}")

    @property
    def uses_image(self) -> bool:
        return self.image_position != "none"

    @property
    def name(self) -> str:
        return f"{'no_img' if self.image_position == 'none' else 'img_' + self.image_position}_{'cross_role' if self.swap else 'no_swap'}"


# Named settings, including the paper's vector names.
SETTINGS: dict[str, Setting] = {
    "no_img_no_swap": Setting("none", False),
    "no_img_cross_role": Setting("none", True),  # A
    "img_in_no_swap": Setting("in", False),
    "img_in_cross_role": Setting("in", True),
    "img_end_no_swap": Setting("end", False),  # newB
    "img_end_cross_role": Setting("end", True),  # newC
    "img_outside_no_swap": Setting("outside", False),  # B
    "img_outside_cross_role": Setting("outside", True),  # C
}
SETTINGS.update({"A": SETTINGS["no_img_cross_role"], "B": SETTINGS["img_outside_no_swap"],
                 "C": SETTINGS["img_outside_cross_role"], "newB": SETTINGS["img_end_no_swap"],
                 "newC": SETTINGS["img_end_cross_role"]})


def default_attn_implementation(fallback: str = "sdpa") -> str:
    return "flash_attention_2" if importlib.util.find_spec("flash_attn") is not None else fallback


class MLLMAdapter:
    """Base class. Subclasses define template strings and model loading."""

    family: str = ""
    default_model_id: str = ""
    user_tag: str = ""
    assistant_tag: str = ""
    default_system: str = ""  # removed from the prompt when ``system_prompt=False``
    # LoRA targets for SFT
    lora_target_modules: tuple[str, ...] = (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    )

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def device(self):
        return self.model.device

    # ------------------------------------------------------------------ prompts
    def swap_roles(self, text: str) -> str:
        text = text.replace(self.assistant_tag, _SWAP_SENTINEL)
        text = text.replace(self.user_tag, self.assistant_tag)
        return text.replace(_SWAP_SENTINEL, self.user_tag)

    def _chat_template(self, instruction: str, with_image: bool, response: str | None = None) -> str:
        raise NotImplementedError

    def _place_image(self, text: str, position: str, training: bool = False) -> str:
        raise NotImplementedError

    def _strip_system(self, text: str, system_prompt: bool) -> str:
        if not system_prompt and self.default_system:
            text = text.replace(self.default_system, "")
        return text

    def build_prompt(self, instruction: str, setting: Setting, system_prompt: bool = False) -> str:
        """Format a generation prompt with the given structural perturbation applied."""
        text = self._chat_template(instruction, with_image=setting.uses_image)
        text = self._strip_system(text, system_prompt)
        if setting.uses_image:
            text = self._place_image(text, setting.image_position)
        if setting.swap:
            text = self.swap_roles(text)
        return text

    def build_training_text(self, prompt: str, response: str, setting: Setting) -> str:
        """Format a full (prompt, response) conversation for SFT with the perturbation applied."""
        text = self._chat_template(prompt, with_image=setting.uses_image, response=response)
        text = self._strip_system(text, system_prompt=False)
        if setting.uses_image:
            text = self._place_image(text, setting.image_position, training=True)
        if setting.swap:
            text = self.swap_roles(text)
        return text

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def _per_prompt_images(prompts: list[str], image) -> list | None:
        if image is None:
            return None
        return list(image) if isinstance(image, (list, tuple)) else [image] * len(prompts)

    def prepare_inputs(self, prompts: list[str], image: Image.Image | list[Image.Image] | None):
        """Tokenize a batch of formatted prompts. ``image`` is shared by all prompts, or one per prompt."""
        images = self._per_prompt_images(prompts, image)
        return self.processor(
            images=images, text=prompts, return_tensors="pt",
            add_special_tokens=False, padding=True, truncation=True,
        ).to(self.device)

    def _decode(self, generated_ids) -> list[str]:
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def _generate_kwargs(self) -> dict:
        return {}

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        image: Image.Image | list[Image.Image] | None = None,
        batch_size: int = 16,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        desc: str = "generate",
    ) -> list[str]:
        """Generate for already-formatted prompts. ``image``: one shared image, one per prompt, or None."""
        outputs: list[str] = []
        for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
            batch = prompts[start:start + batch_size]
            batch_image = image[start:start + batch_size] if isinstance(image, (list, tuple)) else image
            inputs = self.prepare_inputs(batch, batch_image)
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, **self._generate_kwargs())
            prompt_len = inputs["input_ids"].shape[1]  # left padding -> all prompts end at the same index
            outputs.extend(self._decode(out[:, prompt_len:]))
        return outputs

    def run(
        self,
        instructions: list[str],
        setting: Setting,
        image: Image.Image | None = None,
        system_prompt: bool = False,
        **gen_kwargs,
    ) -> list[dict]:
        """Build perturbed prompts, generate, and return records ``{instruction, prompt, response}``."""
        if setting.uses_image and image is None:
            raise ValueError(f"Setting {setting.name} needs an image.")
        prompts = [self.build_prompt(ins, setting, system_prompt) for ins in instructions]
        responses = self.generate(prompts, image if setting.uses_image else None, desc=setting.name, **gen_kwargs)
        return [{"instruction": i, "prompt": p, "response": r} for i, p, r in zip(instructions, prompts, responses)]

    # ------------------------------------------------------------------ activations
    @torch.no_grad()
    def last_token_activations(
        self,
        instructions: list[str],
        setting: Setting,
        image: Image.Image | None = None,
        system_prompt: bool = False,
        desc: str = "activations",
    ) -> np.ndarray:
        """Residual-stream activations of the final prompt token at every layer.

        Returns an array of shape ``(num_samples, num_layers, hidden_dim)`` (embedding layer excluded).
        """
        acts = []
        for ins in tqdm(instructions, desc=desc):
            prompt = self.build_prompt(ins, setting, system_prompt)
            inputs = self.prepare_inputs([prompt], image if setting.uses_image else None)
            out = self.model(**inputs, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states[1:]
            acts.append(torch.stack([layer[0, -1, :] for layer in hs]).float().cpu().numpy())
        return np.stack(acts) if acts else np.empty((0,))

    # ------------------------------------------------------------------ SFT
    def train_label_spec(self) -> dict:
        """Token patterns used by the SFT collator to mask everything except the response.

        ``start_user`` / ``start_assistant``: what the sequence starts with in the normal / swapped case.
        ``until_user`` / ``until_assistant``: mask up to and including the first occurrence.
        ``extra_until``: additionally mask up to and including this pattern (e.g. the image end token).
        """
        raise NotImplementedError

    def ignored_label_ids(self) -> list[int]:
        return []

    def tokenize_for_training(self, texts: list[str], images: list[Image.Image] | None):
        return self.processor(text=texts, images=images, return_tensors="pt", padding=True)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, model_id: str | None = None, adapter_path: str | None = None,
             attn_implementation: str | None = None) -> "MLLMAdapter":
        raise NotImplementedError

    @classmethod
    def load_for_training(cls, model_id: str | None = None, attn_implementation: str | None = None):
        """Return ``(adapter, model)`` with a 4-bit quantized model and right-padding processor."""
        raise NotImplementedError

    def load_lora(self, adapter_path: str | None) -> None:
        if adapter_path:
            logger.info("Loading LoRA adapter from %s", adapter_path)
            self.model.load_adapter(adapter_path)


def _bnb_4bit_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )


# ====================================================================== LLaVA-1.5
class LlavaAdapter(MLLMAdapter):
    """``USER: <image>\\n{instruction} ASSISTANT:``"""

    family = "llava"
    default_model_id = "llava-hf/llava-1.5-7b-hf"
    user_tag = "USER"
    assistant_tag = "ASSISTANT"
    image_token = "<image>"

    def _chat_template(self, instruction, with_image, response=None):
        content = ([{"type": "image"}] if with_image else []) + [{"type": "text", "text": instruction}]
        messages = [{"role": "user", "content": content}]
        if response is None:
            return self.processor.apply_chat_template(messages, add_generation_prompt=True)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def _place_image(self, text, position, training=False):
        if position == "in":
            return text
        text = text.replace(self.image_token, "")
        if position == "end":
            return text.replace("ASSISTANT:", self.image_token + "ASSISTANT:")
        # outside: right after the assistant header (== end of string for a generation prompt)
        return text.replace("ASSISTANT:", "ASSISTANT:" + self.image_token)

    def train_label_spec(self):
        return dict(start_user="<s>USER:", start_assistant="<s>ASSISTANT:",
                    until_user="USER:", until_assistant="ASSISTANT:", extra_until=None)

    def ignored_label_ids(self):
        return [self.tokenizer.convert_tokens_to_ids(self.image_token)]

    @classmethod
    def load(cls, model_id=None, adapter_path=None, attn_implementation=None):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        model_id = model_id or cls.default_model_id
        model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto").eval()
        processor = AutoProcessor.from_pretrained(cls.default_model_id if adapter_path else model_id)
        processor.tokenizer.padding_side = "left"
        adapter = cls(model, processor)
        adapter.load_lora(adapter_path)
        return adapter

    @classmethod
    def load_for_training(cls, model_id=None, attn_implementation=None):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(
            model_id or cls.default_model_id, device_map="auto", torch_dtype=torch.float16,
            quantization_config=_bnb_4bit_config(),
        )
        processor = AutoProcessor.from_pretrained(cls.default_model_id)
        processor.tokenizer.padding_side = "right"
        return cls(model, processor), model


# ====================================================================== Qwen2-VL
class QwenVLAdapter(MLLMAdapter):
    """``<|im_start|>user\\n<|vision_start|><|image_pad|><|vision_end|>{instruction}<|im_end|>\\n<|im_start|>assistant\\n``"""

    family = "qwen"
    default_model_id = "Qwen/Qwen2-VL-7B-Instruct"
    user_tag = "<|im_start|>user"
    assistant_tag = "<|im_start|>assistant"
    default_system = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    image_token = "<|vision_start|><|image_pad|><|vision_end|>"

    def _chat_template(self, instruction, with_image, response=None):
        content = ([{"type": "image"}] if with_image else []) + [{"type": "text", "text": instruction}]
        messages = [{"role": "user", "content": content}]
        if response is None:
            return self.processor.apply_chat_template(messages, add_generation_prompt=True)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def _place_image(self, text, position, training=False):
        if position == "in":
            return text
        text = text.replace(self.image_token, "")
        if position == "end":
            anchor = "<|im_end|>\n<|im_start|>assistant"
            return text.replace(anchor, self.image_token + anchor)
        anchor = "<|im_end|>\n<|im_start|>assistant\n"
        if training:
            return text.replace(anchor, anchor + self.image_token)
        return text + self.image_token

    def train_label_spec(self):
        return dict(start_user="<|im_start|>user\n", start_assistant="<|im_start|>assistant\n",
                    until_user="<|im_start|>user", until_assistant="<|im_start|>assistant",
                    extra_until="<|vision_end|>")

    @classmethod
    def load(cls, model_id=None, adapter_path=None, attn_implementation=None):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        model_id = model_id or cls.default_model_id
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation=attn_implementation or default_attn_implementation(),
        ).eval()
        processor = AutoProcessor.from_pretrained(cls.default_model_id if adapter_path else model_id)
        processor.tokenizer.padding_side = "left"
        adapter = cls(model, processor)
        adapter.load_lora(adapter_path)
        return adapter

    @classmethod
    def load_for_training(cls, model_id=None, attn_implementation=None):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id or cls.default_model_id, device_map="auto", torch_dtype=torch.bfloat16,
            quantization_config=_bnb_4bit_config(),
        )
        processor = AutoProcessor.from_pretrained(cls.default_model_id)
        processor.tokenizer.padding_side = "right"
        return cls(model, processor), model


# ====================================================================== Phi-3.5-Vision
def _pad_and_stack(per_sample: list, pad_token_id: int, side: str) -> dict:
    """Phi-3.5-V's processor cannot batch, so samples are processed one by one and padded here."""
    seqs = [s["input_ids"][0] for s in per_sample]
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        if side == "left":
            input_ids[i, max_len - len(s):] = s
        else:
            input_ids[i, :len(s)] = s
    batch = {"input_ids": input_ids, "attention_mask": (input_ids != pad_token_id).long()}
    if "pixel_values" in per_sample[0]:
        batch["pixel_values"] = torch.cat([s["pixel_values"] for s in per_sample], dim=0)
        batch["image_sizes"] = torch.cat([s["image_sizes"] for s in per_sample], dim=0)
    return batch


def _patch_cache_for_phi() -> None:
    """Phi-3.5-V's remote code uses ``DynamicCache`` methods removed in newer transformers."""
    from transformers import DynamicCache

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)


class PhiVisionAdapter(MLLMAdapter):
    """``<|user|>\\n<|image_1|>\\n{instruction}<|end|>\\n<|assistant|>\\n``"""

    family = "phi"
    default_model_id = "microsoft/Phi-3.5-vision-instruct"
    user_tag = "<|user|>"
    assistant_tag = "<|assistant|>"
    image_token = "<|image_1|>"
    lora_target_modules = ("self_attn.o_proj", "self_attn.qkv_proj", "mlp.gate_up_proj", "mlp.down_proj")

    def _chat_template(self, instruction, with_image, response=None):
        content = ("<|image_1|>\n" if with_image else "") + instruction
        messages = [{"role": "user", "content": content}]
        if response is not None:
            messages.append({"role": "assistant", "content": response})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=response is None)

    def _place_image(self, text, position, training=False):
        if position == "in":
            return text
        if training:
            text = text.replace("<|image_1|>\n", "")
            if position == "end":
                return text.replace("<|assistant|>", "<|image_1|>\n<|assistant|>")
            return text.replace("<|assistant|>", "<|assistant|>\n<|image_1|>")
        text = text.replace("<|image_1|>", "")
        if position == "end":
            return text.replace("<|assistant|>", "<|image_1|><|assistant|>")
        return text + "<|image_1|>\n"

    def prepare_inputs(self, prompts, image):
        images = self._per_prompt_images(prompts, image)
        per_sample = [
            self.processor(images=[images[i]] if images else None, text=p, return_tensors="pt")
            for i, p in enumerate(prompts)
        ]
        batch = _pad_and_stack(per_sample, self.tokenizer.pad_token_id, side="left")
        return {k: v.to(self.device) for k, v in batch.items()}

    def _decode(self, generated_ids):
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def _generate_kwargs(self):
        return {"eos_token_id": self.tokenizer.eos_token_id}

    def train_label_spec(self):
        # Match on the bare special tokens: Phi's tokenizer turns tokenize("<|assistant|>\n") into a
        # single unknown token, which would never match the in-context token sequence.
        return dict(start_user="<s><|user|>", start_assistant="<s><|assistant|>",
                    until_user="<|user|>\n", until_assistant="<|assistant|>", extra_until=None)

    def tokenize_for_training(self, texts, images):
        per_sample = [
            self.processor(images=images[k] if images else None, text=texts[k], return_tensors="pt")
            for k in range(len(texts))
        ]
        return _pad_and_stack(per_sample, self.tokenizer.pad_token_id, side="right")

    @classmethod
    def load(cls, model_id=None, adapter_path=None, attn_implementation=None):
        from transformers import AutoModelForCausalLM, AutoProcessor

        _patch_cache_for_phi()
        model_id = model_id or cls.default_model_id
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto",
            _attn_implementation=attn_implementation or default_attn_implementation("eager"),
        ).eval()
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)
        processor.tokenizer.padding_side = "left"
        adapter = cls(model, processor)
        adapter.load_lora(adapter_path)
        return adapter

    @classmethod
    def load_for_training(cls, model_id=None, attn_implementation=None):
        from transformers import AutoModelForCausalLM, AutoProcessor

        _patch_cache_for_phi()
        model_id = model_id or cls.default_model_id
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map={"": torch.cuda.current_device()}, trust_remote_code=True,
            _attn_implementation=attn_implementation or default_attn_implementation("eager"),
            use_cache=False, quantization_config=_bnb_4bit_config(),
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)
        processor.tokenizer.padding_side = "right"
        return cls(model, processor), model


ADAPTERS: dict[str, type[MLLMAdapter]] = {a.family: a for a in (LlavaAdapter, QwenVLAdapter, PhiVisionAdapter)}


def load_model(family: str, model_id: str | None = None, adapter_path: str | None = None,
               attn_implementation: str | None = None) -> MLLMAdapter:
    """Load an MLLM for inference, optionally with a LoRA adapter (e.g. from ``scripts/train_sft.py``)."""
    if family not in ADAPTERS:
        raise ValueError(f"Unknown model family {family!r}; choose from {list(ADAPTERS)}")
    return ADAPTERS[family].load(model_id=model_id, adapter_path=adapter_path, attn_implementation=attn_implementation)


def adapter_without_model(family: str, processor) -> MLLMAdapter:
    """An adapter for prompt building / tokenization only (no weights loaded)."""
    return ADAPTERS[family](model=None, processor=processor)
