#!/usr/bin/env python
"""Utility check after SFT: VQA accuracy and reward-model score on VQAv2 validation questions.

Notebook counterpart (``notebooks/SFT Fresh/<MODEL>/VQA Reward*.ipynb``).

Uses the ``lmms-lab/VQAv2`` validation set (first shard; ``Graphcore/vqa``, used in the paper, is no
longer on the Hugging Face Hub, so the exact 520 questions differ).

python scripts/vqa_eval.py --model qwen --adapter checkpoints/qwen_mix --reward-model openassistant
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from blindspots.cli import add_generation_args, add_model_args, apply_gpu  # noqa: E402
from blindspots.models import SETTINGS, load_model  # noqa: E402
from blindspots.utils import RESULTS_DIR, hf_login, save_json, set_seed, setup_logging  # noqa: E402

VQA_INSTRUCTION = "Answer the question using a single phrase or word. "
REWARD_MODELS = {
    "openassistant": "OpenAssistant/reward-model-deberta-v3-large-v2",
    "skywork": "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
}


def load_vqa(n: int, repo: str, data_file: str):
    from datasets import load_dataset

    ds = load_dataset(repo, data_files={"validation": data_file}, split="validation")
    ds = ds.select(range(min(n, len(ds))))
    gold = [sorted({a["answer"].lower() for a in ex["answers"]}) for ex in ds]
    return ds["question"], [img.convert("RGB") for img in ds["image"]], gold


def clean(text: str) -> str:
    try:
        import emoji

        text = emoji.replace_emoji(text, replace="")
    except ImportError:
        pass
    return " ".join(re.sub(r"[^a-zA-Z0-9]", " ", text).split())


def accuracy(responses: list[str], gold: list[list[str]], split_on: str) -> float:
    hits = 0
    for resp, answers in zip(responses, gold):
        pred = resp.lower().split(split_on)[0].strip()
        hits += any(a in pred for a in answers)
    return hits / len(responses)


@torch.no_grad()
def reward_scores(kind: str, questions: list[str], responses: list[str], batch_size: int = 16) -> float:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = REWARD_MODELS[kind]
    if kind == "skywork":
        rm = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto", num_labels=1)
    else:
        rm = AutoModelForSequenceClassification.from_pretrained(name).to("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(name)
    scores = []
    for s in tqdm(range(0, len(questions), batch_size), desc=f"reward ({kind})"):
        q, r = questions[s:s + batch_size], responses[s:s + batch_size]
        if kind == "skywork":
            convs = [tok.apply_chat_template([{"role": "user", "content": a}, {"role": "assistant", "content": b}], tokenize=False)
                     for a, b in zip(q, r)]
            inputs = tok(convs, return_tensors="pt", padding=True, truncation=True)
        else:
            inputs = tok(q, r, return_tensors="pt", padding=True, truncation=True)
        scores += rm(**inputs.to(rm.device)).logits[:, 0].float().cpu().tolist()
    return sum(scores) / len(scores)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(p)
    add_generation_args(p)
    p.add_argument("--settings", nargs="+", default=["img_in_no_swap"], choices=sorted(s for s in SETTINGS if SETTINGS[s].uses_image))
    p.add_argument("--n", type=int, default=520)
    p.add_argument("--vqa-repo", default="lmms-lab/VQAv2")
    p.add_argument("--vqa-file", default="data/validation-00000-of-00068.parquet")
    p.add_argument("--reward-model", default="openassistant", choices=list(REWARD_MODELS) + ["none"])
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    apply_gpu(args)
    setup_logging()
    set_seed(args.seed)
    hf_login()

    questions, images, gold = load_vqa(args.n, args.vqa_repo, args.vqa_file)
    adapter = load_model(args.model, args.model_id, args.adapter, args.attn_implementation)
    out_dir = Path(args.output_dir or RESULTS_DIR / "vqa" / args.model)

    results = {}
    for name in args.settings:
        setting = SETTINGS[name]
        prompts = [adapter.build_prompt(VQA_INSTRUCTION + q, setting, args.system_prompt) for q in questions]
        responses = adapter.generate(prompts, images, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                                     do_sample=not args.greedy, desc=name)
        responses = [clean(r) for r in responses]
        acc = accuracy(responses, gold, split_on="\n" if args.model == "phi" else ",")
        results[name] = {"accuracy": acc}
        save_json(out_dir / f"{setting.name}.json",
                  [{"question": q, "prompt": pr, "response": r, "answers": g} for q, pr, r, g in zip(questions, prompts, responses, gold)])
        print(f"{name}: VQA accuracy {acc:.1%}")
        if args.reward_model != "none":
            results[name]["reward"] = reward_scores(args.reward_model, list(questions), responses)
            print(f"{name}: mean reward ({args.reward_model}) {results[name]['reward']:.3f}")
    save_json(out_dir / "summary.json", results)


if __name__ == "__main__":
    main()
