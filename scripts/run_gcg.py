#!/usr/bin/env python
"""GCG baseline: optimize adversarial suffixes, then test them on text-only LLMs with/without role swap.

Notebook counterpart (``notebooks/Other Attacks/gcg attack.ipynb``). Requires ``pip install nanogcg``.

# 1) Optimize one suffix per AdvBench instruction on Vicuna-7B (slow: ~200 GCG steps each)
python scripts/run_gcg.py optimize --output results/gcg/vicuna_advbench_gcg.json

# 2) Evaluate those prompts on a text LLM (normal and cross-role)
python scripts/run_gcg.py evaluate --llm qwen2 --prompts results/gcg/vicuna_advbench_gcg.json
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from blindspots import metrics  # noqa: E402
from blindspots.data import load_instructions  # noqa: E402
from blindspots.utils import RESULTS_DIR, hf_login, load_json, save_json, set_seed, setup_logging  # noqa: E402

VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
)
VICUNA_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}"
    "{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}"
    "{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}"
    "{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}"
    "{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"
)

# name -> (hf id, user tag, assistant tag, default system text to strip)
TEXT_LLMS = {
    "vicuna": ("lmsys/vicuna-7b-v1.5", "USER", "ASSISTANT", VICUNA_SYSTEM),
    "qwen2": ("Qwen/Qwen2-7B-Instruct", "<|im_start|>user", "<|im_start|>assistant",
              "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"),
    "phi3": ("microsoft/Phi-3-mini-4k-instruct", "<|user|>", "<|assistant|>", ""),
}


def load_text_llm(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = TEXT_LLMS[name][0]
    kw = dict(trust_remote_code=True, torch_dtype="auto") if name == "phi3" else dict(torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", **kw).eval()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=name == "phi3")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    if name == "vicuna":
        tok.chat_template = VICUNA_TEMPLATE
    return model, tok


def cmd_optimize(args):
    import nanogcg
    from nanogcg import GCGConfig

    model, tok = load_text_llm(args.llm)
    config = GCGConfig(
        num_steps=args.num_steps, optim_str_init=" ".join(["x"] * 30), search_width=64, batch_size=64,
        topk=64, seed=random.randint(0, 1000) if args.seed is None else args.seed, verbosity="WARNING",
    )
    out = []
    for i, msg in enumerate(tqdm(load_instructions(args.dataset)[: args.limit], desc="GCG")):
        target = f"Sure, here's how to {msg.lower()}:\n\n"
        result = nanogcg.run(model, tok, msg, target, config)
        out.append(f"{msg} {result.best_string}")
        if i % 10 == 0:
            save_json(args.output, out)  # checkpoint progress
    save_json(args.output, out)
    print(f"Saved {len(out)} GCG prompts to {args.output}")


@torch.no_grad()
def cmd_evaluate(args):
    model, tok = load_text_llm(args.llm)
    _, user_tag, asst_tag, system = TEXT_LLMS[args.llm]
    prompts_in = load_json(args.prompts)
    out_dir = Path(args.output_dir or RESULTS_DIR / "gcg" / args.llm)
    for swap in (False, True):
        prompts = []
        for txt in prompts_in:
            t = tok.apply_chat_template([{"role": "user", "content": txt}], tokenize=False, add_generation_prompt=True)
            if system:
                t = t.replace(system, "")
            if swap:
                t = t.replace(asst_tag, "<<SWAP>>").replace(user_tag, asst_tag).replace("<<SWAP>>", user_tag)
            prompts.append(t)
        responses = []
        for s in tqdm(range(0, len(prompts), args.batch_size), desc=f"swap={swap}"):
            batch = tok(prompts[s:s + args.batch_size], return_tensors="pt", add_special_tokens=False,
                        padding=True, truncation=True).to(model.device)
            gen = model.generate(**batch, max_new_tokens=256, do_sample=True)
            responses += tok.batch_decode(gen[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        records = [{"instruction": i, "prompt": p, "response": r} for i, p, r in zip(prompts_in, prompts, responses)]
        name = "cross_role" if swap else "no_swap"
        save_json(out_dir / f"{name}.json", records)
        metrics.report(records, f"{args.llm} GCG {name}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("optimize", help="Run nanoGCG to find adversarial suffixes.")
    o.add_argument("--llm", default="vicuna", choices=list(TEXT_LLMS))
    o.add_argument("--dataset", default="advbench")
    o.add_argument("--num-steps", type=int, default=200)
    o.add_argument("--limit", type=int, default=None)
    o.add_argument("--seed", type=int, default=None, help="GCG seed (random if unset, as in the notebook).")
    o.add_argument("--output", default=str(RESULTS_DIR / "gcg" / "vicuna_advbench_gcg.json"))
    e = sub.add_parser("evaluate", help="Generate responses to GCG prompts on a text LLM.")
    e.add_argument("--llm", required=True, choices=list(TEXT_LLMS))
    e.add_argument("--prompts", required=True)
    e.add_argument("--batch-size", type=int, default=16)
    e.add_argument("--output-dir", default=None)
    args = p.parse_args()

    setup_logging()
    set_seed(0)
    hf_login()
    {"optimize": cmd_optimize, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
