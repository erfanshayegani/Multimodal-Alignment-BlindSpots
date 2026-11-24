#!/usr/bin/env python
"""Judge (instruction, response) pairs with Llama Guard 3 as a second harmfulness metric.

Notebook counterpart (``notebooks/Llama Guard/LlamaGuard.ipynb``). Requires access to the gated
``meta-llama/Llama-Guard-3-8B`` checkpoint (export ``HF_TOKEN``).

python scripts/llama_guard.py results/sft_mix/qwen/harmbench/*.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from blindspots.data import load_instructions  # noqa: E402
from blindspots.utils import hf_login, load_json, save_json, setup_logging  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="Result JSON files with 'response' (and ideally 'instruction') fields.")
    p.add_argument("--dataset", default=None,
                   help="Only needed for old result files without an 'instruction' field: dataset to index by position.")
    p.add_argument("--model-id", default="meta-llama/Llama-Guard-3-8B")
    p.add_argument("--output-dir", default=None, help="Default: <input dir>/llama_guard/")
    args = p.parse_args()

    setup_logging()
    hf_login()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
    fallback = load_instructions(args.dataset) if args.dataset else None

    @torch.no_grad()
    def moderate(question: str, response: str) -> str:
        chat = [{"role": "user", "content": question}, {"role": "assistant", "content": response}]
        ids = tok.apply_chat_template(chat, return_tensors="pt").to(model.device)
        out = model.generate(input_ids=ids, max_new_tokens=100, pad_token_id=0)
        return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)

    for f in args.files:
        records = load_json(f)
        judged, unsafe = [], 0
        for i, r in enumerate(tqdm(records, desc=Path(f).name)):
            question = r.get("original") or r.get("instruction") or (fallback[i] if fallback else None)
            if question is None:
                raise ValueError(f"{f}: records have no 'instruction'; pass --dataset")
            verdict = moderate(question, r["response"]) if r["response"] else "safe"
            unsafe += "unsafe" in verdict
            judged.append({"prompt": question, "response": r["response"], "llama_guard": verdict})
        out_dir = Path(args.output_dir) if args.output_dir else Path(f).parent / "llama_guard"
        save_json(out_dir / Path(f).name, judged)
        print(f"{Path(f).name}: {unsafe}/{len(records)} unsafe ({unsafe / max(len(records), 1):.1%})")


if __name__ == "__main__":
    main()
