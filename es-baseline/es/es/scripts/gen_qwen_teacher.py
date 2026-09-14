#!/usr/bin/env python3
"""
Generate Qwen2.5-3B-Instruct's (safe) responses to the 30 training prompts.

Qwen2.5-3B-Instruct is the safety-aligned reference in saftyAnalysis/REPORT.md
(~12-15% unsafe vs VibeThinker's ~54%). Its refusals are the "teacher" behavior
we shape VibeThinker toward. Output feeds:
  * a report baseline (what a well-aligned 3B does on these exact prompts), and
  * the refusal-marker vocabulary used by the optional teacher-shaping reward term.

Output: teacher/qwen_refusals.jsonl  (one {id, prompt, response} per line)
"""
import json
from pathlib import Path

from datasets import load_from_disk
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parent.parent
QWEN_ID = "Qwen/Qwen2.5-3B-Instruct"


def main():
    ds = load_from_disk(str(ROOT / "datasets" / "safety_train_30"))
    prompts = list(ds["problem"])
    ids = [m["id"] for m in ds["meta"]]

    llm = LLM(model=QWEN_ID, dtype="bfloat16", gpu_memory_utilization=0.85, enforce_eager=False)
    tok = llm.get_tokenizer()
    chat_prompts = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=512)
    outs = llm.generate(chat_prompts, sp)

    out_path = ROOT / "teacher" / "qwen_refusals.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for i, o in zip(ids, outs):
            fh.write(json.dumps({"id": i, "prompt": prompts[ids.index(i)],
                                 "response": o.outputs[0].text}) + "\n")
    print(f"wrote {len(outs)} teacher responses -> {out_path}")
    # quick peek
    for o in outs[:3]:
        print("  -", o.outputs[0].text[:120].replace("\n", " "))


if __name__ == "__main__":
    main()
