#!/usr/bin/env python3
"""
Generate model responses for the 20 SORRY-Bench linguistic mutation styles.

Reuses run_benchmarks.py's backends. Each (model, style) is written to
results/mutations/<model>__<style>.jsonl (resume-safe), one record per prompt,
same schema as run_benchmarks. Reasoning models (VibeThinker, DeepSeek-R1) use a
12k-token budget; others use 1024, matching the base runs.

Styles (20): 6 writing + 5 persuasion + 4 cipher + 5 translation.
Cipher/translation responses are decoded later by decode_mutations.py before
judging (SORRY-Bench evaluates the decoded English response).

Launch per-model in parallel (one GPU each for local models, separate processes
for API models) — see runbook. Example:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python run_mutations.py --models vibethinker-1.5b
    .venv/bin/python run_mutations.py --models gpt-5.5,opus-4.8,sonnet-5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import run_benchmarks as rb

STYLES = [
    # writing
    "slang", "uncommon_dialects", "technical_terms", "role_play", "misspellings", "question",
    # persuasion
    "authority_endorsement", "misrepresentation", "logical_appeal",
    "evidence-based_persuasion", "expert_endorsement",
    # cipher (need decode before judging)
    "ascii", "caesar", "morse", "atbash",
    # translation (need back-translation before judging)
    "translate-fr", "translate-ml", "translate-mr", "translate-ta", "translate-zh-cn",
]
ALL_MODELS = ["opus-4.8", "sonnet-5", "gpt-5.5", "qwen-3b", "vibethinker-1.5b",
              "deepseek-r1-1.5b", "llama-3.2-3b"]


def load_style(data_dir: Path, style: str) -> list[dict]:
    """Parse question_<style>.jsonl robustly.

    Some files (e.g. atbash) contain raw control characters and literal newlines
    inside string values, so line-splitting fractures records. Stream with a
    lenient JSONDecoder (strict=False) over the whole file instead.
    """
    path = data_dir / "sorry-bench" / f"question_{style}.jsonl"
    text = path.read_text(encoding="utf-8", errors="replace")
    dec = json.JSONDecoder(strict=False)
    rows = []
    idx, n = 0, len(text)
    while idx < n:
        while idx < n and text[idx] in " \r\n\t":
            idx += 1
        if idx >= n:
            break
        o, idx = dec.raw_decode(text, idx)
        turns = o.get("turns") or []
        # some records (e.g. in the 'question' style) have a null turn; skip them
        if turns and isinstance(turns[0], str) and turns[0].strip():
            rows.append({"id": str(o["question_id"]), "prompt": turns[0],
                         "meta": {"category": o.get("category"), "style": style}})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--styles", default=",".join(STYLES))
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="./results/mutations")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--openai-model", default="gpt-5.5")
    args = ap.parse_args()

    rb.load_env_file()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]

    reg = rb.build_model_registry(SimpleNamespace(
        openai_model=args.openai_model,
        hf_vibethinker=rb.VIBETHINKER_HF_ID, hf_qwen=rb.QWEN_HF_ID))
    bad = [m for m in models if m not in reg]
    if bad:
        raise SystemExit(f"Unknown models: {bad}")

    # preload prompts per style
    prompts_by_style = {s: load_style(data_dir, s) for s in styles}

    for model in models:
        print(f"\n=== Model: {model} ===")
        try:
            backend = reg[model]()
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {model}: {type(exc).__name__}: {exc}"); continue
        rb.MAX_NEW_TOKENS = 12000 if model in rb.REASONING_MODELS else 1024
        print(f"  max_new_tokens={rb.MAX_NEW_TOKENS}")
        for style in styles:
            rb.run_one(model, backend, style, prompts_by_style[style], out_dir, args.limit)

    print(f"\nDone. Mutation responses under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
