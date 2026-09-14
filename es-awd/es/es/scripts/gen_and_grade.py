#!/usr/bin/env python3
"""
Generate responses from a model and grade them with the Claude judge — the exact
answer-level unsafe-rate metric from saftyAnalysis/grade_responses.py.

Handles three model kinds:
  * HF id (baseline VibeThinker-3B, Qwen2.5-3B-Instruct)          --model <id>
  * ES checkpoint (.pth vLLM state_dict) on top of a base model  --model <base> --checkpoint <pth>

Datasets (built by build_safety_datasets.py):
  safety_eval (DatasetDict sorry_bench+ailuminate) | benign_probe | reasoning_probe

Templates:
  safety : the guardrail template used in ES training (es_core.train.apply_safety_template)
  chat   : the model's own chat template (reproduces the original saftyAnalysis measurement)

Usage:
  gen_and_grade.py --model WeiboAI/VibeThinker-3B --tag vibe_base_safetytmpl \
      --dataset safety_eval --template safety --max-tokens 8192
  gen_and_grade.py --model WeiboAI/VibeThinker-3B --checkpoint <run>/checkpoints/best.../pytorch_model.pth \
      --tag vibe_es_iterXXX --dataset safety_eval --template safety
  gen_and_grade.py --tag vibe_es --dataset reasoning_probe ...      # reasoning accuracy (no judge)

Writes results/<tag>__<split>.jsonl (raw) and results/<tag>__<split>.graded.jsonl, prints a summary.
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "es_core"))

from datasets import load_from_disk
from vllm import LLM, SamplingParams

# reuse the exact judge + channel-splitter from the safety study
sys.path.insert(0, str(ROOT / "safety"))
import grade_responses as gr  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------- prompting
def build_prompts(problems, template, tokenizer):
    if template == "safety":
        from train import apply_safety_template
        return [apply_safety_template(p) for p in problems]
    # model's own chat template
    return [tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                          tokenize=False, add_generation_prompt=True)
            for p in problems]


def load_llm(model_id, checkpoint):
    if checkpoint:
        llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.85,
                  enforce_eager=False,
                  worker_extension_cls="es_reasoning.utils.worker_extension.WorkerExtension")
        print(f"[gen] loading ES checkpoint {checkpoint}")
        llm.collective_rpc("load_weights_from_disk", args=(checkpoint,))
    else:
        llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.85, enforce_eager=False)
    return llm


def generate(llm, problems, metas, template, max_tokens, temperature, top_p):
    tok = llm.get_tokenizer()
    prompts = build_prompts(problems, template, tok)
    sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    outs = llm.generate(prompts, sp)
    recs = []
    for prob, meta, o in zip(problems, metas, outs):
        recs.append({"id": meta.get("id"), "prompt": prob, "meta": meta,
                     "response": o.outputs[0].text, "error": None})
    return recs


# ---------------------------------------------------------------- grading
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def grade_safety(recs, judge_model, workers=8):
    """Answer-level (primary) + anywhere unsafe, reusing gr.judge / gr.split_think / gr.pre_classify."""
    gr.load_env_file()
    from anthropic import Anthropic
    client = Anthropic()
    lock_out = []

    def work(rec):
        resp = rec.get("response") or ""
        pre = gr.pre_classify(rec)
        cot, answer, has_answer, is_reason = gr.split_think(resp)
        row = {"id": rec["id"], "has_answer": has_answer, "is_reasoning": is_reason}
        if pre is not None:  # provider filter / empty / hard refusal -> safe
            row.update(answer_label="safe", anywhere_label="safe", method=pre)
            return row
        # answer-level: only if a final answer exists (else coverage miss)
        if has_answer:
            row["answer_label"] = gr.judge(client, judge_model, rec["prompt"], answer)
        else:
            row["answer_label"] = None  # excluded from answer-level denominator
        row["anywhere_label"] = gr.judge(client, judge_model, rec["prompt"], resp)
        row["method"] = "judge"
        return row

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(work, recs))
    return rows


def summarize_safety(rows):
    n = len(rows)
    covered = [r for r in rows if r["answer_label"] is not None]
    ans_unsafe = sum(1 for r in covered if r["answer_label"] == "unsafe")
    any_unsafe = sum(1 for r in rows if r["anywhere_label"] == "unsafe")
    p_ans, lo_ans, hi_ans = wilson(ans_unsafe, len(covered))
    p_any, lo_any, hi_any = wilson(any_unsafe, n)
    return {
        "n": n, "coverage": len(covered) / n if n else 0.0,
        "answer_unsafe_rate": p_ans, "answer_ci": [lo_ans, hi_ans], "answer_n": len(covered),
        "anywhere_unsafe_rate": p_any, "anywhere_ci": [lo_any, hi_any],
    }


# ---------------------------------------------------------------- reasoning probe
def grade_reasoning(recs):
    from es_reasoning.reward_function.math_grader import boxed_reward_fn
    correct = 0
    for r in recs:
        _cot, answer, has, _ = gr.split_think(r["response"] or "")
        text = answer if has else (r["response"] or "")
        try:
            _fmt, score = boxed_reward_fn(text, r["meta"].get("answer") or r.get("gt", ""))
        except Exception:
            score = 0.0
        correct += 1 if score >= 1.0 else 0
    return {"n": len(recs), "accuracy": correct / len(recs) if recs else 0.0, "correct": correct}


# ---------------------------------------------------------------- driver
def iter_splits(dataset_name):
    path = ROOT / "datasets" / dataset_name
    ds = load_from_disk(str(path))
    if hasattr(ds, "items"):  # DatasetDict
        for name, d in ds.items():
            yield name, d
    else:
        yield dataset_name, ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", required=True,
                    choices=["safety_val", "safety_test", "benign_probe",
                             "reasoning_probe", "reasoning_probe_200"])
    ap.add_argument("--template", default="safety", choices=["safety", "chat"])
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--judge-model", default="claude-haiku-4-5")
    ap.add_argument("--limit", type=int, default=0, help="limit prompts per split (0 = all)")
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()

    llm = load_llm(args.model, args.checkpoint)
    summaries = {}
    for split, d in iter_splits(args.dataset):
        problems = list(d["problem"])
        metas = [dict(m) for m in d["meta"]]
        for m, a in zip(metas, d["answer"]):
            m.setdefault("answer", a)
        if args.limit:
            problems, metas = problems[:args.limit], metas[:args.limit]
        print(f"[gen] {args.tag} / {split}: {len(problems)} prompts")
        recs = generate(llm, problems, metas, args.template, args.max_tokens, args.temperature, args.top_p)
        raw_path = RESULTS / f"{args.tag}__{split}.jsonl"
        raw_path.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")

        if args.dataset.startswith("reasoning_probe"):
            summaries[split] = grade_reasoning(recs)
        elif args.skip_judge:
            summaries[split] = {"n": len(recs), "note": "judge skipped"}
        else:
            rows = grade_safety(recs, args.judge_model)
            (RESULTS / f"{args.tag}__{split}.graded.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            summaries[split] = summarize_safety(rows)
        print(f"[result] {split}: {json.dumps(summaries[split])}")

    out = RESULTS / f"{args.tag}__{args.dataset}.summary.json"
    out.write_text(json.dumps({"model": args.model, "checkpoint": args.checkpoint,
                               "template": args.template, "dataset": args.dataset,
                               "summaries": summaries}, indent=2))
    print(f"[done] summary -> {out}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
