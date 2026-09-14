#!/usr/bin/env python3
"""Generation-averaging eval: K samples/prompt (temp>0) per model, judged once each.

Attacks the real noise source (generation nondeterminism), not the judge (which K=5 proved
is ~deterministic). For one model, generates K samples/prompt on safety_test at temperature
0.6, judges every sample once (answer channel + anywhere), and saves per-generation records
to results/genavg/<tag>.jsonl for the paired analysis (RL/analyze_genavg.py).

Usage:
  gen_average_eval.py --model <id|dir> [--checkpoint <pth>] --tag <tag> [--k 8]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in ("", "es_core", "safety", "scripts"):
    sys.path.insert(0, str(ROOT / _p) if _p else str(ROOT))

from datasets import load_from_disk
from vllm import LLM, SamplingParams
import grade_responses as gr
from anthropic import Anthropic

RES = ROOT / "results" / "genavg"; RES.mkdir(parents=True, exist_ok=True)


def load_llm(model_id, checkpoint, gpu_mem):
    """ES-checkpoint-aware loader with configurable memory (checkpoint needs headroom
    to load its 5.8 GB state_dict on top of vLLM's reservation)."""
    if checkpoint:
        llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=gpu_mem,
                  enforce_eager=False,
                  worker_extension_cls="es_reasoning.utils.worker_extension.WorkerExtension")
        print(f"[gen] loading ES checkpoint {checkpoint}")
        llm.collective_rpc("load_weights_from_disk", args=(checkpoint,))
    else:
        llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=gpu_mem, enforce_eager=False)
    return llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--judge-model", default="claude-opus-4-8")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    # prompts (plain chat, same as the main eval)
    ds = load_from_disk(str(ROOT / "datasets" / "safety_test"))
    prompts, metas, splits = [], [], []
    for split, d in ds.items():
        for prob, m in zip(d["problem"], d["meta"]):
            prompts.append(prob); metas.append(dict(m)); splits.append(split)

    llm = load_llm(args.model, args.checkpoint, args.gpu_mem)
    tok = llm.get_tokenizer()
    templated = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True) for p in prompts]
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    outs = llm.generate(templated, sp)

    # flatten to one record per (prompt, sample)
    gens = []
    for prob, meta, split, o in zip(prompts, metas, splits, outs):
        for ki, comp in enumerate(o.outputs):
            gens.append({"id": meta.get("id"), "split": split, "prompt": prob, "meta": meta,
                         "k": ki, "response": comp.text, "error": None})
    print(f"[{args.tag}] {len(prompts)} prompts x k={args.k} = {len(gens)} generations; judging (once each) ...")

    gr.load_env_file()
    client = Anthropic()

    def work(rec):
        resp = rec["response"] or ""
        pre = gr.pre_classify(rec)
        cot, answer, has_answer, is_reason = gr.split_think(resp)
        rec["has_answer"] = has_answer
        if pre is not None:                      # provider filter / empty / hard refusal -> safe
            rec["answer_label"] = "safe" if has_answer else None
            rec["anywhere_label"] = "safe"; rec["method"] = pre
            return rec
        rec["answer_label"] = gr.judge(client, args.judge_model, rec["prompt"], answer) if has_answer else None
        rec["anywhere_label"] = gr.judge(client, args.judge_model, rec["prompt"], resp)
        rec["method"] = "judge"
        return rec

    with ThreadPoolExecutor(max_workers=24) as ex:
        gens = list(ex.map(work, gens))

    out = RES / f"{args.tag}.jsonl"
    out.write_text("\n".join(json.dumps({k: g[k] for k in
                   ("id", "split", "k", "has_answer", "answer_label", "anywhere_label", "method")})
                   for g in gens), encoding="utf-8")
    # quick per-model rate
    ans = [g for g in gens if g["answer_label"] is not None]
    au = sum(g["anywhere_label"] == "unsafe" for g in gens)
    ak = sum(g["answer_label"] == "unsafe" for g in ans)
    print(f"[{args.tag}] answer-level {ak}/{len(ans)}={ak/max(len(ans),1)*100:.1f}%  "
          f"anywhere {au}/{len(gens)}={au/len(gens)*100:.1f}%  -> {out}")


if __name__ == "__main__":
    main()
