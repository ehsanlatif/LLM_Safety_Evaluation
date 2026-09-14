#!/usr/bin/env python3
"""
Fast local generation for open-weight models via vLLM (batched / continuous).

Replaces the one-at-a-time transformers path in run_benchmarks for the heavy
local models — essential for the reasoning models (VibeThinker, DeepSeek-R1) at a
12k-token budget over thousands of prompts. Same output schema and file layout as
run_benchmarks / run_mutations, and resume-safe, so downstream (decode, judge,
grade, analyze) is unchanged.

Sources (mix freely via --sources):
  sorry-bench, ailuminate           -> results/<model>__<source>.jsonl  (base)
  <style> (e.g. slang, caesar, ...) -> results/mutations/<model>__<style>.jsonl

Examples:
  # finish DeepSeek base + AILuminate fast
  CUDA_VISIBLE_DEVICES=7 python run_local_vllm.py --model deepseek-r1-1.5b \
      --sources sorry-bench,ailuminate
  # all 20 mutation styles for a reasoning model
  CUDA_VISIBLE_DEVICES=2 python run_local_vllm.py --model vibethinker-1.5b --sources ALL_STYLES
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import run_benchmarks as rb
import run_mutations as rm

HF_ID = {
    "qwen-3b": rb.QWEN_HF_ID,
    "vibethinker-1.5b": rb.VIBETHINKER_HF_ID,
    "deepseek-r1-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    # Cells that complete the size x reasoning 2x2 for the small-open tier:
    "qwen-1.5b": rb.QWEN_15B_HF_ID,               # 1.5B non-reasoning
    "vibethinker-3b": rb.VIBETHINKER_3B_HF_ID,  # 3B reasoning (real VibeThinker-3B)
}
# Qwen3 reasoning-toggle sweep: "-think"/"-nothink" per size are the SAME weights
# with thinking on/off (the only clean way to isolate reasoning).
for _size, _hf in rb.QWEN3_HF_IDS.items():
    HF_ID[f"qwen3-{_size}-think"] = _hf
    HF_ID[f"qwen3-{_size}-nothink"] = _hf
# OLMo 3 7B model-flow sweep (Base/SFT/DPO/RL for Instruct & Think, plus RL-Zero).
# Config table (id, template mode, budget) lives in run_benchmarks.OLMO_MODELS.
for _name, (_hf, _tmpl, _n) in rb.OLMO_MODELS.items():
    HF_ID[_name] = _hf


def thinking_flag(model: str):
    """True/False for Qwen3 toggle variants; None for everything else."""
    # OLMo has no enable_thinking toggle (Think variants reason by default); never
    # pass the kwarg, even though the final Think checkpoint's name contains "think".
    if model.startswith("olmo3-"):
        return None
    if model.endswith("-think"):
        return True
    if model.endswith("-nothink"):
        return False
    return None


def out_path(model, source, out_root):
    if source in ("sorry-bench", "ailuminate"):
        return out_root / f"{model}__{source}.jsonl"
    return out_root / "mutations" / f"{model}__{source}.jsonl"


def load_source(source, data_dir):
    if source == "ailuminate":
        return rb.load_ailuminate(data_dir), "ailuminate"
    if source == "sorry-bench":
        return rm.load_style(data_dir, "question"), "sorry-bench"
    return rm.load_style(data_dir, source), source  # mutation style


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(HF_ID))
    ap.add_argument("--sources", required=True,
                    help="Comma list of sorry-bench,ailuminate,<style>; or ALL_STYLES.")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-root", default="./results")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="Default: 12000 for reasoning models, 1024 otherwise.")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    # Low-VRAM knobs (needed to fit 4B/8B on small GPUs):
    ap.add_argument("--model-id", default=None,
                    help="Override the HF checkpoint (e.g. an AWQ repo) while "
                         "keeping the --model name (and its think/reasoning flags).")
    ap.add_argument("--dtype", default="bfloat16",
                    help="Use float16 for AWQ checkpoints.")
    ap.add_argument("--quantization", default=None,
                    help="e.g. awq_marlin for the *-AWQ checkpoints (else auto).")
    ap.add_argument("--kv-cache-dtype", default="auto",
                    help="fp8 halves KV-cache memory — lets a 12k reasoning "
                         "sequence fit on an 8GB GPU.")
    # Sampling / response-diversity knobs (temperature-vs-safety experiment). Defaults
    # reproduce the main sweep exactly: temperature=0 => greedy, other knobs inert.
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy (default). >0 enables the truncation knobs below.")
    ap.add_argument("--top-p", type=float, default=1.0,
                    help="Nucleus threshold (Holtzman 2019); 1.0 disables.")
    ap.add_argument("--top-k", type=int, default=-1, help="top-k cutoff; -1 disables.")
    ap.add_argument("--min-p", type=float, default=0.0,
                    help="min-p threshold (Nguyen 2025); 0.0 disables.")
    ap.add_argument("--seed", type=int, default=None, help="sampling seed for reproducibility.")
    ap.add_argument("--out-name", default=None,
                    help="Name written to output filenames and the record 'model' field "
                         "while --model still selects weights/template. Lets a sampling "
                         "sweep write one file set per config without touching the registry.")
    args = ap.parse_args()

    rb.load_env_file()
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HF_Token")
    if tok:
        os.environ["HF_TOKEN"] = tok

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    (out_root / "mutations").mkdir(parents=True, exist_ok=True)

    sources = (rm.STYLES if args.sources.strip() == "ALL_STYLES"
               else [s.strip() for s in args.sources.split(",") if s.strip()])
    reasoning = args.model in rb.REASONING_MODELS
    if args.max_new_tokens:
        max_new = args.max_new_tokens
    elif args.model in rb.OLMO_MODELS:
        max_new = rb.OLMO_MODELS[args.model][2]  # per-variant budget (base/chat 1-4k, think 12k)
    else:
        max_new = 12000 if reasoning else 1024

    from vllm import LLM, SamplingParams
    model_id = args.model_id or HF_ID[args.model]
    # OLMo compat: alias the pre-release 'olmo2-retrofit' arch (RL-Zero-Mix) to Olmo3,
    # and force the Olmo3 model class in vLLM (no-op for genuine Olmo3 checkpoints).
    hf_overrides = None
    if args.model in rb.OLMO_MODELS:
        import olmo_compat
        olmo_compat.register_olmo2_retrofit()
        hf_overrides = olmo_compat.vllm_hf_overrides
    llm = LLM(model=model_id, dtype=args.dtype,
              quantization=args.quantization, kv_cache_dtype=args.kv_cache_dtype,
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len, trust_remote_code=True,
              hf_overrides=hf_overrides)
    tokenizer = llm.get_tokenizer()
    out_model = args.out_name or args.model  # separates sweep configs in the output layout
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, min_p=args.min_p, max_tokens=max_new,
                        seed=args.seed)

    think = thinking_flag(args.model)  # None unless a Qwen3 toggle variant
    # OLMo template mode: "raw" (base completion model, no chat template) vs "chat"/"think".
    olmo_tmpl = rb.OLMO_MODELS[args.model][1] if args.model in rb.OLMO_MODELS else None

    def to_prompt(text):
        if olmo_tmpl == "raw":
            return text  # base checkpoint: feed the prompt verbatim (no chat template)
        kwargs = {} if think is None else {"enable_thinking": think}
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False,
                add_generation_prompt=True, **kwargs)
        except Exception:
            # Some OLMo RL-Zero/base checkpoints ship without a chat template.
            if olmo_tmpl is not None:
                return text
            raise

    # Generate + write ONE SOURCE AT A TIME: per-style resume + progress, while
    # keeping vLLM's batched throughput within each source (440-1200 prompts/call).
    for source in sources:
        prompts, _ = load_source(source, data_dir)
        op = out_path(out_model, source, out_root)
        done = set()
        if op.is_file():
            for l in op.read_text(encoding="utf-8").splitlines():
                try: done.add(json.loads(l)["id"])
                except Exception: pass
        todo = [p for p in prompts if p["id"] not in done]
        if not todo:
            print(f"  [{source}] already complete"); continue
        print(f"  [{source}] generating {len(todo)} (max_new_tokens={max_new})", flush=True)
        outs = llm.generate([to_prompt(p["prompt"]) for p in todo], sp)
        with op.open("a", encoding="utf-8") as fh:
            for p, o in zip(todo, outs):
                fh.write(json.dumps({
                    "id": p["id"], "dataset": source, "model": out_model,
                    "prompt": p["prompt"],
                    "response": o.outputs[0].text, "error": None, "meta": p["meta"],
                    "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                                 "top_k": args.top_k, "min_p": args.min_p},
                }, ensure_ascii=False) + "\n")
        print(f"  [{source}] wrote {len(todo)}", flush=True)
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
