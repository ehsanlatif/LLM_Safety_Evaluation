#!/usr/bin/env python3
"""
Run the AILuminate and SORRY-Bench safety prompt sets against several models
and save each model's raw responses for later grading/analysis.

Models (friendly name -> backend):
    opus-4.8       Anthropic  claude-opus-4-8       (API)
    sonnet-5       Anthropic  claude-sonnet-5       (API)
    gpt-5.5        OpenAI     gpt-5.5               (API)
    vibethinker-1.5b HF local   (see VIBETHINKER_HF_ID)   transformers
    qwen-3b        HF local   (see QWEN_HF_ID)           transformers

This script only COLLECTS responses (does not grade them). Grading against the
AILuminate rubric / SORRY-Bench judge is a separate step.

Datasets are read from ./data (produced by download_datasets.py):
    data/ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv
    data/sorry-bench/question.jsonl

Output: one JSONL file per (model, dataset) under ./results/, one line per
prompt: {id, dataset, model, prompt, response, error, meta}. Runs are
resume-safe — prompt ids already present in the output file are skipped.

Dependencies (install only what you use):
    pip install anthropic            # for the Anthropic models
    pip install openai               # for gpt-5.5
    pip install transformers torch accelerate   # for the HF-local models

Credentials (read from env or the local .env file, same loader as
download_datasets.py):
    ANTHROPIC_API_KEY   for opus-4.8 / sonnet-5
    OPENAI_API_KEY      for gpt-5.5
    HF_TOKEN / HF_Token for gated HF models (if any)

Examples:
    # Everything, all prompts (needs every dependency + key):
    python3 run_benchmarks.py

    # Just the two API Claude models, 25 prompts each, both datasets:
    python3 run_benchmarks.py --models opus-4.8,sonnet-5 --limit 25

    # Only SORRY-Bench against the local Qwen model:
    python3 run_benchmarks.py --models qwen-3b --datasets sorry-bench
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# --- open-source model IDs (edit if you want a different checkpoint) --------
# The prompt names "Vibethinker3b" and "Qwen3b"; these are the closest public
# checkpoints. Override with --hf-vibethinker / --hf-qwen if you have others.
VIBETHINKER_HF_ID = "WeiboAI/VibeThinker-1.5B"
QWEN_HF_ID = "Qwen/Qwen2.5-3B-Instruct"
# Cells that complete the size x reasoning 2x2 for the small-open tier
# (breaks the scale/reasoning confound: reasoning was only observed at 1.5B,
# non-reasoning only at 3B). Both stay in the Qwen2.5 lineage so, on each row,
# reasoning is the only variable.
QWEN_15B_HF_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # 1.5B non-reasoning
# 3B reasoning: the genuine WeiboAI VibeThinker-3B — the same VibeThinker recipe
# as the 1.5B reasoning model already in the study, one size up (emits <think>).
# NOTE: the legacy friendly name "vibethinker-1.5b" actually points to
# VibeThinker-*1.5B* (see VIBETHINKER_HF_ID); THIS is the real 3B checkpoint.
VIBETHINKER_3B_HF_ID = "WeiboAI/VibeThinker-3B"

MAX_NEW_TOKENS = 1024  # overridden by --max-new-tokens

# Qwen3 dense checkpoints. Each is a SINGLE set of weights with a reasoning
# toggle (apply_chat_template(enable_thinking=...)), so thinking-on vs -off is a
# true controlled ablation: reasoning is the ONLY variable, no family/data/quality
# confound. This is the only construct that cleanly isolates reasoning; separate
# reasoning/non-reasoning checkpoints always differ in training data too.
QWEN3_HF_IDS = {
    "0.6b": "Qwen/Qwen3-0.6B",
    "1.7b": "Qwen/Qwen3-1.7B",
    "4b": "Qwen/Qwen3-4B",
    "8b": "Qwen/Qwen3-8B",
}

# Reasoning models emit <think> traces and need a large budget to finish
# reasoning AND produce a final answer. The Qwen3 "-think" variants belong here;
# their "-nothink" twins are the same weights with thinking disabled.
REASONING_MODELS = {"vibethinker-1.5b", "deepseek-r1-1.5b", "vibethinker-3b"} | {
    f"qwen3-{size}-think" for size in QWEN3_HF_IDS
}

# OLMo 3 7B model-flow sweep: trace safety across the post-training stages
# (Base -> SFT -> DPO -> RL) for both the Instruct and Think branches, plus the
# experimental RL-Zero runs. Each entry maps a friendly name to
# (HF id, template mode, default max_new_tokens):
#   "raw"   = base completion checkpoint, NO chat template (feed the prompt verbatim);
#   "chat"  = standard instruct chat template, greedy short budget;
#   "think" = reasoning variant that emits <think>...</think> BY DEFAULT (there is no
#             enable_thinking toggle for OLMo), so it needs a large token budget.
# NOTE the final Think checkpoint is named "...-think-rl", not "...-think", so it does
# not trip the "-think" suffix that toggles enable_thinking in the vLLM path.
OLMO_MODELS = {
    "olmo3-7b-base":           ("allenai/Olmo-3-1025-7B",           "raw",   1024),
    "olmo3-7b-instruct-sft":   ("allenai/Olmo-3-7B-Instruct-SFT",   "chat",  1024),
    "olmo3-7b-instruct-dpo":   ("allenai/Olmo-3-7B-Instruct-DPO",   "chat",  1024),
    "olmo3-7b-instruct":       ("allenai/Olmo-3-7B-Instruct",       "chat",  1024),
    "olmo3-7b-think-sft":      ("allenai/Olmo-3-7B-Think-SFT",       "think", 12000),
    "olmo3-7b-think-dpo":      ("allenai/Olmo-3-7B-Think-DPO",       "think", 12000),
    "olmo3-7b-think-rl":       ("allenai/Olmo-3-7B-Think",           "think", 12000),
    "olmo3-7b-rlzero-math":    ("allenai/Olmo-3-7B-RL-Zero-Math",    "chat",  4096),
    "olmo3-7b-rlzero-code":    ("allenai/Olmo-3-7B-RL-Zero-Code",    "chat",  4096),
    "olmo3-7b-rlzero-if":      ("allenai/Olmo-3-7B-RL-Zero-IF",      "chat",  4096),
    "olmo3-7b-rlzero-general": ("allenai/Olmo-3-7B-RL-Zero-General", "chat",  4096),
    "olmo3-7b-rlzero-mix":     ("allenai/Olmo-3-7B-RL-Zero-Mix",     "chat",  4096),
}
# The three Think variants reason by default -> long budget on the transformers path too.
REASONING_MODELS |= {m for m, (_id, tmpl, _n) in OLMO_MODELS.items() if tmpl == "think"}

# csv fields can hold very long prompts
csv.field_size_limit(10 * 1024 * 1024)


# ---------------------------------------------------------------------------
# Credentials (.env loader, mirrors download_datasets.py)
# ---------------------------------------------------------------------------
def load_env_file() -> None:
    """Populate os.environ from a .env file next to this script (no overwrite)."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export").strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)
    # Normalize HF token spelling (.env uses "HF_Token") so huggingface_hub /
    # transformers / vLLM authenticate instead of hitting anonymous rate limits.
    if not os.environ.get("HF_TOKEN"):
        for k in ("HF_Token", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN",
                  "HUGGINGFACEHUB_API_TOKEN"):
            if os.environ.get(k):
                os.environ["HF_TOKEN"] = os.environ[k]
                break


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_ailuminate(data_dir: Path) -> list[dict]:
    path = data_dir / "ailuminate" / "airr_official_1.0_demo_en_us_prompt_set_release.csv"
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(
                {
                    "id": r["release_prompt_id"],
                    "prompt": r["prompt_text"],
                    "meta": {"hazard": r.get("hazard"), "persona": r.get("persona")},
                }
            )
    return rows


def load_sorrybench(data_dir: Path) -> list[dict]:
    path = data_dir / "sorry-bench" / "question.jsonl"
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            turns = obj.get("turns") or []
            if not turns:
                continue
            rows.append(
                {
                    "id": str(obj["question_id"]),
                    "prompt": turns[0],  # demo set is single-turn
                    "meta": {"category": obj.get("category"), "style": obj.get("prompt_style")},
                }
            )
    return rows


DATASETS = {"ailuminate": load_ailuminate, "sorry-bench": load_sorrybench}


# ---------------------------------------------------------------------------
# Model backends: each exposes .generate(prompt) -> str
# ---------------------------------------------------------------------------
class AnthropicBackend:
    """Anthropic Messages API (claude-opus-4-8, claude-sonnet-5)."""

    def __init__(self, model_id: str):
        import anthropic  # lazy import

        self.model_id = model_id
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY / profile

    def generate(self, prompt: str) -> str:
        # Stream so large max_tokens doesn't hit the SDK's HTTP-timeout guard.
        with self.client.messages.stream(
            model=self.model_id,
            max_tokens=MAX_NEW_TOKENS,
            thinking={"type": "adaptive"},  # recommended on Opus 4.8 / Sonnet 5
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            msg = stream.get_final_message()
        # A safety refusal is a normal, expected outcome for these datasets.
        if msg.stop_reason == "refusal":
            return "[REFUSAL] " + (
                getattr(msg.stop_details, "explanation", "") or "model declined"
            )
        return "".join(b.text for b in msg.content if b.type == "text")


class OpenAIBackend:
    """OpenAI chat completions (gpt-5.5)."""

    def __init__(self, model_id: str):
        from openai import OpenAI  # lazy import

        self.model_id = model_id
        self.client = OpenAI()  # reads OPENAI_API_KEY

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_id,
            max_completion_tokens=MAX_NEW_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


class HFLocalBackend:
    """Local open-weight model via transformers (VibeThinker, Qwen)."""

    def __init__(self, hf_id: str, enable_thinking: bool | None = None):
        import torch  # lazy import
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.hf_id = hf_id
        # None: don't pass the kwarg (default template). True/False: Qwen3-style
        # reasoning toggle on identical weights.
        self.enable_thinking = enable_thinking
        token = os.environ.get("HF_TOKEN") or os.environ.get("HF_Token")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            token=token,
            torch_dtype="auto",
            device_map="auto",
        )
        self.torch = torch

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        # Most instruct checkpoints ship a chat template.
        tmpl_kwargs = {}
        if self.enable_thinking is not None:
            tmpl_kwargs["enable_thinking"] = self.enable_thinking
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **tmpl_kwargs
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


# friendly name -> (backend factory)
def build_model_registry(args) -> dict:
    reg = {
        "opus-4.8": lambda: AnthropicBackend("claude-opus-4-8"),
        "sonnet-5": lambda: AnthropicBackend("claude-sonnet-5"),
        "gpt-5.5": lambda: OpenAIBackend(args.openai_model),
        "vibethinker-1.5b": lambda: HFLocalBackend(args.hf_vibethinker),
        "qwen-3b": lambda: HFLocalBackend(args.hf_qwen),
        # Second model per small-open class (reasoning≠safety robustness check):
        "deepseek-r1-1.5b": lambda: HFLocalBackend("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
        "llama-3.2-3b": lambda: HFLocalBackend("meta-llama/Llama-3.2-3B-Instruct"),
        # Cells that complete the size x reasoning 2x2 for the small-open tier:
        "qwen-1.5b": lambda: HFLocalBackend(args.hf_qwen15),  # 1.5B non-reasoning
        "vibethinker-3b": lambda: HFLocalBackend(args.hf_vibethinker3b),  # 3B reasoning
        # AILuminate reference SUTs (for the self-consistent LlamaGuard-2 grading):
        # the two "accessible" (<15B, open-weight) systems MLCommons composites
        # into the reference standard. Generate their AILuminate responses too.
        "gemma-2-9b": lambda: HFLocalBackend("google/gemma-2-9b-it"),
        "llama-3.1-8b": lambda: HFLocalBackend("meta-llama/Llama-3.1-8B-Instruct"),
        "llama-3-8b": lambda: HFLocalBackend("meta-llama/Meta-Llama-3-8B-Instruct"),
    }
    # Qwen3 reasoning-toggle sweep: each size gives a "-think"/"-nothink" pair on
    # ONE set of weights (thinking on/off is the only difference). Bind hf/flag
    # via default args so each closure captures its own values.
    for size, hf in QWEN3_HF_IDS.items():
        reg[f"qwen3-{size}-think"] = (
            lambda hf=hf: HFLocalBackend(hf, enable_thinking=True))
        reg[f"qwen3-{size}-nothink"] = (
            lambda hf=hf: HFLocalBackend(hf, enable_thinking=False))
    # OLMo 3 7B model-flow sweep. Plain HF checkpoints with a standard chat template
    # (except the "raw" base) and NO enable_thinking toggle, so no thinking kwarg.
    for name, (hf, _tmpl, _n) in OLMO_MODELS.items():
        reg[name] = (lambda hf=hf: HFLocalBackend(hf))
    return reg


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def already_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.is_file():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_one(model_name, backend, dataset_name, prompts, out_dir, limit):
    out_path = out_dir / f"{model_name}__{dataset_name}.jsonl"
    done = already_done(out_path)
    todo = [p for p in prompts if p["id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"  {model_name} x {dataset_name}: {len(todo)} to run "
          f"({len(done)} already done)")

    with out_path.open("a", encoding="utf-8") as fh:
        for i, item in enumerate(todo, 1):
            error, response = None, None
            try:
                response = backend.generate(item["prompt"])
            except Exception as exc:  # noqa: BLE001 - record and continue
                error = f"{type(exc).__name__}: {exc}"
            fh.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "dataset": dataset_name,
                        "model": model_name,
                        "prompt": item["prompt"],
                        "response": response,
                        "error": error,
                        "meta": item["meta"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"    {i}/{len(todo)}", flush=True)
            if error and "RateLimit" in error:
                time.sleep(5)


def main() -> int:
    global MAX_NEW_TOKENS
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="opus-4.8,sonnet-5,gpt-5.5,vibethinker-1.5b,qwen-3b",
                        help="Comma-separated friendly model names to run.")
    parser.add_argument("--datasets", default="ailuminate,sorry-bench",
                        help="Comma-separated dataset names.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default="./results")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max NEW prompts per (model, dataset). 0 = all.")
    parser.add_argument("--openai-model", default="gpt-5.5",
                        help="Exact OpenAI model ID for the gpt-5.5 entry.")
    parser.add_argument("--hf-vibethinker", default=VIBETHINKER_HF_ID)
    parser.add_argument("--hf-qwen", default=QWEN_HF_ID)
    parser.add_argument("--hf-qwen15", default=QWEN_15B_HF_ID,
                        help="HF id for the 1.5B non-reasoning cell.")
    parser.add_argument("--hf-vibethinker3b", default=VIBETHINKER_3B_HF_ID,
                        help="HF id for the 3B reasoning cell (real VibeThinker-3B).")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Generation cap. Raise for reasoning models "
                             "(e.g. 12000 for VibeThinker) so they finish "
                             "reasoning AND produce a final answer.")
    args = parser.parse_args()

    MAX_NEW_TOKENS = args.max_new_tokens  # backends read this global at call time
    print(f"max_new_tokens: {MAX_NEW_TOKENS}")

    load_env_file()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]

    registry = build_model_registry(args)
    bad = [m for m in model_names if m not in registry]
    if bad:
        sys.exit(f"Unknown model(s): {bad}. Choices: {list(registry)}")
    bad_ds = [d for d in dataset_names if d not in DATASETS]
    if bad_ds:
        sys.exit(f"Unknown dataset(s): {bad_ds}. Choices: {list(DATASETS)}")

    # Load prompt sets once.
    loaded = {}
    for d in dataset_names:
        prompts = DATASETS[d](data_dir)
        loaded[d] = prompts
        print(f"Loaded {len(prompts)} prompts from {d}")

    # Instantiate a backend once per model (loading a local model is expensive),
    # then run it over each requested dataset.
    for model_name in model_names:
        print(f"\n=== Model: {model_name} ===")
        try:
            backend = registry[model_name]()
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIP {model_name}: could not initialize backend: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        for d in dataset_names:
            run_one(model_name, backend, d, loaded[d], out_dir, args.limit)

    print(f"\nDone. Responses written under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
