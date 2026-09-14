# SRM Safety Analysis

Research code and results for a two-part study of LLM safety:

1. **A cross-benchmark, cross-evaluator safety _audit_** (this `main` branch) — how safety
   varies with **scale**, **alignment**, and **reasoning** across model families, on
   **AILuminate v1.0** and **SORRY-Bench**, under three independent evaluators, plus a
   same-weights reasoning ablation and a 20-style adversarial-mutation sweep.
2. **Safety _interventions_ on VibeThinker-3B** (the experiment branches) — can we install
   safety into a small reasoning model's weights with **Evolution Strategies (ES)** or
   **RL (GRPO)**, and what makes it work?

## Repository branches

| Branch | Contents | Headline |
|---|---|---|
| [`main`](../../tree/main) | Benchmark safety audit (7+ models, 3 evaluators, mutations) | Alignment sets the safety level; reasoning adds a causal "safety tax"; persuasion beats encoding as a jailbreak. |
| [`es-baseline`](../../tree/es-baseline) | ES (no-AWD) fine-tuning of VibeThinker-3B → `es/` | Plain ES does essentially nothing to safety (the null control). |
| [`es-awd`](../../tree/es-awd) | ES + **Anchored Weight Decay** → `es/` | AWD is the difference-maker: ~50% relative cut in the plain-chat unsafe rate. |
| [`rl`](../../tree/rl) | **GRPO** counterpart → `rl/` | RL with the same reward/eval, for a matched ES-vs-RL comparison. |

Each experiment branch contains the full `main` audit **plus** its experiment directory
(`es/` or `rl/`). Model weights, W&B runs, and per-iteration dumps are never committed.

## Layout (main)

```
src/          all Python (run from repo root; modules import each other flat)
  run_*.py      generation (vLLM / API / mutations) + dataset download
  eval_*.py     the three evaluators (LLM-judge, SORRY-Bench, AILuminate) + mutation judging
  analyze*.py   summary tables + figures; final_results.py merges evaluators
scripts/      shell orchestration (GPU scheduling)
reports/      REPORT*.md write-ups (audit, reasoning toggle, mutations, OLMo, temperature)
analysis/     committed result CSVs + figures
paper/        BayLearn abstract (LaTeX), references, paper figures
data/         benchmark prompt sets      (should be git-ignored; fetch with src/download_datasets.py)
results/      raw responses / judgments  (should be git-ignored; GBs)
```

> Run everything from the repo root (e.g. `python src/analyze.py`, `bash scripts/run_eval_all.sh`).
> Scripts read/write `data/`, `results/`, `analysis/` relative to the current directory and load
> `.env` from the repo root; the shell scripts add `src/` to `PYTHONPATH`.

## Key findings (audit)

- **Alignment sets the safety level; reasoning does not.** Unaligned small reasoning models are
  the least safe; a better-aligned reasoning model is far safer.
- **Reasoning carries a causal "safety tax."** On identical Qwen3 weights, enabling thinking
  raises the unsafe rate at every scale (0.6–8B) under all three evaluators.
- **Persuasion, not encoding, is the jailbreak.** Semantic-persuasion mutations drive small open
  models to 90–99% fulfillment; ciphers do the opposite.

Full write-ups in [`reports/`](reports/).

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # fill in HF_Token / ANTHROPIC_API_KEY / OPENAI_API_KEY
python src/download_datasets.py  # populates data/
```

## Notes

- Model IDs / friendly names live in `src/run_benchmarks.py` / `src/run_local_vllm.py`.
- The AILuminate grade is a faithful open reproduction (LlamaGuard-2 annotator), not the private
  MLCommons ensemble; SORRY-Bench uses its official judge.
- Research code — expect rough edges. Judges require a GPU + vLLM.
