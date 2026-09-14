# SRM Safety Analysis — `rl` branch

**Branch focus: RL (GRPO) safety fine-tuning of VibeThinker-3B — the reinforcement-learning
counterpart to the ES experiment.**

This branch contains the full [`main`](../../tree/main) benchmark safety audit **plus** the RL
experiment directory [`rl/`](rl/). The ES arms are on
[`es-baseline`](../../tree/es-baseline) (no-AWD) and [`es-awd`](../../tree/es-awd) (+AWD). See
[SRM Safety Analysis on `main`](../../tree/main) for the project overview and branch map.

## What's here

- [`rl/`](rl/) — the GRPO experiment: trainer (`grpo_train.py`), dataset builder
  (`build_rl_dataset.py`) + `rl/data/rl_train_1000`, vLLM weight-sync extension, eval/plot
  scripts, run launchers, the GRPO run outputs (`rl/experiments/` — `metrics.jsonl` +
  `plots/` + config only), plain-chat eval results (`rl/results/`), and the write-ups
  `rl/REPORT.md` and `rl/REPRODUCIBILITY.md`. Start at [`rl/README.md`](rl/README.md).
- Everything from `main` (`src/`, `reports/`, `analysis/`, `paper/`, …).

**Design:** GRPO (group-relative policy gradient with a KL-to-reference penalty) holds the
model, Claude-Opus reward, guardrail prompt, and final eval protocol identical to the ES run,
changing **only the optimizer** — a real gradient step on an HF policy vs ES's gradient-free
weight perturbations — so ES and RL are directly comparable. RL uses a larger balanced
1000-row train set (`rl/data/rl_train_1000`) vs the 30-row ES set.

## Not in git
Model weights (`*.safetensors`, `final/`, `checkpoints/`), W&B runs, and tokenizer blobs are
excluded (see the repo `.gitignore`). The GRPO best/final HF checkpoints live only on local
disk under `ES4Safety/RL/experiments/<run>/`.
