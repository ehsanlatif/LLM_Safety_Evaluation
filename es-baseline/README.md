# SRM Safety Analysis — `es-baseline` branch

**Branch focus: Evolution-Strategies (ES) safety fine-tuning of VibeThinker-3B — the
baseline arm with _no_ Anchored Weight Decay (the null control).**

This branch contains the full [`main`](../../tree/main) benchmark safety audit **plus** the ES
experiment directory [`es/`](es/). The matched ES+AWD arm is on
[`es-awd`](../../tree/es-awd); the GRPO counterpart on [`rl`](../../tree/rl). See
[SRM Safety Analysis on `main`](../../tree/main) for the project overview and branch map.

## What's here

- [`es/`](es/) — the ES experiment: trainer (`es_core/`), safety reward functions
  (`reward/`), eval harness (`safety/`), curated datasets, run launchers (`scripts/`), the
  no-AWD run outputs (`es/experiments/` — `metrics.jsonl` + `eval-output/` only), plain-chat
  eval results (`es/results/`), and the write-ups `es/REPORT.md` (v1) and `es/REPORT_v2.md`
  (v2, canonical). Start at [`es/README.md`](es/README.md).
- Everything from `main` (`src/`, `reports/`, `analysis/`, `paper/`, …).

**Result (see [`es/REPORT_v2.md`](es/REPORT_v2.md)):** at 300 iterations plain ES moved safety
essentially not at all (pooled answer-level 13.6% → 13.0%, p=0.87 vs baseline) and its
`<think>` channel got slightly worse — the control that makes the AWD effect legible.

## Not in git
Model weights (`*.pth`, `*.safetensors`), W&B runs, per-iteration `train-output/`, and the
off-the-shelf base model are excluded (see the repo `.gitignore`). Best-val checkpoints
(iteration 200) are recorded in `es/results/v2_checkpoints.txt`; the ~6 GB weight files live
only on local disk.
