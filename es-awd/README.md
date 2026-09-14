# SRM Safety Analysis — `es-awd` branch

**Branch focus: Evolution-Strategies (ES) safety fine-tuning of VibeThinker-3B — the
ES + Anchored Weight Decay (AWD) arm, the configuration that actually works.**

This branch contains the full [`main`](../../tree/main) benchmark safety audit **plus** the ES
experiment directory [`es/`](es/). The matched no-AWD baseline is on
[`es-baseline`](../../tree/es-baseline); the GRPO counterpart on [`rl`](../../tree/rl). See
[SRM Safety Analysis on `main`](../../tree/main) for the project overview and branch map.

## What's here

- [`es/`](es/) — the ES experiment: trainer (`es_core/`), safety reward functions
  (`reward/`), eval harness (`safety/`), curated datasets, run launchers (`scripts/`), the
  ES+AWD run outputs (`es/experiments/`, the `opus-awdl210` runs — `metrics.jsonl` +
  `eval-output/` only), plain-chat eval results (`es/results/`), and the write-ups
  `es/REPORT.md` (v1) and `es/REPORT_v2.md` (v2, canonical). Start at
  [`es/README.md`](es/README.md).
- Everything from `main` (`src/`, `reports/`, `analysis/`, `paper/`, …).

**Result (see [`es/REPORT_v2.md`](es/REPORT_v2.md)): AWD is the difference-maker.** At 300
iterations, ES+AWD (L2 anchor to the frozen t=0 model, λ=10) roughly halved the plain-chat
unsafe rate (pooled answer-level 13.6% → 6.5%, p=0.039 vs baseline; p=0.014 on the
coverage-robust "anywhere" metric vs the matched no-AWD run), with reasoning within noise and
0% over-refusal. AWD curbs the length-collapse/drift plain ES suffers, so the safety it finds
transfers to the untouched test split.

## Not in git
Model weights (`*.pth`, `*.safetensors`), W&B runs, per-iteration `train-output/`, and the
off-the-shelf base model are excluded (see the repo `.gitignore`). Best-val checkpoint
(iteration 200) is recorded in `es/results/v2_checkpoints.txt`; the ~6 GB weight files live
only on local disk.
