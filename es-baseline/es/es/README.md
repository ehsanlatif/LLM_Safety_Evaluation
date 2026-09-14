# ES safety fine-tuning of VibeThinker-3B — **ES baseline (no AWD)**

Evolution-Strategies (gradient-free) weight fine-tuning of `WeiboAI/VibeThinker-3B`
toward safety, using a granular Claude-Opus reward. **This branch holds the baseline
arm: plain ES with _no_ Anchored Weight Decay.** The matched ES+AWD arm — the one that
actually moved safety — lives on the `es-awd` branch; the GRPO counterpart on `rl`.

**Headline (see [`REPORT_v2.md`](REPORT_v2.md)):** at 300 iterations plain ES did
essentially nothing to safety (pooled answer-level 13.6% → 13.0%, p=0.87 vs baseline),
and its `<think>` channel got slightly worse — the null control against which AWD is
measured.

## Layout
```
es_core/       ES trainer (train.py), reward-shaping utils, vLLM worker extension
reward/        safety reward functions (safety_opus.py = Opus judge, safety_guard.py = LlamaGuard)
safety/        eval harness (grade_responses.py, run_benchmarks_ref.py)
teacher/       teacher refusal data (qwen_refusals.jsonl)
scripts/       dataset builders, plotting, run launchers
datasets/      curated ES train/val/probe/test sets (safety_train_30_v2, reasoning_probe_200, ...)
experiments/   ES runs — the noawd (opus) runs + early pop30-bs30 and pop8-bs6 smoke runs.
               Each run keeps metrics.jsonl (training curve) + eval-output/ (per-iteration
               test scores). Model weights, W&B, and per-iteration train-output are NOT
               committed (git-ignored; see repo .gitignore).
results/       plain-chat eval outputs: vibe_es_noawd__* and the vibe_base_v2__* reference,
               plus figs/ figs_v2/ and report_v2.html (overall study comparison).
REPORT.md      v1 writeup (100-iter, null result)
REPORT_v2.md   v2 writeup (300-iter, final) — the canonical report
```

## Reproduce (needs a GPU, vLLM, and `safety/.env` with an Anthropic key)
```bash
python scripts/build_safety_datasets_v2.py     # balanced 30-sample train set
ITERS=300 scripts/run_all_v2.sh                # ES no-AWD → ES +AWD → greedy Opus eval
python scripts/plot_v2.py && python scripts/summarize_v2.py
```
Best-val checkpoints (iteration 200) are recorded in `results/v2_checkpoints.txt` — the
weight files themselves are not in git (they are ~6 GB each).
