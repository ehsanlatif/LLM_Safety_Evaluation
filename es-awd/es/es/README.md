# ES safety fine-tuning of VibeThinker-3B — **ES + AWD (Anchored Weight Decay)**

Evolution-Strategies (gradient-free) weight fine-tuning of `WeiboAI/VibeThinker-3B`
toward safety, using a granular Claude-Opus reward. **This branch holds the ES+AWD arm:
plain ES plus an L2 anchor (λ=10) that decays the weights toward the frozen t=0 model.**
The matched no-AWD baseline lives on the `es-baseline` branch; the GRPO counterpart on `rl`.

**Headline (see [`REPORT_v2.md`](REPORT_v2.md)): AWD is the difference-maker.** At 300
iterations, ES+AWD roughly halved the plain-chat unsafe rate (pooled answer-level
13.6% → 6.5%, p=0.039 vs baseline; p=0.014 on the coverage-robust "anywhere" metric vs
the matched no-AWD run), with reasoning within noise and 0% over-refusal. AWD curbs the
length-collapse/drift that plain ES suffers, so the safety it finds transfers to the
untouched test split.

## Layout
```
es_core/       ES trainer (train.py), reward-shaping utils, vLLM worker extension
reward/        safety reward functions (safety_opus.py = Opus judge, safety_guard.py = LlamaGuard)
safety/        eval harness (grade_responses.py, run_benchmarks_ref.py)
teacher/       teacher refusal data (qwen_refusals.jsonl)
scripts/       dataset builders, plotting, run launchers
datasets/      curated ES train/val/probe/test sets (safety_train_30_v2, reasoning_probe_200, ...)
experiments/   ES+AWD runs (opus-awdl210): the 100-iter (v1) and 300-iter (v2, canonical) runs.
               Each keeps metrics.jsonl (training curve) + eval-output/ (per-iteration test
               scores). Model weights, W&B, and per-iteration train-output are NOT committed
               (git-ignored; see repo .gitignore).
results/       plain-chat eval outputs: vibe_es_awd__* and the vibe_base_v2__* reference,
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
Best-val checkpoint (iteration 200) is recorded in `results/v2_checkpoints.txt` — the
weight files themselves are not in git (they are ~6 GB each).
