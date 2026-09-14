# RL (GRPO) safety fine-tuning — the RL counterpart to the ES experiment

Same hypothesis as the ES run — *does adding a safety guardrail on top of the
model's reasoning improve safety?* — optimized with **GRPO** (group-relative
policy gradient) instead of Evolution Strategies, holding everything else fixed
so the two are directly comparable:

| Held fixed vs ES | value |
|---|---|
| Model | `WeiboAI/VibeThinker-3B` |
| Reward (in-loop) | `reward/safety_opus.OpusSafetyScorer` — Claude-Opus granular 1–5 → reward∈[0,1] (harmful=fulfillment, benign=helpfulness, penalize-cot blend) |
| Guardrail prompt | `es_core.train.apply_safety_template` / `SAFETY_SYSTEM` |
| In-loop val | `datasets/safety_val_small` (safe-rate = mean reward) |
| Final eval | `datasets/safety_test` + `reasoning_probe_200` + `benign_probe`, plain-chat, Opus judge, greedy — identical to `scripts/run_eval_v2.sh` |
| Rollouts | Ray + vLLM engines (reused from the ES harness) |

**Only the optimizer changes:** GRPO with a KL-to-reference penalty, a real
gradient step on a HF policy model, vs ES's gradient-free weight perturbations.

## Train set — `RL/data/rl_train_1000` (1000 rows)
RL needs more samples than the 30-row ES set, so this is a **balanced 1000**:
- **800 harmful** from SORRY-Bench + AILuminate (disjoint from val/test), split by
  **base-label** (how Opus grades the *base* model's plain-chat answer):
  `unsafe` = base complies (grade ≥ 3), `safe` = base already refuses.
  Target was 400/400, but base-`unsafe` prompts are scarce (~12% base compliance),
  so we take **all available base-unsafe** and top up with base-safe to keep
  harmful = 800 ("base-label, best-effort" — see `build_rl_dataset.py` header and
  `rl_train_1000.audit.jsonl` for the exact counts).
- **200 benign** sampled from `tatsu-lab/alpaca` (disjoint from `benign_probe`).

Each row: `{problem, answer ("safe"|"benign"), kind ("harmful"|"benign"), meta}` —
the same contract the ES trainer read.

Rebuild (one GPU + `safety/.env`):
```bash
CUDA_VISIBLE_DEVICES=0 python RL/build_rl_dataset.py
```

## Train (GRPO, ≤300 iterations)
```bash
RL/run_grpo_safety.sh smoke     # ~4-step pipeline check on 3 GPUs
RL/run_grpo_safety.sh full      # 300-iter run on 8 GPUs (1 learner + 7 vLLM engines)
```
Env overrides: `ITERS PROMPTS GEN LR BETA ENGINES EVAL_FREQ MAXTOK MICRO`.
Outputs land in `RL/experiments/<run>/`: `metrics.jsonl` (plottable with
`scripts/plot_training.py`), `checkpoints/best-*` (best in-loop-val, HF dirs),
and `final/` (last-iteration HF model).

## Evaluate (same protocol as ES)
```bash
RL/run_eval_rl.sh RL/experiments/<run>/final          # baseline vs GRPO, full test
RL/run_eval_rl.sh RL/experiments/<run>/final 40       # quick, 40 prompts/split
```
Writes `results/vibe_grpo__*.summary.json` next to the ES `vibe_es_*` / `vibe_base_v2` files.

## Files
- `build_rl_dataset.py` — builds `data/rl_train_1000` (base-gen + Opus grade + Alpaca benign).
- `grpo_train.py` — GRPO trainer: `Learner` actor (HF policy + frozen ref + AdamW),
  vLLM rollout engines, per-step weight sync, Opus reward, in-loop val.
- `rl_worker_extension.py` — `load_hf_weights`: pushes HF policy weights into the
  vLLM engines (maps HF names → fused qkv/gate_up via vLLM's native `load_weights`).
- `run_grpo_safety.sh` / `run_eval_rl.sh` — launchers.

## GRPO knobs (defaults)
`prompts-per-step 8 · num-generations(G) 8 · mu 1 · lr 1e-6 · beta(KL) 0.04 ·
eps-clip 0.2 · grad-clip 1.0 · micro-bsz 2 · max-tokens 3072 · temp 0.7 · top-p 0.95`.
Advantages are group-normalized per prompt: `A = (r − mean_G) / (std_G + 1e-4)`.
