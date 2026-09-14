#!/usr/bin/env bash
# GRPO safety fine-tuning of VibeThinker-3B — the RL counterpart to the ES run.
#   * SAME granular Claude-Opus reward (reward/safety_opus.py)
#   * SAME safety guardrail template (es_core.train.apply_safety_template)
#   * SAME in-loop val (datasets/safety_val_small) and (via run_eval_rl.sh) the SAME final eval
#   * 1000-sample balanced train set (800 harmful [base safe/unsafe] + 200 benign)
#
# Usage:
#   RL/run_grpo_safety.sh smoke      # ~4-step pipeline check on 3 GPUs
#   RL/run_grpo_safety.sh full       # 300-iter GRPO run on 8 GPUs
# Env overrides: ITERS, PROMPTS, GEN, LR, BETA, ENGINES, EVAL_FREQ, MAXTOK, WANDB
set -euo pipefail
cd "$(dirname "$0")/.."

set -a; source safety/.env 2>/dev/null || true; set +a
export HF_TOKEN="${HF_Token:-${HF_TOKEN:-}}"
export ES4SAFETY_ROOT="$(pwd)"
PY=python

MODE="${1:-full}"
COMMON=(
  --model WeiboAI/VibeThinker-3B
  --train-dataset RL/data/rl_train_1000
  --eval-dataset datasets/safety_val_small
  --judge-model "${JUDGE_MODEL:-claude-opus-4-8}" --judge-workers 24 --penalize-cot
  --train-temperature 0.7 --train-top-p 0.95
)

if [[ "$MODE" == "smoke" ]]; then
  # tiny, on 3 free GPUs (leaves GPU0 alone); proves generate->reward->update->sync->eval
  CUDA_VISIBLE_DEVICES="${SMOKE_GPUS:-1,2,3}" "$PY" RL/grpo_train.py "${COMMON[@]}" \
    --n-iterations "${ITERS:-4}" --eval-freq "${EVAL_FREQ:-2}" \
    --prompts-per-step "${PROMPTS:-4}" --num-generations "${GEN:-4}" \
    --n-engines "${ENGINES:-2}" --micro-bsz 2 --max-tokens "${MAXTOK:-512}" \
    --lr "${LR:-1e-6}" --beta "${BETA:-0.04}" --logging none
else
  # 300 iterations (user cap): 8 prompts x 8 samples = 64 completions/step, 7 rollout engines
  "$PY" RL/grpo_train.py "${COMMON[@]}" \
    --n-iterations "${ITERS:-300}" --eval-freq "${EVAL_FREQ:-50}" \
    --prompts-per-step "${PROMPTS:-8}" --num-generations "${GEN:-8}" \
    --n-engines "${ENGINES:-7}" --micro-bsz "${MICRO:-2}" --max-tokens "${MAXTOK:-3072}" \
    --lr "${LR:-1e-6}" --beta "${BETA:-0.04}" \
    --logging "${WANDB:-wandb}" --wandb-project "${WANDB_PROJECT:-es-safety-grpo}"
fi
