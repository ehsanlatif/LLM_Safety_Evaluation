#!/usr/bin/env bash
# ES safety fine-tuning of VibeThinker-3B — experiment v2:
#   * granular Claude-Opus reward (not binary LlamaGuard)
#   * balanced 30-sample train set (15 harmful uniform safe/unsafe x SORRY/AILuminate + 15 benign)
#   * --teacher-shaping 0 (avoid the length-collapse driver noted in REPORT.md)
#   * Anchored Weight Decay (AWD) on/off, to measure reasoning<->safety with/without AWD
#
# Usage:
#   scripts/run_es_opus_awd.sh smoke [none|awd]   # quick 3-iter pipeline check
#   scripts/run_es_opus_awd.sh full  none          # ES, no AWD
#   scripts/run_es_opus_awd.sh full  awd           # ES + AWD (l2, lambda=10)
set -euo pipefail
cd "$(dirname "$0")/.."

set -a; source safety/.env 2>/dev/null || true; set +a
export HF_TOKEN="${HF_Token:-${HF_TOKEN:-}}"
export ES4SAFETY_ROOT="$(pwd)"
export ES4SAFETY_EXP="$(pwd)/experiments"
PY=python

MODE="${1:-full}"
AWD="${2:-none}"           # none | awd
AWD_ARGS=(--awd-type none --awd-lambda 0.0)
if [[ "$AWD" == "awd" ]]; then AWD_ARGS=(--awd-type l2 --awd-lambda 10.0); fi

# small in-loop val (bounds Opus judge cost during population-eval)
[[ -d datasets/safety_val_small ]] || "$PY" scripts/make_val_small.py 20

COMMON=(
  --model WeiboAI/VibeThinker-3B
  --reward safety --template safety --penalize-cot
  --judge-backend opus --judge-model "${JUDGE_MODEL:-claude-opus-4-8}" --judge-workers 24
  --teacher-shaping 0.0
  --train-dataset datasets/safety_train_30_v2
  --eval-dataset datasets/safety_val_small
  --sigma 0.001 --alpha -1.0 --mu 1
  --train-temperature 0.7 --train-top-p 0.95 --eval-temperature 0.0
  "${AWD_ARGS[@]}"
)

if [[ "$MODE" == "smoke" ]]; then
  "$PY" es_core/train.py "${COMMON[@]}" \
    --population-size 8 --n-iterations 3 --eval-freq 2 \
    --batch-size 6 --mini-batch-size 6 --max-tokens 1024 \
    --n-vllm-engines 4 --logging none
else
  # Env-overridable: ITERS/BATCH/POP/EVAL_FREQ. Defaults raised to 300 iters
  # (next-steps: exit the CI noise floor and give AWD real drift to correct).
  "$PY" es_core/train.py "${COMMON[@]}" \
    --population-size "${POP:-30}" --n-iterations "${ITERS:-300}" --eval-freq "${EVAL_FREQ:-50}" \
    --batch-size "${BATCH:-10}" --mini-batch-size "${BATCH:-10}" --max-tokens 3072 \
    --n-vllm-engines 8 --logging "${WANDB:-wandb}"
fi
