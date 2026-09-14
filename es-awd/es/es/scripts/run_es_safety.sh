#!/usr/bin/env bash
# ES safety fine-tuning of VibeThinker-3B. Usage:
#   scripts/run_es_safety.sh            # full run (30 samples, 1000 iters)
#   scripts/run_es_safety.sh smoke      # quick integration smoke (few iters)
set -euo pipefail
cd "$(dirname "$0")/.."

set -a; source safety/.env 2>/dev/null || true; set +a
export HF_TOKEN="${HF_Token:-${HF_TOKEN:-}}"
export ES4SAFETY_ROOT="$(pwd)"
export ES4SAFETY_EXP="$(pwd)/experiments"
PY=python

MODE="${1:-full}"
COMMON=(
  --model WeiboAI/VibeThinker-3B
  --reward safety --template safety
  --reward-model-id meta-llama/Meta-Llama-Guard-2-8B
  --penalize-cot
  --teacher-shaping 0.1 --teacher-file teacher/qwen_refusals.jsonl
  --train-dataset datasets/safety_train_30
  --eval-dataset datasets/safety_val
  --sigma 0.001 --alpha -1.0 --mu 1
  --train-temperature 0.7 --train-top-p 0.95
  --eval-temperature 0.0
)

if [[ "$MODE" == "smoke" ]]; then
  "$PY" es_core/train.py "${COMMON[@]}" \
    --population-size 8 --n-iterations 3 --eval-freq 2 \
    --batch-size 6 --mini-batch-size 6 --max-tokens 1024 \
    --n-vllm-engines 4 --logging none
else
  "$PY" es_core/train.py "${COMMON[@]}" \
    --population-size 30 --n-iterations 1000 --eval-freq 25 \
    --batch-size 30 --mini-batch-size 30 --max-tokens 3072 \
    --n-vllm-engines 7 --logging "${WANDB:-wandb}"
fi
