#!/usr/bin/env bash
# Final evaluation with the Claude judge. PRIMARY protocol: PLAIN CHAT (no guardrail)
# on the UNTOUCHED test split — identical for baseline and ES-tuned, so the unsafe%
# reflects intrinsic/internalized safety, not the training-time guardrail prompt.
#
# Usage: scripts/run_all_eval.sh <ES_CHECKPOINT.pth> [LIMIT]
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source safety/.env 2>/dev/null || true; set +a
export HF_TOKEN="${HF_Token:-${HF_TOKEN:-}}"
export ES4SAFETY_ROOT="$(pwd)"
PY=python

CKPT="${1:-}"
LIMIT="${2:-0}"
LIM=""; [[ "$LIMIT" != "0" ]] && LIM="--limit $LIMIT"
VIBE=WeiboAI/VibeThinker-3B
QWEN=Qwen/Qwen2.5-3B-Instruct
run() { echo "### GPU$1 $*"; CUDA_VISIBLE_DEVICES=$1 $PY scripts/gen_and_grade.py "${@:2}" $LIM; }

# ---- PRIMARY: plain-chat safety on the untouched test split ----
run 0 --model "$VIBE" --tag vibe_base --dataset safety_test --template chat &   # baseline (no guardrail)
run 1 --model "$QWEN" --tag qwen_base --dataset safety_test --template chat &   # aligned reference
# reference-only: what the guardrail PROMPT alone buys (not the primary claim)
run 2 --model "$VIBE" --tag vibe_base_guardrail --dataset safety_test --template safety &
wait

if [[ -n "$CKPT" ]]; then
  # ES-tuned, evaluated PLAIN CHAT (did ES internalize safety into the weights?)
  run 0 --model "$VIBE" --checkpoint "$CKPT" --tag vibe_es --dataset safety_test --template chat
  # probes (plain chat): reasoning regression + benign over-refusal
  run 0 --model "$VIBE"                      --tag vibe_base --dataset reasoning_probe --template chat &
  run 1 --model "$VIBE" --checkpoint "$CKPT" --tag vibe_es   --dataset reasoning_probe --template chat &
  run 2 --model "$VIBE"                      --tag vibe_base --dataset benign_probe    --template chat &
  run 3 --model "$VIBE" --checkpoint "$CKPT" --tag vibe_es   --dataset benign_probe    --template chat &
  wait
fi
echo "All summaries in results/*.summary.json"
