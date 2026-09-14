#!/usr/bin/env bash
# Generation-averaging eval for all 4 models in parallel (1 GPU each), K samples/prompt.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source safety/.env 2>/dev/null || true; set +a
export HF_TOKEN="${HF_Token:-${HF_TOKEN:-}}"; export ES4SAFETY_ROOT="$(pwd)"
export PYTHONUNBUFFERED=1
PY=python
K="${K:-8}"
VIBE=WeiboAI/VibeThinker-3B
CK_NOAWD=$(python3 -c "import json;print(json.load(open('results/vibe_es_noawd__safety_test.summary.json'))['checkpoint'])")
CK_AWD=$(python3 -c "import json;print(json.load(open('results/vibe_es_awd__safety_test.summary.json'))['checkpoint'])")
GRPO=RL/experiments/grpo-safety-G8-B8-lr1e-06-kl0.04-20260821-203749/checkpoints/best-step150-safe0.9062

g() { local gpu=$1 name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu "$PY" RL/gen_average_eval.py --k "$K" "$@" > "RL/genavg_$name.log" 2>&1; }

g 0 base     --model "$VIBE"                          --tag base     &
g 1 es_noawd --model "$VIBE" --checkpoint "$CK_NOAWD" --tag es_noawd &
g 2 es_awd   --model "$VIBE" --checkpoint "$CK_AWD"   --tag es_awd   &
g 3 grpo     --model "$GRPO"                          --tag grpo     &
wait
echo "=== all genavg done ==="
