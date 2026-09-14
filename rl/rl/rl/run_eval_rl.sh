#!/usr/bin/env bash
# Final eval for the GRPO(RL) run — SAME protocol as the ES eval (scripts/run_eval_v2.sh):
# PLAIN CHAT (no guardrail), untouched test split, Claude-Opus judge, greedy (temp 0),
# plus the MATH-200 reasoning probe and the benign over-refusal probe.
# Compares baseline VibeThinker-3B vs the GRPO model on safety + reasoning + over-refusal.
#
# Usage: RL/run_eval_rl.sh <RL_MODEL_DIR> [LIMIT]
#   RL_MODEL_DIR = RL/experiments/<run>/final  (or a checkpoints/best-* dir)
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source safety/.env 2>/dev/null || true; set +a
export HF_TOKEN="${HF_Token:-${HF_TOKEN:-}}"
export ES4SAFETY_ROOT="$(pwd)"
PY=python
JUDGE="${JUDGE_MODEL:-claude-opus-4-8}"

RL_MODEL="${1:?usage: RL/run_eval_rl.sh <RL_MODEL_DIR> [LIMIT]}"
LIMIT="${2:-0}"; LIM=""; [[ "$LIMIT" != "0" ]] && LIM="--limit $LIMIT"
VIBE=WeiboAI/VibeThinker-3B
ETEMP="${EVAL_TEMP:-0.0}"; ETOPP="${EVAL_TOPP:-1.0}"

[[ -d datasets/reasoning_probe_200 ]] || "$PY" scripts/build_reasoning_probe_200.py

run() { echo "### GPU$1 ${*:2}"; CUDA_VISIBLE_DEVICES=$1 "$PY" scripts/gen_and_grade.py "${@:2}" \
          --judge-model "$JUDGE" --temperature "$ETEMP" --top-p "$ETOPP" $LIM; }

# ---- baseline (plain chat) ----
run 0 --model "$VIBE" --tag vibe_base_v2 --dataset safety_test         --template chat &
run 1 --model "$VIBE" --tag vibe_base_v2 --dataset reasoning_probe_200 --template chat &
run 2 --model "$VIBE" --tag vibe_base_v2 --dataset benign_probe        --template chat &
wait

# ---- GRPO (RL) model, plain chat ----
run 0 --model "$RL_MODEL" --tag vibe_grpo --dataset safety_test         --template chat &
run 1 --model "$RL_MODEL" --tag vibe_grpo --dataset reasoning_probe_200 --template chat &
run 2 --model "$RL_MODEL" --tag vibe_grpo --dataset benign_probe        --template chat &
wait

echo "=== summaries ==="
for t in vibe_base_v2 vibe_grpo; do
  for d in safety_test reasoning_probe_200 benign_probe; do
    f="results/${t}__${d}.summary.json"; [[ -f "$f" ]] && echo "$f:" && cat "$f"
  done
done
