#!/usr/bin/env bash
# Final eval for experiment v2 — PLAIN CHAT (no guardrail) on the UNTOUCHED test
# split, Claude-OPUS judge, plus the MATH reasoning probe and benign over-refusal
# probe. Compares baseline vs ES(no-AWD) vs ES(+AWD) on BOTH safety and reasoning.
#
# Usage: scripts/run_eval_v2.sh <ES_NOAWD.pth> <ES_AWD.pth> [LIMIT]
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source safety/.env 2>/dev/null || true; set +a
export HF_TOKEN="${HF_Token:-${HF_TOKEN:-}}"
export ES4SAFETY_ROOT="$(pwd)"
PY=python
JUDGE="${JUDGE_MODEL:-claude-opus-4-8}"

# build the 200-sample reasoning probe on first use
[[ -d datasets/reasoning_probe_200 ]] || "$PY" scripts/build_reasoning_probe_200.py

CK_NOAWD="${1:-}"
CK_AWD="${2:-}"
LIMIT="${3:-0}"
LIM=""; [[ "$LIMIT" != "0" ]] && LIM="--limit $LIMIT"
VIBE=WeiboAI/VibeThinker-3B
# Greedy decoding (temperature 0) makes the unsafe% / MATH numbers reproducible
# across re-runs (next-steps: stop the temp-0.6 sampling noise). Override with EVAL_TEMP.
ETEMP="${EVAL_TEMP:-0.0}"; ETOPP="${EVAL_TOPP:-1.0}"
run() { echo "### GPU$1 $*"; CUDA_VISIBLE_DEVICES=$1 $PY scripts/gen_and_grade.py "${@:2}" \
          --judge-model "$JUDGE" --temperature "$ETEMP" --top-p "$ETOPP" $LIM; }

# ---- baseline (plain chat): safety + reasoning + benign ----
run 0 --model "$VIBE" --tag vibe_base_v2 --dataset safety_test    --template chat &
run 1 --model "$VIBE" --tag vibe_base_v2 --dataset reasoning_probe_200 --template chat &
run 2 --model "$VIBE" --tag vibe_base_v2 --dataset benign_probe    --template chat &
wait

# ---- ES (no AWD), plain chat ----
if [[ -n "$CK_NOAWD" ]]; then
  run 0 --model "$VIBE" --checkpoint "$CK_NOAWD" --tag vibe_es_noawd --dataset safety_test    --template chat &
  run 1 --model "$VIBE" --checkpoint "$CK_NOAWD" --tag vibe_es_noawd --dataset reasoning_probe_200 --template chat &
  run 2 --model "$VIBE" --checkpoint "$CK_NOAWD" --tag vibe_es_noawd --dataset benign_probe    --template chat &
  wait
fi

# ---- ES (+AWD), plain chat ----
if [[ -n "$CK_AWD" ]]; then
  run 0 --model "$VIBE" --checkpoint "$CK_AWD" --tag vibe_es_awd --dataset safety_test    --template chat &
  run 1 --model "$VIBE" --checkpoint "$CK_AWD" --tag vibe_es_awd --dataset reasoning_probe_200 --template chat &
  run 2 --model "$VIBE" --checkpoint "$CK_AWD" --tag vibe_es_awd --dataset benign_probe    --template chat &
  wait
fi

echo "=== summaries ==="
for t in vibe_base_v2 vibe_es_noawd vibe_es_awd; do
  for d in safety_test reasoning_probe_200 benign_probe; do
    f="results/${t}__${d}.summary.json"; [[ -f "$f" ]] && echo "$f:" && cat "$f"
  done
done
