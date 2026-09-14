#!/bin/bash
# Evaluate + analyze the Qwen3 reasoning-toggle sweep, using all 8 GPUs.
#
# The two OFFICIAL judges are 7-8B and do NOT fit on a single 8GB card, so each
# is sharded across 4 GPUs (not one-model-per-GPU). The unified LLM-judge is an
# Anthropic API call (no GPU), so it runs concurrently on the CPU.
#
#   Phase A (parallel):
#     - LLM-judge (primary answer-level metric)   API / CPU           no GPU
#     - Official SORRY-Bench (ft-Mistral-7B)      GPUs 0-3, TP=4      vLLM
#     - AILuminate annotate (LlamaGuard-2-8B)     GPUs 4-7, device_map
#   Phase B (after A, CPU only):
#     - AILuminate grade  ->  analyze.py  ->  final_results.py
#
# Resume-safe (every stage skips already-done ids). Launch detached:
#     setsid nohup bash scripts/run_eval_all.sh > results/eval_all.log 2>&1 &
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (scripts/ is one level down)
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # make src/ modules importable
PY=.venv/bin/python
mkdir -p results

QWEN3="qwen3-0.6b-nothink,qwen3-0.6b-think,qwen3-1.7b-nothink,qwen3-1.7b-think,qwen3-4b-nothink,qwen3-4b-think,qwen3-8b-nothink,qwen3-8b-think"
# Union with the existing models so the official tables stay complete (resume
# skips anything already judged/annotated).
ALL="opus-4.8,sonnet-5,gpt-5.5,qwen-3b,vibethinker-3b,deepseek-r1-1.5b,llama-3.2-3b,$QWEN3"

echo "=== Phase A: 3 evaluators in parallel  $(date) ==="

# 1) Unified LLM-judge — PRIMARY answer-level metric. Anthropic API, NO GPU.
$PY src/grade_responses.py --models "$QWEN3" --workers 16 \
    > results/eval_llmjudge_qwen3.log 2>&1 &
P1=$!

# 2) Official SORRY-Bench — ft-Mistral-7B judge, GPUs 0-3 (tensor-parallel 4).
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY src/eval_sorrybench.py --models "$ALL" \
    --judge-path ckpts/sorrybench-judge --tensor-parallel 4 --gpu-mem-util 0.90 \
    > results/eval_sorrybench_official.log 2>&1 &
P2=$!

# 3) AILuminate annotate — LlamaGuard-2-8B sharded across GPUs 4-7.
CUDA_VISIBLE_DEVICES=4,5,6,7 $PY src/eval_ailuminate.py --phase annotate \
    --annotate-models "$QWEN3" --batch-size 16 \
    > results/eval_ailuminate_annotate.log 2>&1 &
P3=$!

wait $P1; echo "[A1] LLM-judge exited ($?)  $(date)"
wait $P2; echo "[A2] SORRY-Bench official exited ($?)  $(date)"
wait $P3; echo "[A3] AILuminate annotate exited ($?)  $(date)"

echo "=== Phase B: CPU aggregation + figures  $(date) ==="
$PY src/eval_ailuminate.py --phase grade --suts "$ALL"   # reference SUTs already annotated
$PY src/analyze.py            # summary_overall.csv + fig1-4 (answer-level, primary)
$PY src/final_results.py      # FINAL_master_table.csv + cross-evaluator figures

echo "=== ALL EVAL + ANALYSIS COMPLETE  $(date) ==="
