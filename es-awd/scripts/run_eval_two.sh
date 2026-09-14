#!/bin/bash
# Evaluate the two previously-missing models (responses already generated):
#   qwen-1.5b (Qwen2.5-1.5B, non-reasoning) and vibethinker-3b-real (VibeThinker-3B, reasoning)
# Judges fit on a single 80GB GPU, so no tensor-parallel / quantization needed.
# Official aggregations are run over the FULL model union so the summary CSVs stay complete.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (scripts/ is one level down)
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # make src/ modules importable
PY=.venv/bin/python
NEW="qwen-1.5b,vibethinker-3b-real"
ALL="opus-4.8,sonnet-5,gpt-5.5,qwen-3b,vibethinker-3b,deepseek-r1-1.5b,llama-3.2-3b,qwen-1.5b,vibethinker-3b-real,qwen3-0.6b-nothink,qwen3-0.6b-think,qwen3-1.7b-nothink,qwen3-1.7b-think,qwen3-4b-nothink,qwen3-4b-think,qwen3-8b-nothink,qwen3-8b-think"

echo "=== Phase A: 3 evaluators in parallel  $(date) ==="
$PY src/grade_responses.py --models "$NEW" --workers 16 \
    > results/eval_llmjudge_two.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=0 $PY src/eval_sorrybench.py --models "$ALL" \
    --judge-path ckpts/sorrybench-judge --gpu-mem-util 0.90 \
    > results/eval_sorrybench_two.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=1 $PY src/eval_ailuminate.py --phase annotate \
    --annotate-models "$NEW" --batch-size 32 \
    > results/eval_ailuminate_annotate_two.log 2>&1 &
P3=$!
wait $P1; echo "[A1] LLM-judge exited ($?)  $(date)"
wait $P2; echo "[A2] SORRY-Bench official exited ($?)  $(date)"
wait $P3; echo "[A3] AILuminate annotate exited ($?)  $(date)"

echo "=== Phase B: aggregate + figures  $(date) ==="
$PY src/eval_ailuminate.py --phase grade --suts "$ALL"
$PY src/analyze.py          # writes summary_overall.csv (must precede final_results)
$PY src/final_results.py    # reads summary_overall.csv -> master table
$PY src/make_fig_combined.py
echo "=== DONE  $(date) ==="
