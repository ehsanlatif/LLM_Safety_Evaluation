#!/bin/bash
# OLMo 3 7B safety sweep — DECODE + EVALUATION + GRADING, then analysis.
#
# Reuses the existing judges (all resume-safe, so prior models are only re-aggregated,
# not re-judged). The official SorryBench judge is the fine-tuned Mistral-7B in the HF
# cache (ckpts/ no longer exists), sharded TP=4; AILuminate uses LlamaGuard-2-8B.
#
#   Phase A (parallel): unified Claude judge (API, OLMo only) | SorryBench official
#                       (GPU 0-3) | AILuminate annotate (GPU 4-7)
#   Phase B: AILuminate grade (reference SUTs already annotated)
#   Phase C: decode ciphers/translations -> official Mistral judge over 20 styles
#   Phase D: analysis + tables + trajectory figure/report
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (scripts/ is one level down)
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # make src/ modules importable
PY=.venv/bin/python
export HF_HUB_DISABLE_XET=1   # judges were purged for disk; re-download over plain HTTPS
# Both official judges (ft-Mistral, LlamaGuard-2) are GATED. Export the token from
# .env so every child (vLLM/transformers) authenticates instead of hitting a 401.
export HF_TOKEN=$($PY -c "import run_benchmarks as rb; rb.load_env_file(); import os; print(os.environ.get('HF_TOKEN',''))")
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { echo "!!! no HF_TOKEN found in .env — gated judges will 401"; exit 1; }
JUDGE="sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406"   # ft-Mistral judge (auto-fetched)
mkdir -p results/mutations_decoded results/official_mutations

# 12 new OLMo variants; and the union with the 17 prior configs so the aggregate
# CSVs stay complete (resume skips everything already judged/annotated).
OLMO="olmo3-7b-base,olmo3-7b-instruct-sft,olmo3-7b-instruct-dpo,olmo3-7b-instruct,olmo3-7b-think-sft,olmo3-7b-think-dpo,olmo3-7b-think-rl,olmo3-7b-rlzero-math,olmo3-7b-rlzero-code,olmo3-7b-rlzero-if,olmo3-7b-rlzero-general,olmo3-7b-rlzero-mix"
PRIOR="opus-4.8,sonnet-5,gpt-5.5,qwen-3b,qwen-1.5b,llama-3.2-3b,deepseek-r1-1.5b,vibethinker-1.5b,vibethinker-3b,qwen3-0.6b-nothink,qwen3-0.6b-think,qwen3-1.7b-nothink,qwen3-1.7b-think,qwen3-4b-nothink,qwen3-4b-think,qwen3-8b-nothink,qwen3-8b-think"
UNION="${PRIOR},${OLMO}"

echo "=== Phase A: 3 evaluators in parallel  $(date) ==="
# 1) Unified Claude judge — OLMo only (avoid re-billing the 17 prior models). Per-model
#    graded files; analyze.py still picks up the prior models from their existing files.
$PY src/grade_responses.py --models "$OLMO" --datasets ailuminate,sorry-bench --workers 16 \
    > results/olmo_eval_llmjudge.log 2>&1 &
P1=$!
# 2) Official SorryBench — ft-Mistral-7B, GPUs 0-3 (TP=4). Union keeps the overall CSV whole.
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY src/eval_sorrybench.py --models "$UNION" \
    --judge-path "$JUDGE" --tensor-parallel 4 --gpu-mem-util 0.90 \
    > results/olmo_eval_sorrybench.log 2>&1 &
P2=$!
# 3) AILuminate annotate — LlamaGuard-2-8B, GPUs 4-7 (OLMo only; refs already annotated).
CUDA_VISIBLE_DEVICES=4,5,6,7 $PY src/eval_ailuminate.py --phase annotate \
    --annotate-models "$OLMO" --batch-size 16 \
    > results/olmo_eval_ailuminate_annotate.log 2>&1 &
P3=$!
wait $P1; echo "[A1] Claude judge exited ($?)  $(date)"
wait $P2; echo "[A2] SorryBench official exited ($?)  $(date)"
wait $P3; echo "[A3] AILuminate annotate exited ($?)  $(date)"

echo "=== Phase B: AILuminate grade  $(date) ==="
$PY src/eval_ailuminate.py --phase grade --suts "$UNION"

echo "=== Phase C: mutations decode -> official Mistral judge  $(date) ==="
# Decode the 4 cipher + 5 translation styles for OLMo (NLLB on one GPU).
CUDA_VISIBLE_DEVICES=0 $PY src/decode_mutations.py --models "$OLMO"
# Official ft-Mistral judge over all 20 styles; union keeps mutation_fulfillment.csv whole.
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY src/eval_mutations.py --models "$UNION" \
    --judge-path "$JUDGE" --tensor-parallel 4 --gpu-mem-util 0.90

echo "=== Phase D: analysis + tables + trajectory  $(date) ==="
$PY src/analyze.py            # summary_overall.csv + primary answer-level figures
$PY src/final_results.py      # FINAL_master_table.csv (cross-evaluator)
$PY src/mutation_analysis.py  # resilience_gap.csv + mutation heatmap
$PY src/olmo_trajectory.py    # fig_olmo_trajectory.png + REPORT_olmo.md (headline deliverable)
$PY src/olmo_plots.py         # Olmo-ONLY figures: overall / mutation heatmap / resilience / evaluators
echo "=== OLMo EVAL + ANALYSIS COMPLETE  $(date) ==="
