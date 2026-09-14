#!/bin/bash
# Autonomous downstream pipeline: decode -> judge mutations -> official base grades
# for the 2 new models -> AILuminate LlamaGuard -> analyses. Resume-safe; detach with
# setsid so it survives session teardown. Uses the single visible GPU.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (scripts/ is one level down)
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # make src/ modules importable
L=results/downstream; mkdir -p $L
PY=.venv/bin/python
GMU=0.40   # gpu_memory_utilization (shared cluster)

echo "[1/7] decode ciphers + NLLB translations  $(date)"
$PY src/decode_mutations.py > $L/1_decode.log 2>&1 || { echo "decode FAILED"; exit 1; }

echo "[2/7] judge 20 mutation styles (fine-tuned Mistral)  $(date)"
$PY src/eval_mutations.py --gpu-mem-util $GMU > $L/2_judge_mutations.log 2>&1 || { echo "judge mut FAILED"; exit 1; }

echo "[3/7] official SORRY-Bench base for all 7 (judges only the 2 new; aggregates 7)  $(date)"
$PY src/eval_sorrybench.py --models opus-4.8,sonnet-5,gpt-5.5,qwen-3b,vibethinker-3b,deepseek-r1-1.5b,llama-3.2-3b \
    --gpu-mem-util $GMU > $L/3_sorry_base.log 2>&1 || { echo "sorry base FAILED"; exit 1; }

echo "[4/7] AILuminate annotate 2 new models (LlamaGuard-2)  $(date)"
$PY src/eval_ailuminate.py --phase annotate --annotate-models deepseek-r1-1.5b,llama-3.2-3b \
    --batch-size 6 > $L/4_ail_annotate.log 2>&1 || { echo "ail annotate FAILED"; exit 1; }

echo "[5/7] AILuminate grade all 7 SUTs  $(date)"
$PY src/eval_ailuminate.py --phase grade > $L/5_ail_grade.log 2>&1 || { echo "ail grade FAILED"; exit 1; }

echo "[6/7] resilience-gap analysis  $(date)"
$PY src/mutation_analysis.py > $L/6_mutation_analysis.log 2>&1 || { echo "mut analysis FAILED"; exit 1; }

echo "[7/7] final combined results  $(date)"
$PY src/final_results.py > $L/7_final_results.log 2>&1 || { echo "final results FAILED"; exit 1; }

echo "DOWNSTREAM_COMPLETE  $(date)"
