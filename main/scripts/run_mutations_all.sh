#!/bin/bash
# Full mutation pipeline: generate the 20 SORRY-Bench mutation styles for the
# models that still lack them, then decode -> judge -> resilience-gap analysis.
# The 7 original models already have mutation responses (resume-safe; skipped).
#
# Phase A  generate 20 styles for the 10 new local models, 8 GPUs, queued.
# Phase B  decode ciphers/translations -> official Mistral judge (all styles,
#          all models) -> resilience-gap analysis + refreshed master table.
#
# 4B/8B use the SAME AWQ int4 checkpoints as their base runs so the resilience
# gap (mutation - base fulfillment) compares like with like.
#
# Launch detached:  setsid nohup bash scripts/run_mutations_all.sh > results/mut_all.log 2>&1 &
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (scripts/ is one level down)
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # make src/ modules importable
PY=.venv/bin/python
mkdir -p results/mutations results/mutations_decoded results/official_mutations

AWQ4="--model-id Qwen/Qwen3-4B-AWQ --dtype float16 --quantization awq_marlin"
AWQ8="--model-id Qwen/Qwen3-8B-AWQ --dtype float16 --quantization awq_marlin"

# 10 local models that need mutation generation (7 originals already done).
GEN=(qwen-1.5b vibethinker-3b
     qwen3-0.6b-nothink qwen3-0.6b-think qwen3-1.7b-nothink qwen3-1.7b-think
     qwen3-4b-nothink qwen3-4b-think qwen3-8b-nothink qwen3-8b-think)
declare -A FLAGS
FLAGS[qwen3-4b-nothink]="$AWQ4"
FLAGS[qwen3-4b-think]="$AWQ4 --kv-cache-dtype fp8"
FLAGS[qwen3-8b-nothink]="$AWQ8"
FLAGS[qwen3-8b-think]="$AWQ8 --kv-cache-dtype fp8 --max-model-len 12000 --max-new-tokens 10000"

# full model set for the downstream judge + analysis (union incl. API + originals)
ALL="opus-4.8,sonnet-5,gpt-5.5,qwen-3b,llama-3.2-3b,deepseek-r1-1.5b,vibethinker-1.5b,qwen-1.5b,vibethinker-3b,qwen3-0.6b-nothink,qwen3-0.6b-think,qwen3-1.7b-nothink,qwen3-1.7b-think,qwen3-4b-nothink,qwen3-4b-think,qwen3-8b-nothink,qwen3-8b-think"

echo "=== Phase A: generate 20 mutation styles, 8-GPU queue  $(date) ==="
NGPU=8
declare -A GPID   # gpu index -> pid of the job running on it
for m in "${GEN[@]}"; do
  # wait for a free GPU
  free=""
  while [ -z "$free" ]; do
    for g in $(seq 0 $((NGPU-1))); do
      p=${GPID[$g]:-}
      if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then free=$g; break; fi
    done
    [ -z "$free" ] && sleep 15
  done
  echo "  [gpu $free] $m  $(date)"
  CUDA_VISIBLE_DEVICES=$free $PY src/run_local_vllm.py --model "$m" --sources ALL_STYLES \
      ${FLAGS[$m]:-} > results/mut_gen_${m}.log 2>&1 &
  GPID[$free]=$!
done
wait
echo "=== Phase A done  $(date) ==="

echo "=== Phase B: decode -> judge -> analyze  $(date) ==="
# 1) decode the 4 cipher + 5 translation styles (NLLB on one GPU)
CUDA_VISIBLE_DEVICES=0 $PY src/decode_mutations.py --models "$ALL"
# 2) official fine-tuned Mistral judge over all 20 styles, all models (TP=4)
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY src/eval_mutations.py --models "$ALL" \
    --judge-path ckpts/sorrybench-judge --tensor-parallel 4 --gpu-mem-util 0.90
# 3) resilience-gap analysis (fulfillment uplift vs base) + refreshed master table
$PY src/mutation_analysis.py
$PY src/final_results.py
echo "=== ALL MUTATIONS + ANALYSIS COMPLETE  $(date) ==="
