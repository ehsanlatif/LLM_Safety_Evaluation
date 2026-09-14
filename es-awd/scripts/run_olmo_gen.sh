#!/bin/bash
# OLMo 3 7B safety sweep — PARALLEL GENERATION across the 8 GPUs.
#
# One vLLM process per model, pinned to one GPU (7B fits on a single A100-80GB, TP=1).
# 12 models over 8 GPUs: 8 start immediately, the remaining 4 start as GPUs free up.
# Each process generates ALL sources in a single load: SorryBench 440 + AILuminate
# 1,200 + the 20 mutation styles (8,800). Resume-safe: already-done ids are skipped.
#
# Think variants (olmo3-7b-think-*) get a 12k-token budget automatically (their
# OLMO_MODELS entry); base is fed verbatim (no chat template). See run_local_vllm.py.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (scripts/ is one level down)
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # make src/ modules importable
PY=.venv/bin/python
mkdir -p results/mutations

MODELS=(olmo3-7b-base
        olmo3-7b-instruct-sft olmo3-7b-instruct-dpo olmo3-7b-instruct
        olmo3-7b-think-sft olmo3-7b-think-dpo olmo3-7b-think-rl
        olmo3-7b-rlzero-math olmo3-7b-rlzero-code olmo3-7b-rlzero-if
        olmo3-7b-rlzero-general olmo3-7b-rlzero-mix)

# Full source list = base benchmarks + the 20 mutation styles (from run_mutations.STYLES).
STYLES=$($PY -c "import run_mutations as rm; print(','.join(rm.STYLES))")
SRC="sorry-bench,ailuminate,${STYLES}"
echo "sources: ${SRC}"

# One model per GPU is throughput-optimal: a single 7B vLLM instance already
# saturates the card (100% GPU-util), so co-locating a 2nd model gives NO speedup
# (they time-slice the compute) and risks the KV-cache OOM below.
#   1/GPU:  SLOTS_PER_GPU=1  GMU=0.90   (recommended)
# If you really want 2/GPU: gpu-mem-util is a ceiling vs the card's TOTAL current
# usage, so the 2nd instance (starting after the 1st holds ~24GB) needs a HIGHER
# ceiling, not a smaller one. Use SLOTS_PER_GPU=2, GMU=0.62, and shrink KV with
# --max-model-len 8192 (edit below). Even then expect ~half speed per model.
NGPU=6
SLOTS_PER_GPU=1
GMU=0.90
echo "=== OLMo generation: ${NGPU} GPUs x ${SLOTS_PER_GPU} slots over ${#MODELS[@]} models (gmu=${GMU})  $(date) ==="
declare -A GPID   # "gpu:slot" -> pid of the job in that slot
for m in "${MODELS[@]}"; do
  free=""
  while [ -z "$free" ]; do
    # slot-major scan: fill one model per GPU (0..5) first, then a second per GPU.
    for s in $(seq 0 $((SLOTS_PER_GPU-1))); do
      for g in $(seq 0 $((NGPU-1))); do
        p=${GPID[$g:$s]:-}
        if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then free=$g; fslot=$s; break 2; fi
      done
    done
    [ -z "$free" ] && sleep 15
  done
  echo "  [gpu $free slot $fslot] $m  $(date)"
  CUDA_VISIBLE_DEVICES=$free $PY src/run_local_vllm.py --model "$m" --sources "$SRC" \
      --max-model-len 16384 --gpu-mem-util $GMU \
      > results/olmo_gen_${m}.log 2>&1 &
  GPID[$free:$fslot]=$!
done
wait
echo "=== OLMo generation complete  $(date) ==="
