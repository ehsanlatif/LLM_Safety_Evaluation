#!/bin/bash
# Temperature / response-diversity vs safety experiment on OLMo-3-7B-Instruct-SFT
# (the safest variant: 3.0% SorryBench at greedy). Fully isolated under
# results/temp_sweep/ so it never touches the main roster CSVs.
#
# Grid (8 configs), grounded in the decoding-safety literature:
#   Arm 1  temperature curve at the standard nucleus threshold top_p=0.9
#          (Holtzman 2019); rising T lifts harmful completions (Huang et al.,
#          ICLR 2024, "Catastrophic Jailbreak via Exploiting Generation").
#   Arm 2  diversity-threshold method comparison at fixed T=1.0:
#          top_k=50 (more exploitable than top-p per the 2024-25 jailbreak-oracle
#          work) and min_p=0.1 (Nguyen et al., ICLR 2025) vs the top_p=0.9 point.
#
# Phases: GEN (8 GPUs) -> GRADE (LLM judge, API) ‖ OFFICIAL (ft-Mistral + LlamaGuard-2)
#         -> ANALYZE. Resume-safe throughout. Launch under tmux:
#   tmux new-session -d -s tsweep -c <repo-root> \
#     'bash scripts/run_temp_sweep.sh 2>&1 | tee results/temp_sweep/run.log; exec bash'
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (scripts/ is one level down)
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # make src/ modules importable
PY=.venv/bin/python
export HF_HUB_DISABLE_XET=1
export HF_TOKEN=$($PY -c "import run_benchmarks as rb; rb.load_env_file(); import os; print(os.environ.get('HF_TOKEN',''))")
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { echo "!!! no HF_TOKEN in .env"; exit 1; }

BASE_MODEL=olmo3-7b-instruct-sft
BASE_HF=allenai/Olmo-3-7B-Instruct-SFT
JUDGE="sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406"
SW=results/temp_sweep
mkdir -p "$SW" "$SW/graded" "$SW/official_sorrybench" "$SW/official_ailuminate" "$SW/analysis"

# --- grid (index-aligned arrays) ------------------------------------------------
RUNS=( olmosft-t00-p90 olmosft-t03-p90 olmosft-t07-p90 olmosft-t10-p90 olmosft-t13-p90 olmosft-t16-p90 olmosft-t10-k50 olmosft-t10-mp10 )
TEMP=( 0.0             0.3             0.7             1.0             1.3             1.6             1.0             1.0             )
TOPP=( 0.9             0.9             0.9             0.9             0.9             0.9             1.0             1.0             )
TOPK=( -1              -1              -1              -1              -1              -1              50              -1              )
MINP=( 0.0             0.0             0.0             0.0             0.0             0.0             0.0             0.1             )
ARM=(  temp_curve      temp_curve      temp_curve      temp_curve      temp_curve      temp_curve      method          method          )
N=${#RUNS[@]}

# --- manifest (read by temperature_analysis.py) --------------------------------
{
  echo "run,temperature,top_p,top_k,min_p,arm"
  for i in $(seq 0 $((N-1))); do
    echo "${RUNS[$i]},${TEMP[$i]},${TOPP[$i]},${TOPK[$i]},${MINP[$i]},${ARM[$i]}"
  done
} > "$SW/manifest.csv"
echo "wrote $SW/manifest.csv ($N configs)"

# --- pre-download the SFT weights ONCE (avoid 8-way concurrent download races) --
echo "########## pre-download $BASE_HF  $(date) ##########"
$PY -c "import os; os.environ['HF_HUB_DISABLE_XET']='1'; from huggingface_hub import snapshot_download; snapshot_download('$BASE_HF')" \
  || { echo '!!! SFT weight download failed'; exit 1; }

# --- Phase 1: generation, one config per GPU -----------------------------------
echo "########## Phase 1: generation ($N configs on GPUs 0-7)  $(date) ##########"
for i in $(seq 0 $((N-1))); do
  g=$((i % 8))
  echo "  [gpu $g] ${RUNS[$i]}  T=${TEMP[$i]} top_p=${TOPP[$i]} top_k=${TOPK[$i]} min_p=${MINP[$i]}"
  CUDA_VISIBLE_DEVICES=$g $PY src/run_local_vllm.py --model "$BASE_MODEL" \
      --out-name "${RUNS[$i]}" --out-root "$SW" --sources sorry-bench,ailuminate \
      --temperature "${TEMP[$i]}" --top-p "${TOPP[$i]}" --top-k "${TOPK[$i]}" \
      --min-p "${MINP[$i]}" --seed 0 --gpu-mem-util 0.90 \
      > "$SW/gen_${RUNS[$i]}.log" 2>&1 &
done
wait
echo "  generation done  $(date)"

RUNS_CSV=$(IFS=,; echo "${RUNS[*]}")

# --- Phase 2: LLM judge (API, no GPU) ‖ official judges (GPU) -------------------
echo "########## Phase 2: grade + official judges  $(date) ##########"
$PY src/grade_responses.py --models "$RUNS_CSV" --datasets ailuminate,sorry-bench \
    --in-dir "$SW" --out-dir "$SW/graded" --workers 16 \
    > "$SW/grade_llmjudge.log" 2>&1 &
PL=$!
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY src/eval_sorrybench.py --models "$RUNS_CSV" \
    --in-dir "$SW" --out-dir "$SW/official_sorrybench" --analysis-dir "$SW/analysis" \
    --judge-path "$JUDGE" --tensor-parallel 4 --gpu-mem-util 0.90 \
    > "$SW/eval_sorrybench.log" 2>&1 &
PS=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 $PY src/eval_ailuminate.py --phase annotate \
    --annotate-models "$RUNS_CSV" --in-dir "$SW" --out-dir "$SW/official_ailuminate" \
    --analysis-dir "$SW/analysis" --batch-size 16 \
    > "$SW/eval_ailuminate.log" 2>&1 &
PA=$!
wait $PL; echo "  [grade] LLM judge exited ($?)"
wait $PS; echo "  [sorrybench] official exited ($?)"
wait $PA; echo "  [ailuminate] annotate exited ($?)"

# --- Phase 3: analysis + plots + report ----------------------------------------
echo "########## Phase 3: analysis  $(date) ##########"
$PY src/temperature_analysis.py
echo "########## TEMPERATURE SWEEP COMPLETE  $(date) ##########"
echo "Deliverables: REPORT_temperature.md, analysis/fig_temp_curve.png, analysis/fig_temp_methods.png"
