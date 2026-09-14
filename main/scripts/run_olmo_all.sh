#!/bin/bash
# OLMo 3 7B safety sweep — MASTER DRIVER (idempotent, sequential phases; each phase
# is internally parallel). Launch under tmux:
#   tmux new-session -d -s olmo -c <repo-root> \
#     'bash scripts/run_olmo_all.sh 2>&1 | tee results/olmo_all.log; exec bash'
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root (scripts/ is one level down)
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # make src/ modules importable
PY=.venv/bin/python
mkdir -p results analysis

die() { echo "!!! $1 FAILED  $(date)"; exit 1; }

echo "########## Phase 0: environment  $(date) ##########"
if [ ! -x "$PY" ]; then
  echo "creating .venv"
  python3 -m venv .venv || die "venv"
  $PY -m pip install -U pip wheel || die "pip upgrade"
  $PY -m pip install -r requirements.txt || die "pip install requirements"
fi
# OLMo 3 needs transformers >= 4.57; analysis scripts need matplotlib/numpy.
$PY -m pip install -U "transformers>=4.57.0" matplotlib numpy || die "pip install extras"
$PY - <<'PYEOF' || die "transformers version check"
import transformers
from packaging.version import Version
assert Version(transformers.__version__) >= Version("4.57.0"), transformers.__version__
print("transformers", transformers.__version__, "OK")
PYEOF

echo "########## Phase 1: datasets  $(date) ##########"
if [ ! -f data/sorry-bench/question.jsonl ] || \
   [ ! -f data/ailuminate/airr_official_1.0_demo_en_us_prompt_set_release.csv ]; then
  $PY src/download_datasets.py || die "download_datasets"
else
  echo "datasets already present — skipping download"
fi

echo "########## Phase 2: preflight (load tokenizers/configs)  $(date) ##########"
$PY src/preflight_olmo.py || die "preflight_olmo (a variant is unreachable/gated)"

echo "########## Phase 3: generation (8-GPU queue)  $(date) ##########"
bash scripts/run_olmo_gen.sh || die "run_olmo_gen"

# echo "########## Phase 4: eval + grade + analysis  $(date) ##########"
# bash scripts/run_olmo_eval.sh || die "run_olmo_eval"

# echo "########## ALL DONE  $(date) ##########"
# echo "Deliverables: REPORT_olmo.md, analysis/fig_olmo_trajectory.png, analysis/FINAL_master_table.csv"
