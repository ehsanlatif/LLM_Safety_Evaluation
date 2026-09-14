#!/usr/bin/env bash
# End-to-end experiment v2 driver (sequential — each ES run wants all 8 GPUs):
#   1. ES + Opus reward, NO AWD      (balanced 30-sample data, teacher-shaping 0)
#   2. ES + Opus reward, +AWD (l2,10)
#   3. Final eval of both checkpoints vs baseline: Opus-judge safety on the
#      untouched test split + MATH reasoning probe + benign over-refusal probe.
set -euo pipefail
cd "$(dirname "$0")/.."
# WANDB feeds the trainer's --logging VALUE (must be exactly "wandb" to publish);
# it is NOT wandb's online/offline switch (that is WANDB_MODE, default online).
# Default to publishing to the "es-finetuning" wandb project. Set WANDB=none to disable.
export WANDB="${WANDB:-wandb}"

best_ckpt() {  # $1 = experiment dir -> highest-mean population-eval checkpoint
  python - "$1" <<'PY'
import sys, glob, re, os
d = sys.argv[1]
cands = glob.glob(os.path.join(d, "checkpoints", "*", "pytorch_model.pth"))
def mean(p):
    m = re.search(r"mean([0-9.]+)", os.path.basename(os.path.dirname(p)))
    return float(m.group(1)) if m else -1.0
print(max(cands, key=mean) if cands else "")
PY
}

echo "########## RUN 1/2 : ES + Opus, no AWD ##########"
scripts/run_es_opus_awd.sh full none
NOAWD_DIR=$(ls -dt experiments/es-safety-*opus-noawd-*/ 2>/dev/null | head -1)
NOAWD_CK=$(best_ckpt "$NOAWD_DIR")
echo "no-AWD dir=$NOAWD_DIR ckpt=$NOAWD_CK"

echo "########## RUN 2/2 : ES + Opus, +AWD (l2, 10) ##########"
scripts/run_es_opus_awd.sh full awd
AWD_DIR=$(ls -dt experiments/es-safety-*opus-awdl210-*/ 2>/dev/null | head -1)
AWD_CK=$(best_ckpt "$AWD_DIR")
echo "AWD dir=$AWD_DIR ckpt=$AWD_CK"

echo "########## FINAL EVAL (Opus judge) ##########"
{ echo "noawd_dir=$NOAWD_DIR"; echo "noawd_ckpt=$NOAWD_CK"; echo "awd_dir=$AWD_DIR"; echo "awd_ckpt=$AWD_CK"; } > results/v2_checkpoints.txt
scripts/run_eval_v2.sh "$NOAWD_CK" "$AWD_CK"
echo "########## experiment v2 complete ##########"
