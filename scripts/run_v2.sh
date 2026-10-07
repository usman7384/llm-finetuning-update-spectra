#!/usr/bin/env bash
# Second round of fine-tunes, with fp32 master weights (see configs/v2_model_*).
# Trains each run, deletes its intermediate checkpoints, then saves the SVD of
# its update. Safe to rerun: finished training runs and existing SVDs are
# skipped, and an interrupted run resumes from its last checkpoint.
#
#   bash scripts/run_v2.sh            # from the repo root
#
# Progress goes to logs/run_v2.log. About 45 GPU hours in total on a 6GB card.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec >> logs/run_v2.log 2>&1
export PYTHONUNBUFFERED=1
UV="${UV:-uv}"

step() {
  echo "=== $(date '+%F %T') START: $*"
  if "$UV" run python "$@"; then
    echo "=== $(date '+%F %T') OK: $*"
  else
    echo "=== $(date '+%F %T') FAILED: $*"
    echo "=== ABORTED"
    exit 1
  fi
}

# name, base checkpoint to measure the update against
run() {
  local name=$1 base=$2
  step scripts/train_sft.py --config "configs/v2_${name}.yaml"
  rm -rf "checkpoints/v2/${name}"/checkpoint-*
  if [ ! -f "results/v2/svd_${name}/matrix_summary.csv" ]; then
    step scripts/svd_deltas.py --before "$base" --after "checkpoints/v2/${name}" \
      --tag "v2_${name}" --out "results/v2/svd_${name}"
  fi
}

echo "=== $(date '+%F %T') BEGIN v2"

# controls read their token budget from the SFT run of the same model, so SFT goes first
run sft checkpoints/base
run control_lm checkpoints/base
run control_code checkpoints/base
run control_shuffled checkpoints/base

run smollm2_sft checkpoints/smollm2_base
run smollm2_control_lm checkpoints/smollm2_base
run smollm2_control_code checkpoints/smollm2_base
run smollm2_control_shuffled checkpoints/smollm2_base

# second seeds last: they only matter once the first round is in
run sft_seed43 checkpoints/base
run control_lm_seed43 checkpoints/base
run control_code_seed43 checkpoints/base

echo "=== ALL DONE"
