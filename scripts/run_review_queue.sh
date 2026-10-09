#!/usr/bin/env bash
# GPU experiments added after the review, run one after another:
#
#   1. same-size random updates scored on GSM8K accuracy (random_update_tasks.py)
#   2. Timber's recommended lambda=0.2 on the full GSM8K test set
#   3. Timber on GPQA-Diamond, its own benchmark, lambda in {0.2, 0.5, 0.8},
#      plus the stable-rank and uniform-scaling controls (skipped if the gated
#      dataset is not accessible yet; rerun the script once it is)
#   4. new fine-tunes: code matched on steps, SmolLM2 second seeds (run_v2.sh)
#   5. subspace overlap between updates (CPU-heavy, needs the runs from 4)
#
# Every step is resumable and skips work already done, so the script can be
# restarted at any point. Waits for any running timber_cutoff.py first.
#
#   bash scripts/run_review_queue.sh
#
# Progress goes to logs/review_queue.log.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec >> logs/review_queue.log 2>&1
export PYTHONUNBUFFERED=1
UV="${UV:-uv}"

step() {
  echo "=== $(date '+%F %T') START: $*"
  if "$UV" run python "$@"; then
    echo "=== $(date '+%F %T') OK: $*"
  else
    echo "=== $(date '+%F %T') FAILED: $*"
    return 1
  fi
}

echo "=== $(date '+%F %T') BEGIN review queue"
while powershell -NoProfile -c "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\").CommandLine" | grep -q "timber_cutoff.py"; do
  sleep 120
done
echo "=== $(date '+%F %T') GPU free"

# 1. damage test
step scripts/random_update_tasks.py || exit 1

# 2. Timber at lambda=0.2, full GSM8K; keep the 300-problem summary separate
step scripts/timber_cutoff.py --limit 1319 --variants erank_l0.2 || exit 1
cp results/timber_cutoff.csv results/timber_cutoff_full.csv
step scripts/timber_cutoff.py --limit 300 --variants instruct

# 3. GPQA-Diamond, Timber's benchmark (needs Hub login and accepted terms)
if "$UV" run python -c "from huggingface_hub import hf_hub_download as d; d('Idavidrein/gpqa','gpqa_diamond.csv',repo_type='dataset')" > /dev/null 2>&1; then
  step scripts/timber_cutoff.py --task gpqa --limit 198 --max-new-tokens 2048 --batch-seqs 16 \
    --variants instruct erank_l0.2 erank_l0.5 erank_l0.8 stable_l0.2 uniform_stable_l0.2
else
  echo "=== $(date '+%F %T') SKIPPED GPQA: dataset not accessible (gated)"
fi

# 4. new fine-tunes (run_v2.sh skips the runs that are already done)
bash scripts/run_v2.sh || echo "=== $(date '+%F %T') FAILED: run_v2.sh (see logs/run_v2.log)"

# 5. do same-shape updates point the same way?
step scripts/update_overlap.py

echo "=== $(date '+%F %T') QUEUE DONE"
