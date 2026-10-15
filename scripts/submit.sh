#!/bin/bash
# Submit the full cross-fitting pipeline: 5 fold-training jobs, then scoring.
#   scripts/submit.sh                 (runs/default)
#   RUNS=runs/exp1 scripts/submit.sh --epochs 60
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export RUNS="${RUNS:-runs/default}"
jid=$(sbatch --parsable --export=ALL scripts/train_folds.sbatch "$@")
sbatch --export=ALL --dependency=afterok:"$jid" scripts/score.sbatch
echo "training array $jid -> scoring into $RUNS/scores"
