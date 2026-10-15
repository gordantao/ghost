#!/bin/bash
# Train one cross-fitting fold per array task on Longleaf.
#   sbatch scripts/train_folds.sbatch            (folds 0-4, runs/default)
#   RUNS=runs/exp1 sbatch scripts/train_folds.sbatch --epochs 60
# Extra arguments are passed through to `ghost train`.
#SBATCH --job-name=ghost_train
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda/12.1
PYTHON="${PYTHON:-$HOME/.conda/envs/ghost/bin/python}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PYTHON" -m ghost.cli train --fold "$SLURM_ARRAY_TASK_ID" --n-folds 5 \
    --runs "${RUNS:-runs/default}" "$@"
