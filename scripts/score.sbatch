#!/bin/bash
# Score heliocentricity once every fold has trained. Usually submitted with a dependency:
#   jid=$(sbatch --parsable scripts/train_folds.sbatch)
#   sbatch --dependency=afterok:$jid scripts/score.sbatch
#SBATCH --job-name=ghost_score
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load cuda/12.9
PYTHON="${PYTHON:-$HOME/.conda/envs/ghost/bin/python}"

"$PYTHON" -m ghost.cli score --runs "${RUNS:-runs/default}" "$@"
