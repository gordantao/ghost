#!/bin/bash
# PIT calibration of the ghost nearest-defender distance, once every fold has trained.
#   mkdir -p logs && RUNS=runs/exp1 sbatch scripts/calibration.sbatch
#   mkdir -p logs && RUNS=runs/exp1 sbatch scripts/calibration.sbatch --plays-per-fold 500
# SLURM opens logs/ before the job starts, so it must already exist (submit.sh makes it).
# Extra arguments are passed through to `ghost calibration`.
#SBATCH --job-name=ghost_calibration
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

"$PYTHON" -m ghost.cli calibration --runs "${RUNS:-runs/default}" "$@"
