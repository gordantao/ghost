# Ghost defense heliocentricity

How much defensive attention does a receiver command? A generative "ghost" defense model (a conditional-VAE transformer) predicts how a typical defense would move, given the offense's actual routes, the play context and the pre-snap defensive alignment. A receiver's heliocentricity is how much tighter the real defense played them than the ghosts did. Each play is scored only by the fold model that held it out (cross-fitting).

Data: NFL Big Data Bowl tracking, all 22 players plus the ball, snap through the throw or sack.
- BDB 2023: 2021 season, weeks 1-8, in `data/raw/bdb2023/`
- BDB 2025: 2022 season, weeks 1-9, in `data/raw/bdb2025/`

`data/` is gitignored, so the raw files must be copied in. See `CLAUDE.md` for the architecture and the invariants the tests guard.

## Setup

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest     # ~2 s, synthetic data only
```

The machine running real-data work needs enough memory: train and score on a GPU cluster, not a laptop.

## Pipeline

```bash
ghost preprocess                                   # data/raw/bdb2023 -> data/processed/bdb2023.npz
ghost preprocess --raw data/raw/bdb2023 --raw data/raw/bdb2025 --out data/processed/combined.npz   # both seasons
ghost train --data data/processed/combined.npz --fold 0   # folds 0-4 (--fold all trains on everything)
ghost score                                        # needs every fold -> runs/default/scores/
ghost sample-size                                  # how many ghost samples score needs
ghost animate                                      # random play and its ghost defense -> a GIF
```

- `ghost animate` takes `--play-id N --game-id G`, `--actual-call` and `--seed`.
- `ghost sample-size` takes `--eps 0.05 --quantile 0.9` or `--delta YARDS`.
- Every model and training setting is a flag (`--d-model`, `--obs-frames`, `--epochs`, ...). The learning rate follows a linear warmup then cosine decay: `--lr`, `--lr-schedule {cosine,constant}`, `--warmup-steps`, `--min-lr-frac`.
- `--runs runs/<name>` keeps experiments apart.
- Plays that throw later than `--max-frames` (default 70, a `preprocess` flag) are dropped. Don't re-preprocess between training and scoring.

## Longleaf

Copy `data/raw/bdb2023/` and `data/raw/bdb2025/` over, then:

```bash
conda env create -f environment.yaml && conda activate ghost && pip install -e .
ghost preprocess --raw data/raw/bdb2023 --raw data/raw/bdb2025 --out data/processed/combined.npz
RUNS=runs/exp1 scripts/submit.sh --data data/processed/combined.npz [extra train flags]
RUNS=runs/exp1 sbatch scripts/sample_size.sbatch      # optional, after training
```

`submit.sh` submits a 5-task fold array (`train_folds.sbatch`) and a scoring job that waits for it (`score.sbatch`). Submit from the repo root: the sbatch scripts use `$SLURM_SUBMIT_DIR`.

- `PYTHON=/path/to/bin/python` picks the interpreter. If unset, the first existing of `~/.conda/envs/ghost/bin/python` and `~/.conda/envs/ghost_env/bin/python` is used.
- Each job checks that `import ghost` resolves to this checkout's `src/` and exits if not, so a stale install elsewhere can't silently run old code.
- Python output is unbuffered, so the Slurm logs fill as the job runs.

## Outputs

- `data/processed/*.npz` (+ `.json`): preprocessed cache
- `runs/<name>/fold<k>/{model.pt,metrics.csv}`: per-fold checkpoints and per-epoch metrics
- `runs/<name>/scores/{play_receiver_scores,player_scores}.csv`: heliocentricity per play and receiver, and per player
- `runs/<name>/sample_size/`, `runs/<name>/animations/*.gif`
- `logs/`: Slurm logs
