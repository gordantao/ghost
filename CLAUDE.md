# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Big Data Bowl project that measures receiver **heliocentricity** (gravity): how much tighter the real defense plays a receiver than a "ghost" defense would. The ghost is a conditional-VAE transformer (`src/ghost/model.py`). It sees the whole play's offense and ball tracking, the play context, and the defense only at the snap. From those it samples where a typical defense would go. It never sees player identities.

Data is BDB 2023 (2021 season, weeks 1–8) in `data/raw/bdb2023/`. Every frame has all 22 players plus the ball, so entities are never padded; only time is. Tracking runs from just before the snap to about the throw, so the model covers route development, not ball-in-air coverage. `data/raw/bdb2026/` holds the BDB 2026 data, which is unused. The pre-rewrite code lives outside the repo in `../big_data_bowl_legacy/`.

## Commands

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[dev]'   # local setup
.venv/bin/python -m pytest                               # all tests (~2 s, synthetic data only)
.venv/bin/python -m pytest tests/test_model.py::test_prior_never_sees_hidden_defense

ghost preprocess                       # data/raw/bdb2023 -> data/processed/bdb2023.npz (+ .json metadata)
ghost train --fold 0                   # one cross-fitting fold -> runs/default/fold0/{model.pt,metrics.csv}
ghost train --fold all                 # train on everything (no held-out plays)
ghost score                            # needs every fold -> runs/default/scores/*.csv
```

Every `ModelConfig` and `TrainConfig` field is also a CLI flag (`--d-model`, `--obs-frames`, `--epochs`, ...). Use `--runs runs/<name>` to keep experiments apart.

**Local memory limit:** the dev machine is a MacBook with 8 GB of RAM total. Never run anything locally that could use more than about 4 GB. That rules out training, benchmarking, or scoring on the real cache (`PlayData.load` alone puts about 260 MB on the device, and a default-size model at batch 16 already overwhelmed the OS). Run real-data training and scoring on Longleaf. Locally, stick to `pytest` (tiny synthetic tensors), `ghost preprocess`, and small read-only checks of the data.

**Longleaf:** create the env from `environment.yaml`, run `pip install -e .`, then `ghost preprocess`, then `scripts/submit.sh [extra train flags]`. That submits a 5-task fold array (`train_folds.sbatch`) and a scoring job that waits for it (`score.sbatch`). Set `RUNS=runs/<name>` to choose the output dir. Logs go to `logs/`.

## Architecture

- **`preprocess.py`**: keeps a play only if every frame from the snap on has exactly 11 offense, 11 defense and the ball. Entity slots are fixed: offense sorted by nflId in 0–10 (`OFF`), defense in 11–21 (`DEF`), ball in 22 (`BALL`). Code everywhere indexes by these slices. Left-moving plays are rotated 180°. Angles use the NGS convention (`vx = s·sin(dir)`, `vy = s·cos(dir)`, checked against frame-to-frame motion). x is measured from the ball's x at the snap, and y from the field center. Positions are mapped to coarse groups (`GROUP_VOCAB`); WR, TE and RB are the receivers that get scored.
- **`data.py`**: `PlayData` loads the whole cache onto the device and builds batches by indexing, with no DataLoader. Splits are always **by game**: `game_folds` assigns folds, and `split_indices` also holds out `val_frac` of the training games for checkpoint selection.
- **`model.py`**: tokens are (frame, entity) pairs. Each `FactorizedBlock` attends over time within an entity, then over entities within a frame. Defender tokens at frames `>= obs_frames` have their features zeroed and a learned "hidden" embedding added. The prior sees only that masked input. The posterior encoder (training only) also sees the true defense. A per-play latent `z` goes to the decoder, which outputs defender positions as offsets from the last observed frame. The loss is a proper negative ELBO: a fixed-σ Gaussian (σ = 1 yard) summed over predicted frames, plus β·KL, with β annealed in `train.py`.
- **`score.py`**: each play is scored only by the fold model that held it out (cross-fitting). For each receiver, at every predicted frame, `H_t = E[nearest ghost defender distance] − nearest actual defender distance`, and H is the mean over frames. Positive H means the real defense played the receiver tighter than expected. Outputs are `play_receiver_scores.csv` and `player_scores.csv` (with names from `players.csv`).

## Invariants the tests guard

`tests/test_model.py` checks that ghost samples don't change when defender tracking after `obs_frames` changes, or when time-padded frames change. Any model change must keep these passing. If you add an input pathway, make sure it can't carry hidden defense information.
