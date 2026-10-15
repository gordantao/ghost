# Ghost defense heliocentricity

How much defensive attention does a receiver command? A generative "ghost" defense model predicts how a typical defense would move, given the offense's actual routes and the pre-snap defensive alignment. A receiver's heliocentricity is how much tighter the real defense played them than the ghosts did.

Data: NFL Big Data Bowl 2023 tracking (2021 season, weeks 1–8), all 22 players plus the ball, snap through the throw.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[dev]'
ghost preprocess          # expects data/raw/bdb2023/{plays,players,games,week1..8}.csv
ghost train --fold 0      # repeat for folds 0-4 (or scripts/submit.sh on Longleaf)
ghost score               # -> runs/default/scores/player_scores.csv
```
