"""Which receivers actually ran a route, so pass-blockers are not scored as receivers.

RBs and TEs often stay in to block (2022: about 20% of RB plays and 15% of TE
plays, 0.5% of WR plays), and blockers average a much higher H. A blocker is
not a receiver, so `score` keeps them out of the player aggregates.

Two sources, per play:
  label  BDB 2025 `player_play.csv` has `wasRunningRoute`. It is read at score
         time from the raw dirs in the cache's metadata (`sources`), so the
         cache and its fingerprint are untouched. A player missing from, or NaN
         in, a labelled play counts as not running a route.
  rule   BDB 2023 has no label, and neither does a play whose raw dir isn't on
         disk. The rule uses only the cached tracking: a receiver ran a route
         if he ended the window at least 5 yards from where he started, or
         within 2 yards of the line of scrimmage or beyond it (x >= -2; x is
         measured from the ball at the snap). Blockers hold their depth: an RB
         ends about 4.5 yards behind the line, a TE about 3.6.

Calibration against `wasRunningRoute` on the 2022 plays of data/processed/combined.npz
(WR 23,353, RB 9,274, TE 10,369 player-plays), for "ran a route":

    group   accuracy  precision  recall
    all       0.929     0.959     0.963
    WR        0.986     0.996     0.990
    RB        0.834     0.926     0.861
    TE        0.884     0.895     0.978

Non-WRs are the hard case: a checkdown RB barely moves, and an in-line TE who
blocks starts at the line. Other rules (path length, depth gained, either
combined with these) did not beat this one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .preprocess import OFF

ROUTE_MIN_DISPLACEMENT = 5.0   # yards from the snap position to the last frame
ROUTE_MIN_END_X = -2.0         # yards from the line of scrimmage; at or beyond this counts

KEY = ['game_id', 'play_id']


def rule_ran_route(dyn: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
    """[B, 11] bool: did each offensive player run a route, judged from tracking alone.

    dyn [B, T, 23, F] with x, y first; time_mask [B, T] marks the valid frames.
    """
    pos = dyn[:, :, OFF, :2]
    last = (time_mask.sum(1) - 1).clamp(min=0)
    end = pos[torch.arange(len(pos), device=pos.device), last]                       # [B, 11, 2]
    displacement = (end - pos[:, 0]).norm(dim=-1)
    return (displacement >= ROUTE_MIN_DISPLACEMENT) | (end[..., 0] >= ROUTE_MIN_END_X)


def load_route_labels(sources: list[str | Path]) -> pd.DataFrame | None:
    """`wasRunningRoute` rows (game_id, play_id, nfl_id, ran_route) from every raw dir that has them."""
    tables = []
    for src in sources:
        path = Path(src) / 'player_play.csv'
        if not path.exists() or 'wasRunningRoute' not in pd.read_csv(path, nrows=0).columns:
            continue
        t = pd.read_csv(path, usecols=['gameId', 'playId', 'nflId', 'wasRunningRoute'])
        tables.append(pd.DataFrame({'game_id': t.gameId, 'play_id': t.playId, 'nfl_id': t.nflId,
                                    'ran_route': t.wasRunningRoute.fillna(0).astype(float) > 0}))
    if not tables:
        return None
    return pd.concat(tables, ignore_index=True).drop_duplicates(['game_id', 'play_id', 'nfl_id'])


def route_flags(game_id: np.ndarray, play_id: np.ndarray, nfl_id: np.ndarray, rule: np.ndarray,
                labels: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    """(ran_route, route_source) for rows given as parallel arrays; `rule` is the rule's answer.

    A row uses the label when its play appears in `labels` (a player missing
    from that play then counts as not running a route), and the rule otherwise.
    """
    ran = np.asarray(rule, dtype=bool).copy()
    source = np.full(len(ran), 'rule', dtype=object)
    if labels is not None and len(labels):
        rows = pd.DataFrame({'game_id': game_id, 'play_id': play_id, 'nfl_id': nfl_id})
        plays = labels[KEY].drop_duplicates().assign(_labelled=True)
        use = rows.merge(plays, on=KEY, how='left')._labelled.fillna(False).to_numpy(bool)
        label = rows.merge(labels, on=[*KEY, 'nfl_id'], how='left').ran_route.fillna(False).to_numpy(bool)
        ran[use] = label[use]
        source[use] = 'label'
    return ran, source
