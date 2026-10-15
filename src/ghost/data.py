"""Loading the preprocessed cache and splitting it by game."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class PlayData:
    dyn: torch.Tensor        # [P, T, 23, F] standardized dynamics, zero-padded in time
    length: torch.Tensor     # [P] valid frames per play
    group: torch.Tensor      # [P, 23] position-group index (see preprocess.GROUP_VOCAB)
    glob: torch.Tensor       # [P, G] play-context features
    nfl_id: np.ndarray       # [P, 22]
    game_id: np.ndarray      # [P]
    play_id: np.ndarray      # [P]
    meta: dict

    @classmethod
    def load(cls, path: Path) -> 'PlayData':
        arrays = np.load(path)
        return cls(
            dyn=torch.from_numpy(arrays['dyn']),
            length=torch.from_numpy(arrays['length']),
            group=torch.from_numpy(arrays['group']),
            glob=torch.from_numpy(arrays['glob']),
            nfl_id=arrays['nfl_id'], game_id=arrays['game_id'], play_id=arrays['play_id'],
            meta=json.loads(Path(path).with_suffix('.json').read_text()),
        )

    def __len__(self) -> int:
        return len(self.length)

    def to(self, device: torch.device) -> 'PlayData':
        for name in ('dyn', 'length', 'group', 'glob'):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def batch(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """Slice plays, trimming time padding to the longest play in the batch."""
        length = self.length[idx]
        t_max = int(length.max())
        return {
            'dyn': self.dyn[idx, :t_max],
            'time_mask': torch.arange(t_max, device=length.device)[None] < length[:, None],
            'group': self.group[idx],
            'glob': self.glob[idx],
        }


def game_folds(game_id: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign each play a fold so that all plays from a game share a fold."""
    games = np.unique(game_id)
    order = np.random.default_rng(seed).permutation(len(games))
    fold_of_game = dict(zip(games[order], np.arange(len(games)) % n_folds))
    return np.array([fold_of_game[g] for g in game_id])


def split_indices(game_id: np.ndarray, fold: int | None, n_folds: int, val_frac: float,
                  seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(train, val, held-out) play indices.

    Held-out is the plays in `fold` (empty when fold is None, i.e. train on all).
    A `val_frac` share of the remaining games is used for checkpoint selection.
    """
    all_idx = np.arange(len(game_id))
    if fold is None:
        held_out = all_idx[:0]
        rest = all_idx
    else:
        folds = game_folds(game_id, n_folds, seed)
        held_out, rest = all_idx[folds == fold], all_idx[folds != fold]

    rest_games = np.unique(game_id[rest])
    rng = np.random.default_rng(seed + 1)
    n_val = max(1, round(val_frac * len(rest_games)))
    val_games = rng.choice(rest_games, size=n_val, replace=False)
    is_val = np.isin(game_id[rest], val_games)
    return rest[~is_val], rest[is_val], held_out
