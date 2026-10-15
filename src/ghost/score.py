"""Heliocentricity scores from cross-fitted ghost-defense models.

For each receiver r on a play, at every frame the model had to predict:

    A_t = distance from r to the nearest actual defender
    E_t = mean over K ghost samples of the distance from r to the nearest ghost defender
    H_t = E_t - A_t

H for the (play, receiver) is the mean of H_t. Positive H means the real
defense played r tighter than a typical defense would have given the same
offensive movement; that extra attention is the receiver's "gravity".

Each play is scored only by the model whose fold held it out, so no score
comes from a model that trained on that play.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import PlayData, split_indices
from .model import future_mask
from .preprocess import DEF, GROUP_VOCAB, OFF, RECEIVER_GROUPS
from .train import load_model, pick_device

RECEIVER_IDS = [GROUP_VOCAB.index(g) for g in RECEIVER_GROUPS]


def nearest_defender_distance(receivers: torch.Tensor, defenders: torch.Tensor) -> torch.Tensor:
    """receivers [..., T, R, 2], defenders [..., T, D, 2] -> [..., T, R]."""
    return (receivers[..., :, None, :] - defenders[..., None, :, :]).norm(dim=-1).min(dim=-1).values


@torch.no_grad()
def heliocentricity(samples: torch.Tensor, batch: dict, obs_frames: int) -> dict[str, torch.Tensor]:
    """Per-(play, offensive player) expected and actual separation, and H.

    samples: [B, K, T, 11, 2] ghost defenders. Returns [B, 11] tensors.
    """
    offense = batch['dyn'][:, :, OFF, :2]
    actual = nearest_defender_distance(offense, batch['dyn'][:, :, DEF, :2])         # [B, T, 11]
    expected = nearest_defender_distance(offense[:, None], samples).mean(dim=1)      # [B, T, 11]
    mask = future_mask(batch['time_mask'], obs_frames)[:, :, None].float()
    n = mask.sum(1)
    return {
        'expected_sep': (expected * mask).sum(1) / n,
        'actual_sep': (actual * mask).sum(1) / n,
        'H': ((expected - actual) * mask).sum(1) / n,
        'n_frames': n.expand(-1, offense.shape[2]),
    }


def score_fold(ckpt_path: Path, data: PlayData, n_samples: int, batch_size: int,
               device: torch.device) -> pd.DataFrame:
    model, ckpt = load_model(ckpt_path, device)
    seed = ckpt['train_config']['seed']
    _, _, held_out = split_indices(data.game_id, ckpt['fold'], ckpt['n_folds'],
                                   ckpt['train_config']['val_frac'], seed)
    torch.manual_seed(seed)

    frames = []
    for i in range(0, len(held_out), batch_size):
        idx = held_out[i:i + batch_size]
        batch = data.batch(torch.from_numpy(idx).to(device))
        result = heliocentricity(model.sample(batch, n_samples), batch, model.cfg.obs_frames)
        group = data.group[torch.from_numpy(idx).to(device)][:, OFF].cpu().numpy()
        cols = {k: v.cpu().numpy() for k, v in result.items()}
        play_rows = pd.DataFrame({
            'game_id': np.repeat(data.game_id[idx], 11),
            'play_id': np.repeat(data.play_id[idx], 11),
            'nfl_id': data.nfl_id[idx][:, OFF].ravel(),
            'group': [GROUP_VOCAB[g] for g in group.ravel()],
            **{k: v.ravel() for k, v in cols.items()},
        })
        frames.append(play_rows[np.isin(group.ravel(), RECEIVER_IDS)])
    out = pd.concat(frames, ignore_index=True)
    out['fold'] = ckpt['fold']
    return out


def aggregate_players(plays: pd.DataFrame, players_csv: Path | None) -> pd.DataFrame:
    stats = plays.groupby('nfl_id').agg(
        group=('group', 'first'), n_plays=('H', 'size'),
        H_mean=('H', 'mean'), H_std=('H', 'std'),
        expected_sep=('expected_sep', 'mean'), actual_sep=('actual_sep', 'mean'),
    )
    stats['H_se'] = stats.H_std / np.sqrt(stats.n_plays)
    if players_csv is not None and players_csv.exists():
        names = pd.read_csv(players_csv, usecols=['nflId', 'displayName']).set_index('nflId').displayName
        stats.insert(0, 'name', names.reindex(stats.index).to_numpy())
    return stats.sort_values('H_mean', ascending=False).reset_index()


def score(data_path: Path, runs_dir: Path, out_dir: Path, n_samples: int = 16,
          batch_size: int = 32, players_csv: Path | None = None, device: str = 'auto') -> None:
    dev = pick_device(device)
    data = PlayData.load(data_path).to(dev)
    ckpts = sorted(runs_dir.glob('fold*/model.pt'))
    if not ckpts:
        raise FileNotFoundError(f'no fold*/model.pt checkpoints under {runs_dir}')

    n_folds = {torch.load(c, map_location='cpu', weights_only=False)['n_folds'] for c in ckpts}
    if len(n_folds) != 1 or len(ckpts) != next(iter(n_folds)):
        print(f'warning: found {len(ckpts)} fold checkpoints for n_folds={n_folds}; '
              'plays in missing folds will not be scored')

    plays = pd.concat([score_fold(c, data, n_samples, batch_size, dev) for c in ckpts], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    plays.to_csv(out_dir / 'play_receiver_scores.csv', index=False)
    aggregate_players(plays, players_csv).to_csv(out_dir / 'player_scores.csv', index=False)
    print(f'scored {plays.play_id.size} play-receiver pairs from {len(ckpts)} folds -> {out_dir}')
