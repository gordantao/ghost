"""Heliocentricity scores from cross-fitted ghost-defense models.

For each receiver r on a play, at every frame the model had to predict:

    A_t = distance from r to the nearest actual defender
    E_t = mean over K ghost samples of the distance from r to the nearest ghost defender
    H_t = E_t - A_t

H for the (play, receiver) is the mean of H_t. Positive H means the real
defense played r tighter than a typical defense would have given the same
offensive movement; that extra attention is the receiver's "gravity".

Two ghosts are scored. For H the ghost draws its coverage call from the model's
prior, so H is total gravity, including any that shows up in the call itself.
For H_within the ghost runs the defense's actual call, so H_within is gravity
within the call, and H_call = H - H_within is the part that acts through the
play call.

Each play is scored only by the model whose fold held it out, so no score
comes from a model that trained on that play.
"""
from __future__ import annotations

import os
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
    coverage_vocab = data.meta['coverage_vocab']

    frames = []
    for i in range(0, len(held_out), batch_size):
        idx = held_out[i:i + batch_size]
        batch = data.batch(torch.from_numpy(idx).to(device))
        # Same z noise for both ghosts, so H_call reflects the call, not Monte Carlo noise.
        batch_seed = int(torch.randint(2 ** 62, ()))
        gen = lambda: torch.Generator(device).manual_seed(batch_seed)  # noqa: E731
        total = heliocentricity(model.sample(batch, n_samples, generator=gen()), batch, model.cfg.obs_frames)
        within = heliocentricity(model.sample(batch, n_samples, coverage=batch['coverage'], generator=gen()),
                                 batch, model.cfg.obs_frames)
        group = batch['group'][:, OFF].cpu().numpy()
        cols = {k: v.cpu().numpy() for k, v in total.items()}
        cols['expected_sep_within'] = within['expected_sep'].cpu().numpy()
        cols['H_within'] = within['H'].cpu().numpy()
        cols['H_call'] = cols['H'] - cols['H_within']
        play_rows = pd.DataFrame({
            'game_id': np.repeat(data.game_id[idx], 11),
            'play_id': np.repeat(data.play_id[idx], 11),
            'nfl_id': data.nfl_id[idx][:, OFF].ravel(),
            'group': [GROUP_VOCAB[g] for g in group.ravel()],
            'coverage': np.repeat([coverage_vocab[c] for c in batch['coverage'].cpu().numpy()], 11),
            **{k: v.ravel() for k, v in cols.items()},
        })
        frames.append(play_rows[np.isin(group.ravel(), RECEIVER_IDS)])
    out = pd.concat(frames, ignore_index=True)
    out['fold'] = ckpt['fold']
    return out


def player_names(players_csvs: list[Path] | None) -> pd.Series:
    """nflId -> display name from every players.csv that exists (seasons can each add players)."""
    tables = [pd.read_csv(p, usecols=['nflId', 'displayName']) for p in players_csvs or [] if p.exists()]
    if not tables:
        return pd.Series(dtype=str)
    return pd.concat(tables).drop_duplicates('nflId').set_index('nflId').displayName


def aggregate_players(plays: pd.DataFrame, players_csvs: list[Path] | None) -> pd.DataFrame:
    stats = plays.groupby('nfl_id').agg(
        group=('group', 'first'), n_plays=('H', 'size'),
        H_mean=('H', 'mean'), H_std=('H', 'std'),
        H_within_mean=('H_within', 'mean'), H_within_std=('H_within', 'std'),
        H_call_mean=('H_call', 'mean'), H_call_std=('H_call', 'std'),
        expected_sep=('expected_sep', 'mean'), expected_sep_within=('expected_sep_within', 'mean'),
        actual_sep=('actual_sep', 'mean'),
    )
    for h in ('H', 'H_within', 'H_call'):
        stats[f'{h}_se'] = stats[f'{h}_std'] / np.sqrt(stats.n_plays)
    names = player_names(players_csvs)
    if len(names):
        stats.insert(0, 'name', names.reindex(stats.index).to_numpy())
    return stats.sort_values('H_mean', ascending=False).reset_index()


def check_checkpoints(ckpts: list[Path]) -> tuple[Path, str]:
    """Every fold must come from one cross-fitting run; return its cache path and fingerprint.

    Folds with different n_folds, seed, val_frac or cache would hold out
    overlapping or incomplete sets of plays.
    """
    metas = []
    for c in ckpts:
        ck = torch.load(c, map_location='cpu', weights_only=False)
        metas.append({'fold': ck['fold'], 'n_folds': ck['n_folds'], 'seed': ck['train_config']['seed'],
                      'val_frac': ck['train_config']['val_frac'],
                      'data_path': os.path.normpath(ck['data_path']),
                      'data_fingerprint': ck['data_fingerprint']})
    for key in ('n_folds', 'seed', 'val_frac', 'data_path', 'data_fingerprint'):
        values = {m[key] for m in metas}
        if len(values) != 1:
            raise ValueError(f'fold checkpoints disagree on {key}: '
                             + ', '.join(f'{c.parent.name}={m[key]}' for c, m in zip(ckpts, metas)))
    n_folds = metas[0]['n_folds']
    folds = sorted(m['fold'] for m in metas)
    if folds != list(range(n_folds)):
        raise ValueError(f'expected one checkpoint for each of folds 0..{n_folds - 1}, found {folds}')
    return Path(metas[0]['data_path']), metas[0]['data_fingerprint']


def score(data_path: Path | None, runs_dir: Path, out_dir: Path, n_samples: int = 16,
          batch_size: int = 32, players_csvs: list[Path] | None = None, device: str = 'auto') -> None:
    ckpts = sorted(runs_dir.glob('fold*/model.pt'))
    if not ckpts:
        raise FileNotFoundError(f'no fold*/model.pt checkpoints under {runs_dir}')
    trained_on, fingerprint = check_checkpoints(ckpts)
    if data_path is None:
        data_path = trained_on
    elif data_path.resolve() != trained_on.resolve():
        raise ValueError(f'--data {data_path} differs from the cache the folds trained on ({trained_on})')

    dev = pick_device(device)
    data = PlayData.load(data_path)
    if data.fingerprint() != fingerprint:
        raise ValueError(f'{data_path} has changed since training (play order differs); '
                         'scoring would rebuild the wrong held-out splits')
    data.to(dev)

    plays = pd.concat([score_fold(c, data, n_samples, batch_size, dev) for c in ckpts], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    plays.to_csv(out_dir / 'play_receiver_scores.csv', index=False)
    aggregate_players(plays, players_csvs).to_csv(out_dir / 'player_scores.csv', index=False)
    print(f'scored {plays.play_id.size} play-receiver pairs from {len(ckpts)} folds -> {out_dir}')
