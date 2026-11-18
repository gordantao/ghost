"""Calibration of the ghost's nearest-defender distance, by probability integral transform (PIT).

H depends on the distribution of the distance from a receiver to the nearest
defender, but the ghost is trained and selected on per-defender position error.
This checks that statistic directly. For a held-out play, receiver and predicted
frame t, take the real nearest-defender distance A_t and the K ghost samples'
nearest-ghost-defender distances D_t^(k). The PIT value is

    u = (#{k: D_t^(k) < A_t} + 0.5 * #{k: D_t^(k) == A_t}) / K

For a calibrated ghost u is roughly Uniform(0, 1): mean 0.5, and the real
distance falls below every ghost about as often as above every ghost. Ghosts
that are too loose (farther from receivers than the real defense) push u
toward 0, so mean u < 0.5; ghosts that are too tight push it toward 1. With
finite K, u takes values in {0, 1/(2K), ...} so the histogram is only
approximately flat even for a perfect ghost.

Both ghosts of `score` are checked: the default one that draws its own call
and the one given the actual call.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import PlayData
from .model import future_mask
from .preprocess import DEF, GROUP_VOCAB, OFF
from .score import (RECEIVER_IDS, fold_batches, held_out_plays, load_cross_fit, nearest_defender_distance,
                    receiver_rows, two_ghosts)
from .train import load_model

N_BINS = 10
FRAMES_PER_SECOND = 10
OFFSET_BINS = ['0-0.9s', '1-1.9s', '2-2.9s', '3s+']      # seconds after the snap


@torch.no_grad()
def pit_values(samples: torch.Tensor, batch: dict, obs_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame PIT value u [B, T, 11] of the real nearest-defender distance among the ghosts'.

    samples: [B, K, T, 11, 2] ghost defenders. Also returns the predicted-frame
    mask [B, T] (u is only meaningful where it is True).
    """
    offense = batch['dyn'][:, :, OFF, :2]
    ghost = nearest_defender_distance(offense[:, None], samples)                    # [B, K, T, 11]
    actual = nearest_defender_distance(offense, batch['dyn'][:, :, DEF, :2])      # [B, T, 11]
    a = actual[:, None]
    u = ((ghost < a).float() + 0.5 * (ghost == a).float()).mean(1)
    return u, future_mask(batch['time_mask'], obs_frames)


def offset_bin(frame: np.ndarray) -> np.ndarray:
    """Index into OFFSET_BINS of each frame index (frame 0 is the snap)."""
    return np.minimum(frame // FRAMES_PER_SECOND, len(OFFSET_BINS) - 1)


def pit_stat_rows(u: torch.Tensor, mask: torch.Tensor, batch: dict) -> pd.DataFrame:
    """One row per (group, offset bin) of this batch's receiver-frames, holding sums to aggregate.

    Columns: group, offset, n, u (sum), low (count u == 0), high (count u == 1), h0..h9 (histogram counts).
    """
    group = batch['group'][:, OFF][:, None, :].expand_as(u)                         # [B, T, 11]
    frame = torch.arange(u.shape[1], device=u.device)[None, :, None].expand_as(u)
    keep = mask[:, :, None] & torch.isin(group, torch.tensor(RECEIVER_IDS, device=u.device))
    u, group, frame = u[keep].cpu().numpy(), group[keep].cpu().numpy(), frame[keep].cpu().numpy()
    df = pd.DataFrame({'group': [GROUP_VOCAB[g] for g in group], 'offset': offset_bin(frame),
                       'n': 1, 'u': u, 'low': u == 0, 'high': u == 1})
    bins = np.minimum((u * N_BINS).astype(int), N_BINS - 1)
    for b in range(N_BINS):
        df[f'h{b}'] = bins == b
    return df.groupby(['group', 'offset'], as_index=False).sum()


def summarize(stats: pd.DataFrame) -> pd.DataFrame:
    """Turn summed stat rows (with a `ghost` column) into the overall / by-group / by-offset summary."""
    hist = [f'h{b}' for b in range(N_BINS)]
    stats = stats.assign(offset=[OFFSET_BINS[o] for o in stats.offset])
    parts = []
    for ghost, g in stats.groupby('ghost'):
        for stratum, by in (('overall', None), ('group', 'group'), ('offset', 'offset')):
            sums = g.drop(columns=['ghost', 'group', 'offset']).sum().to_frame().T if by is None else \
                g.groupby(by)[['n', 'u', 'low', 'high', *hist]].sum().reset_index()
            sums.insert(0, 'level', 'all' if by is None else sums[by])
            sums.insert(0, 'stratum', stratum)
            sums.insert(0, 'ghost', ghost)
            parts.append(sums.drop(columns=[c for c in ('group', 'offset') if c in sums]))
    out = pd.concat(parts, ignore_index=True)
    n = out.n
    out['mean_u'] = out.u / n
    out['share_u0'] = out.low / n
    out['share_u1'] = out.high / n
    out[[f'hist{b}' for b in range(N_BINS)]] = out[hist].div(n, axis=0).to_numpy()
    cols = ['ghost', 'stratum', 'level', 'n', 'mean_u', 'share_u0', 'share_u1'] + [f'hist{b}' for b in range(N_BINS)]
    return out[cols]


def calibrate_fold(ckpt_path: Path, data: PlayData, n_samples: int, batch_size: int,
                   plays_per_fold: int | None, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summed PIT stats, per-play-receiver mean u) for one fold's held-out plays."""
    model, ckpt = load_model(ckpt_path, device)
    plays = held_out_plays(ckpt, data)
    if plays_per_fold is not None and plays_per_fold < len(plays):
        rng = np.random.default_rng(ckpt['train_config']['seed'])
        plays = np.sort(rng.choice(plays, plays_per_fold, replace=False))
    obs = model.cfg.obs_frames
    stats, per_play = [], []
    for idx, batch, seed in fold_batches(ckpt, data, plays, batch_size, device):
        cols = {}
        for ghost, samples in zip(('own_call', 'actual_call'), two_ghosts(model, batch, n_samples, seed)):
            u, mask = pit_values(samples, batch, obs)
            stats.append(pit_stat_rows(u, mask, batch).assign(ghost=ghost))
            m = mask[:, :, None].float()
            cols[f'u_mean_{ghost}'] = ((u * m).sum(1) / m.sum(1)).cpu().numpy()
        per_play.append(receiver_rows(data, idx, batch, cols))
    per_play = pd.concat(per_play, ignore_index=True)
    per_play['fold'] = ckpt['fold']
    return pd.concat(stats, ignore_index=True), per_play


def calibration(data_path: Path | None, runs_dir: Path, out_dir: Path, n_samples: int = 16,
                batch_size: int = 32, plays_per_fold: int | None = None, device: str = 'auto') -> pd.DataFrame:
    # Checked up front: a bad value would otherwise fail only after all the sampling.
    if n_samples < 1:
        raise ValueError('need at least 1 ghost sample per play')
    if plays_per_fold is not None and plays_per_fold < 1:
        raise ValueError('plays_per_fold must be >= 1')

    ckpts, data, dev = load_cross_fit(data_path, runs_dir, device)
    folds = [calibrate_fold(c, data, n_samples, batch_size, plays_per_fold, dev) for c in ckpts]
    per_play = pd.concat([f[1] for f in folds], ignore_index=True)
    stats = pd.concat([f[0] for f in folds], ignore_index=True)
    stats = stats.groupby(['ghost', 'group', 'offset'], as_index=False).sum()
    summary = summarize(stats)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_play.to_csv(out_dir / 'pit_play_receiver.csv', index=False)
    summary.to_csv(out_dir / 'pit_summary.csv', index=False)
    overall = summary[summary.stratum == 'overall']
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(overall.to_string(index=False, float_format='%.4g'))
    print(f'PIT of {len(per_play)} play-receiver pairs at K={n_samples} -> {out_dir}')
    return summary
