"""How many ghost samples `ghost score` needs, from the per-play Monte Carlo error.

For a receiver r on a play, ghost sample k gives D_k, the frame-mean distance
from r to the nearest ghost defender. Given the play the samples are i.i.d.,
so with K samples the Monte Carlo standard error of H is s / sqrt(K), where
s = sd(D_k) for that (play, receiver). Nothing here assumes D_k is Gaussian.

This draws a large K for every held-out play, estimates s for each
(play, receiver), and picks the K at which a `quantile` share of them have
standard error <= delta:

    K = ceil((quantile of s across play-receivers / delta) ** 2)

The quantile is the observed s at or above it, not an interpolated value, so
at least that share is guaranteed to meet delta.

By default delta = sqrt(eps * V), where V is the variance of H across
play-receivers net of Monte Carlo noise, so sampling noise is at most an `eps`
share of the real spread. Pass `delta` (yards) to fix it instead.

This is done for H, H_within and H_call. The two ghosts share their z noise
(common random numbers, as in `score`), so H_call's s is the sd of the paired
per-sample difference. For H, s^2 is also split by the law of total variance
into spread within a call and between the calls the ghost drew; a large
between-call share means stratifying over the call would cut the K needed.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import PlayData
from .score import (actual_separation, fold_batches, ghost_separation, held_out_plays, load_cross_fit,
                    receiver_rows, two_ghosts)
from .train import load_model

QUANTITIES = ('H', 'H_within', 'H_call')


def call_decomposition(d: torch.Tensor, call: torch.Tensor, n_coverage: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the per-sample variance of d [B, K, R] by the call each sample drew, call [B, K].

    Returns (within-call, between-call) variances [B, R]; with population
    variances they sum exactly to d.var(1, unbiased=False).
    """
    onehot = F.one_hot(call, n_coverage).to(d.dtype)                           # [B, K, C]
    n = onehot.sum(1)                                                           # [B, C]
    call_mean = torch.einsum('bkc,bkr->bcr', onehot, d) / n.clamp(min=1)[..., None]
    weight = (n / d.shape[1])[..., None]                                        # [B, C, 1]
    between = (weight * (call_mean - d.mean(1, keepdim=True)) ** 2).sum(1)
    within = (d.var(1, unbiased=False) - between).clamp(min=0)
    return within, between


@torch.no_grad()
def batch_spread(model, batch: dict, n_samples: int, seed: int) -> dict[str, torch.Tensor]:
    """H, H_within, H_call and the per-sample sd of each, as [B, 11] tensors.

    Drawn through the same `two_ghosts` as `score`, so for the same plays, K and
    seed, H and H_within equal what `score` computes. The CSVs still differ play
    by play, because this command's batches (and so its batch seeds) differ.
    """
    obs = model.cfg.obs_frames
    total, call, within = two_ghosts(model, batch, n_samples, seed, return_coverage=True)
    d_total = ghost_separation(total, batch, obs)                               # [B, K, 11]
    d_within = ghost_separation(within, batch, obs)
    d_call = d_total - d_within
    actual = actual_separation(batch, obs)
    within_call, between_call = call_decomposition(d_total, call, model.cfg.n_coverage)
    return {
        'H': d_total.mean(1) - actual,
        'H_within': d_within.mean(1) - actual,
        'H_call': d_call.mean(1),
        's_H': d_total.std(1),
        's_H_within': d_within.std(1),
        's_H_call': d_call.std(1),
        's2_H_within_call': within_call,
        's2_H_between_call': between_call,
    }


def spread_fold(ckpt_path: Path, data: PlayData, n_samples: int, batch_size: int,
                plays_per_fold: int | None, device: torch.device) -> pd.DataFrame:
    model, ckpt = load_model(ckpt_path, device)
    plays = held_out_plays(ckpt, data)
    if plays_per_fold is not None and plays_per_fold < len(plays):
        rng = np.random.default_rng(ckpt['train_config']['seed'])
        plays = np.sort(rng.choice(plays, plays_per_fold, replace=False))
    frames = []
    for idx, batch, seed in fold_batches(ckpt, data, plays, batch_size, device):
        spread = batch_spread(model, batch, n_samples, seed)
        frames.append(receiver_rows(data, idx, batch, {k: v.cpu().numpy() for k, v in spread.items()}))
    out = pd.concat(frames, ignore_index=True)
    out['fold'] = ckpt['fold']
    return out


def recommend(spread: pd.DataFrame, n_samples: int, eps: float = 0.05, quantile: float = 0.9,
              delta: float | None = None, current_k: int = 16) -> pd.DataFrame:
    """One row per quantity: the tolerance delta, the K it needs, and how the current K fares.

    A quantity with no spread beyond Monte Carlo noise (V <= 0) has no
    eps-based tolerance; its delta and K are NaN unless `delta` is given.
    """
    rows = []
    for q in QUANTITIES:
        s = spread[f's_{q}']
        mc_var = (s ** 2).mean()
        V = spread[q].var() - mc_var / n_samples        # spread of H across play-receivers, net of MC noise
        tol = delta if delta is not None else math.sqrt(eps * V) if V > 0 else math.nan
        s_q = s.quantile(quantile, interpolation='higher')  # an observed s, so >= `quantile` meet delta
        k = max(2, math.ceil((s_q / tol) ** 2)) if tol > 0 else math.nan
        if math.isnan(k):
            print(f'warning: {q} shows no spread beyond Monte Carlo noise (V = {V:.3g}); pass --delta to size it')
        rows.append({
            'quantity': q, 'V': V, 'delta': tol, 's_rms': math.sqrt(mc_var),
            'quantile': quantile, 's_quantile': s_q, 'K': k,
            'share_within_delta': (s / math.sqrt(k) <= tol).mean() if k > 0 else math.nan,
            'current_K': current_k,
            'share_within_delta_current_K': (s / math.sqrt(current_k) <= tol).mean() if tol > 0 else math.nan,
            'mc_share_of_V_current_K': mc_var / current_k / V if V > 0 else math.nan,
        })
    out = pd.DataFrame(rows)
    total = spread.s2_H_within_call.mean() + spread.s2_H_between_call.mean()
    share = spread.s2_H_between_call.mean() / total if total > 0 else math.nan
    out['between_call_share'] = np.where(out.quantity == 'H', share, math.nan)
    return out


def sample_size(data_path: Path | None, runs_dir: Path, out_dir: Path, n_samples: int = 128,
                batch_size: int = 8, eps: float = 0.05, quantile: float = 0.9, delta: float | None = None,
                plays_per_fold: int | None = None, current_k: int = 16, device: str = 'auto') -> pd.DataFrame:
    # Checked up front: a bad value would otherwise fail only after all the sampling.
    if n_samples < 2:
        raise ValueError('need at least 2 samples per play to estimate their spread')
    if not eps > 0 or not 0 < quantile < 1 or current_k < 1:
        raise ValueError('need eps > 0, 0 < quantile < 1 and current_k >= 1')
    if delta is not None and not delta > 0:
        raise ValueError('delta must be > 0 yards')
    if plays_per_fold is not None and plays_per_fold < 1:
        raise ValueError('plays_per_fold must be >= 1')

    ckpts, data, dev = load_cross_fit(data_path, runs_dir, device)
    spread = pd.concat([spread_fold(c, data, n_samples, batch_size, plays_per_fold, dev) for c in ckpts],
                       ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    spread.to_csv(out_dir / 'play_receiver_spread.csv', index=False)   # saved first: it's the expensive part
    summary = recommend(spread, n_samples, eps, quantile, delta, current_k)
    summary.to_csv(out_dir / 'summary.csv', index=False)
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(summary.to_string(index=False, float_format='%.4g'))
    print(f'{len(spread)} play-receiver pairs at K={n_samples} -> {out_dir}; '
          f'recommended K = {summary.K.max():.0f} (max over {", ".join(QUANTITIES)})')
    return summary
