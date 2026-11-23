"""Heliocentricity scores from cross-fitted ghost-defense models.

For each receiver r on a play, at every frame the model had to predict:

    A_t = distance from r to the nearest actual defender
    E_t = mean over K ghost samples of the distance from r to the nearest ghost defender
    H_t = E_t - A_t

H for the (play, receiver) is the mean of H_t. Positive H means the real
defense played r tighter than a typical defense would have given the same
offensive movement; that extra attention is the receiver's "gravity".

Raw H carries the model's miscalibration: for a calibrated ghost the mean H in
any group defined by model inputs is exactly 0, yet real fold models are biased
by position group and increasingly so at later frames. H_centered subtracts the
mean H_t of each (fold, position group, frame offset) cell before averaging over
the play's frames, so what is left is gravity relative to the group's norm.
Player scores then get game-clustered standard errors, and players with too
few plays to rank (by default the lower quartile by play count) are left out.

H_dens{sigma} is the same comparison for defender density instead of
nearest distance: every defender counts exp(-d^2 / 2 sigma^2), and H_dens is the
real density minus the mean ghost density. Far defenders add almost nothing and
a second close defender counts as much as the first, so it credits double
coverage and (at the wider bandwidth) help, and isn't dominated by how far away
the nearest defender is on loose plays. Positive still means the real defense
crowded the player more than the ghosts did. It is unitless (about "defenders
within sigma yards"), so it isn't comparable to H in size.

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
from .routes import check_label_coverage, load_route_labels, route_flags, rule_ran_route
from .train import load_model, pick_device

RECEIVER_IDS = [GROUP_VOCAB.index(g) for g in RECEIVER_GROUPS]
H_COLS = ('H', 'H_within', 'H_call')
# Defender-density bandwidths in yards: 2 counts tight coverage only, 5 also reaches help defenders.
DENSITY_SIGMAS = (2, 5)
DENS_COLS = tuple(f'H_dens{s}' for s in DENSITY_SIGMAS)
# A centering cell (fold, group, frame offset) needs at least this many scored rows.
# Offsets from the first one with fewer rows onward share one pooled cell, so the
# long tail of rarely-reached late frames doesn't get noisy means.
MIN_CELL_ROWS = 200


def nearest_defender_distance(receivers: torch.Tensor, defenders: torch.Tensor) -> torch.Tensor:
    """receivers [..., T, R, 2], defenders [..., T, D, 2] -> [..., T, R]."""
    return (receivers[..., :, None, :] - defenders[..., None, :, :]).norm(dim=-1).min(dim=-1).values


def defender_density(receivers: torch.Tensor, defenders: torch.Tensor, sigma: float) -> torch.Tensor:
    """receivers [..., T, R, 2], defenders [..., T, D, 2] -> [..., T, R]: sum over defenders of exp(-d^2 / 2 sigma^2)."""
    d2 = (receivers[..., :, None, :] - defenders[..., None, :, :]).pow(2).sum(-1)
    return torch.exp(-d2 / (2 * sigma ** 2)).sum(-1)


@torch.no_grad()
def ghost_separation(samples: torch.Tensor, batch: dict, obs_frames: int) -> torch.Tensor:
    """[B, K, 11] per ghost sample, the frame-mean distance from each offensive player
    to the nearest ghost defender over the predicted frames."""
    offense = batch['dyn'][:, :, OFF, :2]
    dist = nearest_defender_distance(offense[:, None], samples)                      # [B, K, T, 11]
    return masked_frame_mean(dist, batch, obs_frames)


def masked_frame_mean(dist: torch.Tensor, batch: dict, obs_frames: int) -> torch.Tensor:
    """[B, K, T, 11] -> [B, K, 11], the mean over predicted frames."""
    mask = future_mask(batch['time_mask'], obs_frames)[:, None, :, None].float()
    return (dist * mask).sum(2) / mask.sum(2)


@torch.no_grad()
def heliocentricity(samples: torch.Tensor, batch: dict, obs_frames: int) -> dict[str, torch.Tensor]:
    """Per-(play, offensive player) expected and actual separation, and H.

    samples: [B, K, T, 11, 2] ghost defenders. Returns [B, 11] tensors, plus
    `H_t` [B, T, 11], H at every frame (frames the model doesn't predict are
    meaningless; mask them with `future_mask`).
    """
    offense = batch['dyn'][:, :, OFF, :2]
    ghost_dist = nearest_defender_distance(offense[:, None], samples)                # [B, K, T, 11]
    actual_dist = nearest_defender_distance(offense, batch['dyn'][:, :, DEF, :2])    # [B, T, 11]
    expected_sep = masked_frame_mean(ghost_dist, batch, obs_frames).mean(1)
    actual_sep = masked_frame_mean(actual_dist[:, None], batch, obs_frames)[:, 0]
    n = future_mask(batch['time_mask'], obs_frames).sum(1, keepdim=True).float()
    return {
        'expected_sep': expected_sep,
        'actual_sep': actual_sep,
        'H': expected_sep - actual_sep,
        'H_t': ghost_dist.mean(1) - actual_dist,
        'n_frames': n.expand_as(actual_sep),
    }


@torch.no_grad()
def density_gravity(samples: torch.Tensor, batch: dict, obs_frames: int, sigma: float) -> dict[str, torch.Tensor]:
    """Density analogue of `heliocentricity`: [B, 11] expected and actual defender density, and
    H = actual - expected; `H_t` [B, T, 11] per frame."""
    offense = batch['dyn'][:, :, OFF, :2]
    ghost = defender_density(offense[:, None], samples, sigma).mean(1)                 # [B, T, 11]
    actual = defender_density(offense, batch['dyn'][:, :, DEF, :2], sigma)            # [B, T, 11]
    expected = masked_frame_mean(ghost[:, None], batch, obs_frames)[:, 0]
    real = masked_frame_mean(actual[:, None], batch, obs_frames)[:, 0]
    return {'expected': expected, 'actual': real, 'H': real - expected, 'H_t': actual - ghost}


def actual_separation(batch: dict, obs_frames: int) -> torch.Tensor:
    """[B, 11] frame-mean distance from each offensive player to the nearest real defender."""
    return ghost_separation(batch['dyn'][:, None, :, DEF, :2], batch, obs_frames)[:, 0]


def two_ghosts(model, batch: dict, n_samples: int, seed: int, return_coverage: bool = False):
    """(total, within) ghost samples: one draws its own call, the other runs the actual call.

    Both use one seed, so they share their z noise (common random numbers) and
    H_call reflects the call, not Monte Carlo noise. With `return_coverage`,
    also returns the calls the total ghost drew [B, K].
    """
    gen = lambda: torch.Generator(batch['dyn'].device).manual_seed(seed)  # noqa: E731
    total = model.sample(batch, n_samples, generator=gen(), return_coverage=return_coverage)
    within = model.sample(batch, n_samples, coverage=batch['coverage'], generator=gen())
    return (*total, within) if return_coverage else (total, within)


def fold_batches(ckpt: dict, data: PlayData, plays: np.ndarray, batch_size: int, device: torch.device):
    """Yield (play indices, batch, batch seed), seeded from the fold's training seed."""
    torch.manual_seed(ckpt['train_config']['seed'])
    for i in range(0, len(plays), batch_size):
        idx = plays[i:i + batch_size]
        batch = data.batch(torch.from_numpy(idx).to(device))
        yield idx, batch, int(torch.randint(2 ** 62, ()))


def held_out_plays(ckpt: dict, data: PlayData) -> np.ndarray:
    """Indices of the plays this fold's model never trained on."""
    _, _, held_out = split_indices(data.game_id, ckpt['fold'], ckpt['n_folds'],
                                   ckpt['train_config']['val_frac'], ckpt['train_config']['seed'])
    return held_out


def receiver_rows(data: PlayData, idx: np.ndarray, batch: dict, cols: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per receiver (WR, TE, RB) on the batch's plays; `cols` holds [B, 11] arrays.

    `ran_route` / `route_source` say whether the receiver ran a route and
    whether that came from the label or the tracking rule (see routes.py).
    """
    group = batch['group'][:, OFF].cpu().numpy()
    coverage_vocab = data.meta['coverage_vocab']
    game_id, play_id = np.repeat(data.game_id[idx], 11), np.repeat(data.play_id[idx], 11)
    nfl_id = data.nfl_id[idx][:, OFF].ravel()
    ran_route, route_source = route_flags(game_id, play_id, nfl_id,
                                          rule_ran_route(batch['dyn'], batch['time_mask']).cpu().numpy().ravel(),
                                          data.route_labels)
    rows = pd.DataFrame({
        'game_id': game_id,
        'play_id': play_id,
        'nfl_id': nfl_id,
        'ran_route': ran_route,
        'route_source': route_source,
        'group': [GROUP_VOCAB[g] for g in group.ravel()],
        'coverage': np.repeat([coverage_vocab[c] for c in batch['coverage'].cpu().numpy()], 11),
        **{k: v.ravel() for k, v in cols.items()},
    })
    return rows[np.isin(group.ravel(), RECEIVER_IDS)]


def frame_arrays(batch: dict, h_t: torch.Tensor, obs_frames: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """([N, width] per-frame H, [N, width] predicted-frame mask) for each scored receiver, in `receiver_rows` order.

    Batches are trimmed to their longest play, so both are right-padded (0, False) to the cache's `width`.
    """
    group = batch['group'][:, OFF].cpu().numpy().ravel()
    keep = np.isin(group, RECEIVER_IDS)
    h = h_t.permute(0, 2, 1).reshape(-1, h_t.shape[1]).cpu().numpy()[keep]
    valid = np.repeat(future_mask(batch['time_mask'], obs_frames).cpu().numpy(), 11, axis=0)[keep]
    pad = ((0, 0), (0, width - h.shape[1]))
    return np.pad(h, pad), np.pad(valid, pad)


def score_fold(ckpt_path: Path, data: PlayData, n_samples: int, batch_size: int,
               device: torch.device) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Per-(play, receiver) rows, and the per-frame arrays `H_t`, `H_within_t`, `H_dens{sigma}_t` and
    `valid` ([N, T], same row order). Density uses the total ghost (own call) only."""
    model, ckpt = load_model(ckpt_path, device)
    obs = model.cfg.obs_frames
    width = data.dyn.shape[1]
    frames, valids = [], []
    per_frame = {k: [] for k in ('H_t', 'H_within_t', *(f'{c}_t' for c in DENS_COLS))}
    for idx, batch, seed in fold_batches(ckpt, data, held_out_plays(ckpt, data), batch_size, device):
        total_samples, within_samples = two_ghosts(model, batch, n_samples, seed)
        total, within = heliocentricity(total_samples, batch, obs), heliocentricity(within_samples, batch, obs)
        cols = {k: v.cpu().numpy() for k, v in total.items() if k != 'H_t'}
        cols['expected_sep_within'] = within['expected_sep'].cpu().numpy()
        cols['H_within'] = within['H'].cpu().numpy()
        cols['H_call'] = cols['H'] - cols['H_within']
        h, valid = frame_arrays(batch, total['H_t'], obs, width)
        per_frame['H_t'].append(h)
        per_frame['H_within_t'].append(frame_arrays(batch, within['H_t'], obs, width)[0])
        for sigma, c in zip(DENSITY_SIGMAS, DENS_COLS):
            dens = density_gravity(total_samples, batch, obs, sigma)
            cols[c] = dens['H'].cpu().numpy()
            cols[f'dens{sigma}_expected'] = dens['expected'].cpu().numpy()
            cols[f'dens{sigma}_actual'] = dens['actual'].cpu().numpy()
            per_frame[f'{c}_t'].append(frame_arrays(batch, dens['H_t'], obs, width)[0])
        frames.append(receiver_rows(data, idx, batch, cols))
        valids.append(valid)
    out = pd.concat(frames, ignore_index=True)
    out['fold'] = ckpt['fold']
    return out, {**{k: np.concatenate(v) for k, v in per_frame.items()}, 'valid': np.concatenate(valids)}


def offset_buckets(valid: np.ndarray, min_rows: int) -> np.ndarray:
    """[T] the centering cell of each frame offset: itself, except that the first offset
    with fewer than `min_rows` scored rows, and every later one, share one cell."""
    counts = valid.sum(0)
    seen = np.flatnonzero(counts)
    bucket = np.arange(valid.shape[1])
    if len(seen):
        thin = seen[counts[seen] < min_rows]
        thin = thin[thin > seen[0]]
        if len(thin):
            bucket[thin[0]:] = thin[0]
    return bucket


def center_frames(h_t: np.ndarray, valid: np.ndarray, fold: np.ndarray, group: np.ndarray,
                  min_rows: int = MIN_CELL_ROWS) -> np.ndarray:
    """[N] per-row mean over predicted frames of H_t minus its (fold, group, frame offset) cell mean.

    h_t, valid: [N, T]. Late offsets are pooled per `offset_buckets`. Each offset's residuals
    average to 0 over the rows scored there, so a cell's row means are only approximately 0
    when play lengths vary (zero frame-weighted, not row-weighted).
    """
    out = np.zeros(len(h_t))
    n_frames = np.maximum(valid.sum(1), 1)
    cell = pd.MultiIndex.from_arrays([fold, group]).factorize()[0]
    for c in np.unique(cell):
        rows = np.flatnonzero(cell == c)
        h, v = h_t[rows], valid[rows]
        bucket = offset_buckets(v, min_rows)
        k = bucket.max() + 1
        total = np.bincount(np.broadcast_to(bucket, h.shape)[v], weights=h[v], minlength=k)
        count = np.bincount(np.broadcast_to(bucket, h.shape)[v], minlength=k)
        mean = np.where(count > 0, total / np.maximum(count, 1), 0.0)
        out[rows] = np.where(v, h - mean[bucket], 0.0).sum(1) / n_frames[rows]
    return out


def add_centered(plays: pd.DataFrame, arrays: dict[str, np.ndarray], min_rows: int = MIN_CELL_ROWS) -> pd.DataFrame:
    """Add H_centered, H_within_centered and H_call_centered (and H_dens{sigma}_centered for each
    density array present) to the play-receiver rows."""
    for col in ('H', 'H_within', *DENS_COLS):
        if f'{col}_t' in arrays:
            plays[f'{col}_centered'] = center_frames(arrays[f'{col}_t'], arrays['valid'], plays.fold.to_numpy(),
                                                     plays.group.to_numpy(), min_rows)
    plays['H_call_centered'] = plays.H_centered - plays.H_within_centered
    return plays


def player_names(players_csvs: list[Path] | None) -> pd.Series:
    """nflId -> display name from every players.csv that exists (seasons can each add players)."""
    tables = [pd.read_csv(p, usecols=['nflId', 'displayName']) for p in players_csvs or [] if p.exists()]
    if not tables:
        return pd.Series(dtype=str)
    return pd.concat(tables).drop_duplicates('nflId').set_index('nflId').displayName


def clustered_se(plays: pd.DataFrame, col: str) -> pd.Series:
    """nfl_id -> game-clustered standard error of the player's mean of `col`.

    CR-style sandwich for a mean: sqrt(sum_g (sum of residuals in game g)^2) / n, times
    sqrt(G / (G - 1)) for G games. With one play per game it equals std / sqrt(n).
    A player seen in one game has no clustered SE (NaN).
    """
    resid = plays[col] - plays.groupby('nfl_id')[col].transform('mean')
    by_game = resid.groupby([plays.nfl_id, plays.game_id]).sum()
    n_games = by_game.groupby(level=0).size()
    n = plays.groupby('nfl_id').size()
    sq = (by_game ** 2).groupby(level=0).sum()
    se = np.sqrt(sq * n_games / (n_games - 1).where(n_games > 1)) / n
    return se.reindex(n.index)


def lower_quartile_plays(n_plays: pd.Series) -> int:
    """Default `min_plays`: the 25th percentile of players' play counts, so the lower quartile is dropped."""
    return int(np.ceil(n_plays.quantile(0.25)))


def aggregate_players(plays: pd.DataFrame, players_csvs: list[Path] | None,
                      include_blockers: bool = False, min_plays: int | None = 0) -> pd.DataFrame:
    """Per-player means. Plays where the receiver didn't run a route (he stayed in to block) are
    left out unless `include_blockers`. Players with fewer than `min_plays` of the remaining plays
    are dropped; None drops the lower quartile of players by play count. The cutoff used is in
    `.attrs['min_plays']`."""
    if not include_blockers:
        plays = plays[plays.ran_route]
    stats = plays.groupby('nfl_id').agg(
        group=('group', 'first'), n_plays=('H', 'size'),
        H_mean=('H', 'mean'), H_std=('H', 'std'),
        H_within_mean=('H_within', 'mean'), H_within_std=('H_within', 'std'),
        H_call_mean=('H_call', 'mean'), H_call_std=('H_call', 'std'),
        expected_sep=('expected_sep', 'mean'), expected_sep_within=('expected_sep_within', 'mean'),
        actual_sep=('actual_sep', 'mean'),
    )
    for h in H_COLS:
        stats[f'{h}_se'] = stats[f'{h}_std'] / np.sqrt(stats.n_plays)
    for h in (c for c in DENS_COLS if c in plays):
        stats[f'{h}_mean'] = plays.groupby('nfl_id')[h].mean()
        stats[f'{h}_se'] = plays.groupby('nfl_id')[h].std() / np.sqrt(stats.n_plays)
    centered = [f'{h}_centered' for h in H_COLS]
    if all(c in plays for c in centered) and 'game_id' in plays:
        centered += [f'{h}_centered' for h in DENS_COLS if f'{h}_centered' in plays]
        for c in centered:
            stats[f'{c}_mean'] = plays.groupby('nfl_id')[c].mean()
            stats[f'{c}_se'] = clustered_se(plays, c)
        sort_by = 'H_centered_mean'
    else:
        sort_by = 'H_mean'
    if min_plays is None:
        min_plays = lower_quartile_plays(stats.n_plays)
    stats = stats[stats.n_plays >= min_plays]
    names = player_names(players_csvs)
    if len(names):
        stats.insert(0, 'name', names.reindex(stats.index).to_numpy())
    out = stats.sort_values(sort_by, ascending=False).reset_index()
    out.attrs['min_plays'] = min_plays
    return out


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


def load_cross_fit(data_path: Path | None, runs_dir: Path,
                   device: str) -> tuple[list[Path], PlayData, torch.device]:
    """Every fold checkpoint of a run, and the unchanged cache they trained on, on `device`."""
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
    sources = data.meta.get('sources', [])
    # sources are repo-root-relative and the cache sits at <root>/data/processed/<name>.npz
    data.route_labels = load_route_labels(sources, Path(data_path).resolve().parents[2])
    check_label_coverage(data.season, data.game_id, data.play_id, data.route_labels, sources)
    data.to(dev)
    return ckpts, data, dev


def score(data_path: Path | None, runs_dir: Path, out_dir: Path, n_samples: int = 16,
          batch_size: int = 32, players_csvs: list[Path] | None = None, device: str = 'auto',
          include_blockers: bool = False, min_plays: int | None = None) -> None:
    ckpts, data, dev = load_cross_fit(data_path, runs_dir, device)
    folds = [score_fold(c, data, n_samples, batch_size, dev) for c in ckpts]
    plays = pd.concat([f for f, _ in folds], ignore_index=True)
    arrays = {k: np.concatenate([a[k] for _, a in folds]) for k in folds[0][1]}
    del folds
    add_centered(plays, arrays)
    out_dir.mkdir(parents=True, exist_ok=True)
    plays.to_csv(out_dir / 'play_receiver_scores.csv', index=False)
    players = aggregate_players(plays, players_csvs, include_blockers, min_plays)
    players.to_csv(out_dir / 'player_scores.csv', index=False)
    print(f'scored {plays.play_id.size} play-receiver pairs from {len(ckpts)} folds '
          f'({int(plays.ran_route.sum())} ran a route); {len(players)} players with >= '
          f'{players.attrs["min_plays"]} plays -> {out_dir}')
