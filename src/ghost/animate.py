"""Animate one play: real offense and defense with the ghost defense drawn over them.

The ghost comes from the fold model that held the play out, the same model
`score` uses for it, so the picture matches the play's H. Each ghost defender
is shown as a cloud of K samples plus their mean; the mean is drawn as a ring
in the same slot as the real defender it stands in for, so the two can be
compared directly. Receivers are labelled with their H for the play.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import PlayData, split_indices
from .preprocess import BALL, DEF, FIELD_WIDTH, FPS, GROUP_VOCAB, OFF, RECEIVER_GROUPS
from .score import check_checkpoints, heliocentricity, player_names
from .train import load_model, pick_device

# Reference categorical palette, first three slots (validated all-pairs, light surface).
SURFACE, INK, INK_2, RULE = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
OFFENSE, DEFENSE, GHOST = '#2a78d6', '#eb6834', '#1baf7a'


def select_play(game_id: np.ndarray, play_id: np.ndarray, game: int | None, play: int | None,
                rng: np.random.Generator) -> int:
    """Index of the requested play, or of a random one when `play` is None.

    play_id is only unique within a game, so `game` is needed when a play_id
    appears in several games.
    """
    if play is None:
        candidates = np.flatnonzero(game_id == game) if game is not None else np.arange(len(play_id))
    else:
        candidates = np.flatnonzero((play_id == play) & ((game_id == game) if game is not None else True))
    if len(candidates) == 0:
        raise ValueError(f'no play matches game_id={game} play_id={play}')
    if play is not None and len(candidates) > 1:
        games = ', '.join(str(g) for g in game_id[candidates])
        raise ValueError(f'play_id {play} occurs in several games ({games}); pass --game-id')
    return int(rng.choice(candidates))


def holdout_checkpoint(ckpts: list[Path], game_id: np.ndarray, idx: int) -> Path:
    """The fold checkpoint whose held-out split contains play `idx`."""
    for c in ckpts:
        ck = torch.load(c, map_location='cpu', weights_only=False)
        _, _, held_out = split_indices(game_id, ck['fold'], ck['n_folds'],
                                       ck['train_config']['val_frac'], ck['train_config']['seed'])
        if idx in held_out:
            return c
    raise ValueError(f'no fold checkpoint holds out play index {idx}')


@torch.no_grad()
def ghost_play(model, data: PlayData, idx: int, n_samples: int, actual_call: bool,
               seed: int) -> dict[str, np.ndarray]:
    """Real tracking, ghost samples and per-player H for one play (trimmed to its length)."""
    device = next(model.parameters()).device
    batch = data.batch(torch.tensor([idx], device=device))
    gen = torch.Generator(device).manual_seed(seed)
    coverage = batch['coverage'] if actual_call else None
    samples = model.sample(batch, n_samples, coverage=coverage, generator=gen)       # [1, K, T, 11, 2]
    h = heliocentricity(samples, batch, model.cfg.obs_frames)
    return {
        'dyn': batch['dyn'][0, :, :, :2].cpu().numpy(),                              # [T, 23, 2]
        'ghost': samples[0].cpu().numpy(),                                           # [K, T, 11, 2]
        'H': h['H'][0].cpu().numpy(),                                                # [11]
        'group': batch['group'][0].cpu().numpy(),
    }


def render(play: dict, obs_frames: int, title: str, subtitle: str, labels: dict[int, str],
           first_down: float | None, out: Path | None, fps: float = FPS):
    """Draw the animation. Saves a GIF to `out`, or opens a window when `out` is None."""
    import matplotlib
    if out is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D

    dyn, ghost = play['dyn'], play['ghost']
    T = dyn.shape[0]
    half_w = FIELD_WIDTH / 2
    pts = np.concatenate([dyn.reshape(-1, 2), ghost.reshape(-1, 2)])
    x_lo, x_hi = min(pts[:, 0].min(), -5.0) - 3, max(pts[:, 0].max(), 10.0) + 3
    y_lo, y_hi = max(pts[:, 1].min() - 3, -half_w - 1), min(pts[:, 1].max() + 3, half_w + 1)

    fig = plt.figure(figsize=(12, 6.6), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    box = (0.04, 0.1, 0.92, 0.72)                                   # axes rect in figure fractions
    ax = fig.add_axes(box)
    # Equal yards on both axes: pad the shorter span so the play fills the frame.
    target = (12 * box[2]) / (6.6 * box[3])
    if (x_hi - x_lo) / (y_hi - y_lo) < target:
        pad = ((y_hi - y_lo) * target - (x_hi - x_lo)) / 2
        x_lo, x_hi = x_lo - pad, x_hi + pad
    else:
        pad = ((x_hi - x_lo) / target - (y_hi - y_lo)) / 2
        y_lo, y_hi = y_lo - pad, y_hi + pad
    ax.set_facecolor(SURFACE)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect('equal')
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel('yards past the line of scrimmage', color=INK_2, fontsize=9)
    ax.set_yticks([])

    for x in np.arange(np.ceil(x_lo / 5) * 5, x_hi, 5):
        ax.axvline(x, color=RULE, lw=0.8, zorder=0)
    for y in (-half_w, half_w):
        if y_lo < y < y_hi:
            ax.axhline(y, color=INK_2, lw=1.2, zorder=0)
    top = ax.get_xaxis_transform()                                  # x in yards, y in axes fractions
    ax.axvline(0, color=INK_2, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(0, 1.01, 'LOS', transform=top, color=INK_2, fontsize=8, ha='center', va='bottom')
    if first_down is not None and x_lo < first_down < x_hi:
        ax.axvline(first_down, color=INK_2, lw=1.2, ls=(0, (1, 2)), zorder=1)
        ax.text(first_down, 1.01, '1st down', transform=top, color=INK_2, fontsize=8, ha='center', va='bottom')

    fig.text(0.04, 0.955, title, color=INK, fontsize=13, weight='bold', ha='left')
    fig.text(0.04, 0.92, subtitle, color=INK_2, fontsize=9.5, ha='left')
    clock = fig.text(0.96, 0.025, '', color=INK_2, fontsize=9, ha='right')

    # Trails first, so markers sit on top of them.
    real_trails = [ax.plot([], [], color=DEFENSE, lw=1.0, alpha=0.45, zorder=2)[0] for _ in range(11)]
    ghost_trails = [ax.plot([], [], color=GHOST, lw=1.0, alpha=0.6, ls=(0, (3, 2)), zorder=2)[0]
                    for _ in range(11)]
    cloud = ax.scatter([], [], s=10, color=GHOST, alpha=0.22, lw=0, zorder=3)
    ghost_mean = ax.scatter([], [], s=90, facecolor='none', edgecolor=GHOST, lw=2.0, zorder=5)
    defense = ax.scatter([], [], s=90, color=DEFENSE, edgecolor=SURFACE, lw=1.5, zorder=6)
    offense = ax.scatter([], [], s=90, color=OFFENSE, edgecolor=SURFACE, lw=1.5, zorder=6)
    ball = ax.scatter([], [], s=45, marker='D', color=INK_2, edgecolor=SURFACE, lw=1.0, zorder=7)
    names = {i: ax.text(0, 0, text, color=INK, fontsize=8, zorder=8) for i, text in labels.items()}

    handles = [Line2D([], [], ls='', marker='o', ms=9, color=OFFENSE, label='offense'),
               Line2D([], [], ls='', marker='o', ms=9, color=DEFENSE, label='real defense'),
               Line2D([], [], ls='', marker='o', ms=9, mfc='none', mec=GHOST, mew=2, label='ghost (mean)'),
               Line2D([], [], ls='', marker='o', ms=4, color=GHOST, alpha=0.5, label='ghost samples')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.035, 0.905), ncol=4, frameon=False,
               fontsize=9, labelcolor=INK_2, handletextpad=0.3, columnspacing=1.2)

    real_def = dyn[:, DEF]
    mean = ghost.mean(axis=0)                                                        # [T, 11, 2]
    empty = np.empty((0, 2))

    def draw(t: int):
        offense.set_offsets(dyn[t, OFF])
        defense.set_offsets(real_def[t])
        ball.set_offsets(dyn[t, BALL][None])
        predicted = t >= obs_frames
        cloud.set_offsets(ghost[:, t].reshape(-1, 2) if predicted else empty)
        ghost_mean.set_offsets(mean[t] if predicted else empty)
        # The ghost's path starts from the last frame it observed.
        start = obs_frames - 1
        for j in range(11):
            real_trails[j].set_data(real_def[:t + 1, j, 0], real_def[:t + 1, j, 1])
            g = mean[start:t + 1, j] if predicted else empty
            ghost_trails[j].set_data(np.r_[real_def[start:start + 1, j, 0], g[1:, 0]] if predicted else [],
                                     np.r_[real_def[start:start + 1, j, 1], g[1:, 1]] if predicted else [])
        for i, txt in names.items():
            txt.set_position(dyn[t, i] + np.array([0.7, 0.7]))
        phase = 'observed' if not predicted else 'ghost predicting'
        clock.set_text(f'frame {t + 1}/{T}  ·  {t / fps:.1f} s after snap  ·  {phase}')
        return [offense, defense, ball, cloud, ghost_mean, clock, *real_trails, *ghost_trails, *names.values()]

    anim = FuncAnimation(fig, draw, frames=T, interval=1000 / fps, blit=False)
    if out is None:
        plt.show()
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        anim.save(out, writer=PillowWriter(fps=fps))
        plt.close(fig)
    return anim


def animate(runs_dir: Path, out: Path | None, game: int | None = None, play: int | None = None,
            n_samples: int = 16, actual_call: bool = False, players_csvs: list[Path] | None = None,
            seed: int | None = None, device: str = 'cpu', show: bool = False) -> Path | None:
    ckpts = sorted(runs_dir.glob('fold*/model.pt'))
    if not ckpts:
        raise FileNotFoundError(f'no fold*/model.pt checkpoints under {runs_dir}')
    data_path, fingerprint = check_checkpoints(ckpts)
    data = PlayData.load(data_path)
    if data.fingerprint() != fingerprint:
        raise ValueError(f'{data_path} has changed since training (play order differs)')

    rng = np.random.default_rng(seed)
    idx = select_play(data.game_id, data.play_id, game, play, rng)
    game_id, play_id = int(data.game_id[idx]), int(data.play_id[idx])
    ckpt_path = holdout_checkpoint(ckpts, data.game_id, idx)
    model, _ = load_model(ckpt_path, pick_device(device))
    print(f'game {game_id} play {play_id} (index {idx}), ghost from {ckpt_path.parent.name}')

    result = ghost_play(model, data, idx, n_samples, actual_call, seed=int(rng.integers(2 ** 62)))

    names = player_names(players_csvs)
    receiver_ids = {GROUP_VOCAB.index(g) for g in RECEIVER_GROUPS}
    labels = {}
    for i in range(OFF.stop):
        if result['group'][i] in receiver_ids:
            nfl_id = int(data.nfl_id[idx][i])
            name = str(names.get(nfl_id, nfl_id)).split(' ')[-1]
            labels[i] = f'{name}  H {result["H"][i]:+.2f}'

    glob_names = data.meta['global_features']
    first_down = float(data.glob[idx, glob_names.index('yards_to_go')]) * 10.0
    call = data.meta['coverage_vocab'][int(data.coverage[idx])]
    ghost_call = f'actual call ({call})' if actual_call else 'calls drawn from the prior'
    title = f'Game {game_id} · play {play_id}'
    subtitle = (f'Real call: {call} · ghost: {ckpt_path.parent.name}, {n_samples} samples, {ghost_call} · '
                f'H > 0: real defense tighter than the ghost')

    if not show and out is None:
        out = runs_dir / 'animations' / f'{game_id}_{play_id}{"_actual_call" if actual_call else ""}.gif'
    render(result, model.cfg.obs_frames, title, subtitle, labels, first_down, None if show else out)
    if not show:
        print(f'saved {out}')
    return None if show else out
