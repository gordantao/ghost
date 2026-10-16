"""Turn raw BDB 2023 tracking CSVs into a fixed-shape array cache.

Every kept play has exactly 11 offensive players, 11 defenders and the ball in
every frame, so entities are never padded; only time is. Entity order is fixed:
offense (sorted by nflId) in slots 0-10, defense (sorted by nflId) in 11-21,
ball in 22.

Coordinates are standardized so the offense always drives toward +x:
x is yards past the line of scrimmage (ball x at the snap), y is yards from the
field's center line. Frame 0 is the snap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3
FPS = 10.0

N_OFF = N_DEF = 11
N_ENT = N_OFF + N_DEF + 1
OFF = slice(0, N_OFF)
DEF = slice(N_OFF, N_OFF + N_DEF)
BALL = N_ENT - 1

SIDE_OFFENSE, SIDE_DEFENSE, SIDE_BALL = 0, 1, 2
SIDES = np.array([SIDE_OFFENSE] * N_OFF + [SIDE_DEFENSE] * N_DEF + [SIDE_BALL])

POSITION_GROUPS = {
    'QB': 'QB', 'RB': 'RB', 'FB': 'RB', 'WR': 'WR', 'TE': 'TE',
    'T': 'OL', 'G': 'OL', 'C': 'OL',
    'DE': 'DL', 'DT': 'DL', 'NT': 'DL',
    'OLB': 'LB', 'ILB': 'LB', 'MLB': 'LB', 'LB': 'LB',
    'CB': 'DB', 'FS': 'DB', 'SS': 'DB', 'DB': 'DB',
}
GROUP_VOCAB = ['UNK', 'QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'DB', 'BALL']
RECEIVER_GROUPS = ('WR', 'TE', 'RB')

DYN_FEATURES = ['x', 'y', 'vx', 'vy', 'a', 'o_sin', 'o_cos']
# The PFF coverage call is a training target (the model's discrete latent), never an
# input: it is labelled after the play from what the defense did. Rare calls are
# pooled into 'Other'.
COVERAGE_VOCAB = ['Cover-0', 'Cover-1', 'Cover-2', 'Cover-3', 'Quarters', 'Cover-6',
                  '2-Man', 'Red Zone', 'Other']
SNAP_EVENTS = ('ball_snap', 'autoevent_ballsnap')
SCRAMBLE_TYPES = ('SCRAMBLE', 'SCRAMBLE_ROLLOUT_LEFT', 'SCRAMBLE_ROLLOUT_RIGHT')

PLAY_COLS = ['gameId', 'playId', 'possessionTeam', 'quarter', 'gameClock', 'down', 'yardsToGo',
             'preSnapHomeScore', 'preSnapVisitorScore', 'passResult', 'dropBackType', 'pff_passCoverage']
N_QUARTERS = 5  # quarter 5 is overtime


def load_plays(raw: Path) -> pd.DataFrame:
    """Dropback pass plays (sacks included) with play context.

    Scrambles are dropped: once the play breaks down the defense is in
    scramble-drill rules, not its called coverage. Besides the SCRAMBLE*
    dropback types, QB runs (passResult 'R') are dropped; the few left after
    that filter are scrambles labelled DESIGNED_RUN or UNKNOWN.
    """
    plays = pd.read_csv(raw / 'plays.csv', usecols=PLAY_COLS)
    keep = (plays.passResult.notna() & plays.dropBackType.notna()
            & ~plays.dropBackType.isin(SCRAMBLE_TYPES) & plays.passResult.ne('R'))
    plays = plays[keep].reset_index(drop=True)

    # The score is stored home/visitor; the model's frame is the offense.
    games = pd.read_csv(raw / 'games.csv', usecols=['gameId', 'homeTeamAbbr'])
    home = plays.merge(games, on='gameId', how='left').homeTeamAbbr.eq(plays.possessionTeam).to_numpy()
    plays['offenseScore'] = np.where(home, plays.preSnapHomeScore, plays.preSnapVisitorScore)
    plays['defenseScore'] = np.where(home, plays.preSnapVisitorScore, plays.preSnapHomeScore)
    return plays


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Flip left-moving plays so offense drives toward +x; add vx, vy, o_sin, o_cos.

    Angles follow the NGS convention (0 deg = +y, clockwise), so a 180 deg
    rotation of the field adds 180 to `dir` and `o`.
    """
    out = df.copy()
    left = out.playDirection.eq('left').to_numpy()
    out['x'] = np.where(left, FIELD_LENGTH - out.x, out.x)
    out['y'] = np.where(left, FIELD_WIDTH - out.y, out.y)
    for col in ('dir', 'o'):
        out[col] = np.where(left, (out[col] + 180.0) % 360.0, out[col])

    rad = np.deg2rad(out.dir)
    out['vx'] = out.s * np.sin(rad)
    out['vy'] = out.s * np.cos(rad)
    o_rad = np.deg2rad(out.o)
    out['o_sin'] = np.sin(o_rad).fillna(0.0)
    out['o_cos'] = np.cos(o_rad).fillna(0.0)
    return out


def _ball_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """The ball has no `dir`; derive its velocity from frame-to-frame motion."""
    ball = df.nflId.isna()
    key = [df.gameId, df.playId]
    grp_x = df.x.where(ball).groupby(key)
    grp_y = df.y.where(ball).groupby(key)
    vx = (grp_x.diff().fillna(0.0)) * FPS
    vy = (grp_y.diff().fillna(0.0)) * FPS
    df.loc[ball, 'vx'] = vx[ball]
    df.loc[ball, 'vy'] = vy[ball]
    return df


def process_week(path: Path, plays: pd.DataFrame, player_groups: pd.Series,
                 max_frames: int, min_frames: int) -> list[dict]:
    track = pd.read_csv(path)
    track = track.merge(plays[['gameId', 'playId', 'possessionTeam']], on=['gameId', 'playId'])

    snap = (track[track.event.isin(SNAP_EVENTS)]
            .groupby(['gameId', 'playId']).frameId.min().rename('snapFrame'))
    track = track.merge(snap, on=['gameId', 'playId'])
    track['t'] = track.frameId - track.snapFrame
    # Keep one pre-snap frame: ball velocity at the snap is differenced from it.
    track = track[(track.t >= -1) & (track.t < max_frames)]

    track = track.sort_values(['gameId', 'playId', 'nflId', 'frameId'])
    track = _ball_velocity(standardize(track))
    track = track[track.t >= 0]

    track['side'] = np.where(track.nflId.isna(), SIDE_BALL,
                             np.where(track.team == track.possessionTeam, SIDE_OFFENSE, SIDE_DEFENSE))
    track['nflId'] = track.nflId.fillna(-1).astype(np.int64)
    track = track.sort_values(['gameId', 'playId', 't', 'side', 'nflId'])

    records = []
    for (game_id, play_id), play in track.groupby(['gameId', 'playId'], sort=False):
        n_frames = play.t.max() + 1
        if n_frames < min_frames or len(play) != n_frames * N_ENT:
            continue
        sides = play.side.to_numpy().reshape(n_frames, N_ENT)
        ids = play.nflId.to_numpy().reshape(n_frames, N_ENT)
        if not ((sides == SIDES).all() and (ids == ids[0]).all()):
            continue  # not a clean 11v11 + ball in every frame

        dyn = play[DYN_FEATURES].to_numpy(np.float32).reshape(n_frames, N_ENT, len(DYN_FEATURES))
        los_x = dyn[0, BALL, 0]
        dyn[..., 0] -= los_x
        dyn[..., 1] -= FIELD_WIDTH / 2
        if not np.isfinite(dyn).all():
            continue

        groups = [player_groups.get(i, 'UNK') for i in ids[0, :BALL]] + ['BALL']
        records.append({
            'game_id': game_id, 'play_id': play_id, 'dyn': dyn,
            'nfl_id': ids[0, :BALL], 'group': [GROUP_VOCAB.index(g) for g in groups],
            'yards_to_endzone': 110.0 - los_x,
        })
    return records


def encode_globals(plays: pd.DataFrame, yards_to_endzone: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Down and quarter one-hots, scaled distances, clock and score: situation known before the snap."""
    mins, secs = plays.gameClock.str.split(':', expand=True).astype(np.float32).to_numpy().T
    parts = [np.eye(4, dtype=np.float32)[plays.down.clip(1, 4).to_numpy() - 1],
             (plays.yardsToGo.to_numpy(np.float32) / 10.0)[:, None],
             (yards_to_endzone.astype(np.float32) / 100.0)[:, None],
             np.eye(N_QUARTERS, dtype=np.float32)[plays.quarter.clip(1, N_QUARTERS).to_numpy() - 1],
             ((mins * 60 + secs) / 900.0)[:, None],
             (plays.offenseScore.to_numpy(np.float32) / 28.0)[:, None],
             (plays.defenseScore.to_numpy(np.float32) / 28.0)[:, None]]
    names = ([f'down_{d}' for d in range(1, 5)] + ['yards_to_go', 'yards_to_endzone']
             + [f'quarter_{q}' for q in range(1, N_QUARTERS + 1)]
             + ['quarter_clock', 'offense_score', 'defense_score'])
    return np.concatenate(parts, axis=1), names


def encode_coverage(coverage: pd.Series) -> np.ndarray:
    """PFF coverage call -> index into COVERAGE_VOCAB (missing and rare calls -> 'Other')."""
    other = COVERAGE_VOCAB.index('Other')
    return coverage.map({c: i for i, c in enumerate(COVERAGE_VOCAB)}).fillna(other).to_numpy(np.int64)


def preprocess(raw: Path, out: Path, max_frames: int = 50, min_frames: int = 10) -> None:
    plays = load_plays(raw)
    players = pd.read_csv(raw / 'players.csv')
    player_groups = players.set_index('nflId').officialPosition.map(POSITION_GROUPS).fillna('UNK')

    records = []
    for week in sorted(raw.glob('week*.csv'), key=lambda p: int(p.stem[4:])):
        week_records = process_week(week, plays, player_groups, max_frames, min_frames)
        print(f'{week.name}: {len(week_records)} plays')
        records += week_records

    keys = pd.DataFrame({'gameId': [r['game_id'] for r in records],
                         'playId': [r['play_id'] for r in records]})
    play_ctx = keys.merge(plays, on=['gameId', 'playId'], how='left')
    glob, glob_names = encode_globals(play_ctx, np.array([r['yards_to_endzone'] for r in records]))

    n = len(records)
    dyn = np.zeros((n, max_frames, N_ENT, len(DYN_FEATURES)), dtype=np.float32)
    length = np.zeros(n, dtype=np.int64)
    for i, r in enumerate(records):
        length[i] = len(r['dyn'])
        dyn[i, :length[i]] = r['dyn']

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out, dyn=dyn, length=length, glob=glob, coverage=encode_coverage(play_ctx.pff_passCoverage),
        group=np.array([r['group'] for r in records], dtype=np.int64),
        nfl_id=np.stack([r['nfl_id'] for r in records]),
        game_id=keys.gameId.to_numpy(), play_id=keys.playId.to_numpy(),
    )
    out.with_suffix('.json').write_text(json.dumps({
        'dyn_features': DYN_FEATURES, 'global_features': glob_names,
        'group_vocab': GROUP_VOCAB, 'coverage_vocab': COVERAGE_VOCAB,
        'max_frames': max_frames, 'n_plays': n,
    }, indent=2))
    print(f'Saved {n} plays to {out}')
