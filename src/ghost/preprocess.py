"""Turn raw BDB tracking CSVs (2023 and 2025 releases) into a fixed-shape array cache.

Every kept play has exactly 11 offensive players, 11 defenders and the ball in
every frame, so entities are never padded; only time is. Entity order is fixed:
offense (sorted by nflId) in slots 0-10, defense (sorted by nflId) in 11-21,
ball in 22.

Coordinates are standardized so the offense always drives toward +x:
x is yards past the line of scrimmage (ball x at the snap), y is yards from the
field's center line. Frame 0 is the snap and the last frame is the throw (or sack).

The releases differ only in file names, a few column spellings and which plays
they include. A `Source` spec absorbs those differences when files are read;
everything after the readers works on canonical (2025-spelled) columns.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
# 2025 splits some calls into variants; fold them back into the base call.
COVERAGE_ALIASES = {
    'Cover 6-Left': 'Cover-6', 'Cover-6 Right': 'Cover-6',
    'Cover-3 Seam': 'Cover-3', 'Cover-3 Cloud Left': 'Cover-3', 'Cover-3 Cloud Right': 'Cover-3',
    'Cover-3 Double Cloud': 'Cover-3', 'Cover-1 Double': 'Cover-1',
}
SNAP_EVENTS = ('ball_snap', 'autoevent_ballsnap')
# The model covers route development, so a play ends when the ball leaves the QB's hands.
END_EVENTS = ('pass_forward', 'autoevent_passforward', 'pass_shovel', 'qb_sack', 'qb_strip_sack')
RUN_EVENTS = ('handoff', 'run')
NON_PASS_TYPES = ('SCRAMBLE', 'SCRAMBLE_ROLLOUT_LEFT', 'SCRAMBLE_ROLLOUT_RIGHT', 'DESIGNED_RUN', 'QB_SNEAK')

PLAY_COLS = ['gameId', 'playId', 'possessionTeam', 'quarter', 'gameClock', 'down', 'yardsToGo',
             'preSnapHomeScore', 'preSnapVisitorScore', 'passResult', 'dropbackType', 'pff_passCoverage']
# Present only in some releases; missing ones are filled with NaN.
OPTIONAL_PLAY_COLS = ['isDropback', 'timeToThrow', 'timeToSack']
TRACK_COLS = ['gameId', 'playId', 'nflId', 'frameId', 'frameType', 'club', 'playDirection',
              'x', 'y', 's', 'a', 'o', 'dir', 'event']
N_QUARTERS = 5  # quarter 5 is overtime


@dataclass(frozen=True)
class Source:
    """How one BDB release lays out its files, relative to the canonical columns."""
    name: str
    tracking_glob: str
    renames: dict[str, str]        # raw column -> canonical column
    drop_flags: tuple[str, ...]    # plays.csv columns whose truthy rows are dropped


SOURCES = (
    Source('bdb2023', 'week*.csv',
           {'dropBackType': 'dropbackType', 'officialPosition': 'position', 'team': 'club'}, ()),
    Source('bdb2025', 'tracking_week_*.csv', {},
           ('playNullifiedByPenalty', 'qbSpike', 'qbKneel', 'qbSneak')),
)


def detect_source(raw: Path) -> Source:
    """The release whose tracking files are in `raw`."""
    matches = [s for s in SOURCES if any(raw.glob(s.tracking_glob))]
    if len(matches) != 1:
        found = [s.name for s in matches] or 'no tracking files'
        raise ValueError(f'cannot tell which BDB release {raw} holds ({found})')
    return matches[0]


def tracking_files(raw: Path, src: Source) -> list[Path]:
    return sorted(raw.glob(src.tracking_glob), key=lambda p: int(re.findall(r'\d+', p.stem)[-1]))


def _read(path: Path, src: Source, wanted: list[str], **kwargs) -> pd.DataFrame | pd.io.parsers.TextFileReader:
    """Read the raw columns that become `wanted` canonical columns (missing ones are skipped)."""
    raw_name = {v: k for k, v in src.renames.items()}
    header = set(pd.read_csv(path, nrows=0).columns)
    usecols = [raw_name.get(c, c) for c in wanted if raw_name.get(c, c) in header]
    return pd.read_csv(path, usecols=usecols, **kwargs)


def _truthy(col: pd.Series) -> pd.Series:
    return col.astype(str).str.upper().isin(('Y', 'TRUE', '1'))


def load_plays(raw: Path, src: Source) -> pd.DataFrame:
    """Dropback pass plays (sacks included) with play context.

    Run plays, QB runs, sneaks, spikes, kneels and plays nullified by penalty
    are dropped. So are scrambles: once the play breaks down the defense is in
    scramble-drill rules, not its called coverage.
    """
    plays = _read(raw / 'plays.csv', src, PLAY_COLS + OPTIONAL_PLAY_COLS + list(src.drop_flags))
    plays = plays.rename(columns=src.renames)
    for col in OPTIONAL_PLAY_COLS:
        if col not in plays:
            plays[col] = np.nan

    n = len(plays)
    flagged = np.zeros(n, dtype=bool)
    for col in src.drop_flags:
        flagged |= _truthy(plays[col]).to_numpy()
    dropback = (plays.passResult.notna() & plays.dropbackType.notna()
                & (plays.isDropback.isna() | _truthy(plays.isDropback))).to_numpy()
    passing = (~plays.dropbackType.isin(NON_PASS_TYPES) & plays.passResult.ne('R')).to_numpy()
    print(f'{src.name}: {n} plays; dropped {flagged.sum()} flagged, {(~flagged & ~dropback).sum()} not dropbacks, '
          f'{(~flagged & dropback & ~passing).sum()} scrambles/QB runs')
    plays = plays[~flagged & dropback & passing].reset_index(drop=True)

    # The score is stored home/visitor; the model's frame is the offense.
    games = pd.read_csv(raw / 'games.csv', usecols=['gameId', 'homeTeamAbbr'])
    home = plays.merge(games, on='gameId', how='left').homeTeamAbbr.eq(plays.possessionTeam).to_numpy()
    plays['offenseScore'] = np.where(home, plays.preSnapHomeScore, plays.preSnapVisitorScore)
    plays['defenseScore'] = np.where(home, plays.preSnapVisitorScore, plays.preSnapHomeScore)
    return plays


def load_player_groups(raw: Path, src: Source) -> pd.Series:
    players = _read(raw / 'players.csv', src, ['nflId', 'position']).rename(columns=src.renames)
    return players.set_index('nflId').position.map(POSITION_GROUPS).fillna('UNK')


def read_tracking(path: Path, src: Source, plays: pd.DataFrame, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Tracking rows of the kept plays, read in chunks to bound memory.

    Players' pre-snap frames are skipped where the release marks them
    (`frameType`); the ball's are kept because its velocity at the snap is
    differenced from the frame before.
    """
    keys = pd.MultiIndex.from_frame(plays[['gameId', 'playId']])
    floats = {c: np.float32 for c in ('x', 'y', 's', 'a', 'o', 'dir')}
    parts = []
    for chunk in _read(path, src, TRACK_COLS, chunksize=chunksize, dtype=floats):
        chunk = chunk.rename(columns=src.renames)
        keep = pd.MultiIndex.from_frame(chunk[['gameId', 'playId']]).isin(keys)
        if 'frameType' in chunk:
            keep &= chunk.frameType.ne('BEFORE_SNAP').to_numpy() | chunk.nflId.isna().to_numpy()
        parts.append(chunk[keep])
    track = pd.concat(parts, ignore_index=True)
    if 'frameType' not in track:
        track['frameType'] = np.nan
    return track


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


def play_window(track: pd.DataFrame, plays: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Snap frame and throw frame per play, plus counts of plays dropped and why.

    The snap is the first snap event (or the `frameType == 'SNAP'` frame). The
    end is the first END_EVENTS frame at or after it; without one, it is placed
    from plays.csv `timeToThrow` (`timeToSack` for sacks) when available.
    Plays with a handoff or run before the end are dropped.
    """
    key = ['gameId', 'playId']
    n_plays = track[key].drop_duplicates().shape[0]
    snap = (track[track.event.isin(SNAP_EVENTS)].groupby(key).frameId.min()
            .combine_first(track[track.frameType.eq('SNAP')].groupby(key).frameId.min()).rename('snap'))
    win = snap.reset_index().merge(plays[key + ['passResult', 'timeToThrow', 'timeToSack']], on=key)
    ev = track[track.event.notna()][key + ['frameId', 'event']].merge(win[key + ['snap']], on=key)
    ev = ev[ev.frameId >= ev.snap]
    end = ev[ev.event.isin(END_EVENTS)].groupby(key).frameId.min().rename('end')
    win = win.merge(end, on=key, how='left')
    timed = np.where(win.passResult.eq('S'), win.timeToSack, win.timeToThrow)
    from_timing = win.end.isna() & np.isfinite(timed)
    win.loc[from_timing, 'end'] = win.snap[from_timing] + np.round(timed[from_timing] * FPS)
    no_end = win.end.isna()
    win = win[~no_end]

    runs = ev[ev.event.isin(RUN_EVENTS)].merge(win[key + ['end']], on=key)
    run_keys = runs[runs.frameId <= runs.end][key].drop_duplicates()
    is_run = pd.MultiIndex.from_frame(win[key]).isin(pd.MultiIndex.from_frame(run_keys))
    win = win[~is_run]
    counts = {'no_snap': n_plays - len(snap), 'end_from_timing': int(from_timing.sum()),
              'no_end': int(no_end.sum()), 'run_event': int(is_run.sum())}
    return win[key + ['snap', 'end']].astype(np.int64), counts


def process_week(track: pd.DataFrame, plays: pd.DataFrame, player_groups: pd.Series,
                 max_frames: int, min_frames: int) -> tuple[list[dict], dict[str, int]]:
    """Fixed-shape records for one week of canonical tracking, plus drop counts."""
    win, counts = play_window(track, plays)
    too_long = (win.end - win.snap) >= max_frames
    counts['throw_after_max_frames'] = int(too_long.sum())
    win = win[~too_long]

    track = track.merge(plays[['gameId', 'playId', 'possessionTeam']], on=['gameId', 'playId'])
    track = track.merge(win, on=['gameId', 'playId'])
    track['t'] = track.frameId - track.snap
    # Keep one pre-snap frame: ball velocity at the snap is differenced from it.
    track = track[(track.t >= -1) & (track.frameId <= track.end)]

    track = track.sort_values(['gameId', 'playId', 'nflId', 'frameId'])
    track = _ball_velocity(standardize(track))
    track = track[track.t >= 0]

    track['side'] = np.where(track.nflId.isna(), SIDE_BALL,
                             np.where(track.club == track.possessionTeam, SIDE_OFFENSE, SIDE_DEFENSE))
    track['nflId'] = track.nflId.fillna(-1).astype(np.int64)
    track = track.sort_values(['gameId', 'playId', 't', 'side', 'nflId'])

    records = []
    counts['bad_shape'] = 0
    for (game_id, play_id), play in track.groupby(['gameId', 'playId'], sort=False):
        n_frames = play.t.max() + 1
        if n_frames < min_frames or len(play) != n_frames * N_ENT:
            counts['bad_shape'] += 1
            continue
        sides = play.side.to_numpy().reshape(n_frames, N_ENT)
        ids = play.nflId.to_numpy().reshape(n_frames, N_ENT)
        if not ((sides == SIDES).all() and (ids == ids[0]).all()):
            counts['bad_shape'] += 1
            continue  # not a clean 11v11 + ball in every frame

        dyn = play[DYN_FEATURES].to_numpy(np.float32).reshape(n_frames, N_ENT, len(DYN_FEATURES))
        los_x = dyn[0, BALL, 0]
        dyn[..., 0] -= los_x
        dyn[..., 1] -= FIELD_WIDTH / 2
        if not np.isfinite(dyn).all():
            counts['bad_shape'] += 1
            continue

        groups = [player_groups.get(i, 'UNK') for i in ids[0, :BALL]] + ['BALL']
        records.append({
            'game_id': game_id, 'play_id': play_id, 'dyn': dyn,
            'nfl_id': ids[0, :BALL], 'group': [GROUP_VOCAB.index(g) for g in groups],
            'yards_to_endzone': 110.0 - los_x,
        })
    return records, counts


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
    """PFF coverage call -> index into COVERAGE_VOCAB (variants folded in; missing and rare calls -> 'Other')."""
    other = COVERAGE_VOCAB.index('Other')
    vocab = {c: i for i, c in enumerate(COVERAGE_VOCAB)}
    return coverage.replace(COVERAGE_ALIASES).map(vocab).fillna(other).to_numpy(np.int64)


def preprocess_source(raw: Path, max_frames: int, min_frames: int) -> tuple[list[dict], pd.DataFrame]:
    """Records and their play context (one row per record, same order) for one release."""
    src = detect_source(raw)
    plays = load_plays(raw, src)
    player_groups = load_player_groups(raw, src)

    records = []
    for path in tracking_files(raw, src):
        week_records, counts = process_week(read_tracking(path, src, plays), plays, player_groups,
                                            max_frames, min_frames)
        dropped = ', '.join(f'{k} {v}' for k, v in counts.items() if v)
        print(f'{src.name} {path.name}: {len(week_records)} plays' + (f' ({dropped})' if dropped else ''))
        records += week_records

    keys = pd.DataFrame({'gameId': [r['game_id'] for r in records],
                         'playId': [r['play_id'] for r in records]})
    play_ctx = keys.merge(plays, on=['gameId', 'playId'], how='left')
    seasons = pd.read_csv(raw / 'games.csv', usecols=['gameId', 'season'])
    play_ctx = play_ctx.merge(seasons, on='gameId', how='left')
    return records, play_ctx


def preprocess(raws: list[Path], out: Path, max_frames: int = 70, min_frames: int = 10) -> None:
    records, ctx = [], []
    for raw in raws:
        r, c = preprocess_source(raw, max_frames, min_frames)
        records += r
        ctx.append(c)
    play_ctx = pd.concat(ctx, ignore_index=True)
    if play_ctx.duplicated(['gameId', 'playId']).any():
        raise ValueError('the same play appears in more than one raw directory')
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
        game_id=play_ctx.gameId.to_numpy(), play_id=play_ctx.playId.to_numpy(),
        season=play_ctx.season.to_numpy(np.int64),
    )
    out.with_suffix('.json').write_text(json.dumps({
        'dyn_features': DYN_FEATURES, 'global_features': glob_names,
        'group_vocab': GROUP_VOCAB, 'coverage_vocab': COVERAGE_VOCAB,
        'max_frames': max_frames, 'n_plays': n, 'sources': [str(r) for r in raws],
    }, indent=2))
    print(f'Saved {n} plays to {out}')
