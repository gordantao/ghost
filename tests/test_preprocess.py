import numpy as np
import pandas as pd
import pytest

from ghost.preprocess import BALL, DEF, FIELD_LENGTH, FIELD_WIDTH, OFF, process_week, standardize


def test_standardize_left_play_mirrors_right_play():
    right = pd.DataFrame({'playDirection': ['right'], 'x': [30.0], 'y': [10.0],
                          's': [4.0], 'dir': [60.0], 'o': [90.0]})
    left = pd.DataFrame({'playDirection': ['left'], 'x': [FIELD_LENGTH - 30.0], 'y': [FIELD_WIDTH - 10.0],
                         's': [4.0], 'dir': [240.0], 'o': [270.0]})
    cols = ['x', 'y', 'vx', 'vy', 'o_sin', 'o_cos']
    np.testing.assert_allclose(standardize(right)[cols].to_numpy(), standardize(left)[cols].to_numpy(), atol=1e-9)
    assert standardize(right).vx.iloc[0] > 0  # 60 deg points mostly toward +x


def _synthetic_week(tmp_path, n_frames=14, snap_frame=3, drop_player=False):
    rows = []
    for frame in range(1, n_frames + 1):
        event = 'ball_snap' if frame == snap_frame else 'None'
        for team, ids in (('AAA', range(100, 111)), ('BBB', range(200, 211))):
            for i, nfl_id in enumerate(ids):
                if drop_player and nfl_id == 205 and frame == 8:
                    continue
                rows.append({'frameId': frame, 'event': event, 'nflId': nfl_id, 'team': team,
                             'x': 80.0 - i, 'y': 20.0 + i})
        rows.append({'frameId': frame, 'event': event, 'nflId': np.nan, 'team': 'football',
                     'x': 75.0 - frame, 'y': 26.0})
    df = pd.DataFrame(rows).assign(gameId=1, playId=7, playDirection='left',
                                   s=1.0, a=0.5, dis=0.1, o=10.0, dir=20.0)
    df.loc[df.team == 'football', ['o', 'dir']] = np.nan
    path = tmp_path / 'week1.csv'
    df.to_csv(path, index=False)
    plays = pd.DataFrame({'gameId': [1], 'playId': [7], 'possessionTeam': ['AAA']})
    return path, plays


def test_process_week_layout(tmp_path):
    path, plays = _synthetic_week(tmp_path)
    [rec] = process_week(path, plays, pd.Series(dtype=str), max_frames=50, min_frames=5)
    dyn = rec['dyn']
    assert dyn.shape == (12, 23, 7)  # frames from the snap (3) through 14
    assert list(rec['nfl_id'][OFF]) == list(range(100, 111))
    assert list(rec['nfl_id'][DEF]) == list(range(200, 211))
    assert dyn[0, BALL, 0] == pytest.approx(0.0)  # x measured from the ball at the snap
    # Left play: the ball moving toward smaller raw x is moving toward +x after flipping.
    assert dyn[1:, BALL, 2] == pytest.approx(10.0)
    assert np.isfinite(dyn).all()


def test_process_week_drops_incomplete_plays(tmp_path):
    path, plays = _synthetic_week(tmp_path, drop_player=True)
    assert process_week(path, plays, pd.Series(dtype=str), max_frames=50, min_frames=5) == []
