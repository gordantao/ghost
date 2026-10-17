import numpy as np
import pandas as pd
import pytest

from ghost.preprocess import (BALL, COVERAGE_VOCAB, DEF, FIELD_LENGTH, FIELD_WIDTH, OFF, SOURCES, detect_source,
                              encode_coverage, encode_globals, load_plays, preprocess_source, process_week,
                              standardize)

SRC = {s.name: s for s in SOURCES}


def test_standardize_left_play_mirrors_right_play():
    right = pd.DataFrame({'playDirection': ['right'], 'x': [30.0], 'y': [10.0],
                          's': [4.0], 'dir': [60.0], 'o': [90.0]})
    left = pd.DataFrame({'playDirection': ['left'], 'x': [FIELD_LENGTH - 30.0], 'y': [FIELD_WIDTH - 10.0],
                         's': [4.0], 'dir': [240.0], 'o': [270.0]})
    cols = ['x', 'y', 'vx', 'vy', 'o_sin', 'o_cos']
    np.testing.assert_allclose(standardize(right)[cols].to_numpy(), standardize(left)[cols].to_numpy(), atol=1e-9)
    assert standardize(right).vx.iloc[0] > 0  # 60 deg points mostly toward +x


def _track(n_frames=18, snap_frame=3, events=None, drop_player=False, snap_event=True):
    """Canonical tracking for one left-moving play; `events` maps frameId -> event (default: throw at 14)."""
    events = {14: 'pass_forward'} if events is None else events
    if snap_event:
        events = {snap_frame: 'ball_snap', **events}
    rows = []
    for frame in range(1, n_frames + 1):
        frame_type = 'BEFORE_SNAP' if frame < snap_frame else 'SNAP' if frame == snap_frame else 'AFTER_SNAP'
        common = {'frameId': frame, 'event': events.get(frame), 'frameType': frame_type}
        for club, ids in (('AAA', range(100, 111)), ('BBB', range(200, 211))):
            for i, nfl_id in enumerate(ids):
                if drop_player and nfl_id == 205 and frame == 8:
                    continue
                rows.append({**common, 'nflId': nfl_id, 'club': club, 'x': 80.0 - i, 'y': 20.0 + i})
        rows.append({**common, 'nflId': np.nan, 'club': 'football', 'x': 75.0 - frame, 'y': 26.0})
    df = pd.DataFrame(rows).assign(gameId=1, playId=7, playDirection='left', s=1.0, a=0.5, o=10.0, dir=20.0)
    df.loc[df.club == 'football', ['o', 'dir']] = np.nan
    return df


def _plays(**cols):
    return pd.DataFrame({'gameId': [1], 'playId': [7], 'possessionTeam': ['AAA'], 'passResult': ['C'],
                         'timeToThrow': [np.nan], 'timeToSack': [np.nan], **cols})


def _run(track, plays=None, max_frames=50):
    return process_week(track, _plays() if plays is None else plays, pd.Series(dtype=str),
                        max_frames=max_frames, min_frames=5)


def test_process_week_layout():
    [rec], _ = _run(_track())
    dyn = rec['dyn']
    assert dyn.shape == (12, 23, 7)  # frames from the snap (3) through the throw (14)
    assert list(rec['nfl_id'][OFF]) == list(range(100, 111))
    assert list(rec['nfl_id'][DEF]) == list(range(200, 211))
    assert dyn[0, BALL, 0] == pytest.approx(0.0)  # x measured from the ball at the snap
    # Left play: the ball moving toward smaller raw x is moving toward +x after flipping,
    # including at the snap, whose velocity comes from the pre-snap frame.
    assert dyn[:, BALL, 2] == pytest.approx(10.0)
    assert np.isfinite(dyn).all()


def test_process_week_drops_incomplete_plays():
    records, counts = _run(_track(drop_player=True))
    assert records == [] and counts['bad_shape'] == 1


def test_sack_ends_the_play():
    [rec], _ = _run(_track(events={9: 'qb_sack', 14: 'pass_forward'}))
    assert len(rec['dyn']) == 7  # snap (3) through the sack (9)


def test_missing_end_event_falls_back_to_time_to_throw():
    [rec], counts = _run(_track(events={}), _plays(timeToThrow=[0.9]))
    assert len(rec['dyn']) == 10 and counts['end_from_timing'] == 1  # 9 frames after the snap
    records, counts = _run(_track(events={}))
    assert records == [] and counts['no_end'] == 1


def test_sack_without_event_uses_time_to_sack():
    [rec], _ = _run(_track(events={}), _plays(passResult=['S'], timeToThrow=[0.5], timeToSack=[1.2]))
    assert len(rec['dyn']) == 13


def test_throw_after_max_frames_is_dropped_not_truncated():
    records, counts = _run(_track(), max_frames=11)  # throw is 11 frames after the snap
    assert records == [] and counts['throw_after_max_frames'] == 1
    assert len(_run(_track(), max_frames=12)[0]) == 1


def test_run_event_before_the_throw_drops_the_play():
    records, counts = _run(_track(events={6: 'handoff', 14: 'pass_forward'}))
    assert records == [] and counts['run_event'] == 1
    assert len(_run(_track(events={14: 'pass_forward', 16: 'run'}))[0]) == 1  # after the throw is fine


def test_snap_falls_back_to_frame_type():
    [rec], _ = _run(_track(snap_event=False))
    assert len(rec['dyn']) == 12


def _write_raw(path, layout, track, plays):
    """Write canonical tables in a release's raw layout."""
    path.mkdir()
    src = SRC[layout]
    to_raw = {v: k for k, v in src.renames.items()}
    if layout == 'bdb2023':
        track = track.drop(columns='frameType').assign(event=track.event.fillna('None'))
        plays = plays.drop(columns=['isDropback', 'timeToThrow', 'timeToSack'] + list(SRC['bdb2025'].drop_flags))
        track.rename(columns=to_raw).to_csv(path / 'week1.csv', index=False)
    else:
        track.to_csv(path / 'tracking_week_1.csv', index=False)
    plays.rename(columns=to_raw).to_csv(path / 'plays.csv', index=False)
    pd.DataFrame({'gameId': [1], 'season': [2022], 'homeTeamAbbr': ['AAA']}).to_csv(path / 'games.csv', index=False)
    players = pd.DataFrame({'nflId': [100, 200], 'position': ['WR', 'CB']})
    players.rename(columns=to_raw).to_csv(path / 'players.csv', index=False)


def _plays_table(n=1, **cols):
    base = {'gameId': 1, 'playId': range(7, 7 + n), 'possessionTeam': 'AAA', 'quarter': 1, 'gameClock': '15:00',
            'down': 1, 'yardsToGo': 10, 'preSnapHomeScore': 0, 'preSnapVisitorScore': 0, 'passResult': 'C',
            'dropbackType': 'TRADITIONAL', 'pff_passCoverage': 'Cover-3', 'isDropback': True,
            'timeToThrow': np.nan, 'timeToSack': np.nan, 'playNullifiedByPenalty': 'N', 'qbSpike': False,
            'qbKneel': 0, 'qbSneak': False}
    return pd.DataFrame({**base, **cols})


def test_both_layouts_give_identical_records(tmp_path):
    out = {}
    for layout in ('bdb2023', 'bdb2025'):
        _write_raw(tmp_path / layout, layout, _track(), _plays_table())
        assert detect_source(tmp_path / layout).name == layout
        out[layout] = preprocess_source(tmp_path / layout, max_frames=50, min_frames=5)
    (r23, ctx23), (r25, ctx25) = out['bdb2023'], out['bdb2025']
    assert len(r23) == len(r25) == 1
    np.testing.assert_array_equal(r23[0]['dyn'], r25[0]['dyn'])
    assert r23[0]['group'] == r25[0]['group']
    assert list(ctx23.season) == list(ctx25.season) == [2022]


def test_detect_source_rejects_empty_or_mixed_dirs(tmp_path):
    with pytest.raises(ValueError):
        detect_source(tmp_path)
    (tmp_path / 'week1.csv').touch()
    (tmp_path / 'tracking_week_1.csv').touch()
    with pytest.raises(ValueError):
        detect_source(tmp_path)


def _write_plays(tmp_path, plays, layout):
    src = SRC[layout]
    plays.rename(columns={v: k for k, v in src.renames.items()}).to_csv(tmp_path / 'plays.csv', index=False)
    pd.DataFrame({'gameId': [1], 'homeTeamAbbr': ['AAA']}).to_csv(tmp_path / 'games.csv', index=False)
    return load_plays(tmp_path, src)


def test_load_plays_drops_scrambles_2023(tmp_path):
    plays = _plays_table(
        8, passResult=['C', 'R', 'I', 'S', None, 'C', 'R', 'I'],
        dropbackType=['TRADITIONAL', 'SCRAMBLE', 'SCRAMBLE_ROLLOUT_LEFT', 'DESIGNED_ROLLOUT_RIGHT',
                      'TRADITIONAL', 'SCRAMBLE_ROLLOUT_RIGHT', 'DESIGNED_RUN', None],
    ).drop(columns=['isDropback', 'timeToThrow', 'timeToSack', 'playNullifiedByPenalty', 'qbSpike',
                    'qbKneel', 'qbSneak'])
    kept = _write_plays(tmp_path, plays, 'bdb2023')
    assert list(kept.playId) == [7, 10]
    _, names = encode_globals(kept, np.zeros(len(kept)))
    assert not any(n.startswith('dropbackType') for n in names)


def test_load_plays_drops_runs_and_flagged_plays_2025(tmp_path):
    plays = _plays_table(
        8,
        passResult=['C', None, 'C', 'I', 'C', 'R', 'C', 'S'],
        isDropback=[True, False, True, True, True, True, True, True],
        dropbackType=['TRADITIONAL', None, 'TRADITIONAL', 'TRADITIONAL', 'QB_SNEAK', 'DESIGNED_RUN',
                      'TRADITIONAL', 'DESIGNED_ROLLOUT_LEFT'],
        playNullifiedByPenalty=['N', 'N', 'Y', 'N', 'N', 'N', 'N', 'N'],
        qbSpike=[False, None, False, True, False, False, False, False],
        qbKneel=[0, 0, 0, 0, 0, 0, 1, 0],
    )
    kept = _write_plays(tmp_path, plays, 'bdb2025')
    assert list(kept.playId) == [7, 14]


def test_globals_put_score_in_offense_frame(tmp_path):
    plays = _plays_table(2, possessionTeam=['AAA', 'BBB'], quarter=[2, 5], gameClock=['01:30', '10:00'],
                         down=3, yardsToGo=5, preSnapHomeScore=14, preSnapVisitorScore=7)
    glob, names = encode_globals(_write_plays(tmp_path, plays, 'bdb2025'), np.zeros(2))
    col = {n: glob[:, i] for i, n in enumerate(names)}
    np.testing.assert_allclose(col['offense_score'] * 28, [14, 7])  # AAA is home, BBB visits
    np.testing.assert_allclose(col['defense_score'] * 28, [7, 14])
    np.testing.assert_allclose(col['quarter_clock'] * 900, [90, 600])
    assert list(col['quarter_2']) == [1, 0] and list(col['quarter_5']) == [0, 1]


def test_encode_coverage_pools_rare_and_missing_calls():
    calls = pd.Series(['Cover-3', 'Bracket', None, '2-Man'])
    other = COVERAGE_VOCAB.index('Other')
    assert list(encode_coverage(calls)) == [COVERAGE_VOCAB.index('Cover-3'), other, other,
                                            COVERAGE_VOCAB.index('2-Man')]


def test_encode_coverage_folds_2025_variants():
    calls = pd.Series(['Cover 6-Left', 'Cover-6 Right', 'Cover-3 Seam', 'Cover-3 Cloud Left',
                       'Cover-3 Double Cloud', 'Cover-1 Double', 'Goal Line'])
    idx = COVERAGE_VOCAB.index
    assert list(encode_coverage(calls)) == [idx('Cover-6'), idx('Cover-6'), idx('Cover-3'), idx('Cover-3'),
                                            idx('Cover-3'), idx('Cover-1'), idx('Other')]
