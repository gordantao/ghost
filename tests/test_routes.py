import numpy as np
import pytest
import pandas as pd
import torch

from ghost.data import PlayData
from ghost.preprocess import N_ENT, OFF
from ghost.routes import check_label_coverage, load_route_labels, route_flags, rule_ran_route
from ghost.score import aggregate_players, receiver_rows


def _tracks(ends, starts=None, T=6):
    """One play: offensive player i moves in a straight line from starts[i] to ends[i] (x, y)."""
    ends = torch.tensor(ends, dtype=torch.float32)
    starts = torch.tensor(starts, dtype=torch.float32) if starts is not None else torch.zeros_like(ends)
    dyn = torch.zeros(1, T, N_ENT, 7)
    w = torch.linspace(0, 1, T)[:, None, None]
    dyn[0, :, OFF, :2] = starts[None] * (1 - w) + ends[None] * w
    return dyn, torch.ones(1, T, dtype=torch.bool)


def test_rule_flags_blockers_and_route_runners():
    ends = [[8, 0],     # deep route
            [-1, 3],    # short route near the line (ends within 2 yards of it)
            [-4.5, 0],  # RB held in the backfield
            [-3.5, 1],  # TE blocking just behind the line
            [-2, 0]] + [[-4, 0]] * 6
    starts = [[0, 0], [-1, 0], [-5, 0], [-1, 0], [-1, 0]] + [[-4, 0]] * 6
    dyn, mask = _tracks(ends, starts)
    ran = rule_ran_route(dyn, mask)[0]
    assert ran[:5].tolist() == [True, True, False, False, True]   # [4] sits on the -2 boundary
    assert not ran[5:].any()


def test_rule_uses_the_last_valid_frame_not_padding():
    dyn, mask = _tracks([[10, 0]] * 11, [[-8, 0]] * 11)
    mask[0, 2:] = False      # window ends after 2 of 6 frames: x = -4.4, 3.6 yards from the start
    dyn[0, 2:] = 0.0
    assert not rule_ran_route(dyn, mask)[0].any()   # zero padding (x = 0) must not read as arriving at the line


def test_labels_read_from_player_play_and_missing_is_not_a_route(tmp_path):
    (tmp_path / 'raw').mkdir()
    pd.DataFrame({'gameId': [1, 1, 1], 'playId': [10, 10, 10], 'nflId': [100, 101, 102],
                  'wasRunningRoute': [1.0, np.nan, 0.0]}).to_csv(tmp_path / 'raw' / 'player_play.csv', index=False)
    labels = load_route_labels([tmp_path / 'raw', tmp_path / 'nowhere'])
    assert labels.set_index('nfl_id').ran_route.to_dict() == {100: True, 101: False, 102: False}
    assert load_route_labels([tmp_path / 'nowhere']) is None

    # player 103 is absent from labelled play 10 -> not a route; play 11 isn't labelled -> rule
    game, play = np.array([1, 1, 1, 1]), np.array([10, 10, 10, 11])
    ran, source = route_flags(game, play, np.array([100, 101, 103, 100]), np.array([False, True, True, True]), labels)
    assert ran.tolist() == [True, False, False, True]
    assert source.tolist() == ['label', 'label', 'label', 'rule']


def test_label_file_without_the_column_is_ignored(tmp_path):
    pd.DataFrame({'gameId': [1], 'playId': [1], 'nflId': [1]}).to_csv(tmp_path / 'player_play.csv', index=False)
    assert load_route_labels([tmp_path]) is None


def _data(labels):
    nfl = np.tile(np.arange(100, 122), (1, 1))
    return PlayData(dyn=None, length=None, group=None, glob=None, coverage=None, nfl_id=nfl,
                    game_id=np.array([1]), play_id=np.array([10]),
                    meta={'coverage_vocab': ['Cover-0', 'Cover-1']}, route_labels=labels)


def test_receiver_rows_carry_route_columns():
    dyn, mask = _tracks([[8, 0]] + [[-4, 0]] * 10, [[0, 0]] + [[-4, 0]] * 10)
    group = torch.full((1, N_ENT), 3)                 # every offensive player a WR (vocab index 3)
    batch = {'dyn': dyn, 'time_mask': mask, 'group': group, 'coverage': torch.tensor([1])}
    rows = receiver_rows(_data(None), np.array([0]), batch, {'H': np.zeros((1, 11))})
    assert rows.ran_route.tolist() == [True] + [False] * 10
    assert set(rows.route_source) == {'rule'}

    labels = pd.DataFrame({'game_id': [1], 'play_id': [10], 'nfl_id': [101], 'ran_route': [True]})
    rows = receiver_rows(_data(labels), np.array([0]), batch, {'H': np.zeros((1, 11))})
    assert rows.ran_route.sum() == 1 and rows.set_index('nfl_id').ran_route[101]
    assert set(rows.route_source) == {'label'}


def _plays():
    p = pd.DataFrame({'nfl_id': [1, 1, 1, 2], 'group': 'RB', 'H': [1.0, 1.0, 9.0, 0.5],
                      'ran_route': [True, True, False, False], 'expected_sep': 4.0,
                      'expected_sep_within': 3.5, 'actual_sep': 2.0})
    p['H_within'] = p.H
    p['H_call'] = 0.0
    return p


def test_aggregate_players_drops_blockers_by_default():
    stats = aggregate_players(_plays(), None).set_index('nfl_id')
    assert list(stats.index) == [1]                      # player 2 only blocked
    assert stats.loc[1, 'n_plays'] == 2 and stats.loc[1, 'H_mean'] == 1.0


def test_aggregate_players_include_blockers_keeps_every_play():
    stats = aggregate_players(_plays(), None, include_blockers=True).set_index('nfl_id')
    assert stats.loc[1, 'n_plays'] == 3 and stats.loc[1, 'H_mean'] == 11.0 / 3
    assert 2 in stats.index


def test_missing_label_for_labelled_season_raises(tmp_path):
    game, play = np.array([1, 2]), np.array([10, 20])
    season = np.array([2021, 2022])
    labels = pd.DataFrame({'game_id': [1], 'play_id': [10], 'nfl_id': [5], 'ran_route': [True]})
    with pytest.raises(FileNotFoundError, match='1 plays'):
        check_label_coverage(season, game, play, labels, ['data/raw/bdb2025'])
    with pytest.raises(FileNotFoundError):
        check_label_coverage(season, game, play, None, ['data/raw/bdb2025'])
    # 2021 plays need no label; old caches without a season are skipped
    check_label_coverage(np.array([2021, 2021]), game, play, None, [])
    check_label_coverage(None, game, play, None, [])


def test_relative_sources_resolve_against_base(tmp_path):
    raw = tmp_path / 'data' / 'raw' / 'bdb2025'
    raw.mkdir(parents=True)
    pd.DataFrame({'gameId': [1], 'playId': [1], 'nflId': [1], 'wasRunningRoute': [1]}).to_csv(
        raw / 'player_play.csv', index=False)
    assert load_route_labels(['data/raw/bdb2025'], base=tmp_path) is not None


def test_cli_passes_include_blockers(monkeypatch):
    import ghost.score
    import ghost.cli
    seen = {}
    monkeypatch.setattr(ghost.score, 'score', lambda *a, **k: seen.update(k))
    ghost.cli.main(['score'])
    assert seen['include_blockers'] is False
    ghost.cli.main(['score', '--include-blockers'])
    assert seen['include_blockers'] is True
