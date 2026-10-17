from pathlib import Path

import pandas as pd
import pytest
import torch

from ghost.preprocess import DEF
from ghost.score import aggregate_players, check_checkpoints, heliocentricity, player_names


def test_h_is_zero_when_ghosts_match_reality(batch):
    actual = batch['dyn'][:, :, DEF, :2]
    samples = actual[:, None].expand(-1, 3, -1, -1, -1)
    out = heliocentricity(samples, batch, obs_frames=1)
    assert torch.allclose(out['H'], torch.zeros_like(out['H']), atol=1e-5)
    assert out['H'].shape == (3, 11)


def test_h_positive_when_real_defense_is_tighter(batch):
    actual = batch['dyn'][:, :, DEF, :2]
    samples = actual[:, None].expand(-1, 3, -1, -1, -1) + 100.0  # ghosts far away
    out = heliocentricity(samples, batch, obs_frames=1)
    assert (out['H'] > 0).all()
    assert torch.equal(out['n_frames'][:, 0], torch.tensor([11.0, 8.0, 4.0]))


def _write_folds(tmp_path, folds, n_folds=3, seed=0, data_path='data/x.npz'):
    paths = []
    for fold in folds:
        path = tmp_path / f'fold{fold}' / 'model.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'fold': fold, 'n_folds': n_folds, 'data_path': data_path, 'data_fingerprint': 'abc',
                    'train_config': {'seed': seed, 'val_frac': 0.1}}, path)
        paths.append(path)
    return paths


def test_check_checkpoints_accepts_one_complete_run(tmp_path):
    ckpts = _write_folds(tmp_path, range(3))
    assert check_checkpoints(ckpts) == (Path('data/x.npz'), 'abc')


def test_check_checkpoints_rejects_missing_or_stale_folds(tmp_path):
    with pytest.raises(ValueError, match='folds 0..2'):
        check_checkpoints(_write_folds(tmp_path / 'a', [0, 2]))
    stale = _write_folds(tmp_path / 'b', range(3)) + _write_folds(tmp_path / 'b', [3, 4], n_folds=5)
    with pytest.raises(ValueError, match='n_folds'):
        check_checkpoints(stale)


def test_check_checkpoints_rejects_mixed_seeds(tmp_path):
    ckpts = _write_folds(tmp_path, [0, 1]) + _write_folds(tmp_path, [2], seed=1)
    with pytest.raises(ValueError, match='seed'):
        check_checkpoints(ckpts)


def test_aggregate_players_splits_total_and_within_call():
    plays = pd.DataFrame({'nfl_id': [1, 1, 2], 'group': 'WR', 'H': [1.0, 3.0, 0.5],
                          'H_within': [0.5, 1.5, 0.5], 'expected_sep': 4.0,
                          'expected_sep_within': 3.5, 'actual_sep': 2.0})
    plays['H_call'] = plays.H - plays.H_within
    stats = aggregate_players(plays, None).set_index('nfl_id')
    assert stats.loc[1, 'H_mean'] == 2.0 and stats.loc[1, 'H_within_mean'] == 1.0
    assert stats.loc[1, 'H_call_mean'] == 1.0 and stats.loc[1, 'n_plays'] == 2


def test_player_names_merge_seasons_and_skip_missing_files(tmp_path):
    pd.DataFrame({'nflId': [1, 2], 'displayName': ['A', 'B']}).to_csv(tmp_path / 'a.csv', index=False)
    pd.DataFrame({'nflId': [2, 3], 'displayName': ['B', 'C']}).to_csv(tmp_path / 'b.csv', index=False)
    names = player_names([tmp_path / 'a.csv', tmp_path / 'b.csv', tmp_path / 'missing.csv'])
    assert names.to_dict() == {1: 'A', 2: 'B', 3: 'C'}
    assert player_names(None).empty
