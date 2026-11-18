from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ghost.model import GhostDefense, ModelConfig
from ghost.preprocess import DEF, GROUP_VOCAB, N_ENT
from ghost.score import (add_centered, aggregate_players, center_frames, check_checkpoints, clustered_se,
                         heliocentricity, offset_buckets, player_names, pooled_play_var, score_fold, shrink)
import ghost.score as score_mod


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


def test_h_t_averages_to_h(batch):
    actual = batch['dyn'][:, :, DEF, :2]
    out = heliocentricity(actual[:, None].expand(-1, 3, -1, -1, -1) + 3.0, batch, obs_frames=1)
    mask = (batch['time_mask'] & (torch.arange(12) >= 1)[None]).float()[:, :, None]
    assert torch.allclose((out['H_t'] * mask).sum(1) / mask.sum(1), out['H'], atol=1e-5)


def _synthetic_frames(n=400, T=10, obs=2, seed=0):
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 2, n)
    group = np.where(rng.random(n) < 0.5, 'WR', 'TE')
    drift = np.arange(T) * 0.1 + ((group == 'TE') * 0.3 + fold * 0.5)[:, None]       # bias grows with frame offset
    h_t = drift + rng.normal(0, 0.2, (n, T))
    valid = np.broadcast_to(np.arange(T) >= obs, (n, T)).copy()
    return h_t, valid, fold, group


def test_centering_zeroes_every_fold_group_cell_and_removes_drift():
    h_t, valid, fold, group = _synthetic_frames()
    centered = center_frames(h_t, valid, fold, group, min_rows=1)
    for f in (0, 1):
        for g in ('WR', 'TE'):
            assert abs(centered[(fold == f) & (group == g)].mean()) < 1e-9
    # raw H by later frames is dominated by the drift; centered values don't depend on frame offset
    raw = (h_t * valid).sum(1) / valid.sum(1)
    assert raw.mean() > 0.5 and abs(centered.mean()) < 1e-9
    late = center_frames(h_t * (np.arange(10) >= 7), valid & (np.arange(10) >= 7), fold, group, min_rows=1)
    assert abs(late.mean()) < 1e-9


def test_offset_buckets_pool_thin_late_offsets():
    valid = np.zeros((10, 6), bool)
    for t, n in enumerate([0, 0, 10, 10, 4, 1]):
        valid[:n, t] = True
    assert offset_buckets(valid, min_rows=5).tolist() == [0, 1, 2, 3, 4, 4]
    assert offset_buckets(valid, min_rows=1).tolist() == [0, 1, 2, 3, 4, 5]


def test_add_centered_columns():
    h_t, valid, fold, group = _synthetic_frames(n=60)
    plays = pd.DataFrame({'fold': fold, 'group': group})
    add_centered(plays, {'H_t': h_t, 'H_within_t': h_t * 0.5, 'valid': valid}, min_rows=1)
    assert np.allclose(plays.H_call_centered, plays.H_centered - plays.H_within_centered)
    assert np.allclose(plays.H_within_centered, plays.H_centered * 0.5)


def _player_plays(n_games, per_game, shared=0.0, seed=0):
    rng = np.random.default_rng(seed)
    effect = np.repeat(rng.normal(0, 1, n_games) * shared, per_game)
    return pd.DataFrame({'nfl_id': 1, 'game_id': np.repeat(np.arange(n_games), per_game),
                         'x': effect + rng.normal(0, 1, n_games * per_game)})


def test_clustered_se_equals_plain_se_with_one_play_per_game():
    plays = _player_plays(30, 1)
    assert clustered_se(plays, 'x')[1] == pytest.approx(plays.x.std() / np.sqrt(30))


def test_clustered_se_exceeds_plain_se_under_within_game_correlation():
    plays = _player_plays(10, 8, shared=2.0)
    assert clustered_se(plays, 'x')[1] > 1.5 * plays.x.std() / np.sqrt(len(plays))


def test_shrinkage_pulls_small_n_players_harder():
    rng = np.random.default_rng(1)
    n = pd.Series([2] * 30 + [200] * 30)
    se = 1.0 / np.sqrt(n)
    mean = pd.Series(rng.normal(0, 0.8, 60) + rng.normal(0, 1, 60) * se)
    group = pd.Series('WR', index=mean.index)
    shrunk = shrink(mean, se, group, n)
    m = np.average(mean, weights=n)
    pull = ((mean - shrunk) / (mean - m)).abs()
    assert pull[:30].mean() > pull[30:].mean() + 0.3
    assert ((shrunk - m).abs() <= (mean - m).abs() + 1e-12).all()


def test_aggregate_players_adds_centered_columns_and_sorts_by_shrunk():
    rng = np.random.default_rng(2)
    n = 120
    plays = pd.DataFrame({'nfl_id': rng.integers(0, 12, n), 'game_id': rng.integers(0, 40, n), 'group': 'WR',
                          'expected_sep': 4.0, 'expected_sep_within': 3.5, 'actual_sep': 2.0})
    plays['H'] = rng.normal(0, 1, n)
    plays['H_within'] = plays.H * 0.5
    plays['H_call'] = plays.H - plays.H_within
    for h in ('H', 'H_within', 'H_call'):
        plays[f'{h}_centered'] = plays[h] - plays[h].mean()
    stats = aggregate_players(plays, None)
    for c in ('H_centered_mean', 'H_centered_se', 'H_centered_shrunk', 'H_call_centered_shrunk'):
        assert c in stats
    assert stats.H_centered_shrunk.is_monotonic_decreasing


def test_centering_with_variable_play_lengths_removes_length_and_offset_drift():
    rng = np.random.default_rng(3)
    n, T = 2000, 40
    length = rng.integers(10, T + 1, n)
    valid = np.arange(T)[None] < length[:, None]
    valid[:, :2] = False
    h_t = 0.02 * np.arange(T) + rng.normal(0, 0.3, (n, T))
    fold, group = np.zeros(n, int), np.full(n, 'WR')
    centered = center_frames(h_t, valid, fold, group, min_rows=1)
    nf = valid.sum(1)
    raw = np.where(valid, h_t, 0).sum(1) / nf
    assert np.corrcoef(raw, nf)[0, 1] > 0.3 and abs(np.corrcoef(centered, nf)[0, 1]) < 0.1
    assert abs(centered.mean()) < 0.02                                        # approximately, not exactly, 0
    # one offset's drift removed: shifting only offset 20 of the data changes nothing after centering
    shifted = h_t.copy()
    shifted[:, 20] += 5.0
    assert np.allclose(center_frames(shifted, valid, fold, group, min_rows=1), centered)
    assert not np.allclose(raw, np.where(valid, shifted, 0).sum(1) / nf)


def test_shrinkage_floor_keeps_two_game_player_from_topping_the_ranking():
    rng = np.random.default_rng(4)
    rows = [pd.DataFrame({'nfl_id': i, 'game_id': np.arange(60) % 15, 'x': rng.normal(0.1 * rng.normal(), 1, 60)})
            for i in range(50)]
    rows.append(pd.DataFrame({'nfl_id': 99, 'game_id': [0, 1], 'x': [1.5, 1.52]}))
    plays = pd.concat(rows, ignore_index=True)
    n = plays.groupby('nfl_id').size()
    mean = plays.groupby('nfl_id').x.mean()
    group = pd.Series('WR', index=mean.index)
    se = clustered_se(plays, 'x')
    unfloored = shrink(mean, se, group, n)
    floored = shrink(mean, se, group, n, pooled_play_var(plays, 'x', group))
    assert unfloored.idxmax() == 99
    assert floored.idxmax() != 99 and floored[99] < 0.7


def test_score_fold_pads_batches_to_the_cache_width(monkeypatch):
    torch.manual_seed(0)
    cfg = ModelConfig(n_dyn=7, n_glob=5, n_coverage=4, max_frames=20, obs_frames=2, d_model=32, n_heads=2,
                      n_enc_layers=1, n_post_layers=1, n_dec_layers=1, d_latent=4, dropout=0.0)
    model = GhostDefense(cfg).eval()
    n = 40

    class Stub:
        length = torch.randint(5, 20, (n,))
        dyn = torch.randn(n, 20, N_ENT, 7)
        group = torch.full((n, N_ENT), GROUP_VOCAB.index('WR'))
        glob = torch.randn(n, 5)
        coverage = torch.randint(0, 4, (n,))
        game_id = np.repeat(np.arange(10), 4)
        play_id = np.arange(n)
        nfl_id = np.tile(np.arange(N_ENT), (n, 1))
        meta = {'coverage_vocab': list('abcd')}

        def batch(self, idx):
            from ghost.data import PlayData
            return PlayData.batch(self, idx)

    ckpt = {'fold': 0, 'n_folds': 2, 'train_config': {'seed': 0, 'val_frac': 0.1}}
    monkeypatch.setattr(score_mod, 'load_model', lambda p, d: (model, ckpt))
    monkeypatch.setattr(score_mod, 'held_out_plays', lambda c, d: np.arange(n))
    rows, arrays = score_fold(None, Stub(), 2, 8, torch.device('cpu'))
    assert arrays['H_t'].shape == arrays['valid'].shape == (len(rows), 20)
    assert np.isfinite(arrays['H_t']).all() and not arrays['valid'][:, :2].any()
