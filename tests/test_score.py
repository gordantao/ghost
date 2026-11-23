from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ghost.model import GhostDefense, ModelConfig, future_mask
from ghost.preprocess import DEF, GROUP_VOCAB, N_ENT
from ghost.score import (DENS_COLS, add_centered, dens_col, aggregate_players, center_frames, check_checkpoints, clustered_se,
                         defender_density, density_gravity, heliocentricity, lower_quartile_plays, offset_buckets, player_names, score_fold)
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
    plays = pd.DataFrame({'nfl_id': [1, 1, 2], 'group': 'WR', 'ran_route': True, 'H': [1.0, 3.0, 0.5],
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


def test_aggregate_players_adds_centered_columns_and_sorts_by_centered_mean():
    rng = np.random.default_rng(2)
    n = 120
    plays = pd.DataFrame({'nfl_id': rng.integers(0, 12, n), 'game_id': rng.integers(0, 40, n), 'group': 'WR',
                          'ran_route': True, 'expected_sep': 4.0, 'expected_sep_within': 3.5, 'actual_sep': 2.0})
    plays['H'] = rng.normal(0, 1, n)
    plays['H_within'] = plays.H * 0.5
    plays['H_call'] = plays.H - plays.H_within
    for h in ('H', 'H_within', 'H_call'):
        plays[f'{h}_centered'] = plays[h] - plays[h].mean()
    stats = aggregate_players(plays, None)
    for c in ('H_centered_mean', 'H_centered_se', 'H_call_centered_mean', 'H_call_centered_se'):
        assert c in stats
    assert not any(c.endswith('_shrunk') for c in stats)
    assert stats.H_centered_mean.is_monotonic_decreasing


def _counted_plays(counts):
    ids = np.repeat(np.arange(len(counts)), counts)
    return pd.DataFrame({'nfl_id': ids, 'group': 'WR', 'ran_route': True, 'H': 0.0, 'H_within': 0.0,
                         'H_call': 0.0, 'expected_sep': 4.0, 'expected_sep_within': 4.0, 'actual_sep': 4.0})


def test_min_plays_default_drops_the_lower_quartile_by_play_count():
    counts = [1, 2, 3, 4, 10, 20, 30, 40]                     # 25th percentile is 2.75 -> cutoff 3
    stats = aggregate_players(_counted_plays(counts), None, min_plays=None)
    assert stats.attrs['min_plays'] == lower_quartile_plays(pd.Series(counts)) == 3
    assert sorted(stats.n_plays) == [3, 4, 10, 20, 30, 40]


def test_min_plays_counts_route_plays_only():
    plays = _counted_plays([5, 5])
    plays.loc[plays.index[:3], 'ran_route'] = False            # player 0 ran a route on 2 of his 5 plays
    assert aggregate_players(plays, None, min_plays=3).nfl_id.tolist() == [1]
    assert sorted(aggregate_players(plays, None, include_blockers=True, min_plays=3).nfl_id) == [0, 1]


def test_explicit_min_plays_overrides_the_quartile():
    stats = aggregate_players(_counted_plays([1, 2, 3, 4, 10]), None, min_plays=0)
    assert len(stats) == 5 and stats.attrs['min_plays'] == 0


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
        route_labels = None

        def batch(self, idx):
            from ghost.data import PlayData
            return PlayData.batch(self, idx)

    ckpt = {'fold': 0, 'n_folds': 2, 'train_config': {'seed': 0, 'val_frac': 0.1}}
    monkeypatch.setattr(score_mod, 'load_model', lambda p, d: (model, ckpt))
    monkeypatch.setattr(score_mod, 'held_out_plays', lambda c, d: np.arange(n))
    rows, arrays = score_fold(None, Stub(), 2, 8, torch.device('cpu'))
    assert arrays['H_t'].shape == arrays['valid'].shape == (len(rows), 20)
    assert np.isfinite(arrays['H_t']).all() and not arrays['valid'][:, :2].any()
    for c in DENS_COLS:
        assert arrays[f'{c}_t'].shape == (len(rows), 20) and np.isfinite(arrays[f'{c}_t']).all()
        assert c in rows and rows[c].notna().all()


def test_defender_density_counts_every_close_defender():
    rec = torch.zeros(1, 1, 2)
    one = torch.tensor([[[2.0, 0.0], [50.0, 0.0]]])
    two = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    expected = torch.exp(torch.tensor(-4.0 / 8.0))                 # d = 2, sigma = 2
    assert torch.allclose(defender_density(rec, one, 2.0), expected, atol=1e-6)   # the far defender adds ~0
    assert torch.allclose(defender_density(rec, two, 2.0), 2 * expected)          # double coverage counts twice


def test_density_gravity_is_zero_when_ghosts_match_and_positive_when_they_are_looser(batch):
    actual = batch['dyn'][:, :, DEF, :2]
    same = density_gravity(actual[:, None].expand(-1, 3, -1, -1, -1), batch, 1, 2.0)
    assert torch.allclose(same['H'], torch.zeros_like(same['H']), atol=1e-5)
    far = density_gravity(actual[:, None].expand(-1, 3, -1, -1, -1) + 100.0, batch, 1, 2.0)
    assert (far['H'] >= 0).all() and (far['H'] > 0).any()
    mask = future_mask(batch['time_mask'], 1)[:, :, None].float()
    assert torch.allclose((far['H_t'] * mask).sum(1) / mask.sum(1), far['H'], atol=1e-5)


@pytest.mark.parametrize('cols', [DENS_COLS, (dens_col(1), dens_col(2.5), dens_col(10))])
def test_density_columns_are_centered_and_aggregated_when_present(cols):
    rng = np.random.default_rng(5)
    n, T = 300, 10
    valid = np.ones((n, T), bool)
    h_t = rng.normal(0.5, 1, (n, T))
    plays = pd.DataFrame({'nfl_id': rng.integers(0, 6, n), 'game_id': rng.integers(0, 30, n), 'group': 'WR',
                          'fold': 0, 'ran_route': True, 'H': h_t.mean(1), 'H_within': h_t.mean(1),
                          'H_call': 0.0, 'expected_sep': 4.0, 'expected_sep_within': 4.0, 'actual_sep': 4.0})
    for c in cols:
        plays[c] = h_t.mean(1) * 0.1
    arrays = {'H_t': h_t, 'H_within_t': h_t, 'valid': valid, **{f'{c}_t': h_t * 0.1 for c in cols}}
    add_centered(plays, arrays, min_rows=1)
    stats = aggregate_players(plays, None)
    for c in cols:
        assert abs(plays[f'{c}_centered'].mean()) < 1e-9
        for suffix in ('_mean', '_se', '_centered_mean', '_centered_se'):
            assert f'{c}{suffix}' in stats
