import numpy as np
import pandas as pd
import pytest
import torch

from ghost import cli
from ghost.calibration import OFFSET_BINS, calibration, offset_bin, pit_stat_rows, pit_values, summarize
from ghost.data import PlayData
from ghost.model import GhostDefense, ModelConfig, elbo_loss, sample_ade
from ghost.preprocess import DEF, GROUP_VOCAB, N_ENT, OFF
from ghost.score import RECEIVER_IDS
from ghost.train import TrainConfig, evaluate, train

from test_train import _synthetic_cache


def _plays(real_shift=0.0, B=400, T=12, K=16, ghost_shift=0.0, seed=0):
    """Real and ghost defenders drawn from the same distribution, then shifted: [B, T, 23, 7] and [B, K, T, 11, 2]."""
    g = torch.Generator().manual_seed(seed)
    dyn = torch.zeros(B, T, N_ENT, 7)
    dyn[:, :, OFF, :2] = torch.randn(B, T, 11, 2, generator=g) * 5
    dyn[:, :, DEF, :2] = torch.randn(B, T, 11, 2, generator=g) * 5 + real_shift
    samples = torch.randn(B, K, T, 11, 2, generator=g) * 5 + ghost_shift
    batch = {'dyn': dyn, 'time_mask': torch.ones(B, T, dtype=torch.bool),
             'group': torch.full((B, N_ENT), RECEIVER_IDS[0])}
    return samples, batch


def test_pit_is_uniform_when_ghosts_match_reality():
    samples, batch = _plays(K=100)   # K=16 puts u on a 1/16 grid that aliases with 10 bins
    u, mask = pit_values(samples, batch, obs_frames=2)
    u = u[mask[:, :, None].expand_as(u)]
    assert u.mean() == pytest.approx(0.5, abs=0.01)
    hist = torch.histc(u, 10, 0, 1) / u.numel()
    assert hist.min() > 0.09 and hist.max() < 0.11
    assert (u == 0).float().mean() == pytest.approx(1 / 101, abs=0.005)    # exactly 1/(K+1)


def test_pit_skews_low_when_ghosts_are_too_loose_and_high_when_too_tight():
    loose, batch = _plays(ghost_shift=20.0)      # ghosts far from everyone, so nearest ghost distance is large
    u, mask = pit_values(loose, batch, obs_frames=2)
    assert u[mask[:, :, None].expand_as(u)].mean() < 0.1
    tight, batch = _plays(real_shift=20.0)
    u, mask = pit_values(tight, batch, obs_frames=2)
    assert u[mask[:, :, None].expand_as(u)].mean() > 0.9


def test_pit_counts_ties_as_half(batch):
    samples = torch.zeros(3, 4, 12, 11, 2)
    batch['dyn'][:, :, DEF, :2] = 0.0
    batch['dyn'][:, :, OFF, :2] = 1.0           # every real and ghost distance is the same value
    u, mask = pit_values(samples, batch, obs_frames=2)
    assert torch.allclose(u, torch.full_like(u, 0.5))
    assert mask.shape == (3, 12) and not mask[:, :2].any()


def test_rank_histogram_is_flat_at_default_k_for_a_matched_ghost():
    samples, batch = _plays(B=600, K=16)
    u, mask = pit_values(samples, batch, obs_frames=2)
    stats = pit_stat_rows(u, mask, batch, 16).assign(ghost='own_call')
    overall = summarize(stats).query("stratum == 'overall'").iloc[0]
    shares = overall[[f'rank{b}' for b in range(17)]].astype(float)
    assert (shares - 1 / 17).abs().max() < 0.02
    assert overall.share_u0 == pytest.approx(1 / 17, abs=0.01)


def test_offset_bins_are_seconds_at_ten_hertz():
    assert offset_bin(np.array([0, 9, 10, 19, 20, 29, 30, 69])).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert len(OFFSET_BINS) == 4


def test_summary_is_consistent_across_strata():
    samples, batch = _plays(B=30, T=40)
    batch['group'] = torch.tensor(RECEIVER_IDS)[torch.randint(0, len(RECEIVER_IDS), (30, N_ENT))]
    u, mask = pit_values(samples, batch, obs_frames=2)
    stats = pit_stat_rows(u, mask, batch, 16).assign(ghost='own_call')
    summary = summarize(stats)
    overall = summary[summary.stratum == 'overall'].iloc[0]
    n_receivers = sum(int((batch['group'][:, OFF] == g).sum()) for g in RECEIVER_IDS)
    assert overall.n == n_receivers * 38
    for stratum in ('group', 'offset'):
        part = summary[summary.stratum == stratum]
        assert part.n.sum() == overall.n
        assert (part.mean_u * part.n).sum() / overall.n == pytest.approx(overall.mean_u)
    assert set(summary[summary.stratum == 'offset'].level) == set(OFFSET_BINS)
    assert set(summary[summary.stratum == 'group'].level) <= {GROUP_VOCAB[g] for g in RECEIVER_IDS}
    hist = overall[[f'rank{b}' for b in range(17)]].astype(float)
    assert hist.sum() == pytest.approx(1.0)
    assert overall.k == 16 and pd.api.types.is_integer_dtype(summary.n)


def test_non_receivers_are_excluded(batch):
    samples = torch.randn(3, 4, 12, 11, 2)
    batch['group'] = torch.full_like(batch['group'], len(GROUP_VOCAB) - 1)
    assert RECEIVER_IDS.count(len(GROUP_VOCAB) - 1) == 0
    u, mask = pit_values(samples, batch, obs_frames=2)
    assert pit_stat_rows(u, mask, batch, 4).empty


def _small_setup(tmp_path):
    cache = tmp_path / 'cache.npz'
    _synthetic_cache(cache)
    data = PlayData.load(cache)
    return data, np.arange(len(data.game_id))


def test_evaluate_adds_pit_metrics_without_changing_existing_ones(tmp_path):
    data, idx = _small_setup(tmp_path)
    cfg = TrainConfig(batch_size=7, eval_samples=3)
    torch.manual_seed(0)
    model = GhostDefense(ModelConfig(n_dyn=7, n_glob=3, n_coverage=3, max_frames=8, obs_frames=2, d_model=16,
                                     n_heads=2, n_enc_layers=1, n_post_layers=1, n_dec_layers=1, d_latent=4,
                                     dropout=0.0)).eval()
    out = evaluate(model, data, idx, cfg, np.random.default_rng(0))
    assert {'pit_mean', 'pit_low', 'pit_high'} <= set(out)
    assert 0 <= out['pit_low'] <= 1 and 0 <= out['pit_high'] <= 1 and 0 <= out['pit_mean'] <= 1

    # The pre-existing metrics, recomputed with the same generator and no PIT code.
    generator = torch.Generator(device=data.length.device).manual_seed(cfg.seed)
    totals = {}
    for i in range(0, len(idx), cfg.batch_size):
        batch = data.batch(torch.from_numpy(idx[i:i + cfg.batch_size]))
        terms = elbo_loss(model(batch, generator=generator), batch, 2, cfg.beta)
        terms['ade_mean'], terms['ade_min'] = sample_ade(model.sample(batch, 3, generator=generator), batch, 2)
        for k, v in terms.items():
            totals[k] = totals.get(k, 0.0) + float(v) * len(batch['coverage'])
    for k, v in totals.items():
        assert out[k] == pytest.approx(v / len(idx), rel=1e-5)


def test_calibration_rejects_bad_arguments_before_sampling(tmp_path):
    for kwargs in ({'n_samples': 0}, {'plays_per_fold': 0}):
        with pytest.raises(ValueError):
            calibration(None, tmp_path / 'no_runs', tmp_path / 'out', **kwargs)


def test_cli_wires_up_calibration(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr('ghost.calibration.calibration', lambda *a: seen.setdefault('args', a))
    cli.main(['calibration', '--runs', str(tmp_path), '--samples', '4', '--plays-per-fold', '10'])
    data, runs, out, k, bs, per_fold, device = seen['args']
    assert runs == tmp_path and out == tmp_path / 'calibration' and (k, per_fold) == (4, 10)


def _train_folds(tmp_path):
    cache = tmp_path / 'cache.npz'
    _synthetic_cache(cache)
    runs = tmp_path / 'runs'
    small = {'d_model': 16, 'n_heads': 2, 'n_enc_layers': 1, 'n_post_layers': 1, 'n_dec_layers': 1, 'd_latent': 4}
    for fold in range(5):
        train(cache, runs / f'fold{fold}', fold, 5, small, TrainConfig(epochs=1, batch_size=8, eval_samples=2),
              device='cpu')
    return runs


@pytest.mark.parametrize('plays_per_fold', [None, 2])
def test_calibration_end_to_end(tmp_path, plays_per_fold):
    runs = _train_folds(tmp_path)
    out = tmp_path / 'out'
    summary = calibration(None, runs, out, n_samples=4, batch_size=4, plays_per_fold=plays_per_fold, device='cpu')
    per_play = pd.read_csv(out / 'pit_play_receiver.csv')
    assert (out / 'pit_summary.csv').exists() and len(per_play) > 0
    overall = summary[summary.stratum == 'overall'].set_index('ghost')
    assert set(overall.index) == {'own_call', 'actual_call'}
    for ghost in overall.index:
        assert overall.loc[ghost, [f'rank{b}' for b in range(5)]].astype(float).sum() == pytest.approx(1.0)
    assert overall.n.nunique() == 1 and overall.n.iloc[0] > 0
    if plays_per_fold:
        assert per_play.groupby('fold').size().max() <= plays_per_fold * 11
