import math

import numpy as np
import pandas as pd
import pytest
import torch

from ghost.model import future_mask
from ghost.preprocess import OFF
from ghost.sample_size import batch_spread, call_decomposition, recommend, sample_size
from ghost.score import ghost_separation, heliocentricity, nearest_defender_distance


def test_ghost_separation_averages_to_expected_separation(batch):
    samples = torch.randn(3, 5, 12, 11, 2) * 5
    d = ghost_separation(samples, batch, obs_frames=2)
    assert d.shape == (3, 5, 11)
    # The frame-then-sample mean heliocentricity used before ghost_separation existed.
    offense = batch['dyn'][:, :, OFF, :2]
    expected = nearest_defender_distance(offense[:, None], samples).mean(1)
    mask = future_mask(batch['time_mask'], 2)[:, :, None].float()
    assert torch.allclose(d.mean(1), (expected * mask).sum(1) / mask.sum(1), atol=1e-5)
    assert torch.allclose(heliocentricity(samples, batch, 2)['expected_sep'], d.mean(1))


def test_call_decomposition_is_the_law_of_total_variance():
    torch.manual_seed(0)
    d = torch.randn(4, 50, 3)
    call = torch.randint(0, 3, (4, 50))
    within, between = call_decomposition(d, call, n_coverage=5)
    assert torch.allclose(within + between, d.var(1, unbiased=False), atol=1e-5)

    one_call = torch.zeros(4, 50, dtype=torch.long)
    _, between = call_decomposition(d, one_call, n_coverage=5)
    assert torch.allclose(between, torch.zeros_like(between), atol=1e-6)

    per_call = call.float()[..., None].expand(-1, -1, 3)        # constant within each call
    within, _ = call_decomposition(per_call, call, n_coverage=5)
    assert torch.allclose(within, torch.zeros_like(within), atol=1e-6)


def test_batch_spread_matches_what_score_computes(model, batch):
    spread = batch_spread(model, batch, n_samples=6, seed=7)
    gen = lambda: torch.Generator().manual_seed(7)  # noqa: E731
    obs = model.cfg.obs_frames
    total = heliocentricity(model.sample(batch, 6, generator=gen()), batch, obs)
    within = heliocentricity(model.sample(batch, 6, coverage=batch['coverage'], generator=gen()), batch, obs)
    assert torch.allclose(spread['H'], total['H'], atol=1e-5)
    assert torch.allclose(spread['H_within'], within['H'], atol=1e-5)
    assert torch.allclose(spread['H_call'], total['H'] - within['H'], atol=1e-5)
    assert all(v.shape == (3, 11) for v in spread.values())
    assert (spread['s_H'] >= 0).all()


def _spread(n=4000, s=1.0, signal=2.0, k=128, seed=0):
    rng = np.random.default_rng(seed)
    cols = {}
    for q in ('H', 'H_within', 'H_call'):
        cols[q] = rng.normal(0, signal, n) + rng.normal(0, s / math.sqrt(k), n)
        cols[f's_{q}'] = np.full(n, s)
    return pd.DataFrame({**cols, 's2_H_within_call': 0.75, 's2_H_between_call': 0.25})


def test_recommend_sets_delta_from_the_real_spread():
    out = recommend(_spread(), n_samples=128, eps=0.05, quantile=0.9).set_index('quantity')
    # V ~ signal^2 = 4, delta = sqrt(0.05 * 4) ~ 0.447, K = ceil((1 / 0.447)^2) = 5
    assert out.loc['H', 'V'] == pytest.approx(4.0, rel=0.1)
    assert out.loc['H', 'delta'] == pytest.approx(math.sqrt(0.05 * out.loc['H', 'V']))
    assert out.loc['H', 'K'] == math.ceil((1.0 / out.loc['H', 'delta']) ** 2)
    assert out.loc['H', 'share_within_delta'] == 1.0
    assert out.loc['H', 'between_call_share'] == pytest.approx(0.25)
    assert np.isnan(out.loc['H_call', 'between_call_share'])


def test_recommend_uses_the_quantile_of_s_and_a_fixed_delta():
    spread = _spread()
    spread['s_H'] = np.linspace(0.0, 2.0, len(spread))
    out = recommend(spread, n_samples=128, quantile=0.9, delta=0.1).set_index('quantity')
    assert out.loc['H', 'K'] == math.ceil((spread.s_H.quantile(0.9) / 0.1) ** 2)
    assert out.loc['H', 'share_within_delta'] == pytest.approx(0.9, abs=0.01)


def test_recommend_reports_nan_when_h_has_no_real_spread():
    spread = _spread()
    spread[['H', 'H_within', 'H_call']] = 1.0           # no spread at all, so V = -s^2 / K < 0
    out = recommend(spread, n_samples=128)
    assert out.K.isna().all() and out.delta.isna().all()
    assert recommend(spread, n_samples=128, delta=0.2).K.min() >= 2


@pytest.mark.parametrize('kwargs', [{'n_samples': 1}, {'eps': 0.0}, {'quantile': 1.0}, {'current_k': 0},
                                    {'delta': 0.0}, {'plays_per_fold': 0}])
def test_sample_size_rejects_bad_arguments_before_sampling(tmp_path, kwargs):
    with pytest.raises(ValueError):
        sample_size(None, tmp_path / 'no_runs', tmp_path / 'out', **kwargs)
