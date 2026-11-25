import pytest
import torch

from ghost.model import GhostDefense, ModelConfig, elbo_loss, sample_ade
from ghost.preprocess import DEF, OFF


def _model(lookahead, layers=1, deterministic=False):
    torch.manual_seed(0)
    cfg = ModelConfig(n_dyn=7, n_glob=5, n_coverage=4, max_frames=20, obs_frames=2, d_model=32, n_heads=2,
                      n_enc_layers=layers, n_post_layers=1, n_dec_layers=layers, d_latent=4, dropout=0.0,
                      lookahead=lookahead, deterministic=deterministic)
    return GhostDefense(cfg).eval()


@pytest.fixture(params=[-1, 0, 2], ids=lambda la: f'lookahead={la}')
def model(request):
    """Every invariant below must hold for the bidirectional and the causal ghost."""
    return _model(request.param)


def test_shapes_and_backward(model, batch):
    model.train()
    out = model(batch)
    assert out['pred'].shape == (3, 12, 11, 2)
    terms = elbo_loss(out, batch, model.cfg.obs_frames, beta=1.0)
    assert torch.isfinite(terms['loss'])
    terms['loss'].backward()

    samples = model.sample(batch, k=4)
    assert samples.shape == (3, 4, 12, 11, 2)
    mean_ade, min_ade = sample_ade(samples, batch, model.cfg.obs_frames)
    assert min_ade <= mean_ade


def _sample(model, batch, seed=1):
    torch.manual_seed(seed)
    return model.sample(batch, k=2)


def test_prior_never_sees_hidden_defense(model, batch):
    """Ghost samples must not depend on defender tracking after the observed frames."""
    base = _sample(model, batch)
    changed = {**batch, 'dyn': batch['dyn'].clone()}
    changed['dyn'][:, model.cfg.obs_frames:, DEF] += 7.0
    assert torch.allclose(base, _sample(model, changed), atol=1e-5)


def test_prior_never_sees_coverage_call(model, batch):
    """By default the ghost draws its own call; the real (post-play) label must not leak in."""
    base = _sample(model, batch)
    changed = {**batch, 'coverage': (batch['coverage'] + 1) % model.cfg.n_coverage}
    assert torch.allclose(base, _sample(model, changed), atol=1e-5)


def test_call_prior_ignores_label_and_post_snap_play(model, batch):
    """p(c | context) must not see the real label, nor any frame after the observed ones."""
    base = model.encode(batch)[2]
    changed = {**batch, 'coverage': (batch['coverage'] + 1) % model.cfg.n_coverage,
               'dyn': batch['dyn'].clone()}
    changed['dyn'][:, model.cfg.obs_frames:] += 7.0  # offense, ball and defense after the snap
    assert torch.equal(base, model.encode(changed)[2])


def test_fixed_call_path_never_sees_hidden_defense(model, batch):
    torch.manual_seed(1)
    base = model.sample(batch, k=2, coverage=batch['coverage'])
    changed = {**batch, 'dyn': batch['dyn'].clone()}
    changed['dyn'][:, model.cfg.obs_frames:, DEF] += 7.0
    torch.manual_seed(1)
    assert torch.allclose(base, model.sample(changed, k=2, coverage=batch['coverage']), atol=1e-5)


def test_shared_generator_gives_common_noise(model, batch):
    """With the actual call fixed to what the default path drew, both paths match exactly."""
    gen = lambda: torch.Generator().manual_seed(3)  # noqa: E731
    k = 2
    default = model.sample(batch, k=k, generator=gen())
    g = gen()
    torch.randn(3 * k, model.cfg.d_latent, generator=g)
    drawn = torch.multinomial(model.encode(batch)[2].softmax(-1), k, replacement=True, generator=g)
    for j in range(k):
        fixed = model.sample(batch, k=k, coverage=drawn[:, j], generator=gen())
        assert torch.allclose(default[:, j], fixed[:, j], atol=1e-5)


def test_fixed_coverage_call_steers_samples(model, batch):
    torch.manual_seed(1)
    a = model.sample(batch, k=2, coverage=torch.zeros(3, dtype=torch.long))
    torch.manual_seed(1)
    b = model.sample(batch, k=2, coverage=torch.ones(3, dtype=torch.long))
    assert not torch.allclose(a, b, atol=1e-5)


def test_coverage_terms_in_loss(model, batch):
    model.train()
    terms = elbo_loss(model(batch), batch, model.cfg.obs_frames, beta=1.0)
    assert terms['coverage_nll'] > 0 and 0 <= terms['coverage_acc'] <= 1
    model.zero_grad()
    terms['coverage_nll'].backward()
    assert model.coverage_head[-1].weight.grad.abs().sum() > 0


def test_prior_uses_observed_defense_and_offense(model, batch):
    base = _sample(model, batch)
    for t, ents in ((0, DEF), (5, OFF)):
        changed = {**batch, 'dyn': batch['dyn'].clone()}
        changed['dyn'][:, t, ents] += 7.0
        assert not torch.allclose(base, _sample(model, changed), atol=1e-5)


def test_padding_does_not_leak(model, batch):
    """Values in padded frames must not change predictions for valid frames."""
    base = _sample(model, batch)
    changed = {**batch, 'dyn': batch['dyn'].clone()}
    changed['dyn'][~batch['time_mask']] = 99.0
    valid = batch['time_mask'][:, None, :, None, None].expand_as(base)
    assert torch.allclose(base[valid], _sample(model, changed)[valid], atol=1e-5)


@pytest.mark.parametrize('obs_frames', [0, 20])
def test_obs_frames_must_leave_observed_and_predicted_frames(obs_frames):
    with pytest.raises(ValueError):
        ModelConfig(n_dyn=7, n_glob=5, n_coverage=4, max_frames=20, obs_frames=obs_frames)


def test_forward_generator_makes_the_posterior_draw_reproducible(model, batch):
    gen = lambda: torch.Generator().manual_seed(3)  # noqa: E731
    model.eval()
    with torch.no_grad():
        a, b = model(batch, generator=gen()), model(batch, generator=gen())
        c = model(batch, generator=torch.Generator().manual_seed(4))
    assert torch.equal(a['pred'], b['pred'])  # seeded validation carries no run-to-run noise
    assert not torch.equal(a['pred'], c['pred'])  # but still samples z rather than taking the mean


def test_return_coverage_reports_the_drawn_calls(model, batch):
    gen = lambda: torch.Generator().manual_seed(5)  # noqa: E731
    samples, call = model.sample(batch, k=4, generator=gen(), return_coverage=True)
    assert call.shape == (3, 4)
    assert torch.equal(samples, model.sample(batch, k=4, generator=gen()))
    for j in range(4):
        fixed = model.sample(batch, k=4, coverage=call[:, j], generator=gen())
        assert torch.allclose(samples[:, j], fixed[:, j], atol=1e-5)
    _, fixed_call = model.sample(batch, k=4, coverage=batch['coverage'], return_coverage=True)
    assert torch.equal(fixed_call, batch['coverage'][:, None].expand(-1, 4))


@pytest.mark.parametrize('lookahead,layers', [(0, 1), (2, 1), (2, 3)])
def test_causal_ghost_never_sees_offense_beyond_lookahead(batch, lookahead, layers):
    """Frame t's ghost defense can't depend on offense or ball after t + lookahead (encoder, z prior,
    decoder), however many layers are stacked."""
    model, t0 = _model(lookahead, layers), 6
    base = _sample(model, batch)
    changed = {**batch, 'dyn': batch['dyn'].clone()}
    changed['dyn'][:, t0:] += 7.0
    after = _sample(model, changed)
    keep = t0 - lookahead
    assert torch.allclose(base[:, :, :keep], after[:, :, :keep], atol=1e-5)
    assert not torch.allclose(base[:, :, keep:], after[:, :, keep:], atol=1e-5)


def test_unrestricted_ghost_sees_future_offense(batch):
    """The default (lookahead=-1) model reads the whole play, so the test above is a real constraint."""
    model = _model(-1)
    base = _sample(model, batch)
    changed = {**batch, 'dyn': batch['dyn'].clone()}
    changed['dyn'][:, 6:] += 7.0
    assert not torch.allclose(base[:, :, :6], _sample(model, changed)[:, :, :6], atol=1e-5)


def test_lookahead_adds_no_parameters():
    """Old checkpoints (no `lookahead` in their config) load as lookahead=-1 with the same state dict."""
    assert _model(-1).state_dict().keys() == _model(0).state_dict().keys()
    assert ModelConfig(n_dyn=7, n_glob=5, n_coverage=4, max_frames=20).lookahead == -1
    with pytest.raises(ValueError):
        ModelConfig(n_dyn=7, n_glob=5, n_coverage=4, max_frames=20, lookahead=-2)


@pytest.fixture(params=[-1, 0], ids=lambda la: f'deterministic,lookahead={la}')
def det_model(request):
    return _model(request.param, deterministic=True)


def test_deterministic_baseline_trains_and_repeats_one_prediction(det_model, batch):
    det_model.train()
    out = det_model(batch)
    terms = elbo_loss(out, batch, det_model.cfg.obs_frames, beta=1.0)
    assert torch.isfinite(terms['loss']) and terms['kl'] == 0
    terms['loss'].backward()
    det_model.eval()
    a = det_model.sample(batch, k=3, generator=torch.Generator().manual_seed(1))
    b = det_model.sample(batch, k=3, generator=torch.Generator().manual_seed(2))
    assert a.shape == (3, 3, 12, 11, 2)
    assert torch.equal(a, b) and torch.equal(a[:, 0], a[:, 2])        # no randomness anywhere
    assert torch.allclose(a[:, 0], det_model(batch)['pred'], atol=1e-4)   # eval forward = the prediction


def test_deterministic_baseline_never_sees_hidden_defense_label_or_padding(det_model, batch):
    base = _sample(det_model, batch)
    changed = {**batch, 'dyn': batch['dyn'].clone(), 'coverage': (batch['coverage'] + 1) % det_model.cfg.n_coverage}
    changed['dyn'][:, det_model.cfg.obs_frames:, DEF] += 7.0
    changed['dyn'][~batch['time_mask']] = 99.0
    valid = batch['time_mask'][:, None, :, None, None].expand_as(base)
    assert torch.allclose(base[valid], _sample(det_model, changed)[valid], atol=1e-5)


def test_deterministic_causal_baseline_never_sees_future_offense(batch):
    model = _model(0, layers=2, deterministic=True)
    t0 = 6
    base = _sample(model, batch)
    changed = {**batch, 'dyn': batch['dyn'].clone()}
    changed['dyn'][:, t0:, OFF] += 7.0
    changed['dyn'][:, t0:, 22] += 7.0
    assert torch.allclose(base[:, :, :t0], _sample(model, changed)[:, :, :t0], atol=1e-5)
