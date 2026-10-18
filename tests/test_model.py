import pytest
import torch

from ghost.model import ModelConfig, elbo_loss, sample_ade
from ghost.preprocess import DEF, OFF


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


def test_eval_forward_decodes_the_posterior_mean(model, batch):
    model.eval()
    with torch.no_grad():
        a, b = model(batch), model(batch)
    assert torch.equal(a['pred'], b['pred'])  # validation loss carries no sampling noise
    model.train()
    with torch.no_grad():
        c, d = model(batch), model(batch)
    assert not torch.equal(c['pred'], d['pred'])  # training still samples z
