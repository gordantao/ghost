import torch

from ghost.model import elbo_loss, sample_ade
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
