import torch

from ghost.preprocess import DEF
from ghost.score import heliocentricity


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
