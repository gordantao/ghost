import pytest
import torch

from ghost.model import GhostDefense, ModelConfig
from ghost.preprocess import N_ENT


@pytest.fixture
def batch():
    torch.manual_seed(0)
    B, T, F, G = 3, 12, 7, 5
    length = torch.tensor([12, 9, 5])
    dyn = torch.randn(B, T, N_ENT, F) * 5
    time_mask = torch.arange(T)[None] < length[:, None]
    dyn[~time_mask] = 0.0
    return {
        'dyn': dyn,
        'time_mask': time_mask,
        'group': torch.randint(0, 10, (B, N_ENT)),
        'glob': torch.randn(B, G),
    }


@pytest.fixture
def model():
    torch.manual_seed(0)
    cfg = ModelConfig(n_dyn=7, n_glob=5, max_frames=20, obs_frames=2, d_model=32, n_heads=2,
                      n_enc_layers=1, n_post_layers=1, n_dec_layers=1, d_latent=4, dropout=0.0)
    return GhostDefense(cfg).eval()
