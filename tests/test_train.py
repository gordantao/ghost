import csv
import json
import math

import numpy as np
import pytest
import torch

from ghost.data import split_indices
from ghost.preprocess import N_ENT
from ghost.train import TrainConfig, lr_factor, train


def test_cosine_warms_up_then_decays_to_floor():
    cfg = TrainConfig(warmup_steps=10, min_lr_frac=0.05)
    total = 111
    f = [lr_factor(s, total, cfg) for s in range(total)]
    assert f[0] == pytest.approx(0.1) and f[9] == pytest.approx(1.0)  # linear warmup to the peak
    assert all(a <= b for a, b in zip(f[:10], f[1:10]))
    assert all(a >= b for a, b in zip(f[9:], f[10:]))  # never increases after warmup
    assert f[-1] == pytest.approx(0.05)  # lands on the floor at the last step
    assert f[60] == pytest.approx(0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * 50 / 100)))  # halfway


def test_constant_schedule_and_short_runs():
    assert lr_factor(1234, 2000, TrainConfig(lr_schedule='constant')) == 1.0
    cfg = TrainConfig(warmup_steps=500)
    f = [lr_factor(s, 20, cfg) for s in range(20)]  # warmup is capped at half the run
    assert max(f) == pytest.approx(1.0) and f[-1] == pytest.approx(cfg.min_lr_frac)


def test_schedule_edge_cases():
    no_warmup = TrainConfig(warmup_steps=0)
    assert lr_factor(0, 10, no_warmup) == pytest.approx(1.0)
    assert lr_factor(9, 10, no_warmup) == pytest.approx(no_warmup.min_lr_frac)
    flat = TrainConfig(warmup_steps=3, min_lr_frac=1.0)  # warmup, then constant at the peak
    assert [lr_factor(s, 10, flat) for s in range(3, 10)] == pytest.approx([1.0] * 7)
    assert lr_factor(0, 1, TrainConfig()) == pytest.approx(1.0)  # a one-step run trains at the peak


@pytest.mark.parametrize('bad', [{'lr_schedule': 'step'}, {'warmup_steps': -5},
                                 {'min_lr_frac': -0.1}, {'min_lr_frac': 1.5}])
def test_bad_schedule_settings_are_rejected_at_construction(bad):
    with pytest.raises(ValueError):
        TrainConfig(**bad)


def _synthetic_cache(path, n_games=10, plays_per_game=3, max_frames=8):
    rng = np.random.default_rng(0)
    n = n_games * plays_per_game
    length = rng.integers(4, max_frames + 1, n)
    dyn = rng.normal(size=(n, max_frames, N_ENT, 7)).astype(np.float32)
    dyn[np.arange(max_frames)[None] >= length[:, None]] = 0.0
    np.savez(path, dyn=dyn, length=length, glob=rng.normal(size=(n, 3)).astype(np.float32),
             coverage=rng.integers(0, 3, n), group=rng.integers(0, 10, (n, N_ENT)),
             nfl_id=rng.integers(1, 99, (n, N_ENT - 1)), game_id=np.repeat(np.arange(n_games), plays_per_game),
             play_id=np.tile(np.arange(plays_per_game), n_games))
    path.with_suffix('.json').write_text(json.dumps({'max_frames': max_frames, 'coverage_vocab': ['a', 'b', 'c']}))


def test_train_steps_the_schedule_once_per_batch_and_logs_the_rate_used(tmp_path, monkeypatch):
    cache = tmp_path / 'cache.npz'
    _synthetic_cache(cache)
    cfg = TrainConfig(epochs=2, batch_size=4, warmup_steps=2, beta_warmup=1, eval_samples=2)
    rates = []
    step = torch.optim.AdamW.step

    def recording_step(self, *args, **kwargs):
        rates.append(self.param_groups[0]['lr'])
        return step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, 'step', recording_step)
    small = {'d_model': 16, 'n_heads': 2, 'n_enc_layers': 1, 'n_post_layers': 1, 'n_dec_layers': 1, 'd_latent': 4}
    train(cache, tmp_path / 'run', None, 5, small, cfg, device='cpu')

    train_idx, _, _ = split_indices(np.repeat(np.arange(10), 3), None, 5, cfg.val_frac, cfg.seed)
    per_epoch = math.ceil(len(train_idx) / cfg.batch_size)
    assert len(rates) == cfg.epochs * per_epoch
    assert rates[0] == pytest.approx(cfg.lr / cfg.warmup_steps)
    assert rates[-1] == pytest.approx(cfg.lr * cfg.min_lr_frac)  # the final step reaches the floor

    with (tmp_path / 'run' / 'metrics.csv').open() as f:
        logged = [float(row['lr']) for row in csv.DictReader(f)]
    expected = [np.mean(rates[e * per_epoch:(e + 1) * per_epoch]) for e in range(cfg.epochs)]
    assert logged == pytest.approx(expected)  # each row is the mean rate that epoch actually trained at
    assert (tmp_path / 'run' / 'model.pt').exists()
