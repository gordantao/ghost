"""Training one ghost-defense model (one cross-fitting fold, or all data)."""
from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .data import PlayData, split_indices
from .model import GhostDefense, ModelConfig, elbo_loss, sample_ade


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    beta: float = 1.0           # final KL weight
    beta_warmup: int = 10       # epochs to anneal KL weight from 0 to beta
    grad_clip: float = 1.0
    val_frac: float = 0.1
    eval_samples: int = 6
    seed: int = 0


def pick_device(name: str = 'auto') -> torch.device:
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def batches(idx: np.ndarray, size: int, shuffle: bool, rng: np.random.Generator):
    if shuffle:
        idx = rng.permutation(idx)
    for i in range(0, len(idx), size):
        yield torch.from_numpy(idx[i:i + size])


@torch.no_grad()
def evaluate(model: GhostDefense, data: PlayData, idx: np.ndarray, cfg: TrainConfig,
             rng: np.random.Generator) -> dict[str, float]:
    model.eval()
    totals, n = {}, 0
    for b in batches(idx, cfg.batch_size, False, rng):
        batch = data.batch(b.to(data.length.device))
        terms = elbo_loss(model(batch), batch, model.cfg.obs_frames, cfg.beta)
        terms['ade_mean'], terms['ade_min'] = sample_ade(
            model.sample(batch, cfg.eval_samples), batch, model.cfg.obs_frames)
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + float(value) * len(b)
        n += len(b)
    return {key: value / n for key, value in totals.items()}


def train(data_path: Path, out_dir: Path, fold: int | None, n_folds: int,
          model_overrides: dict, cfg: TrainConfig, device: str = 'auto') -> Path:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = pick_device(device)

    data = PlayData.load(data_path)
    train_idx, val_idx, held_out = split_indices(data.game_id, fold, n_folds, cfg.val_frac, cfg.seed)
    data.to(dev)
    print(f'device={dev} fold={fold} train={len(train_idx)} val={len(val_idx)} held_out={len(held_out)}')

    model_cfg = ModelConfig(n_dyn=data.dyn.shape[-1], n_glob=data.glob.shape[-1],
                            max_frames=data.meta['max_frames'], **model_overrides)
    model = GhostDefense(model_cfg).to(dev)
    print(f'parameters: {sum(p.numel() for p in model.parameters()):,}')
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / 'model.pt'
    log_path = out_dir / 'metrics.csv'
    best = float('inf')
    with log_path.open('w', newline='') as log_file:
        log = None
        for epoch in range(1, cfg.epochs + 1):
            start = time.time()
            model.train()
            beta = cfg.beta * min(1.0, epoch / max(cfg.beta_warmup, 1))
            train_loss, n = 0.0, 0
            for b in batches(train_idx, cfg.batch_size, True, rng):
                batch = data.batch(b.to(dev))
                terms = elbo_loss(model(batch), batch, model_cfg.obs_frames, beta)
                opt.zero_grad()
                terms['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                train_loss += float(terms['loss']) * len(b)
                n += len(b)

            val = evaluate(model, data, val_idx, cfg, rng)
            row = {'epoch': epoch, 'beta': beta, 'train_loss': train_loss / n,
                   **{f'val_{k}': v for k, v in val.items()}, 'seconds': time.time() - start}
            if log is None:
                log = csv.DictWriter(log_file, fieldnames=list(row))
                log.writeheader()
            log.writerow(row)
            log_file.flush()
            print(' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()))

            # Compare checkpoints only once the KL weight has reached its final value.
            if epoch >= min(cfg.beta_warmup, cfg.epochs) and val['loss'] < best:
                best = val['loss']
                torch.save({
                    'model_config': asdict(model_cfg), 'train_config': asdict(cfg),
                    'state_dict': model.state_dict(), 'epoch': epoch,
                    'fold': fold, 'n_folds': n_folds, 'data_path': str(data_path),
                }, ckpt_path)
    print(f'best val loss {best:.4g}; saved {ckpt_path}')
    return ckpt_path


def load_model(ckpt_path: Path, device: torch.device) -> tuple[GhostDefense, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GhostDefense(ModelConfig(**ckpt['model_config'])).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt
