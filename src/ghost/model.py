"""Ghost defense: a conditional VAE over defender trajectories.

Given the whole play's offense and ball tracking, the play context, and the
defense only for the first `obs_frames` frames, the model generates where the
11 defenders go for the rest of the play. It never sees player identities, so
its samples describe how a typical defense reacts to the offense's actual
movement.

Tokens are (frame, entity) pairs. Each block attends over time within an
entity, then over entities within a frame.

The latent has two parts: the defense's coverage call c (one of
preprocess.COVERAGE_VOCAB) and a continuous per-play vector z for everything
else. The prior sees only the masked input and gives p(c | context) and
p(z | context, c). In training c is observed (PFF's label), so the decoder is
teacher-forced with it and the prior's classifier is trained by cross-entropy;
the posterior q(z | context, true defense, c) is used only in training. At
sampling time c is drawn from the prior unless a call is given explicitly, so
the real call never reaches the ghost by default.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import DEF, GROUP_VOCAB, SIDE_DEFENSE, SIDES

POS_SCALE = 10.0  # yards
VEL_SCALE = 5.0   # yards / s
FEATURE_SCALE = (POS_SCALE, POS_SCALE, VEL_SCALE, VEL_SCALE, VEL_SCALE, 1.0, 1.0)


@dataclass
class ModelConfig:
    n_dyn: int
    n_glob: int
    n_coverage: int
    max_frames: int
    obs_frames: int = 1
    d_model: int = 128
    n_heads: int = 4
    n_enc_layers: int = 3
    n_post_layers: int = 1
    n_dec_layers: int = 2
    d_latent: int = 32
    dropout: float = 0.1

    def __post_init__(self):
        # obs_frames=0 would anchor predictions on real defense from the last frame.
        if not 1 <= self.obs_frames < self.max_frames:
            raise ValueError(f'obs_frames must be in [1, max_frames={self.max_frames}), got {self.obs_frames}')


class FactorizedBlock(nn.Module):
    """Temporal attention within each entity, then social attention within each frame."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        layer = lambda: nn.TransformerEncoderLayer(  # noqa: E731
            d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.temporal = layer()
        self.social = layer()

    def forward(self, x: torch.Tensor, time_pad: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        x = x.transpose(1, 2).reshape(B * N, T, D)
        x = self.temporal(x, src_key_padding_mask=time_pad.repeat_interleave(N, dim=0))
        x = x.reshape(B, N, T, D).transpose(1, 2).reshape(B * T, N, D)
        return self.social(x).reshape(B, T, N, D)


def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians, summed over the last dim."""
    return 0.5 * (logvar_p - logvar_q
                  + (logvar_q.exp() + (mu_q - mu_p) ** 2) / logvar_p.exp() - 1).sum(-1)


class GhostDefense(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.register_buffer('feature_scale', torch.tensor(FEATURE_SCALE), persistent=False)
        self.register_buffer('is_defense', torch.from_numpy(SIDES == SIDE_DEFENSE), persistent=False)
        self.register_buffer('sides', torch.from_numpy(SIDES), persistent=False)

        self.dyn_in = nn.Linear(cfg.n_dyn, d)
        self.glob_in = nn.Linear(cfg.n_glob, d)
        self.group_emb = nn.Embedding(len(GROUP_VOCAB), d)
        self.side_emb = nn.Embedding(3, d)
        self.time_emb = nn.Embedding(cfg.max_frames, d)
        self.hidden_emb = nn.Parameter(torch.zeros(d))

        blocks = lambda n: nn.ModuleList(  # noqa: E731
            FactorizedBlock(d, cfg.n_heads, cfg.dropout) for _ in range(n))
        self.encoder = blocks(cfg.n_enc_layers)
        self.post_encoder = blocks(cfg.n_post_layers)
        self.decoder = blocks(cfg.n_dec_layers)

        self.coverage_emb = nn.Embedding(cfg.n_coverage, d)
        self.coverage_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.n_coverage))
        self.prior_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 2 * cfg.d_latent))
        self.post_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 2 * cfg.d_latent))
        self.latent_in = nn.Linear(cfg.d_latent, d)
        self.out = nn.Linear(d, 2)

    def hidden_mask(self, T: int, device) -> torch.Tensor:
        """[T, 23] True where a token is a defender frame the model may not see."""
        late = torch.arange(T, device=device) >= self.cfg.obs_frames
        return late[:, None] & self.is_defense[None, :]

    def embed(self, batch: dict, hide_defense: bool) -> torch.Tensor:
        dyn, glob, group = batch['dyn'], batch['glob'], batch['group']
        B, T = dyn.shape[:2]
        x = dyn / self.feature_scale
        tok = (self.group_emb(group)[:, None] + self.side_emb(self.sides)
               + self.glob_in(glob)[:, None, None] + self.time_emb.weight[:T, None])
        if hide_defense:
            hidden = self.hidden_mask(T, dyn.device)
            x = x.masked_fill(hidden[None, :, :, None], 0.0)
            tok = tok + hidden[None, :, :, None] * self.hidden_emb
        return tok + self.dyn_in(x)

    @staticmethod
    def run(blocks: nn.ModuleList, x: torch.Tensor, time_pad: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            x = block(x, time_pad)
        return x

    @staticmethod
    def pool(h: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        w = time_mask[:, :, None, None].float()
        return (h * w).sum((1, 2)) / (w.sum((1, 2)) * h.shape[2])

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Context tokens, pooled context and coverage-call logits, from defense-masked input only.

        The call logits come from a separate pass over only the first `obs_frames`
        frames. Later offense movement reacts to the coverage (hot routes, sight
        adjustments, quick throws), so letting p(c | context) see it would let
        the ghost read the real call back off the offense.
        """
        h = self.run(self.encoder, self.embed(batch, hide_defense=True), ~batch['time_mask'])
        ctx = self.pool(h, batch['time_mask'])
        n = self.cfg.obs_frames
        snap = {key: batch[key][:, :n] for key in ('dyn', 'time_mask')}
        snap.update(group=batch['group'], glob=batch['glob'])
        h_snap = self.run(self.encoder, self.embed(snap, hide_defense=True), ~snap['time_mask'])
        return h, ctx, self.coverage_head(self.pool(h_snap, snap['time_mask']))

    def prior(self, ctx: torch.Tensor, coverage: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """p(z | context, c)."""
        return self.prior_head(torch.cat([ctx, self.coverage_emb(coverage)], -1)).chunk(2, dim=-1)

    def posterior(self, batch: dict, ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q(z | context, true defense, c); training only."""
        g = self.run(self.post_encoder, self.embed(batch, hide_defense=False), ~batch['time_mask'])
        post_ctx = torch.cat([ctx, self.pool(g, batch['time_mask']), self.coverage_emb(batch['coverage'])], -1)
        return self.post_head(post_ctx).chunk(2, dim=-1)

    def decode(self, h: torch.Tensor, z: torch.Tensor, coverage: torch.Tensor, batch: dict) -> torch.Tensor:
        """Defender positions [B, T, 11, 2] in yards, as offsets from the last observed frame."""
        x = h + (self.latent_in(z) + self.coverage_emb(coverage))[:, None, None]
        x = self.run(self.decoder, x, ~batch['time_mask'])
        anchor = batch['dyn'][:, self.cfg.obs_frames - 1, DEF, :2]
        return anchor[:, None] + POS_SCALE * self.out(x[:, :, DEF])

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """Training pass: decode from a posterior sample with the true call; return loss terms.

        In eval mode the posterior mean is decoded instead of a sample, so
        validation loss is deterministic and checkpoint selection doesn't
        chase Monte Carlo noise.
        """
        h, ctx, coverage_logits = self.encode(batch)
        mu_p, logvar_p = self.prior(ctx, batch['coverage'])
        mu_q, logvar_q = self.posterior(batch, ctx)
        z = mu_q + torch.randn_like(mu_q) * (0.5 * logvar_q).exp() if self.training else mu_q
        pred = self.decode(h, z, batch['coverage'], batch)
        return {'pred': pred, 'kl': gaussian_kl(mu_q, logvar_q, mu_p, logvar_p),
                'coverage_logits': coverage_logits}

    @torch.no_grad()
    def sample(self, batch: dict, k: int, coverage: torch.Tensor | None = None,
               generator: torch.Generator | None = None) -> torch.Tensor:
        """K ghost-defense futures from the prior: [B, K, T, 11, 2].

        By default each sample draws its own call from p(c | context) and never
        reads batch['coverage']. Pass `coverage` [B] to fix the call instead.
        The z noise is drawn before the call, so two calls with identically
        seeded generators share it (common random numbers).
        """
        h, ctx, coverage_logits = self.encode(batch)
        B = h.shape[0]
        rep = lambda t: t.repeat_interleave(k, dim=0)  # noqa: E731
        eps = torch.randn(B * k, self.cfg.d_latent, device=h.device, generator=generator)
        if coverage is None:
            c = torch.multinomial(coverage_logits.softmax(-1), k, replacement=True,
                                  generator=generator).reshape(B * k)
        else:
            c = rep(coverage)
        mu, logvar = self.prior(rep(ctx), c)
        z = mu + eps * (0.5 * logvar).exp()
        batch_k = {'dyn': rep(batch['dyn']), 'time_mask': rep(batch['time_mask'])}
        pred = self.decode(rep(h), z, c, batch_k)
        return pred.reshape(B, k, *pred.shape[1:])


def future_mask(time_mask: torch.Tensor, obs_frames: int) -> torch.Tensor:
    """[B, T] True for valid frames the model has to predict."""
    t = torch.arange(time_mask.shape[1], device=time_mask.device)
    return time_mask & (t >= obs_frames)[None]


def elbo_loss(out: dict, batch: dict, obs_frames: int, beta: float,
              sigma: float = 1.0) -> dict[str, torch.Tensor]:
    """Negative ELBO per play with a fixed-variance Gaussian likelihood (sigma in yards).

    The coverage call is observed, so its prior log-likelihood -log p(c | context)
    enters the bound directly (unweighted by beta).
    """
    target = batch['dyn'][:, :, DEF, :2]
    mask = future_mask(batch['time_mask'], obs_frames)[:, :, None].float()
    sq_err = ((out['pred'] - target) ** 2).sum(-1)
    recon = 0.5 * (sq_err * mask).sum((1, 2)) / sigma ** 2
    coverage_nll = F.cross_entropy(out['coverage_logits'], batch['coverage'], reduction='none')
    n = mask.sum() * target.shape[2]
    return {
        'loss': (recon + beta * out['kl'] + coverage_nll).mean(),
        'recon': recon.mean(),
        'kl': out['kl'].mean(),
        'coverage_nll': coverage_nll.mean(),
        'coverage_acc': (out['coverage_logits'].argmax(-1) == batch['coverage']).float().mean(),
        'ade': (sq_err.sqrt() * mask).sum() / n,
    }


def sample_ade(samples: torch.Tensor, batch: dict, obs_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean ADE over samples, min-over-samples ADE), averaged over plays, in yards."""
    target = batch['dyn'][:, None, :, DEF, :2]
    mask = future_mask(batch['time_mask'], obs_frames)[:, None, :, None].float()
    dist = (samples - target).norm(dim=-1)
    ade = (dist * mask).sum((2, 3)) / (mask.sum((2, 3)) * dist.shape[-1])  # [B, K]
    return ade.mean(), ade.min(dim=1).values.mean()

