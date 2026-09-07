"""
DTG v2 — Digital Twin Generator rebuilt with official unlearn.ai NBM.

Architecture faithful to arXiv:2405.01488:
  1. ImputationModule  — autoencoder handling sparse observations
  2. NBM (official)    — BiasNet=mean predictor, PrecisionNet, WeightsNet
     Context per timepoint: [z_imputed (32), c (9), t_norm (1)] -> ny=d features
  3. TTE Head          — DeepHit discrete survival (60 bins)

Key fixes vs v1:
  - PrecisionNet: log-clip [log(1e-3), log(1e3)] then exp  (was softplus+0.5)
  - WeightsNet:   single linear layer / sqrt(ny)           (was deep MLP)
  - Hidden units: Ising {-1,+1}                            (was Bernoulli {0,1})
  - CD loss:      per-timepoint on observed visits          (was energy clamping)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nbm import NBM


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mlp(dims, dropout=0.0):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ── 1. Imputation Module ──────────────────────────────────────────────────────

class ImputationModule(nn.Module):
    """Masked autoencoder — encodes observed baseline into z (z_dim=32)."""

    def __init__(self, d: int, z_dim: int = 32):
        super().__init__()
        self.d     = d
        self.z_dim = z_dim
        self.encoder = _mlp([d * 2, 128, 64, z_dim])
        self.decoder = _mlp([z_dim, 64, 128, d])

    def forward(self, x0_obs, mask0):
        inp  = torch.cat([x0_obs * mask0, mask0], dim=-1)
        z    = self.encoder(inp)
        x_hat = self.decoder(z)

        # Reconstruction loss — drop 30% of observed features artificially
        B, d   = x0_obs.shape
        drop   = (torch.rand(B, d, device=x0_obs.device) > 0.3).float()
        masked = drop * mask0
        inp2   = torch.cat([x0_obs * masked, masked], dim=-1)
        x_hat2 = self.decoder(self.encoder(inp2))

        obs    = mask0.bool()
        imp_loss = F.mse_loss(x_hat2[obs], x0_obs[obs]) if obs.any() else \
                   torch.tensor(0.0, device=x0_obs.device)
        return z, x_hat, imp_loss


# ── 2. TTE Head (DeepHit) ────────────────────────────────────────────────────

class TTEHead(nn.Module):
    """Discrete-time survival prediction over S bins (DeepHit-style).

    Input: concat of [z_latent, x0_baseline, c_static].
    Giving x0 directly lets the head use raw UPDRS scores without compressing
    through z, which significantly improves C-index in high-dim feature settings.
    """

    def __init__(self, z_dim: int, d: int, c_dim: int, S: int = 60):
        super().__init__()
        self.S   = S
        self.net = _mlp([z_dim + d + c_dim, 256, 128, S], dropout=0.15)

    def forward(self, z, x0, c):
        return self.net(torch.cat([z, x0, c], dim=-1))

    def ranking_loss(self, logits, event_times, event_inds, sigma=0.1):
        S      = self.S
        device = logits.device
        B      = logits.size(0)

        pmf    = torch.softmax(logits, dim=-1)
        bins   = torch.linspace(0, 1, S, device=device)

        # Negative log-likelihood at event bin
        et_idx = (event_times * (S - 1)).long().clamp(0, S - 1)
        nll    = -torch.log(pmf[torch.arange(B), et_idx] + 1e-8)
        nll    = (nll * event_inds).mean()

        # Pairwise ranking
        E_T    = (pmf * bins).sum(dim=-1)
        ri     = E_T.unsqueeze(0) - E_T.unsqueeze(1)          # (B,B)
        ti_tj  = event_times.unsqueeze(0) - event_times.unsqueeze(1)
        ei     = event_inds.unsqueeze(1).expand(B, B)
        # pair[i,j]=1 when patient i has event AND patient i died BEFORE patient j
        pair   = ei * (ti_tj > 0).float()
        rank   = (pair * torch.exp(-ri / sigma)).mean()

        return nll + rank


# ── 3. DTG v2 ─────────────────────────────────────────────────────────────────

class DTG_v2(nn.Module):
    """
    Digital Twin Generator v2.

    Parameters
    ----------
    d     : int   number of longitudinal features
    c_dim : int   static context dimension
    T     : int   number of timepoints
    nh    : int   NBM hidden units (paper uses 64)
    z_dim : int   imputation latent dimension
    S     : int   TTE bins
    """

    def __init__(self, d: int, c_dim: int, T: int = 6,
                 nh: int = 64, z_dim: int = 64, S: int = 60):
        super().__init__()
        self.d     = d
        self.c_dim = c_dim
        self.T     = T
        self.nh    = nh
        self.z_dim = z_dim
        self.S     = S

        self.imputer  = ImputationModule(d, z_dim)

        # NBM context = [z, c, t_norm]  ->  features at time t
        nx = z_dim + c_dim + 1
        self.nbm = NBM(nx=nx, ny=d, nh=nh, visible_unit_type="gaussian")

        # TTE head: z + raw x0 baseline + c for richer survival signal
        self.tte_head = TTEHead(z_dim, d, c_dim, S)

    # ── Forward ───────────────────────────────────────────────────────────────

    def _context(self, z, c, t_norm):
        """Build per-timepoint context vector."""
        B = z.size(0)
        t_vec = torch.full((B, 1), float(t_norm), device=z.device)
        return torch.cat([z, c, t_vec], dim=-1)

    def forward(self, x0_obs, mask0, c, times):
        """
        Returns
        -------
        mu       : (B, T, d)  mean trajectory from NBM bias
        x0_hat   : (B, d)     imputed baseline
        tte_logits: (B, S)    survival logits
        """
        B       = x0_obs.size(0)
        t_max   = float(times[-1]) if times[-1] > 0 else 1.0
        z, x0_hat, _ = self.imputer(x0_obs, mask0)

        mu = torch.zeros(B, self.T, self.d, device=x0_obs.device)
        for ti, t in enumerate(times):
            ctx       = self._context(z, c, float(t) / t_max)
            mu[:, ti, :] = self.nbm.mean(ctx)

        tte_logits = self.tte_head(z, x0_obs, c)
        return mu, x0_hat, tte_logits

    # ── Losses ────────────────────────────────────────────────────────────────

    def compute_losses(self, batch, times, lambdas=None):
        if lambdas is None:
            lambdas = {"imp": 1.0, "pred": 2.0, "cd": 0.05, "tte": 0.5}

        x0_obs  = batch["x0_obs"]
        mask0   = batch["mask0"]
        c       = batch["c"]
        X_traj  = batch["X_traj"]
        M_traj  = batch["M_traj"]
        et      = batch["event_time"]
        ei      = batch["event_ind"]

        t_max   = float(times[-1]) if times[-1] > 0 else 1.0
        z, x0_hat, imp_loss = self.imputer(x0_obs, mask0)

        pred_loss = torch.tensor(0.0, device=x0_obs.device)
        cd_loss   = torch.tensor(0.0, device=x0_obs.device)
        n_steps   = 0

        for ti, t in enumerate(times):
            obs_pat = M_traj[:, ti, :].any(dim=-1)
            if not obs_pat.any():
                continue
            ctx  = self._context(z[obs_pat], c[obs_pat], float(t) / t_max)
            y    = X_traj[obs_pat, ti, :]
            m    = M_traj[obs_pat, ti, :]

            bias = self.nbm.bias_net(ctx)

            # Masked MSE: divide by observed count, not total d (avoids 3x gradient dilution)
            obs_count = m.sum() + 1e-8
            mse = ((bias - y).pow(2) * m).sum() / obs_count
            pred_loss = pred_loss + mse

            # CD with effective y: unobserved features use bias prediction so they
            # contribute zero energy to positive phase (same as negative phase).
            y_eff = y * m + bias.detach() * (1.0 - m)
            res = self.nbm.compute_loss(y_eff, ctx, mc_steps=4)
            if torch.isfinite(res["CD_loss"]):
                cd_loss = cd_loss + res["CD_loss"].clamp(-2.0, 2.0)

            n_steps += 1

        if n_steps > 0:
            pred_loss = pred_loss / n_steps
            cd_loss   = cd_loss   / n_steps

        # TTE loss — x0_obs masked so unobserved features stay zero
        tte_logits = self.tte_head(z, x0_obs * mask0, c)
        tte_loss   = self.tte_head.ranking_loss(tte_logits, et, ei)

        total = (lambdas["imp"]  * imp_loss  +
                 lambdas["pred"] * pred_loss +
                 lambdas["cd"]   * cd_loss   +
                 lambdas["tte"]  * tte_loss)

        if not torch.isfinite(total):
            total = torch.tensor(0.0, device=x0_obs.device, requires_grad=True)

        return {"total": total, "imp": imp_loss, "pred": pred_loss,
                "cd": cd_loss, "tte": tte_loss}

    # ── Digital twin generation ───────────────────────────────────────────────

    @torch.no_grad()
    def generate_twin(self, x0_obs, mask0, c, times, n_samples: int = 50):
        """
        Sample n_samples stochastic trajectories per patient.

        Returns
        -------
        (B, n_samples, T, d)
        """
        B     = x0_obs.size(0)
        t_max = float(times[-1]) if times[-1] > 0 else 1.0
        z, _, _ = self.imputer(x0_obs, mask0)

        out = torch.zeros(B, n_samples, self.T, self.d, device=x0_obs.device)
        for ti, t in enumerate(times):
            ctx = self._context(z, c, float(t) / t_max)
            for s in range(n_samples):
                out[:, s, ti, :] = self.nbm.sample(ctx, mc_steps=32, denoise=False)
        return out