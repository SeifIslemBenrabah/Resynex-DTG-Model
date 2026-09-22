"""
DTG v3 — Improved Digital Twin Generator targeting C-index > 0.79.

Key improvements over v2:
  1. Trajectory-pooled TTE head: [z, x0, c, mu_mean, mu_delta]
     NBM trajectory directly informs survival prediction via co-adaptation gradient.
  2. Deeper imputer with LayerNorm for stable gradient flow (z_dim=128 default).
  3. Curriculum lambda_tte: 0 for first W warm-up epochs, then joint training.
     Avoids the instability of two-phase (frozen repr => TTE can't co-adapt).
  4. forward() returns trajectory mu used for both RMSE eval and TTE pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nbm import NBM

# Bound applied to the contrastive-divergence term. Applied through tanh rather
# than clamp so the gradient survives saturation; see compute_losses.
_CD_BOUND = 2.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mlp(dims, dropout=0.0, use_ln=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if use_ln:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ── 1. Imputation Module (v3: deeper + LayerNorm) ─────────────────────────────

class ImputationModule_v3(nn.Module):
    """Deeper masked autoencoder with LayerNorm for stable gradients."""

    def __init__(self, d: int, z_dim: int = 128):
        super().__init__()
        self.d     = d
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(d * 2, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128),   nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, z_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128), nn.GELU(),
            nn.Linear(128, 256),   nn.GELU(),
            nn.Linear(256, d),
        )

    def forward(self, x0_obs, mask0):
        inp   = torch.cat([x0_obs * mask0, mask0], dim=-1)
        z     = self.encoder(inp)
        x_hat = self.decoder(z)

        # Denoising reconstruction loss: randomly drop 30% of observed features
        B, d = x0_obs.shape
        drop    = (torch.rand(B, d, device=x0_obs.device) > 0.3).float()
        masked  = drop * mask0
        inp2    = torch.cat([x0_obs * masked, masked], dim=-1)
        x_hat2  = self.decoder(self.encoder(inp2))

        obs = mask0.bool()
        imp_loss = F.mse_loss(x_hat2[obs], x0_obs[obs]) if obs.any() else \
                   torch.tensor(0.0, device=x0_obs.device)
        return z, x_hat, imp_loss


# ── 2. TTE Head v3 (trajectory-pooled DeepHit) ────────────────────────────────

class TTEHead_v3(nn.Module):
    """DeepHit TTE that receives full trajectory pooling signals.

    Input = [z | x0*mask | c | mu_mean | mu_delta]
    where:
      mu_mean  = average NBM bias across all T timepoints (disease level)
      mu_delta = mu_T[-1] - mu_T[0]                     (progression rate)

    The gradient from TTE loss flows through mu_mean/mu_delta back into the NBM
    bias_net — the key co-adaptation mechanism missing from two-phase training.
    """

    def __init__(self, z_dim: int, d: int, c_dim: int, S: int = 60):
        super().__init__()
        self.S       = S
        total_in = z_dim + d + c_dim + 2 * d  # +mu_mean +mu_delta
        self.net = _mlp([total_in, 512, 256, 128, S], dropout=0.25, use_ln=True)

    def forward(self, z, x0_masked, c, mu_mean, mu_delta):
        inp = torch.cat([z, x0_masked, c, mu_mean, mu_delta], dim=-1)
        return self.net(inp)

    def ranking_loss(self, logits, event_times, event_inds, sigma=0.1):
        S      = self.S
        device = logits.device
        B      = logits.size(0)

        pmf  = torch.softmax(logits, dim=-1)
        bins = torch.linspace(0, 1, S, device=device)

        et_idx = (event_times * (S - 1)).long().clamp(0, S - 1)

        # Likelihood, DeepHit's two-part form. Event patients are rewarded
        # for mass at their observed event bin. Censored patients previously
        # contributed nothing here (only to the ranking term below), so
        # nothing taught the model to keep predicted-event probability low
        # for a patient who went years without the event -- this second term
        # rewards placing the survival mass (the probability of the event
        # occurring after the patient's last observed, event-free visit)
        # correctly for them instead.
        event_nll  = -torch.log(pmf[torch.arange(B), et_idx] + 1e-8)
        surv_after = 1.0 - torch.cumsum(pmf, dim=-1)
        censor_nll = -torch.log(surv_after[torch.arange(B), et_idx].clamp(min=1e-8))
        nll = (event_nll * event_inds + censor_nll * (1.0 - event_inds)).mean()

        # Pairwise ranking loss (DeepHit)
        E_T    = (pmf * bins).sum(dim=-1)
        ri     = E_T.unsqueeze(0) - E_T.unsqueeze(1)
        ti_tj  = event_times.unsqueeze(0) - event_times.unsqueeze(1)
        ei     = event_inds.unsqueeze(1).expand(B, B)
        # pair[i,j]=1 when patient i has event AND patient i died BEFORE patient j
        # → concordant: patient with event and earlier death should have lower E_T
        pair   = ei * (ti_tj > 0).float()
        rank   = (pair * torch.exp(-ri / sigma)).mean()

        return nll + rank


# ── 3. DTG v3 ─────────────────────────────────────────────────────────────────

class DTG_v3(nn.Module):
    """
    Digital Twin Generator v3.

    Parameters
    ----------
    d       : int   longitudinal feature dimension
    c_dim   : int   static context dimension
    T       : int   number of timepoints
    nh      : int   NBM hidden units
    z_dim   : int   imputation latent dimension (default 128)
    S       : int   TTE bins (60)
    """

    def __init__(self, d: int, c_dim: int, T: int = 6,
                 nh: int = 64, z_dim: int = 128, S: int = 60):
        super().__init__()
        self.d     = d
        self.c_dim = c_dim
        self.T     = T
        self.nh    = nh
        self.z_dim = z_dim
        self.S     = S

        self.imputer  = ImputationModule_v3(d, z_dim)

        nx = z_dim + c_dim + 1
        self.nbm = NBM(nx=nx, ny=d, nh=nh, visible_unit_type="gaussian")

        self.tte_head = TTEHead_v3(z_dim, d, c_dim, S)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _context(self, z, c, t_norm):
        B     = z.size(0)
        t_vec = torch.full((B, 1), float(t_norm), device=z.device)
        return torch.cat([z, c, t_vec], dim=-1)

    def _build_traj_pool(self, z, c, times):
        """Return mu (B,T,d), mu_mean (B,d), mu_delta (B,d) from NBM bias_net."""
        t_max = float(times[-1]) if float(times[-1]) > 0 else 1.0
        all_bias = []
        for t in times:
            ctx = self._context(z, c, float(t) / t_max)
            all_bias.append(self.nbm.bias_net(ctx))
        mu       = torch.stack(all_bias, dim=1)    # (B, T, d)
        mu_mean  = mu.mean(dim=1)                  # (B, d)
        mu_delta = mu[:, -1, :] - mu[:, 0, :]     # (B, d)
        return mu, mu_mean, mu_delta

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x0_obs, mask0, c, times):
        """
        Returns
        -------
        mu          : (B, T, d)  mean trajectory (NBM bias)
        x0_hat      : (B, d)     imputed baseline
        tte_logits  : (B, S)     discrete survival logits
        """
        z, x0_hat, _ = self.imputer(x0_obs, mask0)
        mu, mu_mean, mu_delta = self._build_traj_pool(z, c, times)
        tte_logits = self.tte_head(z, x0_obs * mask0, c, mu_mean, mu_delta)
        return mu, x0_hat, tte_logits

    # ── Losses ────────────────────────────────────────────────────────────────

    def compute_losses(self, batch, times, lambdas=None):
        if lambdas is None:
            lambdas = {"imp": 1.0, "pred": 2.0, "cd": 0.05, "tte": 0.5}

        x0_obs = batch["x0_obs"]
        mask0  = batch["mask0"]
        c      = batch["c"]
        X_traj = batch["X_traj"]
        M_traj = batch["M_traj"]
        et     = batch["event_time"]
        ei     = batch["event_ind"]

        t_max  = float(times[-1]) if float(times[-1]) > 0 else 1.0
        z, x0_hat, imp_loss = self.imputer(x0_obs, mask0)

        # ── Trajectory losses ─────────────────────────────────────────────────
        pred_loss = torch.tensor(0.0, device=x0_obs.device)
        cd_loss   = torch.tensor(0.0, device=x0_obs.device)
        all_bias  = []   # collect biases for trajectory pooling (used by TTE below)
        n_steps   = 0

        for ti, t in enumerate(times):
            ctx  = self._context(z, c, float(t) / t_max)
            bias = self.nbm.bias_net(ctx)       # (B, d) — differentiable
            all_bias.append(bias)

            obs_pat = M_traj[:, ti, :].any(dim=-1)
            if not obs_pat.any():
                continue

            y = X_traj[obs_pat, ti, :]
            m = M_traj[obs_pat, ti, :]
            b = bias[obs_pat]

            # Masked MSE divided by observed count
            obs_count = m.sum() + 1e-8
            mse = ((b - y).pow(2) * m).sum() / obs_count
            pred_loss = pred_loss + mse

            # CD loss with effective y (unobserved positions use bias → zero CD energy)
            ctx_obs = ctx[obs_pat]
            y_eff   = y * m + b.detach() * (1.0 - m)
            res     = self.nbm.compute_loss(y_eff, ctx_obs, mc_steps=4)
            if torch.isfinite(res["CD_loss"]):
                # Bound the contrastive-divergence term smoothly rather than with
                # a hard clamp. torch.clamp has exactly zero gradient outside its
                # range, so once this term saturates the energy model stops
                # receiving any learning signal at all while still appearing in
                # the reported loss. tanh has the same bounding effect and keeps
                # a non-zero gradient everywhere, so a saturated step slows the
                # energy model rather than switching it off.
                cd_loss = cd_loss + _CD_BOUND * torch.tanh(res["CD_loss"] / _CD_BOUND)

            n_steps += 1

        if n_steps > 0:
            pred_loss = pred_loss / n_steps
            cd_loss   = cd_loss   / n_steps

        # ── TTE loss with trajectory pooling ─────────────────────────────────
        mu_all   = torch.stack(all_bias, dim=1)     # (B, T, d) — reuses computed biases
        mu_mean  = mu_all.mean(dim=1)               # (B, d)
        mu_delta = mu_all[:, -1, :] - mu_all[:, 0, :]  # (B, d)

        tte_logits = self.tte_head(z, x0_obs * mask0, c, mu_mean, mu_delta)
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
        """Sample n_samples stochastic trajectories. Returns (B, n_samples, T, d)."""
        B     = x0_obs.size(0)
        t_max = float(times[-1]) if float(times[-1]) > 0 else 1.0
        z, _, _ = self.imputer(x0_obs, mask0)
        out = torch.zeros(B, n_samples, self.T, self.d, device=x0_obs.device)
        for ti, t in enumerate(times):
            ctx = self._context(z, c, float(t) / t_max)
            for s in range(n_samples):
                out[:, s, ti, :] = self.nbm.sample(ctx, mc_steps=32, denoise=False)
        return out