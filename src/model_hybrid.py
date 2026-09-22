"""
Hybrid architecture: the parts of each model that are carried by evidence.

Taken from the deployed platform model
--------------------------------------
  * Sinusoidal time encoding. The research model feeds the query time as a
    single scalar t/t_max, which makes any non-linear time dependence something
    the network must construct from one input. The platform model encodes it as
    [t, sin 2*pi*t, cos 2*pi*t, sin 4*pi*t, cos 4*pi*t], which is the standard
    Fourier feature construction and makes curvature in time directly available.

  * A dedicated point predictor with residual blocks, rather than the two-layer
    bias network of the reference NBM. Depth here is cheap and the target is the
    thing the whole model is judged on.

  * Huber loss instead of squared error on the trajectory. Squared error is
    dominated by the few large misses the ordinal items produce; Huber is
    quadratic within one standard deviation and linear beyond it, so those
    misses stop steering the fit.

Taken from the research model
-----------------------------
  * Trajectory pooling into a DeepHit head. This is what produces a concordance
    of 0.90; the platform's Weibull AFT head has no equivalent published figure.
  * The masked denoising autoencoder with an explicit missingness mask.
  * The aggregated subscale panel.
  * The three NBM corrections made in this work: a tanh bound on the contrastive
    term instead of a hard clamp, a precision range matched to standardized
    units, and per-dimension normalisation of the contrastive loss.

Structure
---------
    baseline + mask -> imputer -> z
    [z | c | fourier(t)] -> flow -> mu(t)          (point prediction)
    mu(t) is also the centre of the energy landscape; the NBM models the
    residual y - mu(t), so sampling is mu(t) + nbm.sample().
    pooled (mu_bar, delta_mu) -> DeepHit -> survival
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_v3 import DTG_v3

_CD_BOUND = 2.0


def fourier_time(t_scalar, batch, device):
    """(B,5) Fourier encoding of a normalized scalar time in [0,1]."""
    t = torch.full((batch, 1), float(t_scalar), device=device)
    two_pi = 2.0 * math.pi
    return torch.cat([t,
                      torch.sin(two_pi * t), torch.cos(two_pi * t),
                      torch.sin(2 * two_pi * t), torch.cos(2 * two_pi * t)],
                     dim=-1)


class FlowPredictor(nn.Module):
    """Residual-block point predictor: (z, c, fourier(t)) -> mu."""

    def __init__(self, n_in, n_out, hidden=128, n_blocks=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_in, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                          nn.SiLU(), nn.Dropout(dropout),
                          nn.Linear(hidden, hidden))
            for _ in range(n_blocks)
        ])
        self.out_proj = nn.Linear(hidden, n_out)

    def forward(self, x):
        h = self.input_proj(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out_proj(h)


class HybridDTG(DTG_v3):
    """DTG_v3 with a Fourier-time residual flow predictor in place of the
    reference bias network, and Huber trajectory loss."""

    def __init__(self, *a, flow_hidden=128, flow_blocks=3, huber_delta=1.0, **kw):
        super().__init__(*a, **kw)
        # z + static context + 5 time features
        n_in = self.z_dim + self.c_dim + 5
        self.flow = FlowPredictor(n_in, self.d, hidden=flow_hidden,
                                  n_blocks=flow_blocks)
        self.huber_delta = huber_delta

    # ── context and prediction ────────────────────────────────────────────────

    def _ctx_fourier(self, z, c, t_norm):
        te = fourier_time(t_norm, z.size(0), z.device)
        return torch.cat([z, c, te], dim=-1)

    def _mu(self, z, c, t_norm):
        return self.flow(self._ctx_fourier(z, c, t_norm))

    def _nbm_ctx(self, z, c, t_norm):
        """Context handed to the NBM. Kept at the reference width so nbm.py is
        unchanged: [z | c | t]."""
        t_vec = torch.full((z.size(0), 1), float(t_norm), device=z.device)
        return torch.cat([z, c, t_vec], dim=-1)

    # ── overrides ─────────────────────────────────────────────────────────────

    def forward(self, x0_obs, mask0, c, times):
        z, x0_hat, _ = self.imputer(x0_obs, mask0)
        t_max = float(times[-1]) if float(times[-1]) > 0 else 1.0
        mu = torch.stack([self._mu(z, c, float(t) / t_max) for t in times], dim=1)
        tte_logits = self.tte_head(z, x0_obs * mask0, c,
                                   mu.mean(dim=1), mu[:, -1, :] - mu[:, 0, :])
        return mu, x0_hat, tte_logits

    def compute_losses(self, batch, times, lambdas=None):
        if lambdas is None:
            lambdas = {"imp": 1.0, "pred": 2.0, "cd": 0.1, "tte": 0.5, "var": 0.0}

        x0_obs, mask0, c = batch["x0_obs"], batch["mask0"], batch["c"]
        X_traj, M_traj   = batch["X_traj"], batch["M_traj"]
        et, ei           = batch["event_time"], batch["event_ind"]

        t_max = float(times[-1]) if float(times[-1]) > 0 else 1.0
        z, x0_hat, imp_loss = self.imputer(x0_obs, mask0)

        pred_loss = torch.tensor(0.0, device=x0_obs.device)
        cd_loss   = torch.tensor(0.0, device=x0_obs.device)
        var_loss  = torch.tensor(0.0, device=x0_obs.device)
        all_mu, n_steps = [], 0

        for ti, t in enumerate(times):
            tn   = float(t) / t_max
            mu_t = self._mu(z, c, tn)
            all_mu.append(mu_t)

            obs_pat = M_traj[:, ti, :].any(dim=-1)
            if not obs_pat.any():
                continue

            y = X_traj[obs_pat, ti, :]
            m = M_traj[obs_pat, ti, :]
            b = mu_t[obs_pat]

            # Huber on observed entries only. reduction='none' so the mask can be
            # applied before averaging; averaging first would let unobserved
            # entries dilute the loss.
            per = F.huber_loss(b, y, delta=self.huber_delta, reduction="none")
            pred_loss = pred_loss + (per * m).sum() / (m.sum() + 1e-8)

            # The energy model works on the residual around the flow prediction,
            # so its landscape stays centred on the point prediction without
            # nbm.py needing to know about the flow.
            ctx   = self._nbm_ctx(z, c, tn)[obs_pat]
            y_res = y - b
            y_eff = y_res * m                      # unobserved -> 0 residual
            res   = self.nbm.compute_loss(y_eff, ctx, mc_steps=4)
            if torch.isfinite(res["CD_loss"]):
                cd_loss = cd_loss + _CD_BOUND * torch.tanh(res["CD_loss"] / _CD_BOUND)
            # Dispersion term. The contrastive term shapes the landscape but
            # nothing in the objective ties the width of the predictive
            # distribution to the width of the observed residuals, so the
            # sampler is free to be systematically over- or under-dispersed
            # while every per-patient metric stays good. This penalises the
            # mismatch between the model's own variance, 1/P, and the squared
            # residual it should equal in expectation.
            if torch.isfinite(res["var_mse"]):
                var_loss = var_loss + torch.tanh(res["var_mse"] / _CD_BOUND) * _CD_BOUND
            n_steps += 1

        if n_steps > 0:
            pred_loss = pred_loss / n_steps
            cd_loss   = cd_loss / n_steps
            var_loss  = var_loss / n_steps

        mu_all = torch.stack(all_mu, dim=1)
        tte_logits = self.tte_head(z, x0_obs * mask0, c,
                                   mu_all.mean(dim=1),
                                   mu_all[:, -1, :] - mu_all[:, 0, :])
        tte_loss = self.tte_head.ranking_loss(tte_logits, et, ei)

        total = (lambdas["imp"]  * imp_loss  + lambdas["pred"] * pred_loss +
                 lambdas["cd"]   * cd_loss   + lambdas["tte"]  * tte_loss +
                 lambdas.get("var", 0.0) * var_loss)
        if not torch.isfinite(total):
            total = torch.tensor(0.0, device=x0_obs.device, requires_grad=True)

        return {"total": total, "imp": imp_loss, "pred": pred_loss,
                "cd": cd_loss, "tte": tte_loss, "var": var_loss}

    @torch.no_grad()
    def generate_twin(self, x0_obs, mask0, c, times, n_samples: int = 50):
        B = x0_obs.size(0)
        t_max = float(times[-1]) if float(times[-1]) > 0 else 1.0
        z, _, _ = self.imputer(x0_obs, mask0)
        out = torch.zeros(B, n_samples, self.T, self.d, device=x0_obs.device)
        for ti, t in enumerate(times):
            tn   = float(t) / t_max
            mu_t = self._mu(z, c, tn)
            ctx  = self._nbm_ctx(z, c, tn)
            for s in range(n_samples):
                out[:, s, ti, :] = mu_t + self.nbm.sample(ctx, mc_steps=32,
                                                          denoise=False)
        return out
