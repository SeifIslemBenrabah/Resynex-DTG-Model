"""
Training wrapper for PlatformDTG.

The architecture itself is defined once, in the service's own inference module,
and imported here. This file adds only what training needs and inference does
not: the loss. Keeping the class definition in a single place means a model
trained here cannot drift from the class the service reconstructs to load it.

Loss terms
----------
  imp    reconstruction of the imputer
  pred   Huber on the observed outcomes, robust to the few large misses the
         ordinal-derived outcomes produce
  cd     contrastive divergence, bounded through tanh and normalised per
         visible unit
  var    dispersion: the mismatch between the model's own variance, 1/P, and
         the squared residual it should equal in expectation. NBMCore exposes
         cd_loss but no variance diagnostic, so this is computed here. Without
         it nothing ties the width of the predictive distribution to the width
         of the observed residuals, and the sampler is free to be systematically
         over- or under-dispersed while every per-patient metric stays good.
  tte    DeepHit ranking loss on the pooled trajectory, phased in by curriculum
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# PlatformDTG is defined once, in the platform repository's inference module
# (Resynex-Platform/ms-digital-twin/model/platform_dtg.py), and imported here
# rather than redefined, so a model trained here cannot drift from the class
# the service reconstructs to load it. Point RESYNEX_PLATFORM_PATH at that
# ms-digital-twin/ directory if it isn't cloned as a sibling of this repo.
_platform_path = os.environ.get("RESYNEX_PLATFORM_PATH") or str(
    Path(__file__).resolve().parents[2].parent / "Resynex-Platform" / "ms-digital-twin")
sys.path.insert(0, _platform_path)
try:
    from model.platform_dtg import PlatformDTG as _PlatformDTG, _CD_BOUND  # noqa: E402
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        f"Could not import PlatformDTG from '{_platform_path}'. Clone "
        "Resynex-Platform as a sibling of this repository, or set the "
        "RESYNEX_PLATFORM_PATH environment variable to its ms-digital-twin/ "
        "directory."
    ) from e

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from model_v3 import TTEHead_v3                                        # noqa: E402


class PlatformDTG(_PlatformDTG):
    """The deployed architecture, plus the training objective."""

    def _ranking_loss(self, logits, event_time, event_ind, sigma=0.1):
        # The head shipped to the service carries no loss, so the reference
        # DeepHit ranking loss is reused from the research implementation.
        return TTEHead_v3.ranking_loss(self, logits, event_time, event_ind,
                                       sigma)

    def losses(self, baseline, t_future, outcomes, event_time, event_ind,
               lambdas=None, mc_steps=4):
        if lambdas is None:
            lambdas = {"imp": 1.0, "pred": 2.0, "cd": 0.1, "var": 0.3,
                       "tte": 0.5}
        dev = baseline.device

        x_norm = self._norm_baseline(baseline)
        x_imp, imp_loss = self.imputer(x_norm)

        y_norm = self._norm_outcome(outcomes)
        obs = (~torch.isnan(y_norm)).float()
        y_fill = torch.nan_to_num(y_norm)
        denom = obs.sum().clamp(min=1.0)

        y_flow = self.flow(x_imp, t_future)
        per = F.huber_loss(y_flow, y_fill, delta=1.0, reduction="none")
        pred_loss = (per * obs).sum() / denom

        ctx = self._build_context(x_imp, t_future)
        resid = y_fill - y_flow.detach()

        cd = self.nbm.cd_loss(resid, ctx, mc_steps)
        if not torch.isfinite(cd):
            cd = torch.tensor(0.0, device=dev)

        P = self.nbm.precision_net(ctx)
        var_sq = (1.0 / P - (resid ** 2).clamp(min=1e-6)) ** 2
        var = (var_sq * obs).sum() / denom
        var = (_CD_BOUND * torch.tanh(var / _CD_BOUND)
               if torch.isfinite(var) else torch.tensor(0.0, device=dev))

        mu_bar, d_mu = self._pool(x_imp)
        logits = self._head(x_imp, mu_bar, d_mu)
        tte_loss = self._ranking_loss(logits, event_time, event_ind)

        total = (lambdas["imp"] * imp_loss + lambdas["pred"] * pred_loss +
                 lambdas["cd"] * cd + lambdas.get("var", 0.0) * var +
                 lambdas["tte"] * tte_loss)
        if not torch.isfinite(total):
            total = torch.tensor(0.0, device=dev, requires_grad=True)

        return {"total": total, "imp": imp_loss, "pred": pred_loss,
                "cd": cd, "var": var, "tte": tte_loss}
