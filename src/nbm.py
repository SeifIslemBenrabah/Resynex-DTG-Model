"""
Official Neural Boltzmann Machine from unlearn.ai
Source: https://github.com/unlearnai/neural-boltzmann-machines
"""

import math
import torch
from torch import nn


def logcosh(x):
    return x - math.log(2.0) + torch.nn.functional.softplus(-2.0 * x)


def visible_times_weights(visible, weights):
    return torch.bmm(visible.unsqueeze(dim=1), weights).squeeze(dim=1)


def hidden_times_weights(hidden, weights):
    return torch.bmm(weights, hidden.unsqueeze(dim=2)).squeeze(dim=2)


class BiasNet(nn.Module):
    """Maps context x (nx) -> bias of visible units (ny)."""
    def __init__(self, nx, ny, visible_unit_type="gaussian"):
        super().__init__()
        net = nn.Sequential(nn.Linear(nx, 32), nn.ReLU(), nn.Linear(32, ny))
        if visible_unit_type == "ising":
            net.append(nn.Tanh())
        self.net = net

    def forward(self, x):
        return self.net(x)


class PrecisionNet(nn.Module):
    """Maps context x (nx) -> precision (inverse variance) of visible units (ny).
    Clips in log-space to [log(pmin), log(pmax)] then exponentiates.
    """
    def __init__(self, nx, ny, pmin=1e-3, pmax=1e3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nx, 32), nn.ReLU(), nn.Linear(32, ny))
        self.lpmin = math.log(pmin)
        self.lpmax = math.log(pmax)

    def forward(self, x):
        return self.net(x).clip(self.lpmin, self.lpmax).exp()


class WeightsNet(nn.Module):
    """Maps context x (nx) -> weight matrix (ny, nh), normalised by sqrt(ny).
    Single linear layer — faithful to the official implementation.
    """
    def __init__(self, nx, ny, nh):
        super().__init__()
        self._ny   = ny
        self._nh   = nh
        self._norm = math.sqrt(ny)
        self.net   = nn.Sequential(nn.Linear(nx, ny * nh))

    def forward(self, x):
        return self.net(x).reshape(-1, self._ny, self._nh) / self._norm


class NBM(nn.Module):
    """
    Neural Boltzmann Machine (Gaussian visible, Ising {-1,+1} hidden).

    Energy:  U(y; x) = 0.5 * (y-mu)^T P (y-mu) - sum_j log cosh(W_j^T (y-mu))
    where mu, P, W are outputs of BiasNet, PrecisionNet, WeightsNet given x.
    """

    def __init__(self, nx, ny, nh, visible_unit_type="gaussian"):
        super().__init__()
        assert visible_unit_type in {"gaussian", "ising"}
        self.visible_unit_type = visible_unit_type
        self.bias_net      = BiasNet(nx, ny, visible_unit_type)
        self.precision_net = PrecisionNet(nx, ny)
        self.weights_net   = WeightsNet(nx, ny, nh)
        self._ny = ny

    def _free_energy(self, y, bias, precision, weights):
        diff        = y - bias
        self_energy = 0.5 * (diff * precision * diff).sum(dim=-1)
        phi         = logcosh(visible_times_weights(diff, weights)).sum(dim=-1)
        return self_energy - phi

    @torch.no_grad()
    def _sample_hid(self, y, bias, weights):
        diff   = y - bias
        logits = visible_times_weights(diff, weights)
        proba  = torch.sigmoid(2.0 * logits)
        return 2.0 * torch.bernoulli(proba) - 1.0   # Ising {-1, +1}

    @torch.no_grad()
    def _sample_vis_gaussian(self, h, bias, precision, weights, denoise=False):
        field = hidden_times_weights(h, weights)
        y     = bias + field / precision
        if not denoise:
            y = y + torch.randn_like(bias) / precision.sqrt()
        return y

    def compute_loss(self, y, x, mc_steps=4):
        """Contrastive Divergence loss.

        Args:
            y  : (B, ny)  observed (target) data
            x  : (B, nx)  context (condition)
            mc_steps : int  Gibbs steps for negative phase
        """
        bias      = self.bias_net(x)
        precision = self.precision_net(x)
        weights   = self.weights_net(x)

        y_model = bias.clone().detach()
        for _ in range(max(1, mc_steps)):
            h_model = self._sample_hid(y_model, bias, weights)
            y_model = self._sample_vis_gaussian(h_model, bias, precision, weights)
        y_model = y_model.detach()

        pos = self._free_energy(y.flatten(start_dim=1), bias, precision, weights)
        neg = self._free_energy(y_model,                 bias, precision, weights)
        cd_loss = (pos - neg).mean()

        # Diagnostic auxiliaries
        diff     = y.flatten(start_dim=1) - bias
        bias_mse = (diff * diff).mean()
        var_mse  = ((1.0 / precision - (diff ** 2).clamp(1e-6)).pow(2)).mean()

        return {"CD_loss": cd_loss, "bias_mse": bias_mse, "var_mse": var_mse}

    @torch.no_grad()
    def sample(self, x, mc_steps=32, denoise=False):
        """Generate samples given context x."""
        bias      = self.bias_net(x)
        precision = self.precision_net(x)
        weights   = self.weights_net(x)

        y_model = bias.clone()
        h_model = None
        for _ in range(max(1, mc_steps)):
            h_model = self._sample_hid(y_model, bias, weights)
            y_model = self._sample_vis_gaussian(h_model, bias, precision, weights,
                                                denoise=False)
        if denoise and h_model is not None:
            y_model = self._sample_vis_gaussian(h_model, bias, precision, weights,
                                                denoise=True)
        return y_model

    def mean(self, x):
        """Return the bias (mean of the visible distribution) given x."""
        return self.bias_net(x)