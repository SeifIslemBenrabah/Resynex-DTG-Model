"""
Full metric set for the aggregated-panel model on the held-out test split.

Reports the four metrics the evaluation plan specifies, with patient-level
bootstrap intervals on the two headline figures, plus the per-group breakdown.

Usage:
    python metrics_agg.py --ckpt outputs_var/model_combined.pt
"""

import argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from lifelines.utils import concordance_index

from preprocess_v2 import TIMEPOINTS, c_dim, T, S
from aggregate_panel import build_aggregated, AGG_COLS, d_agg
from train_v3 import PPMIDataset, recompute_events
from model_hybrid import HybridDTG

GROUPS = {
    "Imaging (DaTscan)": ["MIA_CAUDATE_L", "MIA_CAUDATE_R",
                          "MIA_PUTAMEN_L", "MIA_PUTAMEN_R"],
    "CSF biomarkers":    ["asyn", "tau", "ptau"],
    "Cognitive / mood":  ["moca", "gds", "ess"],
    "Summed MDS-UPDRS":  ["NP3_TOTAL", "NP3_TREMOR", "NP3_RIGID", "NP3_BRADY",
                          "NP3_AXIAL", "NP3_BULBAR", "NP2_TOTAL",
                          "NP1C_TOTAL", "NP1P_TOTAL"],
}


@torch.no_grad()
def collect(model, loader, times):
    P, Y, M, R, ET, EI = [], [], [], [], [], []
    model.eval()
    for b in loader:
        mu, _, tte = model(b["x0_obs"], b["mask0"], b["c"], times)
        P.append(mu.numpy()); Y.append(b["X_traj"].numpy())
        M.append(b["M_traj"].numpy())
        pmf = torch.softmax(tte, dim=-1)
        bins = torch.linspace(0, 1, tte.size(1))
        R.append((pmf * bins).sum(dim=-1).numpy())
        ET.append(b["event_time"].numpy()); EI.append(b["event_ind"].numpy())
    return (np.concatenate(P), np.concatenate(Y), np.concatenate(M),
            np.concatenate(R), np.concatenate(ET), np.concatenate(EI))


def rmse_mae(P, Y, M, idx=None):
    if idx is not None:
        P, Y, M = P[idx], Y[idx], M[idx]
    n = M.sum()
    if n == 0:
        return float("nan"), float("nan")
    return (float(np.sqrt((((P - Y) ** 2) * M).sum() / n)),
            float((np.abs(P - Y) * M).sum() / n))


def r2(P, Y, M, cols=None):
    if cols is not None:
        j = [AGG_COLS.index(c) for c in cols if c in AGG_COLS]
        P, Y, M = P[:, :, j], Y[:, :, j], M[:, :, j]
    sse = (((P - Y) ** 2) * M).sum()
    mean = (Y * M).sum(axis=(0, 1)) / np.maximum(M.sum(axis=(0, 1)), 1e-8)
    sst = (((Y - mean) ** 2) * M).sum()
    return float(1 - sse / max(sst, 1e-8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs_var/model_combined.pt")
    p.add_argument("--out",  default="outputs_var/metrics.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--event_thresh", type=float, default=25.0)
    p.add_argument("--boot", type=int, default=2000)
    a = p.parse_args()

    Xa, Ma, C, sc, X69, M69, sc69, _ = build_aggregated(verbose=False)
    ET_, EI_ = recompute_events(X69, M69, sc69, a.event_thresh)
    ds = PPMIDataset(Xa, Ma, C, ET_, EI_)
    N = len(ds); nv = max(1, int(N * .15)); nt = max(1, int(N * .15))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    _tr, _va, te = random_split(ds, [N - nv - nt, nv, nt])
    loader = DataLoader(te, batch_size=16, shuffle=False)

    model = HybridDTG(d=d_agg, c_dim=c_dim, T=T, nh=64, z_dim=128, S=S)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state", ck))
    times = torch.tensor(TIMEPOINTS, dtype=torch.float32)

    P, Y, M, R, ETv, EIv = collect(model, loader, times)
    rm, ma = rmse_mae(P, Y, M)
    c_idx = concordance_index(ETv, R, EIv)

    # patient-level bootstrap
    rng = np.random.default_rng(a.seed)
    n = len(P); bs_r, bs_c = [], []
    for _ in range(a.boot):
        idx = rng.integers(0, n, n)
        r_, _ = rmse_mae(P, Y, M, idx)
        bs_r.append(r_)
        try:
            if EIv[idx].sum() > 0:
                bs_c.append(concordance_index(ETv[idx], R[idx], EIv[idx]))
        except Exception:
            pass
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    line = "=" * 62
    print(f"\n{line}\n  TEST METRICS  ({n} held-out patients)\n{line}")
    print(f"  C-index : {c_idx:.4f}   95% CI [{ci(bs_c)[0]:.4f}, {ci(bs_c)[1]:.4f}]")
    print(f"  RMSE    : {rm:.4f}   95% CI [{ci(bs_r)[0]:.4f}, {ci(bs_r)[1]:.4f}]")
    print(f"  MAE     : {ma:.4f}")
    print(f"  RMSE/MAE ratio : {rm/max(ma,1e-9):.2f}   (1.25 if errors were normal)")

    print(f"\n  {'group':22s} {'RMSE':>8s} {'MAE':>8s} {'R2':>8s}")
    groups = {}
    for g, cols in GROUPS.items():
        j = [AGG_COLS.index(c) for c in cols if c in AGG_COLS]
        rg, mg = rmse_mae(P[:, :, j], Y[:, :, j], M[:, :, j])
        r2g = r2(P, Y, M, cols)
        print(f"  {g:22s} {rg:8.3f} {mg:8.3f} {r2g:8.3f}")
        groups[g] = {"rmse": rg, "mae": mg, "r2": r2g, "n_features": len(j)}
    print(line)

    json.dump({"c_index": c_idx, "c_index_ci": ci(bs_c),
               "rmse": rm, "rmse_ci": ci(bs_r), "mae": ma,
               "rmse_mae_ratio": rm / max(ma, 1e-9),
               "groups": groups, "n_test": int(n)},
              open(a.out, "w"), indent=2)
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
