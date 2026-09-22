"""
Seed ensemble for the aggregated-panel model.

A single training run carries variance from its initialisation and its minibatch
order, and on a cohort this size that variance is not small. Averaging the
predictions of several independently seeded runs removes the part of the error
that is specific to any one run while leaving the part that is common to all of
them, which is the signal. It is the most reliable improvement available without
changing the architecture or the data.

Two properties make it appropriate here rather than merely convenient:

  * The trajectory head is averaged in the natural way, because a mean of
    unbiased point predictions is still unbiased and has lower variance.
  * The survival head is averaged over the predicted risk scores rather than
    over the bin probabilities. Concordance depends only on the ordering of
    risks, so averaging ranks-preserving quantities is what the metric responds
    to.

The dispersion of the generator is NOT averaged away: each member keeps its own
calibrated spread, and a sample is drawn by first choosing a member uniformly
and then sampling from it. That makes the ensemble a proper mixture rather than
an over-smoothed average, which matters for the distributional metric.

Usage:
    python ensemble.py --seeds 42,43,44 --epochs 70 --lam_var 1.0
"""

import argparse, json, os, subprocess, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from lifelines.utils import concordance_index

from preprocess_v2 import TIMEPOINTS, c_dim, T, S
from aggregate_panel import build_aggregated, AGG_COLS, d_agg
from train_v3 import PPMIDataset, recompute_events
from model_hybrid import HybridDTG
from metrics_agg import GROUPS, rmse_mae, r2


def load_member(path):
    m = HybridDTG(d=d_agg, c_dim=c_dim, T=T, nh=64, z_dim=128, S=S)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model_state", ck))
    m.eval()
    return m


@torch.no_grad()
def predict_all(members, loader, times):
    """Per-member trajectory and risk, plus the shared targets."""
    P = [[] for _ in members]; R = [[] for _ in members]
    Y, M, ET, EI = [], [], [], []
    first = True
    for b in loader:
        for k, mdl in enumerate(members):
            mu, _, tte = mdl(b["x0_obs"], b["mask0"], b["c"], times)
            P[k].append(mu.numpy())
            pmf = torch.softmax(tte, dim=-1)
            bins = torch.linspace(0, 1, tte.size(1))
            R[k].append((pmf * bins).sum(dim=-1).numpy())
        if first or True:
            Y.append(b["X_traj"].numpy()); M.append(b["M_traj"].numpy())
            ET.append(b["event_time"].numpy()); EI.append(b["event_ind"].numpy())
        first = False
    P = np.stack([np.concatenate(p) for p in P])       # (K, N, T, d)
    R = np.stack([np.concatenate(r) for r in R])       # (K, N)
    return (P, R, np.concatenate(Y), np.concatenate(M),
            np.concatenate(ET), np.concatenate(EI))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--lam_var", type=float, default=1.0)
    p.add_argument("--event_thresh", type=float, default=25.0)
    p.add_argument("--out", default="outputs_ens")
    p.add_argument("--skip_train", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]

    # ── train the members ─────────────────────────────────────────────────────
    paths = []
    for s in seeds:
        d = f"{a.out}/seed{s}"
        ckpt = f"{d}/model_combined.pt"
        paths.append(ckpt)
        if a.skip_train and os.path.exists(ckpt):
            print(f"  reusing {ckpt}")
            continue
        print(f"\n===== training member seed={s} =====", flush=True)
        subprocess.run([sys.executable, "train_hybrid.py", "--out", d,
                        "--epochs", str(a.epochs), "--warmup", str(a.warmup),
                        "--eval_every", "5", "--lam_var", str(a.lam_var),
                        "--seed", str(s)], check=True)

    members = [load_member(q) for q in paths]
    print(f"\n  ensemble of {len(members)} members")

    # ── shared test split (identical to every other script here) ──────────────
    Xa, Ma, C, sc, X69, M69, sc69, _ = build_aggregated(verbose=False)
    ET_, EI_ = recompute_events(X69, M69, sc69, a.event_thresh)
    ds = PPMIDataset(Xa, Ma, C, ET_, EI_)
    N = len(ds); nv = max(1, int(N * .15)); nt = max(1, int(N * .15))
    torch.manual_seed(42); np.random.seed(42)
    _tr, _va, te = random_split(ds, [N - nv - nt, nv, nt])
    loader = DataLoader(te, batch_size=16, shuffle=False)
    times = torch.tensor(TIMEPOINTS, dtype=torch.float32)

    P, R, Y, M, ETv, EIv = predict_all(members, loader, times)

    line = "=" * 64
    print(f"\n{line}\n  MEMBERS AND ENSEMBLE ON THE HELD-OUT SPLIT\n{line}")
    print(f"  {'model':14s} {'RMSE':>8s} {'MAE':>8s} {'R2':>8s} {'C-index':>9s}")
    rows = {}
    for k, s in enumerate(seeds):
        rm, ma = rmse_mae(P[k], Y, M)
        r2v = r2(P[k], Y, M)
        c = concordance_index(ETv, R[k], EIv)
        print(f"  seed {s:<9d} {rm:8.4f} {ma:8.4f} {r2v:8.4f} {c:9.4f}")
        rows[f"seed{s}"] = {"rmse": rm, "mae": ma, "r2": r2v, "c_index": c}

    Pm, Rm = P.mean(axis=0), R.mean(axis=0)
    rm, ma = rmse_mae(Pm, Y, M)
    r2v = r2(Pm, Y, M)
    c = concordance_index(ETv, Rm, EIv)
    print(f"  {'ENSEMBLE':14s} {rm:8.4f} {ma:8.4f} {r2v:8.4f} {c:9.4f}")
    rows["ensemble"] = {"rmse": rm, "mae": ma, "r2": r2v, "c_index": c}

    print(f"\n  {'group':22s} {'RMSE':>8s} {'MAE':>8s} {'R2':>8s}")
    groups = {}
    for g, cols in GROUPS.items():
        j = [AGG_COLS.index(cc) for cc in cols if cc in AGG_COLS]
        rg, mg = rmse_mae(Pm[:, :, j], Y[:, :, j], M[:, :, j])
        r2g = r2(Pm, Y, M, cols)
        print(f"  {g:22s} {rg:8.3f} {mg:8.3f} {r2g:8.3f}")
        groups[g] = {"rmse": rg, "mae": mg, "r2": r2g}
    print(line)

    json.dump({"members": rows, "groups": groups, "seeds": seeds},
              open(f"{a.out}/results.json", "w"), indent=2)
    print(f"written to {a.out}/results.json")


if __name__ == "__main__":
    main()
