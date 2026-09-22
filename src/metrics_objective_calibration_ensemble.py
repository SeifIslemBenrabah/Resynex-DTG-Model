"""
Ensemble PIT / MMD / calibration for the objective-endpoint (PPMI) model.

Follows ensemble.py's own established convention exactly (see its docstring):
point predictions (mu) and risk scores are averaged across members; the
generative sample for each draw is a proper MIXTURE -- pick a member
uniformly at random, then draw one sample from that member's own NBM --
rather than an over-smoothed average. This is the same principle already
used for this project's headline R2/C-index ensemble numbers, applied here
to the distributional metrics (PIT, MMD) that single-seed runs turned out
not to measure reliably.

Usage:
    python metrics_objective_calibration_ensemble.py \
        --ckpts outputs_exp_nh64_seed123/model.pt,outputs_exp_nh64_seed456/model.pt,outputs_exp_nh64_seed789/model.pt \
        --split_seed 123 --nh 64 --bins 6
"""
import argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from lifelines.utils import concordance_index

from preprocess_v2 import TIMEPOINTS, T, c_dim
from aggregate_panel import build_aggregated, AGG_COLS, OBJECTIVE_COLS, d_agg
from train_v3 import PPMIDataset, recompute_events
from train_objective import ObjectiveDTG

PANEL = [AGG_COLS.index(c) for c in OBJECTIVE_COLS]
CSF_COLS_ = ["asyn", "tau", "ptau"]
DATSCAN_COLS_ = ["MIA_CAUDATE_L", "MIA_CAUDATE_R", "MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]
CSF_PANEL = [AGG_COLS.index(c) for c in CSF_COLS_]
DATSCAN_PANEL = [AGG_COLS.index(c) for c in DATSCAN_COLS_]


def mmd_rbf(X, Y, gammas=(0.25, 0.5, 1.0, 2.0, 4.0), max_n=600, seed=0):
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]

    def sq(A, B):
        return (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T

    dxx, dyy, dxy = sq(X, X), sq(Y, Y), sq(X, Y)
    med = np.median(dxy[dxy > 0]) if (dxy > 0).any() else 1.0
    n, m = len(X), len(Y)
    tot = 0.0
    for g in gammas:
        s = g / max(med, 1e-9)
        Kxx, Kyy, Kxy = np.exp(-s * dxx), np.exp(-s * dyy), np.exp(-s * dxy)
        np.fill_diagonal(Kxx, 0.0); np.fill_diagonal(Kyy, 0.0)
        tot += (Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
                - 2.0 * Kxy.mean())
    return float(tot / len(gammas))


def mmd_report(gen_rows_, real_rows_, n_repeats=20, seed=0):
    Gp = np.concatenate(gen_rows_, axis=0)
    Rp = np.concatenate(real_rows_, axis=0)
    n_samples = Gp.shape[1]
    rng = np.random.default_rng(seed)
    m2s, fls = [], []
    for _ in range(n_repeats):
        k = rng.integers(0, n_samples)
        m2s.append(mmd_rbf(Gp[:, k, :], Rp, seed=int(rng.integers(0, 1_000_000))))
        perm = rng.permutation(len(Rp))
        h = len(Rp) // 2
        fls.append(mmd_rbf(Rp[perm[:h]], Rp[perm[h:]],
                           seed=int(rng.integers(0, 1_000_000))))
    m2, fl = float(np.mean(m2s)), float(np.mean(fls))
    return {"mmd2": m2, "mmd2_std": float(np.std(m2s)),
            "floor": fl, "floor_std": float(np.std(fls)),
            "ratio": m2 / max(fl, 1e-9), "n_rows": int(len(Rp)), "n_repeats": n_repeats}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", required=True, help="comma-separated model.pt paths")
    p.add_argument("--split_seed", type=int, default=123,
                   help="seed the ensemble members were all trained/held-out with")
    p.add_argument("--nh", type=int, default=64)
    p.add_argument("--bins", type=int, default=6)
    p.add_argument("--event_thresh", type=float, default=25.0)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--mc_steps", type=int, default=16)
    p.add_argument("--out", default="metrics_objective_ensemble.json")
    a = p.parse_args()
    ckpts = a.ckpts.split(",")
    K = len(ckpts)

    Xa, Ma, C, sc, X69, M69, sc69, _ = build_aggregated(verbose=False)
    ET_, EI_ = recompute_events(X69, M69, sc69, a.event_thresh)
    ds = PPMIDataset(Xa, Ma, C, ET_, EI_)
    N = len(ds); nv = max(1, int(N * .15)); nt = max(1, int(N * .15))
    torch.manual_seed(a.split_seed); np.random.seed(a.split_seed)
    _tr, va, te = random_split(ds, [N - nv - nt, nv, nt])
    loader = DataLoader(te, batch_size=16, shuffle=False)
    times = torch.tensor(TIMEPOINTS, dtype=torch.float32)

    members = []
    for path in ckpts:
        m = ObjectiveDTG(d=d_agg, c_dim=c_dim, T=T, nh=a.nh, z_dim=128, S=a.bins)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m.load_state_dict(ck.get("model_state", ck))
        m.eval()
        members.append(m)
    print(f"  loaded {K} ensemble members")

    rng = np.random.default_rng(0)
    all_ET, all_EI, all_risk = [], [], []
    all_P, all_Y, all_M = [], [], []
    gen_rows, real_rows = [], []
    gen_rows_csf, real_rows_csf = [], []
    gen_rows_dat, real_rows_dat = [], []
    pit_vals, pit_by_outcome = [], {}

    for b in loader:
        # point prediction: average mu across members (unbiased, lower variance)
        mus, ttes = [], []
        for m in members:
            mu_k, _, tte_k = m(b["x0_obs"], b["mask0"], b["c"], times)
            mus.append(mu_k); ttes.append(tte_k)
        mu = torch.stack(mus).mean(dim=0)

        risks = []
        for tte_k in ttes:
            pmf_k = torch.softmax(tte_k, dim=-1)
            bins = torch.linspace(0, 1, tte_k.size(1))
            risks.append((pmf_k * bins).sum(dim=-1))
        risk = torch.stack(risks).mean(dim=0)

        Pj = mu.numpy()[:, :, PANEL]
        Yj = b["X_traj"].numpy()[:, :, PANEL]
        Mj = b["M_traj"].numpy()[:, :, PANEL]
        all_P.append(Pj); all_Y.append(Yj); all_M.append(Mj)
        all_ET.append(b["event_time"].numpy())
        all_EI.append(b["event_ind"].numpy())
        all_risk.append(risk.numpy())

        # generative samples: proper mixture. For each of n_samples draws,
        # pick a member uniformly, then sample from that member's own NBM,
        # exactly as ensemble.py's docstring specifies.
        B = b["x0_obs"].size(0)
        pool = torch.zeros(B, a.n_samples, len(TIMEPOINTS), d_agg)
        member_idx = rng.integers(0, K, size=a.n_samples)
        for s, k in enumerate(member_idx):
            draw = members[k].generate_twin(b["x0_obs"], b["mask0"], b["c"], times,
                                            n_samples=1)  # (B,1,T,d)
            pool[:, s] = draw[:, 0]
        pool_j = pool.numpy()[:, :, :, PANEL]

        obs = b["M_traj"].bool()
        M = b["M_traj"]; X = b["X_traj"]
        for ti in range(X.shape[1]):
            sel = obs[:, ti, :][:, PANEL].all(dim=1)
            if sel.any():
                real_rows.append(X[sel, ti][:, PANEL].numpy())
                gen_rows.append(pool_j[sel.numpy(), :, ti, :])
            sel_csf = obs[:, ti, :][:, CSF_PANEL].all(dim=1)
            if sel_csf.any():
                real_rows_csf.append(X[sel_csf, ti][:, CSF_PANEL].numpy())
                gen_rows_csf.append(pool.numpy()[sel_csf.numpy(), :, ti][:, :, CSF_PANEL])
            sel_dat = obs[:, ti, :][:, DATSCAN_PANEL].all(dim=1)
            if sel_dat.any():
                real_rows_dat.append(X[sel_dat, ti][:, DATSCAN_PANEL].numpy())
                gen_rows_dat.append(pool.numpy()[sel_dat.numpy(), :, ti][:, :, DATSCAN_PANEL])

            Mij = M[:, ti, :][:, PANEL].bool()
            if Mij.any():
                y_true = b["X_traj"][:, ti, :][:, PANEL]
                samples_ti = torch.from_numpy(pool_j[:, :, ti, :])
                below = (samples_ti < y_true.unsqueeze(1)).float().mean(dim=1)
                equal = (samples_ti == y_true.unsqueeze(1)).float().mean(dim=1)
                pit_grid = below + 0.5 * equal
                pit = pit_grid[Mij]
                pit_vals.append(pit.numpy())
                for jc, col in enumerate(OBJECTIVE_COLS):
                    m_c = Mij[:, jc]
                    if m_c.any():
                        pit_by_outcome.setdefault(col, []).append(pit_grid[m_c, jc].numpy())

    ET_all = np.concatenate(all_ET); EI_all = np.concatenate(all_EI)
    risk_all = np.concatenate(all_risk)
    c_index = float(concordance_index(ET_all, risk_all, EI_all))

    P_all = np.concatenate(all_P); Y_all = np.concatenate(all_Y); M_all = np.concatenate(all_M)
    col_mean = (Y_all * M_all).sum(axis=(0, 1)) / np.maximum(M_all.sum(axis=(0, 1)), 1e-8)
    sse = (((P_all - Y_all) ** 2) * M_all).sum()
    sst = (((Y_all - col_mean) ** 2) * M_all).sum()
    r2 = float(1 - sse / max(sst, 1e-8))

    pit = np.concatenate(pit_vals)
    levels = (0.5, 0.8, 0.9, 0.95)
    cov = {str(L): float(np.mean(np.abs(pit - 0.5) <= L / 2)) for L in levels}
    srt = np.sort(pit); nn = len(srt)
    ks = float(np.max(np.abs(srt - (np.arange(1, nn + 1) - 0.5) / nn)))

    joint_report = mmd_report(gen_rows, real_rows)
    mmd_csf = mmd_report(gen_rows_csf, real_rows_csf)
    mmd_dat = mmd_report(gen_rows_dat, real_rows_dat)

    print("\n" + "=" * 66)
    print(f"  ENSEMBLE ({K} members) -- CALIBRATION & MMD")
    print("=" * 66)
    print(f"  C-index : {c_index:.4f}   R2 : {r2:.4f}")
    print(f"\n  CALIBRATION ({len(pit):,} observed cells, {a.n_samples} mixture samples)")
    for L in levels:
        print(f"    nominal {L:.2f}   empirical {cov[str(L)]:.3f}")
    print(f"  PIT mean {pit.mean():.3f}   KS vs uniform {ks:.3f}")
    for name, d in (("joint 7-column panel", joint_report),
                    ("CSF panel", mmd_csf), ("DaTscan panel", mmd_dat)):
        print(f"\n  DISTRIBUTIONAL -- {name} ({d['n_rows']} rows, {d['n_repeats']} repeats)")
        print(f"  generated vs real    MMD2 = {d['mmd2']:.5f} (+/- {d['mmd2_std']:.5f})")
        print(f"  real vs real (floor) MMD2 = {d['floor']:.5f} (+/- {d['floor_std']:.5f})")
    print("=" * 66)

    json.dump({
        "ckpts": ckpts, "K": K, "split_seed": a.split_seed,
        "c_index": c_index, "r2": r2,
        "calibration": {"coverage": cov, "pit_mean": float(pit.mean()),
                        "ks": ks, "n_cells": int(len(pit))},
        "mmd_joint7": joint_report, "mmd_csf_panel": mmd_csf, "mmd_datscan_panel": mmd_dat,
    }, open(a.out, "w"), indent=2)
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
