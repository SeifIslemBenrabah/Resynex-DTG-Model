"""
Calibration coverage / PIT / MMD for the objective-endpoint model (seed 456),
adapted from metrics_platform_dtg.py's methodology (built for the deployed
platform model) and metrics_full.py's distributional-comparison pattern
(built for the same DTG_v3/HybridDTG family this model belongs to).

This closes the gap flagged before treating pd_biomarkers as production-ready:
final_model.py reports C-index/R2/RMSE/MAE for this model, but nothing had
measured whether its predictive *distribution* is honest (calibration) or
realistic (MMD) -- the two properties the deployed dashboard's uncertainty
band and "digital twin" framing actually rest on.

Usage:
    python metrics_objective_calibration.py --seed 456 --n_samples 200
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

PANEL = [AGG_COLS.index(c) for c in OBJECTIVE_COLS]   # 7 of the 19 agg columns

# MMD split by clinical procedure rather than requiring all 7 objective
# columns jointly observed. CSF biomarkers (lumbar puncture) and DaTscan
# imaging are different procedures done at different visits; requiring both
# simultaneously observed for a "complete row" throws away most of the
# cohort for no methodological reason -- it tests something ("how often are
# both procedures done at once") that has nothing to do with whether the
# generator is realistic. Splitting into two panels measured separately
# raises usable pooled rows from ~164 (both required) to ~1,277 (CSF alone)
# and ~3,840 (DaTscan alone) on the full cohort.
CSF_COLS_ = ["asyn", "tau", "ptau"]
DATSCAN_COLS_ = ["MIA_CAUDATE_L", "MIA_CAUDATE_R", "MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]
CSF_PANEL = [AGG_COLS.index(c) for c in CSF_COLS_]
DATSCAN_PANEL = [AGG_COLS.index(c) for c in DATSCAN_COLS_]


def mmd_rbf(X, Y, gammas=(0.25, 0.5, 1.0, 2.0, 4.0), max_n=600, seed=0):
    """Unbiased MMD^2 with a mixture of Gaussian kernels (identical estimator
    to metrics_platform_dtg.py / metrics_full.py, for a directly comparable
    figure)."""
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


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=456, help="which trained member to evaluate")
    p.add_argument("--ckpt_dir", default="outputs_final_multiseed")
    p.add_argument("--ckpt", default=None, help="explicit checkpoint path, overrides --ckpt_dir/--seed")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--var_scale", type=float, default=1.0,
                   help="post-hoc multiplicative inflation of the sampled residual (1.0 = none)")
    p.add_argument("--event_thresh", type=float, default=25.0)
    p.add_argument("--bins", type=int, default=6)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--out", default="metrics_objective_calibration.json")
    p.add_argument("--nh", type=int, default=64)
    a = p.parse_args()

    Xa, Ma, C, sc, X69, M69, sc69, _ = build_aggregated(verbose=False)
    ET_, EI_ = recompute_events(X69, M69, sc69, a.event_thresh)
    ds = PPMIDataset(Xa, Ma, C, ET_, EI_)

    # identical split protocol to train_objective.py, so this evaluates the
    # SAME 233-patient held-out test set that checkpoint was actually trained
    # against. The split is reseeded from a.seed, matching train_objective.py's
    # `torch.manual_seed(a.seed)` immediately before its own random_split call
    # (previously this was hardcoded to 42, which silently evaluated seed-456
    # checkpoints against a different partition than the one they were trained
    # and held out on -- a leakage bug caught by cross-checking against
    # train_objective.py's own end-of-run test print).
    N = len(ds); nv = max(1, int(N * .15)); nt = max(1, int(N * .15))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    _tr, va, te = random_split(ds, [N - nv - nt, nv, nt])
    eval_set = va if a.split == "val" else te
    loader = DataLoader(eval_set, batch_size=16, shuffle=False)
    times = torch.tensor(TIMEPOINTS, dtype=torch.float32)

    model = ObjectiveDTG(d=d_agg, c_dim=c_dim, T=T, nh=a.nh, z_dim=128, S=a.bins)
    ckpt_path = a.ckpt or f"{a.ckpt_dir}/seed{a.seed}/model.pt"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state", ck))
    model.eval()

    all_ET, all_EI, all_risk = [], [], []
    n_obs = sq_err = abs_err = 0.0
    all_P, all_Y, all_M = [], [], []   # accumulated for a pooled R^2, exactly as final_model.py computes it
    gen_rows, real_rows = [], []       # pooled (patient, visit) rows, PANEL cols only (all 7 jointly observed)
    gen_rows_csf, real_rows_csf = [], []          # CSF panel alone
    gen_rows_dat, real_rows_dat = [], []          # DaTscan panel alone
    pit_vals = []
    pit_by_outcome = {}
    sq_err_by_outcome = {c: 0.0 for c in OBJECTIVE_COLS}
    abs_err_by_outcome = {c: 0.0 for c in OBJECTIVE_COLS}
    n_obs_by_outcome = {c: 0.0 for c in OBJECTIVE_COLS}

    for b in loader:
        mu, _, tte = model(b["x0_obs"], b["mask0"], b["c"], times)   # mu: (B,T,19)
        X, M = b["X_traj"], b["M_traj"]                              # (B,T,19)

        j = PANEL
        Pj, Yj, Mj = mu[:, :, j], X[:, :, j], M[:, :, j]
        n = Mj.sum().item()
        n_obs += n
        sq_err += (((Pj - Yj) ** 2) * Mj).sum().item()
        abs_err += ((Pj - Yj).abs() * Mj).sum().item()
        all_P.append(Pj.numpy()); all_Y.append(Yj.numpy()); all_M.append(Mj.numpy())
        for jc, col in enumerate(OBJECTIVE_COLS):
            pc, yc, mc = Pj[:, :, jc], Yj[:, :, jc], Mj[:, :, jc]
            sq_err_by_outcome[col] += (((pc - yc) ** 2) * mc).sum().item()
            abs_err_by_outcome[col] += ((pc - yc).abs() * mc).sum().item()
            n_obs_by_outcome[col] += mc.sum().item()

        pmf = torch.softmax(tte, dim=-1)
        bins = torch.linspace(0, 1, tte.size(1))
        risk = (pmf * bins).sum(dim=-1)
        all_ET.append(b["event_time"].numpy())
        all_EI.append(b["event_ind"].numpy())
        all_risk.append(risk.numpy())

        # ── sampled trajectories for calibration + MMD ──────────────────────
        pool = model.generate_twin(b["x0_obs"], b["mask0"], b["c"], times,
                                   n_samples=a.n_samples)             # (B,n,T,19)
        if a.var_scale != 1.0:
            # Post-hoc variance inflation: widen the sampled residual around
            # the (unchanged) point prediction mu, without touching mu
            # itself or retraining anything. Fit on --split val, applied
            # as-is on --split test.
            pool = mu.unsqueeze(1) + (pool - mu.unsqueeze(1)) * a.var_scale
        pool_j = pool[:, :, :, j]                                     # (B,n,T,7)

        obs = M.bool()
        for ti in range(X.shape[1]):
            sel = obs[:, ti, :][:, j].all(dim=1)   # PANEL complete at this visit
            if sel.any():
                real_rows.append(X[sel, ti][:, j].numpy())
                # Keep every posterior draw (not just draw 0), so the MMD
                # step below can average over many draws instead of being
                # a single-draw point estimate with its own sampling noise.
                gen_rows.append(pool_j[sel, :, ti, :].numpy())  # (n_sel, n_samples, 7)

            # Same comparison, split by clinical procedure so a visit only
            # needs its OWN panel complete, not both -- see CSF_PANEL /
            # DATSCAN_PANEL comment above for why.
            sel_csf = obs[:, ti, :][:, CSF_PANEL].all(dim=1)
            if sel_csf.any():
                real_rows_csf.append(X[sel_csf, ti][:, CSF_PANEL].numpy())
                gen_rows_csf.append(pool[sel_csf, :, ti][:, :, CSF_PANEL].numpy())
            sel_dat = obs[:, ti, :][:, DATSCAN_PANEL].all(dim=1)
            if sel_dat.any():
                real_rows_dat.append(X[sel_dat, ti][:, DATSCAN_PANEL].numpy())
                gen_rows_dat.append(pool[sel_dat, :, ti][:, :, DATSCAN_PANEL].numpy())

            # PIT, cell by cell, over whichever PANEL entries ARE observed
            # (not requiring the whole panel complete, unlike the MMD rows)
            Mij = M[:, ti, :][:, j].bool()                            # (B,7)
            if Mij.any():
                y_true = Yj[:, ti, :]                                 # (B,7)
                samples_ti = pool_j[:, :, ti, :]                      # (B,n,7)
                below = (samples_ti < y_true.unsqueeze(1)).float().mean(dim=1)
                equal = (samples_ti == y_true.unsqueeze(1)).float().mean(dim=1)
                pit_grid = below + 0.5 * equal                        # (B,7)
                pit = pit_grid[Mij]
                pit_vals.append(pit.numpy())
                for jc, col in enumerate(OBJECTIVE_COLS):
                    m_c = Mij[:, jc]
                    if m_c.any():
                        pit_by_outcome.setdefault(col, []).append(pit_grid[m_c, jc].numpy())

    ET_all = np.concatenate(all_ET); EI_all = np.concatenate(all_EI)
    risk_all = np.concatenate(all_risk)
    c_index = float(concordance_index(ET_all, risk_all, EI_all))
    rmse = float(np.sqrt(sq_err / n_obs)); mae = float(abs_err / n_obs)

    # pooled R^2 across the 7 objective columns, computed the same way
    # final_model.py's own metrics() does: each column's mean is its own
    # per-column observed mean over the whole test set, not the batch.
    P_all = np.concatenate(all_P); Y_all = np.concatenate(all_Y); M_all = np.concatenate(all_M)
    col_mean = (Y_all * M_all).sum(axis=(0, 1)) / np.maximum(M_all.sum(axis=(0, 1)), 1e-8)
    sse = (((P_all - Y_all) ** 2) * M_all).sum()
    sst = (((Y_all - col_mean) ** 2) * M_all).sum()
    r2 = float(1 - sse / max(sst, 1e-8))

    pit = np.concatenate(pit_vals)
    levels = (0.5, 0.8, 0.9, 0.95)
    cov = {L: float(np.mean(np.abs(pit - 0.5) <= L / 2)) for L in levels}
    srt = np.sort(pit); nn = len(srt)
    ks = float(np.max(np.abs(srt - (np.arange(1, nn + 1) - 0.5) / nn)))

    # per-outcome R2 / RMSE / MAE, and per-outcome calibration coverage/PIT
    per_outcome = {}
    for jc, col in enumerate(OBJECTIVE_COLS):
        n_c = n_obs_by_outcome[col]
        rmse_c = float(np.sqrt(sq_err_by_outcome[col] / n_c)) if n_c > 0 else float("nan")
        mae_c = float(abs_err_by_outcome[col] / n_c) if n_c > 0 else float("nan")
        Yc = Y_all[:, :, jc]; Mc = M_all[:, :, jc]; Pc = P_all[:, :, jc]
        mean_c = (Yc * Mc).sum() / max(Mc.sum(), 1e-8)
        sse_c = (((Pc - Yc) ** 2) * Mc).sum(); sst_c = (((Yc - mean_c) ** 2) * Mc).sum()
        r2_c = float(1 - sse_c / max(sst_c, 1e-8))
        pit_c = np.concatenate(pit_by_outcome[col]) if col in pit_by_outcome else np.array([])
        cov_c = {L: float(np.mean(np.abs(pit_c - 0.5) <= L / 2)) for L in levels} if len(pit_c) else {}
        per_outcome[col] = {"r2": r2_c, "rmse": rmse_c, "mae": mae_c, "n_obs": int(n_c),
                            "coverage": {str(L): v for L, v in cov_c.items()},
                            "pit_mean": float(pit_c.mean()) if len(pit_c) else None,
                            "n_cells": int(len(pit_c))}

    def _mmd_report(gen_rows_, real_rows_, n_repeats=20, seed=0):
        """Averages MMD2 (and the real-vs-real floor) over many posterior
        draws and many random floor splits, instead of the single draw /
        single split point estimate. A single draw carries its own sampling
        noise on top of whatever the model actually does, which is large
        enough at these panel sizes (dozens to a few hundred rows) to
        dominate genuine differences between configurations -- this
        averages that noise down instead of reporting one draw of it.
        """
        Gp = np.concatenate(gen_rows_, axis=0)   # (N, n_samples, ny)
        Rp = np.concatenate(real_rows_, axis=0)  # (N, ny)
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
                "ratio": m2 / max(fl, 1e-9), "n_rows": int(len(Rp)),
                "n_repeats": n_repeats}

    joint_report = _mmd_report(gen_rows, real_rows)
    mmd2, floor, ratio = joint_report["mmd2"], joint_report["floor"], joint_report["ratio"]
    mmd_csf = _mmd_report(gen_rows_csf, real_rows_csf)
    mmd_dat = _mmd_report(gen_rows_dat, real_rows_dat)

    # bootstrap CI on C-index (patient-level resample)
    rng = np.random.default_rng(42); n = len(ET_all)
    bc = []
    for _ in range(a.boot):
        idx = rng.integers(0, n, n)
        try:
            if EI_all[idx].sum() > 0:
                bc.append(concordance_index(ET_all[idx], risk_all[idx], EI_all[idx]))
        except Exception:
            pass
    ci = [float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))]

    line = "=" * 66
    print(f"\n{line}\n  OBJECTIVE MODEL (seed {a.seed}) — CALIBRATION & MMD\n{line}")
    print(f"  C-index : {c_index:.4f}   95% CI {np.round(ci,4).tolist()}")
    print(f"  R2      : {r2:.4f}")
    print(f"  RMSE    : {rmse:.4f}   MAE : {mae:.4f}   (objective-endpoint columns only)")
    print(f"\n  {'outcome':16s} {'R2':>7s} {'RMSE':>8s} {'MAE':>8s} {'n_obs':>7s} {'cov@0.9':>8s} {'PIT':>6s}")
    for col in sorted(OBJECTIVE_COLS, key=lambda c: -per_outcome[c]["r2"]):
        po = per_outcome[col]
        c90 = po["coverage"].get("0.9", float("nan"))
        pitm = po["pit_mean"] if po["pit_mean"] is not None else float("nan")
        print(f"  {col:16s} {po['r2']:7.3f} {po['rmse']:8.3f} {po['mae']:8.3f} {po['n_obs']:7.0f} {c90:8.3f} {pitm:6.3f}")
    print(f"\n  CALIBRATION  ({len(pit):,} observed cells, {a.n_samples} samples)")
    print(f"  {'nominal':>9s} {'empirical':>11s} {'gap':>9s}")
    for L in levels:
        print(f"  {L:9.2f} {cov[L]:11.3f} {cov[L]-L:+9.3f}")
    print(f"  PIT mean {pit.mean():.3f}   KS vs uniform {ks:.3f}")
    print(f"\n  DISTRIBUTIONAL — joint 7-column panel ({joint_report['n_rows']} complete rows;"
          f" averaged over {joint_report['n_repeats']} draws/splits; underpowered, kept for continuity)")
    print(f"  generated vs real    MMD2 = {mmd2:.5f} (+/- {joint_report['mmd2_std']:.5f})")
    print(f"  real vs real (floor) MMD2 = {floor:.5f} (+/- {joint_report['floor_std']:.5f})")
    print(f"  ratio                {ratio:.2f}x")
    print(f"\n  DISTRIBUTIONAL — CSF panel alone ({mmd_csf['n_rows']} complete rows)")
    print(f"  generated vs real    MMD2 = {mmd_csf['mmd2']:.5f}")
    print(f"  real vs real (floor) MMD2 = {mmd_csf['floor']:.5f}")
    print(f"  ratio                {mmd_csf['ratio']:.2f}x")
    print(f"\n  DISTRIBUTIONAL — DaTscan panel alone ({mmd_dat['n_rows']} complete rows)")
    print(f"  generated vs real    MMD2 = {mmd_dat['mmd2']:.5f}")
    print(f"  real vs real (floor) MMD2 = {mmd_dat['floor']:.5f}")
    print(f"  ratio                {mmd_dat['ratio']:.2f}x")
    print(line)

    json.dump({
        "seed": a.seed, "split": a.split, "var_scale": a.var_scale,
        "n_patients": int(len(eval_set)),
        "c_index": c_index, "c_index_ci": ci, "r2": r2, "rmse": rmse, "mae": mae,
        "calibration": {"coverage": {str(k): v for k, v in cov.items()},
                       "pit_mean": float(pit.mean()), "ks": ks,
                       "n_cells": int(len(pit))},
        "mmd_joint7": joint_report,
        "mmd_csf_panel": mmd_csf,
        "mmd_datscan_panel": mmd_dat,
        "per_outcome": per_outcome,
        "panel_columns": OBJECTIVE_COLS,
    }, open(a.out, "w"), indent=2)
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
