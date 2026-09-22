"""
Ensemble PIT / MMD / calibration for the ADNI (Alzheimer's) model.

Same convention as metrics_objective_calibration_ensemble.py and this
project's own ensemble.py: point predictions (flow/mu) are averaged across
members; generative samples are a proper mixture (pick a member uniformly,
draw one sample from it), not an over-smoothed average.

Usage:
    python metrics_adni_calibration_ensemble.py \
        --ckpts models_out/adni_notau_s123/weights.pt,models_out/adni_notau_s42/weights.pt,models_out/adni_notau_s456/weights.pt,models_out/adni_notau_s789/weights.pt,models_out/adni_notau_s1000/weights.pt \
        --split_seed 123 --nh 32
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
from lifelines.utils import concordance_index

# See model_platform_dtg.py for why this import crosses into the platform
# repository, and how to point RESYNEX_PLATFORM_PATH at it if it isn't
# cloned as this repo's sibling.
_platform_path = os.environ.get("RESYNEX_PLATFORM_PATH") or str(
    Path(__file__).resolve().parents[2].parent / "Resynex-Platform" / "ms-digital-twin")
sys.path.insert(0, _platform_path)
from model.nbm import FeatureNormalizer  # noqa: E402

from adni_dataset_notau import build, BASELINE_COLS, OUTCOME_COLS  # noqa: E402
from model_platform_dtg import PlatformDTG  # noqa: E402


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


def mmd_panel(cols, gen_all, Y_te, M_te, n_repeats=20, seed=0):
    idxs = [OUTCOME_COLS.index(c) for c in cols]
    comp = M_te[:, idxs].all(axis=1)
    real = Y_te[comp][:, idxs]
    gen = gen_all[comp][:, :, idxs]
    n = int(comp.sum())
    if n <= 4:
        return {"mmd2": float("nan"), "floor": float("nan"),
                "ratio": float("nan"), "n_rows": n}
    rng = np.random.default_rng(seed)
    n_samples = gen.shape[1]
    m2s, fls = [], []
    for _ in range(n_repeats):
        k = rng.integers(0, n_samples)
        m2s.append(mmd_rbf(gen[:, k, :], real, seed=int(rng.integers(0, 1_000_000))))
        perm = rng.permutation(n)
        half = n // 2
        fls.append(mmd_rbf(real[perm[:half]], real[perm[half:]],
                           seed=int(rng.integers(0, 1_000_000))))
    m2, fl = float(np.mean(m2s)), float(np.mean(fls))
    return {"mmd2": m2, "mmd2_std": float(np.std(m2s)),
            "floor": fl, "floor_std": float(np.std(fls)),
            "ratio": m2 / max(fl, 1e-9), "n_rows": n, "n_repeats": n_repeats}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", required=True)
    p.add_argument("--split_seed", type=int, default=123)
    p.add_argument("--nh", type=int, default=32)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--mc_steps", type=int, default=32)
    p.add_argument("--out", default="metrics_adni_ensemble.json")
    a = p.parse_args()
    ckpts = a.ckpts.split(",")
    K = len(ckpts)

    X, Y, Tf, PID, ET, EI = build(verbose=False)
    n_in, n_out = X.shape[1], Y.shape[1]

    rng = np.random.default_rng(a.split_seed)
    pats = np.unique(PID); rng.shuffle(pats)
    nv = max(1, int(len(pats) * .15)); nt = max(1, int(len(pats) * .15))
    te_p, va_p = set(pats[:nt]), set(pats[nt:nt + nv])
    te = np.array([q in te_p for q in PID])
    tr = ~(te | np.array([q in va_p for q in PID]))

    bn, on = FeatureNormalizer(), FeatureNormalizer()
    bn.fit(X[tr]); on.fit(Y[tr])
    msk = (~np.isnan(Y)).astype(np.float32)
    Yf = np.nan_to_num(Y)

    tt = lambda v: torch.tensor(v, dtype=torch.float32)
    Xt, Tt = tt(X), tt(Tf)

    curr_idx = [BASELINE_COLS.index("curr_" + c) for c in OUTCOME_COLS]
    members = []
    for path in ckpts:
        m = PlatformDTG(n_baseline=n_in, n_outcomes=n_out, nh=a.nh,
                        embed_dim=192, flow_layers=3, S=60, curr_idx=curr_idx)
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        m.attach_normalizers(bn, on)
        m.eval()
        members.append(m)
    print(f"  loaded {K} ensemble members, test pairs={int(te.sum())}")

    idx = np.where(te)[0]
    Xi, Ti = Xt[idx], Tt[idx]

    # point prediction: average flow (mu) across members
    flows = []
    for m in members:
        _, _, _, flow_k, _ = m.predict(Xi, Ti, mc_steps=a.mc_steps, n_samples=8)
        flows.append(flow_k)
    flow = torch.stack(flows).mean(dim=0).numpy()

    r2 = 1 - ((flow - Yf[idx]) ** 2 * msk[idx]).sum(0) / \
        np.maximum((((Yf[idx] - (Yf[idx] * msk[idx]).sum(0) /
                     np.maximum(msk[idx].sum(0), 1)) ** 2) * msk[idx]).sum(0), 1e-8)

    risks = []
    for m in members:
        risks.append(m.risk_score(Xi))
    risk = torch.stack(risks).mean(dim=0).numpy()
    c = concordance_index(ET[idx], risk, EI[idx]) if EI[idx].sum() else float("nan")

    # generative samples: proper mixture across members
    rng2 = np.random.default_rng(1)
    N = len(idx)
    gen_all = np.zeros((N, a.n_samples, n_out), dtype=np.float32)
    member_idx = rng2.integers(0, K, size=a.n_samples)
    for s, k in enumerate(member_idx):
        _, _, samples_k, _, _ = members[k].predict(Xi, Ti, mc_steps=a.mc_steps, n_samples=1)
        gen_all[:, s, :] = samples_k[:, 0, :].numpy()

    Y_te, M_te = Yf[idx], msk[idx]
    pit_vals, pit_by_outcome = [], {c: [] for c in OUTCOME_COLS}
    for j, col in enumerate(OUTCOME_COLS):
        m_ = M_te[:, j].astype(bool)
        if not m_.any():
            continue
        y_true = Y_te[m_, j][:, None]
        s_j = gen_all[m_, :, j]
        below = (s_j < y_true).mean(axis=1)
        equal = (s_j == y_true).mean(axis=1)
        pit = below + 0.5 * equal
        pit_vals.append(pit)
        pit_by_outcome[col] = pit.tolist()

    pit = np.concatenate(pit_vals)
    levels = (0.5, 0.8, 0.9, 0.95)
    cov = {str(L): float(np.mean(np.abs(pit - 0.5) <= L / 2)) for L in levels}
    srt = np.sort(pit); nn = len(srt)
    ks = float(np.max(np.abs(srt - (np.arange(1, nn + 1) / nn))))

    joint_all = mmd_panel(OUTCOME_COLS, gen_all, Y_te, M_te)
    panel_cognitive = mmd_panel(["CDRSB", "MMSCORE", "TOTAL13"], gen_all, Y_te, M_te)
    panel_no_abeta = mmd_panel([c for c in OUTCOME_COLS if c != "ABETA"], gen_all, Y_te, M_te)

    print("\n" + "=" * 66)
    print(f"  ADNI ENSEMBLE ({K} members) -- CALIBRATION & MMD")
    print("=" * 66)
    print(f"  C-index : {c:.4f}   Average R2 : {np.nanmean(r2):.4f}")
    print(f"\n  {'outcome':12s} {'R2':>7s} {'cov@0.9':>8s} {'PIT':>6s}")
    per_outcome = {}
    for j, col in enumerate(OUTCOME_COLS):
        pj = np.array(pit_by_outcome[col])
        c90 = float(np.mean(np.abs(pj - 0.5) <= 0.45)) if len(pj) else float("nan")
        pm = float(pj.mean()) if len(pj) else float("nan")
        print(f"  {col:12s} {r2[j]:7.3f} {c90:8.3f} {pm:6.3f}")
        per_outcome[col] = {"r2": float(r2[j]), "coverage_0.9": c90, "pit_mean": pm}

    print(f"\n  CALIBRATION ({len(pit):,} observed cells, {a.n_samples} mixture samples)")
    for L in levels:
        print(f"    nominal {L:.2f}   empirical {cov[str(L)]:.3f}")
    print(f"  PIT mean {pit.mean():.3f}   KS vs uniform {ks:.3f}")

    for name, d in (("joint, all 6", joint_all), ("no ABETA", panel_no_abeta),
                    ("cognitive", panel_cognitive)):
        print(f"\n  DISTRIBUTIONAL -- {name} ({d['n_rows']} rows, {d.get('n_repeats','-')} repeats)")
        print(f"  generated vs real    MMD2 = {d['mmd2']:.5f}")
        print(f"  real vs real (floor) MMD2 = {d['floor']:.5f}")
    print("=" * 66)

    json.dump({
        "ckpts": ckpts, "K": K, "c_index": float(c), "avg_r2": float(np.nanmean(r2)),
        "calibration": {"coverage": cov, "pit_mean": float(pit.mean()), "ks": ks,
                        "n_cells": len(pit)},
        "mmd_joint6": joint_all, "mmd_no_abeta": panel_no_abeta,
        "mmd_cognitive": panel_cognitive, "per_outcome": per_outcome,
    }, open(a.out, "w"), indent=2)
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
