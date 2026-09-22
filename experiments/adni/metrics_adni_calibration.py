"""
Calibration coverage / PIT / MMD for the ADNI (Alzheimer's) model, trained
standalone (no fine-tuning from the PPMI checkpoint -- train_adni_notau.py
initializes PlatformDTG fresh and trains only on ADNI data).

Mirrors metrics_objective_calibration.py's methodology (same PIT definition,
same mmd_rbf estimator) so the two are directly comparable, adapted to:
  - PlatformDTG (DeepHit + trajectory pooling), not HybridDTG/ObjectiveDTG
  - the ADNI (patient, visit-pair) dataset shape, not the PPMI 6-timepoint grid
  - one joint 6-outcome panel (no CSF/DaTscan procedure split -- ADNI's
    outcomes are not split across two distinct clinical procedures the way
    PPMI's are)

Usage:
    python metrics_adni_calibration.py --seed 123 \
        --ckpt models_out/adni_notau_s123/weights.pt --n_samples 200
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
    """Identical estimator to metrics_objective_calibration.py's mmd_rbf."""
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
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--ckpt", default="models_out/adni_notau_s123/weights.pt")
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--mc_steps", type=int, default=32)
    p.add_argument("--nh", type=int, default=32)
    p.add_argument("--out", default="metrics_adni_calibration.json")
    a = p.parse_args()

    X, Y, Tf, PID, ET, EI = build(verbose=False)
    n_in, n_out = X.shape[1], Y.shape[1]

    # Reproduce train_adni_notau.py's exact patient-level split (same seed,
    # same rng calls), so the test set and refitted normalizers match the
    # ones this checkpoint was actually trained and held out against.
    rng = np.random.default_rng(a.seed)
    pats = np.unique(PID); rng.shuffle(pats)
    nv = max(1, int(len(pats) * .15)); nt = max(1, int(len(pats) * .15))
    te_p, va_p = set(pats[:nt]), set(pats[nt:nt + nv])
    te = np.array([q in te_p for q in PID])
    tr = ~(te | np.array([q in va_p for q in PID]))
    print(f"  patients test={nt}   pairs test={te.sum()}")

    bn, on = FeatureNormalizer(), FeatureNormalizer()
    bn.fit(X[tr]); on.fit(Y[tr])
    msk = (~np.isnan(Y)).astype(np.float32)
    Yf = np.nan_to_num(Y)

    tt = lambda v: torch.tensor(v, dtype=torch.float32)
    Xt, Tt = tt(X), tt(Tf)

    curr_idx = [BASELINE_COLS.index("curr_" + c) for c in OUTCOME_COLS]
    model = PlatformDTG(n_baseline=n_in, n_outcomes=n_out, nh=a.nh,
                        embed_dim=192, flow_layers=3, S=60, curr_idx=curr_idx)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu",
                                     weights_only=False))
    model.attach_normalizers(bn, on)
    model.eval()

    idx = np.where(te)[0]
    mean, std, samples, flow, tte_q = model.predict(
        Xt[idx], Tt[idx], mc_steps=a.mc_steps, n_samples=a.n_samples)
    risk = model.risk_score(Xt[idx]).numpy() if hasattr(model, "risk_score") \
        else None

    Y_te, M_te = Yf[idx], msk[idx]
    samples_np = samples.numpy()  # (N, n_samples, n_out)

    r2 = 1 - (((flow.numpy() - Y_te) ** 2) * M_te).sum(0) / \
        np.maximum((((Y_te - (Y_te * M_te).sum(0) / np.maximum(M_te.sum(0), 1))
                     ** 2) * M_te).sum(0), 1e-8)

    c = concordance_index(ET[idx], (1.0 / (1.0 + np.exp(-flow.numpy()[:, 0]))
                                    if risk is None else risk), EI[idx]) \
        if EI[idx].sum() else float("nan")

    # ── PIT, cell by cell ────────────────────────────────────────────────
    pit_vals, pit_by_outcome = [], {c: [] for c in OUTCOME_COLS}
    for j, col in enumerate(OUTCOME_COLS):
        m = M_te[:, j].astype(bool)
        if not m.any():
            continue
        y_true = Y_te[m, j][:, None]
        s_j = samples_np[m, :, j]
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

    # ── MMD: joint panel (all 6), plus sub-panels to localize the gap ──
    # Averaged over many posterior draws and many real-vs-real floor splits,
    # not a single draw / single split point estimate -- at panel sizes this
    # small (dozens to a few hundred rows), a single draw's own sampling
    # noise is comparable to or larger than genuine model differences, so a
    # one-shot MMD number is not a reliable basis for comparing checkpoints.
    def mmd_panel(cols, n_repeats=20, seed=0):
        idxs = [OUTCOME_COLS.index(c) for c in cols]
        comp = M_te[:, idxs].all(axis=1)
        real = Y_te[comp][:, idxs]
        gen_all = samples_np[comp][:, :, idxs]  # (n, n_samples, len(cols))
        n = int(comp.sum())
        if n <= 4:
            return {"mmd2": float("nan"), "floor": float("nan"),
                    "ratio": float("nan"), "n_rows": n}
        rng = np.random.default_rng(seed)
        n_samples = gen_all.shape[1]
        m2s, fls = [], []
        for _ in range(n_repeats):
            k = rng.integers(0, n_samples)
            m2s.append(mmd_rbf(gen_all[:, k, :], real,
                               seed=int(rng.integers(0, 1_000_000))))
            perm = rng.permutation(n)
            half = n // 2
            fls.append(mmd_rbf(real[perm[:half]], real[perm[half:]],
                               seed=int(rng.integers(0, 1_000_000))))
        m2, fl = float(np.mean(m2s)), float(np.mean(fls))
        return {"mmd2": m2, "mmd2_std": float(np.std(m2s)),
                "floor": fl, "floor_std": float(np.std(fls)),
                "ratio": m2 / max(fl, 1e-9), "n_rows": n, "n_repeats": n_repeats}

    joint_all = mmd_panel(OUTCOME_COLS)
    mmd2, floor, ratio = joint_all["mmd2"], joint_all["floor"], joint_all["ratio"]

    panel_cognitive = mmd_panel(["CDRSB", "MMSCORE", "TOTAL13"])
    panel_imaging = mmd_panel(["VENT_NORM", "HIPPO_NORM"])
    panel_no_abeta = mmd_panel([c for c in OUTCOME_COLS if c != "ABETA"])

    print("\n" + "=" * 60)
    print("  ADNI MODEL (standalone, seed", a.seed, ") -- CALIBRATION & MMD")
    print("=" * 60)
    print(f"  C-index : {c:.4f}   Average R2 : {np.nanmean(r2):.4f}")
    print(f"\n  {'outcome':12s} {'R2':>7s} {'n_obs':>7s} {'cov@0.9':>8s} {'PIT':>6s}")
    per_outcome = {}
    for j, col in enumerate(OUTCOME_COLS):
        pj = np.array(pit_by_outcome[col])
        c90 = float(np.mean(np.abs(pj - 0.5) <= 0.45)) if len(pj) else float("nan")
        pm = float(pj.mean()) if len(pj) else float("nan")
        print(f"  {col:12s} {r2[j]:7.3f} {int(M_te[:,j].sum()):7d} {c90:8.3f} {pm:6.3f}")
        per_outcome[col] = {"r2": float(r2[j]), "n_obs": int(M_te[:, j].sum()),
                            "coverage_0.9": c90, "pit_mean": pm}

    print(f"\n  CALIBRATION ({len(pit):,} observed cells, {a.n_samples} samples)")
    for L in levels:
        print(f"    nominal {L:.2f}   empirical {cov[str(L)]:.3f}")
    print(f"  PIT mean {pit.mean():.3f}   KS vs uniform {ks:.3f}")

    def _pr(name, d):
        print(f"\n  DISTRIBUTIONAL -- {name} ({d['n_rows']} complete rows)")
        print(f"  generated vs real    MMD2 = {d['mmd2']:.5f}")
        print(f"  real vs real (floor) MMD2 = {d['floor']:.5f}")
        print(f"  ratio                {d['ratio']:.2f}x")

    _pr("joint, all 6 outcomes", joint_all)
    _pr("joint, all 6 EXCEPT ABETA", panel_no_abeta)
    _pr("cognitive panel (CDRSB, MMSCORE, TOTAL13)", panel_cognitive)
    _pr("imaging panel (VENT_NORM, HIPPO_NORM)", panel_imaging)
    print("=" * 60)

    out = {
        "seed": a.seed, "c_index": float(c), "avg_r2": float(np.nanmean(r2)),
        "calibration": {"coverage": cov, "pit_mean": float(pit.mean()),
                        "ks": ks, "n_cells": len(pit)},
        "mmd_joint6": joint_all,
        "mmd_no_abeta": panel_no_abeta,
        "mmd_cognitive": panel_cognitive,
        "mmd_imaging": panel_imaging,
        "per_outcome": per_outcome,
    }
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
