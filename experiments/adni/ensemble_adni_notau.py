"""Evaluate the ADNI (no-TAU, 6-outcome) seed ensemble on the held-out patients."""
import json, os, sys
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
from model.nbm import FeatureNormalizer                 # noqa: E402

from adni_dataset_notau import build, BASELINE_COLS, OUTCOME_COLS   # noqa: E402
from model_platform_dtg import PlatformDTG              # noqa: E402

SEEDS = [42, 123, 456, 789, 1000]


def per_outcome(pred, true, mask):
    n = np.maximum(mask.sum(axis=0), 1)
    rmse = np.sqrt((((pred - true) ** 2) * mask).sum(axis=0) / n)
    mae = (np.abs(pred - true) * mask).sum(axis=0) / n
    mu = (true * mask).sum(axis=0) / n
    sse = (((pred - true) ** 2) * mask).sum(axis=0)
    sst = (((true - mu) ** 2) * mask).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1 - sse / np.maximum(sst, 1e-8)
    return r2, rmse, mae


def main():
    X, Y, Tf, PID, ET, EI = build(verbose=False)
    n_in, n_out = X.shape[1], Y.shape[1]

    rng = np.random.default_rng(42)
    pats = np.unique(PID); rng.shuffle(pats)
    nv = max(1, int(len(pats) * .15)); nt = max(1, int(len(pats) * .15))
    te_p = set(pats[:nt])
    te = np.array([q in te_p for q in PID])
    tr = ~np.array([q in te_p or q in set(pats[nt:nt + nv]) for q in PID])

    bn, on = FeatureNormalizer(), FeatureNormalizer()
    bn.fit(X[tr]); on.fit(Y[tr])
    msk = (~np.isnan(Y)).astype(np.float32)
    Yf = np.nan_to_num(Y)

    Xt = torch.tensor(X[te], dtype=torch.float32)
    Tt = torch.tensor(Tf[te], dtype=torch.float32)
    curr_idx = [BASELINE_COLS.index("curr_" + c) for c in OUTCOME_COLS]

    P, R = [], []
    for s in SEEDS:
        m = PlatformDTG(n_baseline=n_in, n_outcomes=n_out, nh=32,
                        embed_dim=192, flow_layers=3, S=60, curr_idx=curr_idx)
        m.load_state_dict(torch.load(f"models_out/adni_notau_s{s}/weights.pt",
                                     map_location="cpu"), strict=False)
        m.attach_normalizers(bn, on); m.eval()
        with torch.no_grad():
            _, _, _, f, _ = m.predict(Xt, Tt, mc_steps=16, n_samples=16)
            R.append(m.risk_score(Xt).numpy())
        P.append(f.numpy())
    P = np.stack(P); R = np.stack(R)

    line = "=" * 62
    print(f"\n{line}\n  ADNI (no-TAU) — members and ensemble ({te.sum()} held-out pairs)\n{line}")
    print(f"  {'model':12s} {'R2':>8s} {'C-index':>9s}")
    for k, s in enumerate(SEEDS):
        r2, _, _ = per_outcome(P[k], Yf[te], msk[te])
        c = concordance_index(ET[te], R[k], EI[te])
        print(f"  seed {s:<7d} {np.nanmean(r2):8.4f} {c:9.4f}")

    Pm, Rm = P.mean(axis=0), R.mean(axis=0)
    r2, rmse, mae = per_outcome(Pm, Yf[te], msk[te])
    c = concordance_index(ET[te], Rm, EI[te])
    print(f"  {'ENSEMBLE':12s} {np.nanmean(r2):8.4f} {c:9.4f}")

    print(f"\n  {'measure':14s} {'R2':>8s} {'RMSE':>10s} {'MAE':>10s}")
    perf = {}
    for j, cn in enumerate(OUTCOME_COLS):
        print(f"  {cn:14s} {r2[j]:8.3f} {rmse[j]:10.3f} {mae[j]:10.3f}")
        perf[cn] = {"r2": round(float(r2[j]), 3),
                    "rmse": round(float(rmse[j]), 3),
                    "mae": round(float(mae[j]), 3)}
    perf["average_r2"] = round(float(np.nanmean(r2)), 3)
    print(line)

    json.dump({"dataset": "ADNI (ADNIMERGE2, TAU removed, VENT_NORM log-transformed)",
               "seeds": SEEDS, "n_baseline": n_in, "n_outcomes": n_out,
               "model_performance": perf, "c_index": round(float(c), 4)},
              open("models_out/adni_notau_ensemble.json", "w"), indent=2)
    print("written to models_out/adni_notau_ensemble.json")


if __name__ == "__main__":
    main()
