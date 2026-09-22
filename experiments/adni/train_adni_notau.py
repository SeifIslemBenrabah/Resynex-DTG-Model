"""
Train the same architecture on ADNI (Alzheimer's disease), to test claim C2.

Nothing about the model is adapted to the new disease. The four components, the
loss terms and their weights, the curriculum schedule, the optimizer and the
checkpoint-selection rule are the ones used for Parkinson's disease. Only
n_baseline and n_outcomes differ, and both are read from the data.

Usage:
    python train_adni.py --epochs 60 --out models_out/adni_notau
"""

import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from lifelines.utils import concordance_index

# See model_platform_dtg.py for why this import crosses into the platform
# repository, and how to point RESYNEX_PLATFORM_PATH at it if it isn't
# cloned as this repo's sibling.
_platform_path = os.environ.get("RESYNEX_PLATFORM_PATH") or str(
    Path(__file__).resolve().parents[2].parent / "Resynex-Platform" / "ms-digital-twin")
sys.path.insert(0, _platform_path)
from model.nbm import FeatureNormalizer                 # noqa: E402

from adni_dataset_notau import build, BASELINE_COLS, OUTCOME_COLS, MAX_DAYS  # noqa: E402
from model_platform_dtg import PlatformDTG              # noqa: E402


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
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--lam_var", type=float, default=0.3)
    p.add_argument("--lam_cd", type=float, default=0.1)
    p.add_argument("--pretrain_epochs", type=int, default=0,
                   help="epochs of imp+pred only (cd=var=tte=0) before the "
                        "usual joint curriculum begins, so the point "
                        "predictor converges before the harder energy/"
                        "survival objectives start competing for gradient")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=4)
    p.add_argument("--nh", type=int, default=32,
                   help="NBM hidden units (interaction/correlation modes); "
                        "32 is the value used for every checkpoint reported "
                        "so far, kept as the default for backward "
                        "compatibility")
    p.add_argument("--out", default="models_out/adni_notau")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    X, Y, Tf, PID, ET, EI = build()
    n_in, n_out = X.shape[1], Y.shape[1]

    rng = np.random.default_rng(a.seed)
    pats = np.unique(PID); rng.shuffle(pats)
    nv = max(1, int(len(pats) * .15)); nt = max(1, int(len(pats) * .15))
    te_p, va_p = set(pats[:nt]), set(pats[nt:nt + nv])
    te = np.array([q in te_p for q in PID]); va = np.array([q in va_p for q in PID])
    tr = ~(te | va)
    print(f"  patients train={len(pats)-nv-nt} val={nv} test={nt}")
    print(f"  pairs    train={tr.sum()} val={va.sum()} test={te.sum()}")

    bn, on = FeatureNormalizer(), FeatureNormalizer()
    bn.fit(X[tr]); on.fit(Y[tr])
    msk = (~np.isnan(Y)).astype(np.float32)
    Yf = np.nan_to_num(Y)

    tt = lambda v: torch.tensor(v, dtype=torch.float32)
    Xt, Yt, Tt, ETt, EIt = tt(X), tt(Y), tt(Tf), tt(ET), tt(EI)

    torch.manual_seed(a.seed)
    # The curr_* block is outcome-dimensional here too, so the head is wired the
    # same way as for Parkinson's disease.
    curr_idx = [BASELINE_COLS.index("curr_" + c) for c in OUTCOME_COLS]
    model = PlatformDTG(n_baseline=n_in, n_outcomes=n_out, nh=a.nh,
                        embed_dim=192, flow_layers=3, S=60, curr_idx=curr_idx)
    model.attach_normalizers(bn, on)
    print(f"  PlatformDTG  n_baseline={n_in}  n_outcomes={n_out}  "
          f"params={sum(q.numel() for q in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs,
                                                       eta_min=1e-6)
    idx = np.where(tr)[0]
    best, best_state = -9.9, None

    for ep in range(1, a.epochs + 1):
        if ep <= a.pretrain_epochs:
            lambdas = {"imp": 1.0, "pred": 2.0, "cd": 0.0, "var": 0.0, "tte": 0.0}
        else:
            e = ep - a.pretrain_epochs
            w = max(1, a.warmup)
            lam = 0.0 if e <= w else min(1.0, (e - w) / max(1, (a.epochs - a.pretrain_epochs) - w))
            lambdas = {"imp": 1.0, "pred": 2.0, "cd": a.lam_cd, "var": a.lam_var,
                       "tte": lam}
        model.train(); rng.shuffle(idx)
        for s in tqdm(range(0, len(idx), a.batch), desc=f"ep{ep:3d}",
                      leave=False, ncols=56):
            b = idx[s:s + a.batch]
            loss = model.losses(Xt[b], Tt[b], Yt[b], ETt[b], EIt[b],
                                lambdas)["total"]
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        sched.step()

        if ep % a.eval_every == 0 or ep == a.epochs:
            model.eval()
            with torch.no_grad():
                _, _, _, fv, _ = model.predict(Xt[va], Tt[va], mc_steps=8,
                                               n_samples=8)
                risk = model.risk_score(Xt[va]).numpy()
            r2v, _, _ = per_outcome(fv.numpy(), Yf[va], msk[va])
            r2m = float(np.nanmean(r2v))
            try:
                cv = concordance_index(ET[va], risk, EI[va]) if EI[va].sum() else 0.0
            except Exception:
                cv = 0.0
            score = r2m + 2.0 * (cv - 0.5)
            if score > best:
                best = score
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  ep{ep:3d}  val R2={r2m:.4f}  C={cv:.4f}", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, _, _, ft, _ = model.predict(Xt[te], Tt[te], mc_steps=16, n_samples=32)
        risk = model.risk_score(Xt[te]).numpy()
    r2t, rmse, mae = per_outcome(ft.numpy(), Yf[te], msk[te])
    c = concordance_index(ET[te], risk, EI[te])

    line = "=" * 60
    print(f"\n{line}\n  ADNI (no-TAU) TEST — same architecture, no structural change\n{line}")
    perf = {}
    for j, cn in enumerate(OUTCOME_COLS):
        print(f"  {cn:14s} R2={r2t[j]:7.3f}  RMSE={rmse[j]:9.3f}  MAE={mae[j]:9.3f}")
        perf[cn] = {"r2": round(float(r2t[j]), 3),
                    "rmse": round(float(rmse[j]), 3),
                    "mae": round(float(mae[j]), 3)}
    avg = float(np.nanmean(r2t))
    perf["average_r2"] = round(avg, 3)
    print(f"  {'AVERAGE':14s} R2={avg:7.3f}")
    print(f"  {'C-index':14s}    {c:7.4f}   (chance 0.50)")
    print(line)

    torch.save(model.state_dict(), f"{a.out}/weights.pt")
    json.dump({"dataset": "ADNI (ADNIMERGE2, TAU removed, VENT_NORM log-transformed)", "n_baseline": n_in,
               "n_outcomes": n_out, "n_patients": int(len(pats)),
               "n_pairs": int(len(X)), "model_performance": perf,
               "c_index": round(float(c), 4),
               "architecture": "PlatformDTG (DeepHit + trajectory pooling)"},
              open(f"{a.out}/results.json", "w"), indent=2)
    print(f"  written to {a.out}/results.json")


if __name__ == "__main__":
    main()
