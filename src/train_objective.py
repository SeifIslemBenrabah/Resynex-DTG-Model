"""
Train the generator on the objective-endpoint panel.

Target
------
Seven instrument-measured endpoints: three cerebrospinal biomarkers and four
striatal binding ratios. These are the readings a trial adopts when it wants an
endpoint that does not depend on a clinician's judgement.

What is NOT changed
-------------------
  * The INPUT is the full 19-feature panel. The clinician-rated subscales remain
    available to the model as predictors; they are simply not among the
    quantities it is asked to reproduce.
  * The clinical EVENT is still the first visit at which the summed MDS-UPDRS
    Part III score crosses its threshold, computed from the raw registry by the
    same code path as every other run here. The survival task is therefore
    identical and the concordance index stays comparable.

Why restricting the target is expected to help rather than merely to flatter:
the 69-to-19 change already showed that targets whose measurement noise
dominates their signal consume capacity and degrade the features that do carry
signal. Imaging rose from 0.664 to 0.838 without its data changing at all. This
takes that observation one step further.

Usage:
    python train_objective.py --epochs 70 --warmup 10 --lam_var 1.0
"""

import argparse, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from lifelines.utils import concordance_index

from preprocess_v2 import TIMEPOINTS, c_dim, T, S
from aggregate_panel import build_aggregated, AGG_COLS, OBJECTIVE_COLS, d_agg
from train_v3 import PPMIDataset, recompute_events
from model_hybrid import HybridDTG

TGT = [AGG_COLS.index(c) for c in OBJECTIVE_COLS]


class ObjectiveDTG(HybridDTG):
    """HybridDTG whose trajectory losses are restricted to the target columns.

    The mask is zeroed outside the targets, so untargeted columns contribute no
    gradient to the trajectory or contrastive terms. Everything else -- the
    imputer, the pooled summaries feeding the survival head, the event labels --
    is untouched.
    """

    def compute_losses(self, batch, times, lambdas=None):
        b = dict(batch)
        keep = torch.zeros(self.d, dtype=b["M_traj"].dtype)
        keep[TGT] = 1.0
        b["M_traj"] = b["M_traj"] * keep.view(1, 1, -1)
        return super().compute_losses(b, times, lambdas)


def rmse_mae_r2(P, Y, M, cols):
    j = [AGG_COLS.index(c) for c in cols]
    P, Y, M = P[:, :, j], Y[:, :, j], M[:, :, j]
    n = M.sum()
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rm = float(np.sqrt((((P - Y) ** 2) * M).sum() / n))
    ma = float((np.abs(P - Y) * M).sum() / n)
    mean = (Y * M).sum(axis=(0, 1)) / np.maximum(M.sum(axis=(0, 1)), 1e-8)
    sse = (((P - Y) ** 2) * M).sum(); sst = (((Y - mean) ** 2) * M).sum()
    return rm, ma, float(1 - sse / max(sst, 1e-8))


@torch.no_grad()
def collect(model, loader, times):
    P, Y, M, R, ET, EI = [], [], [], [], [], []
    model.eval()
    for b in loader:
        mu, _, tte = model(b["x0_obs"], b["mask0"], b["c"], times)
        P.append(mu.numpy()); Y.append(b["X_traj"].numpy()); M.append(b["M_traj"].numpy())
        pmf = torch.softmax(tte, dim=-1)
        bins = torch.linspace(0, 1, tte.size(1))
        R.append((pmf * bins).sum(dim=-1).numpy())
        ET.append(b["event_time"].numpy()); EI.append(b["event_ind"].numpy())
    return (np.concatenate(P), np.concatenate(Y), np.concatenate(M),
            np.concatenate(R), np.concatenate(ET), np.concatenate(EI))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs_obj")
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lam_var", type=float, default=1.0)
    p.add_argument("--lam_cd", type=float, default=0.5)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--event_thresh", type=float, default=25.0)
    p.add_argument("--bins", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nh", type=int, default=64,
                   help="NBM hidden units; 64 is the value used for every "
                        "checkpoint reported so far, kept as default")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    Xa, Ma, C, sc, X69, M69, sc69, _ = build_aggregated(verbose=False)
    ET_, EI_ = recompute_events(X69, M69, sc69, a.event_thresh)
    ds = PPMIDataset(Xa, Ma, C, ET_, EI_)
    N = len(ds); nv = max(1, int(N * .15)); nt = max(1, int(N * .15))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tr, va, te = random_split(ds, [N - nv - nt, nv, nt])
    train_ld = DataLoader(tr, batch_size=a.batch, shuffle=True, drop_last=True)
    val_ld = DataLoader(va, batch_size=a.batch, shuffle=False)
    test_ld = DataLoader(te, batch_size=a.batch, shuffle=False)
    print(f"  targets: {len(OBJECTIVE_COLS)} of {d_agg}  ({', '.join(OBJECTIVE_COLS)})")

    torch.manual_seed(a.seed)
    model = ObjectiveDTG(d=d_agg, c_dim=c_dim, T=T, nh=a.nh, z_dim=128, S=a.bins)
    opt = torch.optim.AdamW([
        {"params": list(model.nbm.parameters()),      "weight_decay": 0.5},
        {"params": list(model.flow.parameters()),     "weight_decay": 1e-4},
        {"params": list(model.imputer.parameters()),  "weight_decay": 1e-4},
        {"params": list(model.tte_head.parameters()), "weight_decay": 1e-3},
    ], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-6)
    times = torch.tensor(TIMEPOINTS, dtype=torch.float32)

    best, best_state = -9.9, None
    for ep in range(1, a.epochs + 1):
        lam = 0.0 if ep <= a.warmup else min(
            1.0, (ep - a.warmup) / max(1, a.epochs - a.warmup))
        lambdas = {"imp": 1.0, "pred": 2.0, "cd": a.lam_cd, "tte": lam,
                   "var": a.lam_var}
        model.train()
        for b in tqdm(train_ld, desc=f"ep{ep:3d}", leave=False, ncols=58):
            loss = model.compute_losses(b, times, lambdas)["total"]
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        sched.step()

        if ep % a.eval_every == 0 or ep == a.epochs:
            P, Y, M, R, ETv, EIv = collect(model, val_ld, times)
            _, _, r2v = rmse_mae_r2(P, Y, M, OBJECTIVE_COLS)
            try:
                cv = concordance_index(ETv, R, EIv) if EIv.sum() else 0.0
            except Exception:
                cv = 0.0
            score = r2v + 2.0 * (cv - 0.5)
            if score > best:
                best = score
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  ep{ep:3d}  val R2={r2v:.4f}  C={cv:.4f}", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    torch.save({"model_state": model.state_dict()}, f"{a.out}/model.pt")

    P, Y, M, R, ETv, EIv = collect(model, test_ld, times)
    rm, ma, r2v = rmse_mae_r2(P, Y, M, OBJECTIVE_COLS)
    c = concordance_index(ETv, R, EIv)

    line = "=" * 60
    print(f"\n{line}\n  TEST — objective-endpoint panel\n{line}")
    print(f"  C-index : {c:.4f}")
    print(f"  RMSE    : {rm:.4f}")
    print(f"  MAE     : {ma:.4f}")
    print(f"  R2      : {r2v:.4f}")
    print(f"\n  {'group':22s} {'RMSE':>8s} {'MAE':>8s} {'R2':>8s}")
    out = {"overall": {"rmse": rm, "mae": ma, "r2": r2v, "c_index": c}, "groups": {}}
    for g, cols in (("Imaging (DaTscan)",
                     ["MIA_CAUDATE_L", "MIA_CAUDATE_R", "MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]),
                    ("CSF biomarkers", ["asyn", "tau", "ptau"])):
        rg, mg, r2g = rmse_mae_r2(P, Y, M, cols)
        print(f"  {g:22s} {rg:8.3f} {mg:8.3f} {r2g:8.3f}")
        out["groups"][g] = {"rmse": rg, "mae": mg, "r2": r2g}
    print(f"\n  {'endpoint':16s} {'R2':>8s}")
    for cl in OBJECTIVE_COLS:
        _, _, r1 = rmse_mae_r2(P, Y, M, [cl])
        print(f"  {cl:16s} {r1:8.3f}")
    print(line)

    json.dump(out, open(f"{a.out}/results.json", "w"), indent=2)
    print(f"written to {a.out}/results.json")


if __name__ == "__main__":
    main()
