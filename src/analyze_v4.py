"""
Post-hoc analysis of the reported DTG v4 checkpoint.

Produces the three numbers Chapter 6 of the PFE report says were not measured:

  1. Bootstrap confidence intervals on the test C-index and trajectory RMSE.
  2. Correlation between predicted and observed trajectories (Pearson and
     Spearman), overall and per visit, which is the rho that the PROCOVA
     sample-size calculation needs.
  3. The resulting control-arm sample-size reduction factor (1 - rho^2).

Reproduces the exact split of train_v3.py by re-seeding identically before
random_split, so the test set here is the test set the checkpoint was scored on.

Usage:
    python analyze_v4.py --ckpt outputs_v4/best_model.pt --out outputs_v4/analysis.json
"""

import argparse, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from lifelines.utils import concordance_index
from scipy import stats

from preprocess_v2 import load_ppmi_v2, TIMEPOINTS, d, c_dim, T, S
from train_v3 import PPMIDataset, recompute_events
from model_v3 import DTG_v3


def build_test_loader(seed, event_thresh, batch):
    """Rebuild the identical partition used by train_v3.py."""
    X, M, C, _, _, _patients, scaler, _meta = load_ppmi_v2()
    ET, EI = recompute_events(X, M, scaler, event_thresh)
    dataset = PPMIDataset(X, M, C, ET, EI)

    N = len(dataset)
    n_val = max(1, int(N * 0.15))
    n_test = max(1, int(N * 0.15))
    n_train = N - n_val - n_test

    torch.manual_seed(seed)
    np.random.seed(seed)
    _, _, test_ds = random_split(dataset, [n_train, n_val, n_test])
    print(f"  split: train={n_train} val={n_val} test={n_test}")
    return DataLoader(test_ds, batch_size=batch, shuffle=False, drop_last=False)


@torch.no_grad()
def collect(model, loader, times, device):
    """One pass over the test set, keeping per-patient quantities."""
    model.eval()
    risks, ev_times, ev_inds = [], [], []
    per_patient_sq, per_patient_n = [], []
    pred_flat, true_flat, visit_idx = [], [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        mu, _, tte_logits = model(batch["x0_obs"], batch["mask0"], batch["c"], times)

        obs = batch["M_traj"].bool()
        err2 = ((mu - batch["X_traj"]) ** 2) * obs.float()
        per_patient_sq.append(err2.sum(dim=(1, 2)).cpu().numpy())
        per_patient_n.append(obs.float().sum(dim=(1, 2)).cpu().numpy())

        # paired predicted/observed values, only where observed
        B = mu.shape[0]
        for b in range(B):
            for ti in range(T):
                sel = obs[b, ti]
                if sel.any():
                    pred_flat.append(mu[b, ti][sel].cpu().numpy())
                    true_flat.append(batch["X_traj"][b, ti][sel].cpu().numpy())
                    visit_idx.append(np.full(int(sel.sum()), ti))

        pmf = torch.softmax(tte_logits, dim=-1)
        bins = torch.linspace(0, 1, tte_logits.size(1), device=device)
        risks.append((pmf * bins).sum(dim=-1).cpu().numpy())
        ev_times.append(batch["event_time"].cpu().numpy())
        ev_inds.append(batch["event_ind"].cpu().numpy())

    return dict(
        risks=np.concatenate(risks),
        ev_times=np.concatenate(ev_times),
        ev_inds=np.concatenate(ev_inds),
        sq=np.concatenate(per_patient_sq),
        n=np.concatenate(per_patient_n),
        pred=np.concatenate(pred_flat),
        true=np.concatenate(true_flat),
        visit=np.concatenate(visit_idx),
    )


def bootstrap(res, B=2000, seed=42):
    """Patient-level resampling: the patient is the independent unit, not the visit."""
    rng = np.random.default_rng(seed)
    N = len(res["risks"])
    cs, rs = [], []
    for _ in range(B):
        idx = rng.integers(0, N, N)
        et, ei, rk = res["ev_times"][idx], res["ev_inds"][idx], res["risks"][idx]
        if ei.sum() > 1:
            try:
                cs.append(concordance_index(et, rk, ei))
            except Exception:
                pass
        tot_n = res["n"][idx].sum()
        if tot_n > 0:
            rs.append(np.sqrt(res["sq"][idx].sum() / tot_n))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return q(cs), q(rs), len(cs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs_v4/best_model.pt")
    p.add_argument("--out", default="outputs_v4/analysis.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--event_thresh", type=float, default=25.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--nh", type=int, default=64)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--boot", type=int, default=2000)
    a = p.parse_args()

    device = "cpu"
    print("Rebuilding the reported test split...")
    test_ld = build_test_loader(a.seed, a.event_thresh, a.batch)

    model = DTG_v3(d=d, c_dim=c_dim, T=T, nh=a.nh, z_dim=a.z_dim, S=S).to(device)
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    sd = ck.get("model_state") or ck.get("model") or ck
    model.load_state_dict(sd)
    print(f"Loaded {a.ckpt} (epoch {ck.get('epoch', '?')})")

    times = torch.tensor(TIMEPOINTS, dtype=torch.float32, device=device)
    res = collect(model, test_ld, times, device)

    c_point = concordance_index(res["ev_times"], res["risks"], res["ev_inds"])
    rmse_point = float(np.sqrt(res["sq"].sum() / res["n"].sum()))

    print(f"\nPoint estimates: C-index={c_point:.4f}  RMSE={rmse_point:.4f}")
    print(f"Bootstrapping ({a.boot} patient-level resamples)...")
    (c_lo, c_hi), (r_lo, r_hi), nboot = bootstrap(res, a.boot, a.seed)

    # --- correlation between predicted and observed (the PROCOVA rho) ---
    pear = float(stats.pearsonr(res["pred"], res["true"])[0])
    spear = float(stats.spearmanr(res["pred"], res["true"])[0])
    per_visit = {}
    for ti, month in enumerate(TIMEPOINTS):
        m = res["visit"] == ti
        if m.sum() > 10:
            per_visit[f"month_{month}"] = {
                "n_obs": int(m.sum()),
                "pearson": float(stats.pearsonr(res["pred"][m], res["true"][m])[0]),
            }

    var_reduction = 1.0 - pear ** 2

    out = {
        "checkpoint": a.ckpt,
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "n_test_patients": int(len(res["risks"])),
        "n_observed_values": int(len(res["pred"])),
        "c_index": {"point": float(c_point), "ci95": [c_lo, c_hi], "n_bootstrap": nboot},
        "rmse": {"point": rmse_point, "ci95": [r_lo, r_hi]},
        "trajectory_correlation": {
            "pearson_r": pear,
            "spearman_rho": spear,
            "per_visit": per_visit,
        },
        "procova": {
            "rho": pear,
            "variance_retained": var_reduction,
            "control_arm_reduction_pct": 100.0 * (1.0 - var_reduction),
        },
    }

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)

    print(f"\n{'='*62}")
    print(f"  C-index : {c_point:.4f}   95% CI [{c_lo:.4f}, {c_hi:.4f}]")
    print(f"  RMSE    : {rmse_point:.4f}   95% CI [{r_lo:.4f}, {r_hi:.4f}]")
    print(f"  Pearson r (pred vs obs) : {pear:.4f}")
    print(f"  Spearman rho            : {spear:.4f}")
    print(f"  Variance retained (1-r^2)      : {var_reduction:.4f}")
    print(f"  Control-arm reduction          : {100*(1-var_reduction):.1f}%")
    print(f"{'='*62}")
    print(f"written to {a.out}")


if __name__ == "__main__":
    main()
