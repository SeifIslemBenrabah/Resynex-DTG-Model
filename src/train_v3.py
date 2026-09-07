"""
DTG PPMI v3 training — targeting C-index > 0.79.

Key changes vs v2:
  - Trajectory-pooled TTE head (gradient co-adapts NBM + TTE jointly)
  - Curriculum lambda_tte: 0 for warmup epochs, then linearly ramps to 1.0
  - Lower NP3 event threshold: 25 (vs 33) for richer survival signal
  - Larger z_dim=128 (default), deeper imputer with LayerNorm
  - 200 total epochs, all joint — NO two-phase separation
  - C-index evaluated every 5 epochs from warmup_end onwards
  - Best checkpoint saved on val C-index (not RMSE)

Usage:
    cd C:\\Users\\admin\\Desktop\\DTG_PPMI_v2
    python train_v3.py
    python train_v3.py --epochs 200 --warmup 60 --z_dim 128 --out outputs_v3
"""

import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from tqdm import tqdm

from preprocess_v2 import (
    load_ppmi_v2, FEAT_COLS, STATIC_COLS, TIMEPOINTS,
    d, c_dim, T, S, NP3_ITEMS,
)
from model_v3 import DTG_v3


# ── Dataset ───────────────────────────────────────────────────────────────────

class PPMIDataset(Dataset):
    def __init__(self, X, M, C, ET, EI):
        self.X  = torch.tensor(np.nan_to_num(X, nan=0.0), dtype=torch.float32)
        self.M  = torch.tensor(M,  dtype=torch.float32)
        self.C  = torch.tensor(C,  dtype=torch.float32)
        self.ET = torch.tensor(ET, dtype=torch.float32)
        self.EI = torch.tensor(EI, dtype=torch.float32)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return {"x0_obs":     self.X[idx, 0, :],
                "mask0":      self.M[idx, 0, :],
                "c":          self.C[idx],
                "X_traj":     self.X[idx],
                "M_traj":     self.M[idx],
                "event_time": self.ET[idx],
                "event_ind":  self.EI[idx]}


# ── Event re-computation ──────────────────────────────────────────────────────

def recompute_events(X, M, scaler, event_thresh=25.0):
    """
    Override the preprocess TTE with a lower NP3 threshold.
    Returns ET (normalised [0,1]), EI.
    """
    np3_cols_idx = [FEAT_COLS.index(c) for c in NP3_ITEMS]
    raw_X = X * scaler.scale_ + scaler.mean_   # un-scale  (N, T, d)

    N = X.shape[0]
    np3_raw = np.full((N, T), np.nan, dtype=np.float32)
    for ti in range(T):
        obs_ti  = M[:, ti, :][:, np3_cols_idx].astype(bool)
        vals_ti = raw_X[:, ti, :][:, np3_cols_idx]
        np3_ti  = np.where(obs_ti, vals_ti, np.nan)
        np3_raw[:, ti] = np.nansum(np3_ti, axis=1)
        no_obs = (obs_ti.sum(axis=1) == 0)
        np3_raw[no_obs, ti] = np.nan

    ET_raw = np.full(N, 60.0, dtype=np.float32)
    EI     = np.zeros(N, dtype=np.float32)
    for i in range(N):
        for ti, t in enumerate(TIMEPOINTS):
            if not np.isnan(np3_raw[i, ti]) and np3_raw[i, ti] >= event_thresh:
                ET_raw[i] = float(t)
                EI[i]     = 1.0
                break

    events = int(EI.sum())
    print(f"  NP3 >= {event_thresh} events: {events} / {N}  ({events/N*100:.1f}%)")
    return (ET_raw / 60.0).astype(np.float32), EI


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, times, device):
    model.eval()
    all_losses, pred_risks, true_times, true_inds = [], [], [], []
    sq_err, n_obs = 0.0, 0

    for batch in loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        losses = model.compute_losses(batch, times)
        all_losses.append(losses["total"].item())

        mu, _, tte_logits = model(batch["x0_obs"], batch["mask0"], batch["c"], times)

        obs    = batch["M_traj"].bool()
        sq_err += ((mu - batch["X_traj"]) ** 2 * obs.float()).sum().item()
        n_obs  += obs.float().sum().item()

        pmf  = torch.softmax(tte_logits, dim=-1)
        bins = torch.linspace(0, 1, tte_logits.size(1), device=device)
        risk = (pmf * bins).sum(dim=-1)
        pred_risks.append(risk.cpu().numpy())
        true_times.append(batch["event_time"].cpu().numpy())
        true_inds.append(batch["event_ind"].cpu().numpy())

    pred_risks = np.concatenate(pred_risks)
    true_times = np.concatenate(true_times)
    true_inds  = np.concatenate(true_inds)

    try:
        c_idx = concordance_index(true_times, pred_risks, true_inds) \
                if true_inds.sum() > 0 else float("nan")
    except Exception:
        c_idx = float("nan")

    return {"val_loss": float(np.mean(all_losses)),
            "rmse":     float(np.sqrt(sq_err / max(n_obs, 1))),
            "c_index":  c_idx}


# ── Training ──────────────────────────────────────────────────────────────────

def lambda_schedule(epoch, warmup, total):
    """Curriculum: 0 until warmup, then linear ramp to 1.0 by total."""
    if epoch <= warmup:
        return 0.0
    ramp_epochs = max(1, total - warmup)
    return min(1.0, (epoch - warmup) / ramp_epochs)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, M, C, ET_old, EI_old, patnos, scaler, meta = load_ppmi_v2()

    # Override events with lower threshold
    ET, EI = recompute_events(X, M, scaler, event_thresh=args.event_thresh)

    dataset = PPMIDataset(X, M, C, ET, EI)
    N       = len(dataset)
    n_val   = max(1, int(N * 0.15))
    n_test  = max(1, int(N * 0.15))
    n_train = N - n_val - n_test

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, drop_last=False)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, drop_last=False)
    print(f"Split: train={n_train}  val={n_val}  test={n_test}")

    model  = DTG_v3(d=d, c_dim=c_dim, T=T, nh=args.nh, z_dim=args.z_dim, S=S).to(device)
    times  = torch.tensor(TIMEPOINTS, dtype=torch.float32, device=device)
    os.makedirs(args.out, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nDTG v3  --  d={d}  c={c_dim}  T={T}  nh={args.nh}  z={args.z_dim}  "
          f"S={S}  params={n_params:,}")
    print(f"Schedule: warmup={args.warmup} epochs then curriculum ramp to epoch {args.epochs}")

    # Single optimizer for all parameters (joint training throughout)
    opt = torch.optim.AdamW([
        {"params": list(model.nbm.bias_net.parameters()),      "weight_decay": 0.5},
        {"params": list(model.nbm.precision_net.parameters()), "weight_decay": 0.5},
        {"params": list(model.nbm.weights_net.parameters()),   "weight_decay": 1.0},
        {"params": list(model.imputer.parameters()),           "weight_decay": 1e-4},
        {"params": list(model.tte_head.parameters()),          "weight_decay": 1e-3},
    ], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    history = {k: [] for k in ["epoch","train_loss","val_loss","rmse","c_index",
                                 "pred_loss","cd_loss","tte_loss","lambda_tte"]}
    best_c      = 0.0
    best_rmse   = float("inf")
    epoch_start = 1

    # ── Optional checkpoint resume ─────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)

        if getattr(args, "reset_tte", False):
            # Load only imputer + NBM; leave TTE head freshly initialised.
            # Use-case: fixing the ranking-loss sign bug when TTE was trained
            # with wrong loss, but imputer/NBM are good.
            partial = {k: v for k, v in resume_ckpt["model_state"].items()
                       if not k.startswith("tte_head.")}
            model.load_state_dict(partial, strict=False)
            epoch_start = args.warmup + 1
            best_rmse   = resume_ckpt["metrics"]["rmse"]
            for _ in range(epoch_start - 1):
                sched.step()
            print(f"\nPartial resume (imputer+NBM) from '{args.resume}'")
            print(f"  TTE head re-initialised. Starting from epoch {epoch_start}")
        else:
            model.load_state_dict(resume_ckpt["model_state"])
            epoch_start = resume_ckpt["epoch"] + 1
            best_rmse   = resume_ckpt["metrics"]["rmse"]
            for _ in range(epoch_start - 1):
                sched.step()
            print(f"\nResumed from '{args.resume}'")
            print(f"  checkpoint epoch: {resume_ckpt['epoch']}  "
                  f"RMSE: {resume_ckpt['metrics']['rmse']:.4f}")
            print(f"  Continuing from epoch {epoch_start}")

    for epoch in range(epoch_start, args.epochs + 1):
        lam_tte = lambda_schedule(epoch, args.warmup, args.epochs)
        lambdas = {"imp": 1.0, "pred": 2.0, "cd": 0.1, "tte": lam_tte}

        model.train()
        ep = {"total":[], "pred":[], "cd":[], "imp":[], "tte":[]}

        for batch in tqdm(train_ld, desc=f"Ep {epoch:3d}", leave=False, ncols=80):
            batch  = {k: v.to(device) for k, v in batch.items()}
            losses = model.compute_losses(batch, times, lambdas)
            if not torch.isfinite(losses["total"]):
                continue
            opt.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in ep:
                v = losses.get(k, torch.tensor(0.0))
                if torch.isfinite(v): ep[k].append(v.item())

        sched.step()

        eval_this = (epoch % args.eval_every == 0 or epoch == args.epochs
                     or epoch == args.warmup)
        if eval_this:
            tl = float(np.mean(ep["total"])) if ep["total"] else float("nan")
            pl = float(np.mean(ep["pred"]))  if ep["pred"]  else 0.0
            cl = float(np.mean(ep["cd"]))    if ep["cd"]    else 0.0
            tl2= float(np.mean(ep["tte"]))   if ep["tte"]   else 0.0

            m = evaluate(model, val_ld, times, device)

            history["epoch"].append(epoch)
            history["train_loss"].append(tl)
            history["val_loss"].append(m["val_loss"])
            history["rmse"].append(m["rmse"])
            history["c_index"].append(m["c_index"])
            history["pred_loss"].append(pl)
            history["cd_loss"].append(cl)
            history["tte_loss"].append(tl2)
            history["lambda_tte"].append(lam_tte)

            ci_str = f"C={m['c_index']:.4f}" if lam_tte > 0 else "C=n/a"
            print(f"  Ep {epoch:3d} lam={lam_tte:.2f} | "
                  f"pred={pl:.3f} cd={cl:.3f} tte={tl2:.3f} | "
                  f"RMSE={m['rmse']:.4f} {ci_str}")

            # Save best RMSE checkpoint (regardless of phase)
            if m["rmse"] < best_rmse:
                best_rmse = m["rmse"]
                torch.save({"epoch":epoch,"lambda_tte":lam_tte,
                            "model_state":model.state_dict(),"metrics":m,
                            "args":vars(args),"meta":meta,
                            "scaler_mean":scaler.mean_.tolist(),
                            "scaler_scale":scaler.scale_.tolist()},
                           Path(args.out)/"best_rmse.pt")

            # Save best C-index checkpoint (only once TTE has warmed up)
            if lam_tte > 0 and not np.isnan(m["c_index"]) and m["c_index"] > best_c:
                best_c = m["c_index"]
                torch.save({"epoch":epoch,"lambda_tte":lam_tte,
                            "model_state":model.state_dict(),"metrics":m,
                            "args":vars(args),"meta":meta,
                            "scaler_mean":scaler.mean_.tolist(),
                            "scaler_scale":scaler.scale_.tolist()},
                           Path(args.out)/"best_model.pt")
                print(f"  ** New best C-index: {best_c:.4f} (epoch {epoch})")

    # ── Test evaluation ───────────────────────────────────────────────────────
    ckpt_path = Path(args.out) / "best_model.pt"
    if not ckpt_path.exists():
        # Warmup not done yet or no events — fall back to best RMSE
        ckpt_path = Path(args.out) / "best_rmse.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(model, test_ld, times, device)

    print("\n" + "=" * 55)
    print("TEST SET RESULTS (DTG v3)")
    print(f"  RMSE    : {test_m['rmse']:.4f}")
    print(f"  C-index : {test_m['c_index']:.4f}")
    print(f"  Best val C-index : {ckpt['metrics']['c_index']:.4f}  epoch {ckpt['epoch']}")
    print("=" * 55)

    (Path(args.out) / "history.json").write_text(json.dumps(history))

    summary = {"test_rmse": test_m["rmse"], "test_c_index": test_m["c_index"],
               "best_val_c_index": ckpt["metrics"]["c_index"],
               "checkpoint_epoch": ckpt["epoch"],
               "event_thresh": args.event_thresh,
               "n_train": n_train, "n_val": n_val, "n_test": n_test,
               "d": d, "c_dim": c_dim, "T": T,
               "nh": args.nh, "z_dim": args.z_dim, "n_params": n_params}
    (Path(args.out) / "test_results.json").write_text(json.dumps(summary, indent=2))

    _plot_history(history, args.out)
    _plot_trajectories(model, test_ds, times, device, scaler, FEAT_COLS, args.out)
    print(f"Outputs saved to {args.out}/")
    return model, scaler, test_m


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_history(history, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    x = history["epoch"] if history["epoch"] else range(len(history["train_loss"]))

    axes[0].plot(x, history["train_loss"], label="train")
    axes[0].plot(x, history["val_loss"],   label="val")
    axes[0].set_title("Total Loss"); axes[0].legend()

    axes[1].plot(x, history["pred_loss"],  label="pred (MSE)", color="darkorange")
    axes[1].plot(x, history["cd_loss"],    label="CD (NBM)",   color="forestgreen")
    axes[1].plot(x, history["tte_loss"],   label="TTE",        color="steelblue")
    axes[1].set_title("Loss Components"); axes[1].legend()

    axes[2].plot(x, history["rmse"], color="darkorange")
    axes[2].set_title("Trajectory RMSE")

    axes[3].plot(x, history["c_index"], color="steelblue", marker="o", markersize=3)
    axes[3].axhline(0.5, ls="--", color="gray", lw=1, label="random")
    axes[3].axhline(0.79, ls="--", color="red", lw=1, label="target")
    axes[3].set_ylim(0.3, 1.0)
    axes[3].set_title("TTE C-index (target > 0.79)")
    axes[3].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(Path(out_dir) / "training_curves.png", dpi=150)
    plt.close()
    print("  Saved training_curves.png")


def _plot_trajectories(model, test_ds, times, device, scaler, feat_names, out_dir,
                       n_patients=4):
    model.eval()
    indices = np.random.choice(len(test_ds), min(n_patients, len(test_ds)), replace=False)
    batch   = {k: torch.stack([test_ds[i][k] for i in indices]).to(device)
               for k in test_ds[0].keys()}

    with torch.no_grad():
        mu, _, _ = model(batch["x0_obs"], batch["mask0"], batch["c"], times)
        twin     = model.generate_twin(batch["x0_obs"], batch["mask0"],
                                       batch["c"], times, n_samples=50)

    mu_np   = mu.cpu().numpy()
    twin_np = twin.cpu().numpy()
    obs_np  = batch["X_traj"].cpu().numpy()
    mask_np = batch["M_traj"].cpu().numpy()

    n_feat = min(4, mu_np.shape[-1])
    t_axis = [int(t.item()) for t in times]

    fig, axes = plt.subplots(n_patients, n_feat, figsize=(4*n_feat, 3.5*n_patients))
    if n_patients == 1: axes = axes[np.newaxis, :]

    for i in range(n_patients):
        for j in range(n_feat):
            ax = axes[i, j]
            lo = np.percentile(twin_np[i, :, :, j], 5,  axis=0)
            hi = np.percentile(twin_np[i, :, :, j], 95, axis=0)
            lo = lo * scaler.scale_[j] + scaler.mean_[j]
            hi = hi * scaler.scale_[j] + scaler.mean_[j]
            mu_i = mu_np[i, :, j] * scaler.scale_[j] + scaler.mean_[j]

            ax.fill_between(t_axis, lo, hi, alpha=0.25, color="royalblue")
            ax.plot(t_axis, mu_i, "royalblue", lw=2, label="mean")

            obs_mask = mask_np[i, :, j].astype(bool)
            if obs_mask.any():
                obs_i = obs_np[i, obs_mask, j] * scaler.scale_[j] + scaler.mean_[j]
                ax.scatter(np.array(t_axis)[obs_mask], obs_i,
                           color="black", zorder=5, s=30, label="Observed")

            ax.set_title(f"P{i+1} - {feat_names[j]}", fontsize=8)
            ax.set_xlabel("Months")
            if j == 0: ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(Path(out_dir) / "sample_trajectories.png", dpi=150)
    plt.close()
    print("  Saved sample_trajectories.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DTG v3 on PPMI — trajectory-pooled TTE")
    p.add_argument("--out",          default="outputs_v3")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--warmup",       type=int,   default=60,
                   help="epochs before TTE loss is activated (curriculum warmup)")
    p.add_argument("--batch",        type=int,   default=16)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--nh",           type=int,   default=64)
    p.add_argument("--z_dim",        type=int,   default=128)
    p.add_argument("--eval_every",   type=int,   default=5)
    p.add_argument("--event_thresh", type=float, default=25.0,
                   help="NP3 sum threshold for PD motor event (default 25, was 33)")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--resume",       default=None,
                   help="path to a checkpoint .pt file to resume from")
    p.add_argument("--reset_tte",    action="store_true",
                   help="when resuming: reload only imputer+NBM, reinit TTE head fresh")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)