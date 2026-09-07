"""
DTG PPMI v2 training loop.

Usage:
    python train_v2.py
    python train_v2.py --epochs 200 --batch 32 --out outputs_v2
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

from preprocess_v2 import load_ppmi_v2, FEAT_COLS, STATIC_COLS, TIMEPOINTS, d, c_dim, T, S
from model_v2 import DTG_v2


# ── Dataset ───────────────────────────────────────────────────────────────────

class PPMIDataset(Dataset):
    def __init__(self, X, M, C, ET, EI):
        self.X  = torch.tensor(np.nan_to_num(X, nan=0.0), dtype=torch.float32)
        self.M  = torch.tensor(M,  dtype=torch.float32)
        self.C  = torch.tensor(C,  dtype=torch.float32)
        self.ET = torch.tensor(ET, dtype=torch.float32)
        self.EI = torch.tensor(EI, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "x0_obs":     self.X[idx, 0, :],
            "mask0":      self.M[idx, 0, :],
            "c":          self.C[idx],
            "X_traj":     self.X[idx],
            "M_traj":     self.M[idx],
            "event_time": self.ET[idx],
            "event_ind":  self.EI[idx],
        }


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

        mu, x_hat, tte_logits = model(
            batch["x0_obs"], batch["mask0"], batch["c"], times)

        obs  = batch["M_traj"].bool()
        diff = (mu - batch["X_traj"]) ** 2
        sq_err += (diff * obs.float()).sum().item()
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

    return {
        "val_loss": float(np.mean(all_losses)),
        "rmse":     float(np.sqrt(sq_err / max(n_obs, 1))),
        "c_index":  c_idx,
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, M, C, ET, EI, patnos, scaler, meta = load_ppmi_v2()

    dataset = PPMIDataset(X, M, C, ET, EI)
    N       = len(dataset)
    n_val   = max(1, int(N * 0.15))
    n_test  = max(1, int(N * 0.15))
    n_train = N - n_val - n_test

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, drop_last=False)
    print(f"Split -- train:{n_train}  val:{n_val}  test:{n_test}")

    model = DTG_v2(d=d, c_dim=c_dim, T=T, nh=args.nh, z_dim=args.z_dim, S=S).to(device)

    times   = torch.tensor(TIMEPOINTS, dtype=torch.float32, device=device)
    os.makedirs(args.out, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nDTG v2  --  d={d}  c={c_dim}  T={T}  nh={args.nh}  z={args.z_dim}  "
          f"S={S}  params={n_params:,}")

    history = {"train_loss": [], "val_loss": [], "rmse": [], "c_index": [],
               "imp_loss": [], "pred_loss": [], "cd_loss": [], "tte_loss": [],
               "phase": []}

    # ── Phase 1: Trajectory learning (imputer + NBM, no TTE) ─────────────────
    print(f"\n[Phase 1] Trajectory learning — {args.phase1} epochs")
    lambdas_p1 = {"imp": 1.0, "pred": 2.0, "cd": 0.1, "tte": 0.0}

    opt1 = torch.optim.AdamW([
        {"params": list(model.nbm.bias_net.parameters()),       "weight_decay": 0.5},
        {"params": list(model.nbm.precision_net.parameters()),  "weight_decay": 0.5},
        {"params": list(model.nbm.weights_net.parameters()),    "weight_decay": 1.0},
        {"params": list(model.imputer.parameters()),            "weight_decay": 1e-4},
    ], lr=args.lr)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(
                 opt1, T_max=args.phase1, eta_min=1e-5)

    best_rmse = float("inf")

    for epoch in range(1, args.phase1 + 1):
        model.train()
        ep_losses = {"total": [], "imp": [], "pred": [], "cd": [], "tte": []}

        for batch in tqdm(train_loader, desc=f"P1 {epoch:3d}", leave=False, ncols=80):
            batch  = {k: v.to(device) for k, v in batch.items()}
            losses = model.compute_losses(batch, times, lambdas_p1)
            if not torch.isfinite(losses["total"]):
                continue
            opt1.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt1.step()
            for k in ep_losses:
                v = losses.get(k, torch.tensor(0.0))
                if torch.isfinite(v):
                    ep_losses[k].append(v.item())

        sched1.step()

        if epoch % args.eval_every == 0 or epoch == args.phase1:
            tl  = float(np.mean(ep_losses["total"])) if ep_losses["total"] else float("nan")
            il  = float(np.mean(ep_losses["imp"]))   if ep_losses["imp"]   else 0.0
            pl  = float(np.mean(ep_losses["pred"]))  if ep_losses["pred"]  else 0.0
            cl  = float(np.mean(ep_losses["cd"]))    if ep_losses["cd"]    else 0.0

            metrics = evaluate(model, val_loader, times, device)
            history["train_loss"].append(tl)
            history["val_loss"].append(metrics["val_loss"])
            history["rmse"].append(metrics["rmse"])
            history["c_index"].append(float("nan"))
            history["imp_loss"].append(il)
            history["pred_loss"].append(pl)
            history["cd_loss"].append(cl)
            history["tte_loss"].append(0.0)
            history["phase"].append(1)

            print(f"  P1 ep {epoch:3d} | "
                  f"train={tl:.4f} (imp={il:.3f} pred={pl:.3f} cd={cl:.3f}) | "
                  f"val={metrics['val_loss']:.4f} | RMSE={metrics['rmse']:.4f}")

            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                torch.save({
                    "epoch": epoch, "phase": 1,
                    "model_state":  model.state_dict(),
                    "metrics":      metrics,
                    "args":         vars(args),
                    "meta":         meta,
                    "scaler_mean":  scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                }, Path(args.out) / "phase1_best.pt")

    print(f"  Phase 1 done. Best val RMSE: {best_rmse:.4f}")

    # ── Phase 2: Freeze imputer+NBM, train TTE head only ─────────────────────
    print(f"\n[Phase 2] TTE head training — {args.phase2} epochs (imputer+NBM frozen)")
    for p in model.imputer.parameters():
        p.requires_grad = False
    for p in model.nbm.parameters():
        p.requires_grad = False

    lambdas_p2 = {"imp": 0.0, "pred": 0.0, "cd": 0.0, "tte": 1.0}

    opt2  = torch.optim.AdamW(model.tte_head.parameters(), lr=args.lr_tte,
                               weight_decay=1e-3)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                 opt2, T_max=args.phase2, eta_min=1e-6)

    best_c = 0.0

    for epoch in range(1, args.phase2 + 1):
        model.train()
        ep_tte = []

        for batch in tqdm(train_loader, desc=f"P2 {epoch:3d}", leave=False, ncols=80):
            batch  = {k: v.to(device) for k, v in batch.items()}
            losses = model.compute_losses(batch, times, lambdas_p2)
            if not torch.isfinite(losses["total"]):
                continue
            opt2.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.tte_head.parameters(), 1.0)
            opt2.step()
            v = losses.get("tte", torch.tensor(0.0))
            if torch.isfinite(v):
                ep_tte.append(v.item())

        sched2.step()

        if epoch % args.eval_every == 0 or epoch == args.phase2:
            ttl     = float(np.mean(ep_tte)) if ep_tte else float("nan")
            metrics = evaluate(model, val_loader, times, device)
            history["train_loss"].append(ttl)
            history["val_loss"].append(metrics["val_loss"])
            history["rmse"].append(metrics["rmse"])
            history["c_index"].append(metrics["c_index"])
            history["imp_loss"].append(0.0)
            history["pred_loss"].append(0.0)
            history["cd_loss"].append(0.0)
            history["tte_loss"].append(ttl)
            history["phase"].append(2)

            print(f"  P2 ep {epoch:3d} | tte={ttl:.4f} | "
                  f"val={metrics['val_loss']:.4f} | RMSE={metrics['rmse']:.4f} | "
                  f"C-idx={metrics['c_index']:.4f}")

            if metrics["c_index"] > best_c:
                best_c = metrics["c_index"]
                torch.save({
                    "epoch": epoch, "phase": 2,
                    "model_state":  model.state_dict(),
                    "opt_state":    opt2.state_dict(),
                    "metrics":      metrics,
                    "args":         vars(args),
                    "meta":         meta,
                    "scaler_mean":  scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                }, Path(args.out) / "best_model.pt")

    # ── Final test ────────────────────────────────────────────────────────────
    ckpt = torch.load(Path(args.out) / "best_model.pt", map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, times, device)

    print("\n" + "=" * 55)
    print("TEST SET RESULTS")
    print(f"  RMSE    : {test_metrics['rmse']:.4f}")
    print(f"  C-index : {test_metrics['c_index']:.4f}")
    print(f"  (best val C-index : {ckpt['metrics']['c_index']:.4f}  epoch {ckpt['epoch']})")
    print("=" * 55)

    (Path(args.out) / "history.json").write_text(json.dumps(history))
    _plot_history(history, args.out)
    _plot_trajectories(model, test_ds, times, device, scaler, FEAT_COLS, args.out)

    summary = {
        "test_rmse":        test_metrics["rmse"],
        "test_c_index":     test_metrics["c_index"],
        "best_val_c_index": ckpt["metrics"]["c_index"],
        "checkpoint_epoch": ckpt["epoch"],
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "d": d, "c_dim": c_dim, "T": T, "nh": args.nh, "z_dim": args.z_dim,
        "n_params": n_params,
    }
    (Path(args.out) / "test_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nOutputs saved to {args.out}/")
    return model, scaler, test_metrics


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_history(history, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    x = range(len(history["train_loss"]))

    axes[0].plot(x, history["train_loss"], label="train")
    axes[0].plot(x, history["val_loss"],   label="val")
    axes[0].set_title("Total Loss"); axes[0].legend()

    axes[1].plot(x, history["imp_loss"],  label="imputation", color="forestgreen")
    axes[1].plot(x, history["cd_loss"],   label="CD (NBM)",   color="darkorange")
    axes[1].plot(x, history["tte_loss"],  label="TTE",        color="steelblue")
    axes[1].set_title("Loss Components"); axes[1].legend()

    axes[2].plot(x, history["rmse"], color="darkorange")
    axes[2].set_title("Trajectory RMSE")

    axes[3].plot(x, history["c_index"], color="steelblue")
    axes[3].axhline(0.5, ls="--", color="gray", lw=1)
    axes[3].set_ylim(0.3, 1.0)
    axes[3].set_title("TTE C-index")

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

    n_plot_feats = min(4, mu_np.shape[-1])
    t_axis       = [int(t.item()) for t in times]

    fig, axes = plt.subplots(n_patients, n_plot_feats,
                             figsize=(4 * n_plot_feats, 3.5 * n_patients))
    if n_patients == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_patients):
        for j in range(n_plot_feats):
            ax = axes[i, j]
            fi = j  # first n_plot_feats features (UPDRS items)

            lo = np.percentile(twin_np[i, :, :, fi], 5,  axis=0)
            hi = np.percentile(twin_np[i, :, :, fi], 95, axis=0)
            lo = lo * scaler.scale_[fi] + scaler.mean_[fi]
            hi = hi * scaler.scale_[fi] + scaler.mean_[fi]
            mu_i = mu_np[i, :, fi] * scaler.scale_[fi] + scaler.mean_[fi]

            ax.fill_between(t_axis, lo, hi, alpha=0.25, color="royalblue",
                            label="5-95%")
            ax.plot(t_axis, mu_i, "royalblue", lw=2, label="mean")

            obs_mask = mask_np[i, :, fi].astype(bool)
            if obs_mask.any():
                obs_i = obs_np[i, obs_mask, fi] * scaler.scale_[fi] + scaler.mean_[fi]
                ax.scatter(np.array(t_axis)[obs_mask], obs_i,
                           color="black", zorder=5, s=30, label="Observed")

            ax.set_title(f"P{i+1} - {feat_names[fi]}", fontsize=8)
            ax.set_xlabel("Months")
            if j == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(Path(out_dir) / "sample_trajectories.png", dpi=150)
    plt.close()
    print("  Saved sample_trajectories.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DTG v2 on PPMI (59 UPDRS items)")
    p.add_argument("--out",        default="outputs_v2")
    p.add_argument("--phase1",     type=int,   default=120,  help="epochs for trajectory phase")
    p.add_argument("--phase2",     type=int,   default=80,   help="epochs for TTE-only phase")
    p.add_argument("--batch",      type=int,   default=32)
    p.add_argument("--lr",         type=float, default=3e-4, help="lr for phase 1")
    p.add_argument("--lr_tte",     type=float, default=1e-3, help="lr for TTE phase 2")
    p.add_argument("--nh",         type=int,   default=64)
    p.add_argument("--z_dim",      type=int,   default=64)
    p.add_argument("--eval_every", type=int,   default=10)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)