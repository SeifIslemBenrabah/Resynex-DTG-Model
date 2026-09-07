"""
DTG training on SUPPORT / METABRIC cancer dataset.

Usage:
    python train_cancer.py                    # SUPPORT (default)
    python train_cancer.py --metabric         # METABRIC breast cancer
    python train_cancer.py --epochs 200 --batch 64
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

from preprocess_cancer import (
    load_cancer, FEAT_COLS, STATIC_COLS, TIMEPOINTS,
    d, c_dim, T, S,
    METABRIC_FEAT_NAMES, METABRIC_STATIC_NAMES
)
from model import DTG


# ── Dataset ───────────────────────────────────────────────────────────────────

class CancerDataset(Dataset):
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

    c_idx = concordance_index(true_times, pred_risks, true_inds) \
            if true_inds.sum() > 0 else float("nan")

    return {
        "val_loss": float(np.mean(all_losses)),
        "rmse":     float(np.sqrt(sq_err / max(n_obs, 1))),
        "c_index":  c_idx,
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_m = getattr(args, "metabric", False)
    print(f"\nLoading {'METABRIC' if use_m else 'SUPPORT'} data...")
    X, M, C, ET, EI, ids, scaler, meta = load_cancer(use_metabric=use_m)

    feat_names   = meta["feat_cols"]
    static_names = meta["static_cols"]
    d_run  = X.shape[2]
    c_run  = C.shape[1]

    dataset  = CancerDataset(X, M, C, ET, EI)
    N        = len(dataset)
    n_val    = max(1, int(N * 0.15))
    n_test   = max(1, int(N * 0.15))
    n_train  = N - n_val - n_test

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, drop_last=False)

    print(f"Split -- train:{n_train}  val:{n_val}  test:{n_test}")

    model = DTG(d=d_run, c_dim=c_run, T=T, K=args.hidden, S=S).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs, eta_min=1e-5)

    times   = torch.tensor(TIMEPOINTS, dtype=torch.float32, device=device)
    lambdas = {"imp": 1.0, "pred": 1.0, "rbm": 0.01, "tte": 0.5}

    os.makedirs(args.out, exist_ok=True)
    best_c  = 0.0
    history = {"train_loss": [], "val_loss": [], "rmse": [], "c_index": []}

    print(f"\nTraining DTG for {args.epochs} epochs  (d={d_run}, c={c_run}, T={T}, K={args.hidden})")

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:3d}", leave=False, ncols=80):
            batch  = {k: v.to(device) for k, v in batch.items()}
            losses = model.compute_losses(batch, times, lambdas)

            if not torch.isfinite(losses["total"]):
                continue
            opt.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_losses.append(losses["total"].item())

        sched.step()
        train_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, val_loader, times, device)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(metrics["val_loss"])
            history["rmse"].append(metrics["rmse"])
            history["c_index"].append(metrics["c_index"])

            print(f"  Epoch {epoch:3d} | "
                  f"train={train_loss:.4f} | "
                  f"val={metrics['val_loss']:.4f} | "
                  f"RMSE={metrics['rmse']:.4f} | "
                  f"C-idx={metrics['c_index']:.4f}")

            if metrics["c_index"] > best_c:
                best_c = metrics["c_index"]
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "opt_state":   opt.state_dict(),
                    "metrics":     metrics,
                    "args":        vars(args),
                    "meta":        meta,
                    "feat_names":  feat_names,
                    "static_names": static_names,
                }, Path(args.out) / "best_model.pt")

    # ── Final test evaluation ─────────────────────────────────────────────────
    ckpt = torch.load(Path(args.out) / "best_model.pt", map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, times, device)

    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print(f"  RMSE    : {test_metrics['rmse']:.4f}")
    print(f"  C-index : {test_metrics['c_index']:.4f}")
    print("=" * 50)

    (Path(args.out) / "history.json").write_text(json.dumps(history))
    _plot_history(history, args.out)
    _plot_trajectories(model, test_ds, times, device, scaler, feat_names, args.out)

    summary = {
        "test_rmse":    test_metrics["rmse"],
        "test_c_index": test_metrics["c_index"],
        "best_val_c_index": ckpt["metrics"]["c_index"],
        "checkpoint_epoch": ckpt["epoch"],
        "dataset": meta["dataset"],
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
    }
    (Path(args.out) / "test_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nOutputs saved to {args.out}/")
    return model, scaler, test_metrics


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_history(history, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    x = range(len(history["train_loss"]))
    axes[0].plot(x, history["train_loss"], label="train")
    axes[0].plot(x, history["val_loss"],   label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(x, history["rmse"], color="darkorange")
    axes[1].set_title("Trajectory RMSE (T=0 only)")
    axes[2].plot(x, history["c_index"], color="steelblue")
    axes[2].axhline(0.5, ls="--", color="gray", lw=1)
    axes[2].set_ylim(0.4, 1.0); axes[2].set_title("TTE C-index")
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
        mu, _, _  = model(batch["x0_obs"], batch["mask0"], batch["c"], times)
        twin      = model.generate_twin(batch["x0_obs"], batch["mask0"],
                                        batch["c"], times, n_samples=50)

    mu_np   = mu.cpu().numpy()
    twin_np = twin.cpu().numpy()       # (B, 50, T, d)
    obs_np  = batch["X_traj"].cpu().numpy()
    mask_np = batch["M_traj"].cpu().numpy()

    d_run = mu_np.shape[-1]
    n_plot_feats = min(3, d_run)
    feat_ids     = list(range(n_plot_feats))
    t_axis       = TIMEPOINTS

    fig, axes = plt.subplots(n_patients, n_plot_feats,
                             figsize=(4 * n_plot_feats, 3.5 * n_patients))
    if n_patients == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_patients):
        for j, fi in enumerate(feat_ids):
            ax = axes[i, j]
            lo = np.percentile(twin_np[i, :, :, fi], 5,  axis=0)
            hi = np.percentile(twin_np[i, :, :, fi], 95, axis=0)
            lo = lo * scaler.scale_[fi] + scaler.mean_[fi]
            hi = hi * scaler.scale_[fi] + scaler.mean_[fi]
            mu_i = mu_np[i, :, fi] * scaler.scale_[fi] + scaler.mean_[fi]

            ax.fill_between(t_axis, lo, hi, alpha=0.25, color="royalblue",
                            label="DTG 5-95%")
            ax.plot(t_axis, mu_i, "royalblue", lw=2, label="DTG mean")

            obs_mask = mask_np[i, :, fi].astype(bool)
            if obs_mask.any():
                obs_i = obs_np[i, obs_mask, fi] * scaler.scale_[fi] + scaler.mean_[fi]
                ax.scatter(np.array(t_axis)[obs_mask], obs_i,
                           color="black", zorder=5, s=30, label="Observed")

            ax.set_title(f"Patient {i+1} — {feat_names[fi]}", fontsize=9)
            ax.set_xlabel("Days")
            if j == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(Path(out_dir) / "sample_trajectories.png", dpi=150)
    plt.close()
    print("  Saved sample_trajectories.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DTG on cancer/clinical dataset")
    p.add_argument("--metabric",    action="store_true",
                   help="Use METABRIC breast-cancer dataset (default: SUPPORT)")
    p.add_argument("--out",         default="outputs")
    p.add_argument("--epochs",      type=int,   default=150)
    p.add_argument("--batch",       type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--hidden",      type=int,   default=16)
    p.add_argument("--eval_every",  type=int,   default=10)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)