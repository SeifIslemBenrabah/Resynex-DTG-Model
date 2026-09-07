"""
DTG_LIVER — Train Digital Twin Generator on PBC2 liver disease data.

Dataset is FREE and auto-downloaded from public R datasets repository.
No registration required.

Usage:
    cd C:\\Users\\admin\\Desktop\\DTG_LIVER
    python train_liver.py
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

sys.path.insert(0, str(Path(__file__).parent.parent / "DTG_PPMI_v2"))
from model_v2 import DTG_v2

from preprocess_liver import load_pbc2, FEAT_COLS, STATIC_COLS, TIMEPOINTS, d, c_dim, T, S


class LiverDataset(Dataset):
    def __init__(self, X, M, C, ET, EI):
        self.X  = torch.tensor(np.nan_to_num(X, nan=0.0), dtype=torch.float32)
        self.M  = torch.tensor(M,  dtype=torch.float32)
        self.C  = torch.tensor(C,  dtype=torch.float32)
        self.ET = torch.tensor(ET, dtype=torch.float32)
        self.EI = torch.tensor(EI, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return {"x0_obs": self.X[idx, 0, :], "mask0": self.M[idx, 0, :],
                "c": self.C[idx], "X_traj": self.X[idx], "M_traj": self.M[idx],
                "event_time": self.ET[idx], "event_ind": self.EI[idx]}


@torch.no_grad()
def evaluate(model, loader, times, device):
    model.eval()
    all_losses, pred_risks, true_times, true_inds = [], [], [], []
    sq_err, n_obs = 0.0, 0
    for batch in loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        losses = model.compute_losses(batch, times)
        if torch.isfinite(losses["total"]): all_losses.append(losses["total"].item())
        mu, _, tte_logits = model(batch["x0_obs"], batch["mask0"], batch["c"], times)
        obs = batch["M_traj"].bool()
        sq_err += ((mu - batch["X_traj"])**2 * obs.float()).sum().item()
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
        c_idx = concordance_index(true_times, pred_risks, true_inds) if true_inds.sum() > 0 else float("nan")
    except Exception:
        c_idx = float("nan")
    return {"val_loss": float(np.mean(all_losses)) if all_losses else float("nan"),
            "rmse":     float(np.sqrt(sq_err / max(n_obs, 1))),
            "c_index":  c_idx}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    X, M, C, ET, EI, ids, scaler, meta = load_pbc2()

    dataset = LiverDataset(X, M, C, ET, EI)
    N       = len(dataset)
    n_val   = max(1, int(N * 0.15))
    n_test  = max(1, int(N * 0.15))
    n_train = N - n_val - n_test
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, drop_last=False)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, drop_last=False)
    print(f"Split: train={n_train}  val={n_val}  test={n_test}")

    model  = DTG_v2(d=d, c_dim=c_dim, T=T, nh=args.nh, z_dim=args.z_dim, S=S).to(device)
    times  = torch.tensor(TIMEPOINTS, dtype=torch.float32, device=device)
    os.makedirs(args.out, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nDTG_LIVER (PBC2): d={d} c={c_dim} T={T} nh={args.nh} z={args.z_dim} params={n_params:,}")

    history = {"train_loss":[], "val_loss":[], "rmse":[], "c_index":[],
               "pred_loss":[], "cd_loss":[], "tte_loss":[], "phase":[]}

    # ── Phase 1: Trajectory ────────────────────────────────────────────────────
    print(f"\n[Phase 1] Trajectory — {args.phase1} epochs")
    lambdas_p1 = {"imp": 1.0, "pred": 2.0, "cd": 0.1, "tte": 0.0}
    opt1   = torch.optim.AdamW([
        {"params": list(model.nbm.bias_net.parameters()),      "weight_decay": 0.5},
        {"params": list(model.nbm.precision_net.parameters()), "weight_decay": 0.5},
        {"params": list(model.nbm.weights_net.parameters()),   "weight_decay": 1.0},
        {"params": list(model.imputer.parameters()),           "weight_decay": 1e-4},
    ], lr=args.lr)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=args.phase1, eta_min=1e-5)
    best_rmse = float("inf")

    for epoch in range(1, args.phase1 + 1):
        model.train()
        ep = {"total":[], "pred":[], "cd":[], "imp":[]}
        for batch in tqdm(train_ld, desc=f"P1 {epoch:3d}", leave=False, ncols=80):
            batch  = {k: v.to(device) for k, v in batch.items()}
            losses = model.compute_losses(batch, times, lambdas_p1)
            if not torch.isfinite(losses["total"]): continue
            opt1.zero_grad(); losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt1.step()
            for k in ep:
                v = losses.get(k, torch.tensor(0.0))
                if torch.isfinite(v): ep[k].append(v.item())
        sched1.step()
        if epoch % args.eval_every == 0 or epoch == args.phase1:
            tl = float(np.mean(ep["total"])) if ep["total"] else float("nan")
            pl = float(np.mean(ep["pred"]))  if ep["pred"]  else 0.0
            cl = float(np.mean(ep["cd"]))    if ep["cd"]    else 0.0
            m  = evaluate(model, val_ld, times, device)
            history["train_loss"].append(tl); history["val_loss"].append(m["val_loss"])
            history["rmse"].append(m["rmse"]); history["c_index"].append(float("nan"))
            history["pred_loss"].append(pl); history["cd_loss"].append(cl)
            history["tte_loss"].append(0.0); history["phase"].append(1)
            print(f"  P1 {epoch:3d} | pred={pl:.3f} | RMSE={m['rmse']:.4f}")
            if m["rmse"] < best_rmse:
                best_rmse = m["rmse"]
                torch.save({"epoch":epoch,"phase":1,"model_state":model.state_dict(),
                            "metrics":m,"args":vars(args),"meta":meta,
                            "scaler_mean":scaler.mean_.tolist(),
                            "scaler_scale":scaler.scale_.tolist()},
                           Path(args.out)/"phase1_best.pt")

    # ── Phase 2: TTE head ──────────────────────────────────────────────────────
    print(f"\n[Phase 2] TTE head — {args.phase2} epochs")
    for p in model.imputer.parameters(): p.requires_grad = False
    for p in model.nbm.parameters():     p.requires_grad = False
    opt2   = torch.optim.AdamW(model.tte_head.parameters(), lr=args.lr_tte, weight_decay=1e-3)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.phase2, eta_min=1e-6)
    lambdas_p2 = {"imp":0.0, "pred":0.0, "cd":0.0, "tte":1.0}
    best_c  = 0.0

    for epoch in range(1, args.phase2 + 1):
        model.train()
        ep_tte = []
        for batch in tqdm(train_ld, desc=f"P2 {epoch:3d}", leave=False, ncols=80):
            batch  = {k: v.to(device) for k, v in batch.items()}
            losses = model.compute_losses(batch, times, lambdas_p2)
            if not torch.isfinite(losses["total"]): continue
            opt2.zero_grad(); losses["total"].backward()
            nn.utils.clip_grad_norm_(model.tte_head.parameters(), 1.0); opt2.step()
            v = losses.get("tte", torch.tensor(0.0))
            if torch.isfinite(v): ep_tte.append(v.item())
        sched2.step()
        if epoch % args.eval_every == 0 or epoch == args.phase2:
            ttl = float(np.mean(ep_tte)) if ep_tte else float("nan")
            m   = evaluate(model, val_ld, times, device)
            history["train_loss"].append(ttl); history["val_loss"].append(m["val_loss"])
            history["rmse"].append(m["rmse"]); history["c_index"].append(m["c_index"])
            history["pred_loss"].append(0.0); history["cd_loss"].append(0.0)
            history["tte_loss"].append(ttl); history["phase"].append(2)
            print(f"  P2 {epoch:3d} | tte={ttl:.4f} | RMSE={m['rmse']:.4f} | C-idx={m['c_index']:.4f}")
            if m["c_index"] > best_c:
                best_c = m["c_index"]
                torch.save({"epoch":epoch,"phase":2,"model_state":model.state_dict(),
                            "metrics":m,"args":vars(args),"meta":meta,
                            "scaler_mean":scaler.mean_.tolist(),
                            "scaler_scale":scaler.scale_.tolist()},
                           Path(args.out)/"best_model.pt")

    ckpt   = torch.load(Path(args.out)/"best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(model, test_ld, times, device)
    print(f"\n{'='*50}\nTEST SET RESULTS (PBC2 Liver)\n  RMSE: {test_m['rmse']:.4f}  C-index: {test_m['c_index']:.4f}\n{'='*50}")

    (Path(args.out)/"history.json").write_text(json.dumps(history))
    summary = {"test_rmse":test_m["rmse"],"test_c_index":test_m["c_index"],
               "best_val_c_index":ckpt["metrics"]["c_index"],"checkpoint_epoch":ckpt["epoch"],
               "n_train":n_train,"n_val":n_val,"n_test":n_test,
               "d":d,"c_dim":c_dim,"T":T,"n_params":n_params,"dataset":"PBC2 (pbcseq)"}
    (Path(args.out)/"test_results.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    x = range(len(history["train_loss"]))
    p2 = next((i for i, p in enumerate(history["phase"]) if p == 2), len(x))
    axes[0].plot(x, history["train_loss"], label="train"); axes[0].plot(x, history["val_loss"], label="val")
    if p2 < len(x): axes[0].axvline(p2 - 0.5, ls="--", color="gray")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(x, history["rmse"], color="darkorange"); axes[1].set_title("Biomarker RMSE")
    axes[2].plot(x, history["c_index"], color="steelblue")
    axes[2].axhline(0.5, ls="--", color="gray"); axes[2].set_ylim(0.3, 1.0)
    axes[2].set_title("C-index (PBC mortality)")
    plt.tight_layout(); plt.savefig(Path(args.out)/"training_curves.png", dpi=150); plt.close()
    print(f"Outputs saved to {args.out}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out",        default="outputs_liver")
    p.add_argument("--phase1",     type=int,   default=120)
    p.add_argument("--phase2",     type=int,   default=80)
    p.add_argument("--batch",      type=int,   default=16)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--lr_tte",     type=float, default=1e-3)
    p.add_argument("--nh",         type=int,   default=32)
    p.add_argument("--z_dim",      type=int,   default=32)
    p.add_argument("--eval_every", type=int,   default=10)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    train(args)