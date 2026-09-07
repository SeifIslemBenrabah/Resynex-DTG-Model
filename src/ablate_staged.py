"""
Controlled ablation: staged pre-training vs curriculum joint training.

The published comparison (outputs_v2 vs outputs_v4) confounds three variables
at once: the training protocol, the latent width (64 vs 128), and the event
definition (train_v2.py uses the preprocess default NP3 >= 33, train_v3.py
recomputes at NP3 >= 25). A C-index computed under two different event
definitions is not measuring the same task, so that comparison cannot support
the claim it was meant to support.

This script runs the STAGED protocol under exactly the configuration of the
reported curriculum run: same architecture (DTG_v3), same z_dim, same event
threshold, same seed, same split. The only thing that differs from
outputs_v4 is the schedule:

    curriculum (v4) : all parameters trained jointly throughout;
                      lambda_tte ramps 0 -> 1 from epoch W to E.
    staged  (here)  : phase 1 trains imputer + NBM with lambda_tte = 0;
                      then imputer and NBM are FROZEN and phase 2 trains
                      the TTE head alone.

Usage:
    python ablate_staged.py --phase1 60 --phase2 240 --z_dim 128 \
        --event_thresh 25 --out outputs_staged_z128
"""

import argparse, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from preprocess_v2 import load_ppmi_v2, TIMEPOINTS, d, c_dim, T, S
from train_v3 import PPMIDataset, recompute_events, evaluate
from model_v3 import DTG_v3


def build(seed, event_thresh, batch):
    X, M, C, _, _, _pat, scaler, _meta = load_ppmi_v2()
    ET, EI = recompute_events(X, M, scaler, event_thresh)
    ds = PPMIDataset(X, M, C, ET, EI)

    N = len(ds)
    n_val = max(1, int(N * 0.15))
    n_test = max(1, int(N * 0.15))
    n_train = N - n_val - n_test

    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va, te = random_split(ds, [n_train, n_val, n_test])
    print(f"Split: train={n_train}  val={n_val}  test={n_test}")
    return (DataLoader(tr, batch_size=batch, shuffle=True, drop_last=True),
            DataLoader(va, batch_size=batch, shuffle=False),
            DataLoader(te, batch_size=batch, shuffle=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs_staged_z128")
    p.add_argument("--phase1", type=int, default=60)
    p.add_argument("--phase2", type=int, default=240)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_tte", type=float, default=1e-3)
    p.add_argument("--nh", type=int, default=64)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--event_thresh", type=float, default=25.0)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    device = "cpu"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    train_ld, val_ld, test_ld = build(a.seed, a.event_thresh, a.batch)
    model = DTG_v3(d=d, c_dim=c_dim, T=T, nh=a.nh, z_dim=a.z_dim, S=S).to(device)
    times = torch.tensor(TIMEPOINTS, dtype=torch.float32, device=device)

    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"\nSTAGED ablation  --  z={a.z_dim}  nh={a.nh}  thresh={a.event_thresh}  "
          f"params={n_params:,}")
    print(f"Phase 1: {a.phase1} epochs (imputer+NBM)   "
          f"Phase 2: {a.phase2} epochs (TTE head only, rest frozen)\n")

    history = {k: [] for k in ["epoch", "phase", "train_loss", "rmse", "c_index"]}

    # ── Phase 1: trajectory only (lambda_tte = 0) ────────────────────────────
    opt1 = torch.optim.AdamW([
        {"params": list(model.nbm.bias_net.parameters()),      "weight_decay": 0.5},
        {"params": list(model.nbm.precision_net.parameters()), "weight_decay": 0.5},
        {"params": list(model.nbm.weights_net.parameters()),   "weight_decay": 1.0},
        {"params": list(model.imputer.parameters()),           "weight_decay": 1e-4},
    ], lr=a.lr)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=a.phase1, eta_min=1e-6)
    lam1 = {"imp": 1.0, "pred": 2.0, "cd": 0.1, "tte": 0.0}

    for ep in range(1, a.phase1 + 1):
        model.train(); tot = []
        for batch in tqdm(train_ld, desc=f"P1 {ep:3d}", leave=False, ncols=70):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model.compute_losses(batch, times, lam1)["total"]
            opt1.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt1.step(); tot.append(loss.item())
        sched1.step()
        if ep % a.eval_every == 0 or ep == a.phase1:
            m = evaluate(model, val_ld, times, device)
            history["epoch"].append(ep); history["phase"].append(1)
            history["train_loss"].append(float(np.mean(tot)) if tot else float("nan"))
            history["rmse"].append(m["rmse"]); history["c_index"].append(m["c_index"])
            print(f"  P1 ep {ep:3d} | loss={np.mean(tot):.3f} RMSE={m['rmse']:.4f}")

    torch.save({"epoch": a.phase1, "phase": 1, "model_state": model.state_dict()},
               os.path.join(a.out, "phase1.pt"))

    # ── Freeze everything except the TTE head ────────────────────────────────
    for q in model.imputer.parameters(): q.requires_grad = False
    for q in model.nbm.parameters():     q.requires_grad = False
    frozen = sum(q.numel() for q in model.parameters() if not q.requires_grad)
    trainable = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"\nFrozen {frozen:,} parameters; {trainable:,} remain trainable (TTE head).\n")

    # ── Phase 2: TTE head alone ──────────────────────────────────────────────
    opt2 = torch.optim.AdamW(model.tte_head.parameters(), lr=a.lr_tte, weight_decay=1e-3)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=a.phase2, eta_min=1e-6)
    lam2 = {"imp": 0.0, "pred": 0.0, "cd": 0.0, "tte": 1.0}

    best_c, best_ep = 0.0, -1
    for ep in range(1, a.phase2 + 1):
        model.train(); tot = []
        for batch in tqdm(train_ld, desc=f"P2 {ep:3d}", leave=False, ncols=70):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model.compute_losses(batch, times, lam2)["total"]
            opt2.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.tte_head.parameters(), 5.0)
                opt2.step(); tot.append(loss.item())
        sched2.step()
        if ep % a.eval_every == 0 or ep == a.phase2:
            m = evaluate(model, val_ld, times, device)
            history["epoch"].append(a.phase1 + ep); history["phase"].append(2)
            history["train_loss"].append(float(np.mean(tot)) if tot else float("nan"))
            history["rmse"].append(m["rmse"]); history["c_index"].append(m["c_index"])
            print(f"  P2 ep {ep:3d} | loss={np.mean(tot):.3f} C={m['c_index']:.4f}")
            if m["c_index"] == m["c_index"] and m["c_index"] > best_c:
                best_c, best_ep = m["c_index"], a.phase1 + ep
                torch.save({"epoch": best_ep, "phase": 2, "metrics": m,
                            "model_state": model.state_dict()},
                           os.path.join(a.out, "best_model.pt"))
                print(f"    ** new best val C-index {best_c:.4f}")

    # ── Test ─────────────────────────────────────────────────────────────────
    ck = torch.load(os.path.join(a.out, "best_model.pt"), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model_state"])
    tm = evaluate(model, test_ld, times, device)

    res = {"protocol": "staged (phase1 trajectory, freeze, phase2 TTE head)",
           "test_rmse": tm["rmse"], "test_c_index": tm["c_index"],
           "best_val_c_index": best_c, "checkpoint_epoch": best_ep,
           "event_thresh": a.event_thresh, "z_dim": a.z_dim, "nh": a.nh,
           "phase1": a.phase1, "phase2": a.phase2, "seed": a.seed,
           "n_params": n_params}
    json.dump(res, open(os.path.join(a.out, "test_results.json"), "w"), indent=2)
    json.dump(history, open(os.path.join(a.out, "history.json"), "w"), indent=2)

    print(f"\n{'='*60}")
    print(f"  STAGED  z={a.z_dim}  thresh={a.event_thresh}")
    print(f"  test C-index : {tm['c_index']:.4f}   (best val {best_c:.4f} @ ep {best_ep})")
    print(f"  test RMSE    : {tm['rmse']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
