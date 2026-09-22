"""
Train the hybrid architecture and compare it against both parents.

Checkpoint selection
--------------------
The earlier experiment programme selected on validation R2 alone, which is why
the variants that improved R2 all reported poor concordance: a checkpoint chosen
to maximise one metric has no reason to be good at the other. This run tracks
both and keeps three checkpoints -- best R2, best concordance, and best on a
combined score -- so the trade-off can be read rather than assumed.

Usage:
    python train_hybrid.py --epochs 300 --warmup 60
"""

import argparse, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from preprocess_v2 import TIMEPOINTS, c_dim, T, S
from aggregate_panel import build_aggregated, AGG_COLS, d_agg
from train_v3 import PPMIDataset, recompute_events, evaluate
from model_hybrid import HybridDTG


@torch.no_grad()
def collect(model, loader, times):
    P, Y, M = [], [], []
    model.eval()
    for b in loader:
        mu, _, _ = model(b["x0_obs"], b["mask0"], b["c"], times)
        P.append(mu.numpy()); Y.append(b["X_traj"].numpy()); M.append(b["M_traj"].numpy())
    return np.concatenate(P), np.concatenate(Y), np.concatenate(M)


def r2_stats(P, Y, M):
    sse = (((P - Y) ** 2) * M).sum(axis=(0, 1))
    n   = M.sum(axis=(0, 1))
    mean = (Y * M).sum(axis=(0, 1)) / np.maximum(n, 1e-8)
    sst = (((Y - mean) ** 2) * M).sum(axis=(0, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        per = 1.0 - sse / np.maximum(sst, 1e-8)
    return per, float(1.0 - sse.sum() / max(sst.sum(), 1e-8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs_hybrid")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--nh", type=int, default=64)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--flow_hidden", type=int, default=128)
    p.add_argument("--flow_blocks", type=int, default=3)
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--lam_var", type=float, default=0.0)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--event_thresh", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    Xa, Ma, C, sc, X69, M69, sc69, _ = build_aggregated(verbose=False)
    ET, EI = recompute_events(X69, M69, sc69, a.event_thresh)
    ds = PPMIDataset(Xa, Ma, C, ET, EI)
    N = len(ds)
    n_val = max(1, int(N * 0.15)); n_test = max(1, int(N * 0.15))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tr, va, te = random_split(ds, [N - n_val - n_test, n_val, n_test])
    train_ld = DataLoader(tr, batch_size=a.batch, shuffle=True, drop_last=True)
    val_ld   = DataLoader(va, batch_size=a.batch, shuffle=False)
    test_ld  = DataLoader(te, batch_size=a.batch, shuffle=False)
    print(f"Split: train={len(tr)} val={len(va)} test={len(te)}")

    torch.manual_seed(a.seed)
    model = HybridDTG(d=d_agg, c_dim=c_dim, T=T, nh=a.nh, z_dim=a.z_dim, S=S,
                      flow_hidden=a.flow_hidden, flow_blocks=a.flow_blocks,
                      huber_delta=a.huber_delta)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"HybridDTG  d={d_agg}  params={n_par:,}")

    opt = torch.optim.AdamW([
        {"params": list(model.nbm.parameters()),      "weight_decay": 0.5},
        {"params": list(model.flow.parameters()),     "weight_decay": 1e-4},
        {"params": list(model.imputer.parameters()),  "weight_decay": 1e-4},
        {"params": list(model.tte_head.parameters()), "weight_decay": 1e-3},
    ], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-6)
    times = torch.tensor(TIMEPOINTS, dtype=torch.float32)

    best = {"r2": (-9.9, None), "c": (-9.9, None), "combined": (-9.9, None)}
    hist = []
    for ep in range(1, a.epochs + 1):
        lam = 0.0 if ep <= a.warmup else min(
            1.0, (ep - a.warmup) / max(1, a.epochs - a.warmup))
        lambdas = {"imp": 1.0, "pred": 2.0, "cd": 0.1, "tte": lam,
                   "var": a.lam_var}
        model.train()
        for b in tqdm(train_ld, desc=f"ep{ep:3d}", leave=False, ncols=60):
            loss = model.compute_losses(b, times, lambdas)["total"]
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        sched.step()

        if ep % a.eval_every == 0 or ep == a.epochs:
            P, Y, M = collect(model, val_ld, times)
            _, r2v = r2_stats(P, Y, M)
            m = evaluate(model, val_ld, times, "cpu")
            cv = m["c_index"] if m["c_index"] == m["c_index"] else 0.0
            # Combined score. Concordance starts at 0.5 for a coin flip, so its
            # useful range is [0.5, 1]; rescaling puts both on [0, 1] before
            # they are added, otherwise concordance dominates by construction.
            comb = r2v + 2.0 * (cv - 0.5)
            state = {k: v.clone() for k, v in model.state_dict().items()}
            if r2v  > best["r2"][0]:       best["r2"] = (r2v, state)
            if cv   > best["c"][0]:        best["c"] = (cv, state)
            if comb > best["combined"][0]: best["combined"] = (comb, state)
            hist.append({"epoch": ep, "r2": r2v, "c_index": cv, "combined": comb})
            print(f"  ep{ep:3d}  val R2={r2v:.4f}  C={cv:.4f}  comb={comb:.4f}",
                  flush=True)

    results = {}
    line = "=" * 66
    print(f"\n{line}\n  TEST RESULTS BY CHECKPOINT SELECTION RULE\n{line}")
    print(f"  {'selected on':14s} {'test R2':>9s} {'C-index':>9s} {'RMSE':>9s}")
    for key in ("r2", "c", "combined"):
        score, state = best[key]
        if state is None:
            continue
        model.load_state_dict(state)
        P, Y, M = collect(model, test_ld, times)
        per, r2t = r2_stats(P, Y, M)
        mt = evaluate(model, test_ld, times, "cpu")
        results[key] = {"test_r2": r2t, "test_c_index": mt["c_index"],
                        "test_rmse": mt["rmse"],
                        "per_feature": {AGG_COLS[j]: float(per[j])
                                        for j in range(d_agg)}}
        print(f"  {key:14s} {r2t:9.4f} {mt['c_index']:9.4f} {mt['rmse']:9.4f}")
        torch.save({"model_state": state}, f"{a.out}/model_{key}.pt")
    print(line)

    sel = results.get("combined", {})
    if sel:
        print(f"\n  Per-feature R2 of the combined-selection checkpoint:")
        for name, v in sel["per_feature"].items():
            flag = "  <-- above 0.70" if v > 0.70 else ""
            print(f"    {name:16s} {v:7.3f}{flag}")

    json.dump({"results": results, "history": hist, "config": vars(a),
               "n_params": n_par}, open(f"{a.out}/results.json", "w"), indent=2)
    print(f"\nwritten to {a.out}/results.json")


if __name__ == "__main__":
    main()
