"""
DTG_LIVER — Primary Biliary Cirrhosis (PBC) longitudinal preprocessing.

Dataset: Mayo Clinic PBC2 (pbcseq) — publicly available, NO registration needed.
Source:  https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/pbcseq.csv
         (R survival package 'pbcseq' dataset)

312 PD patients, up to 16 visits each, follow-up 0-14 years.
Reference: Fleming & Harrington (1991) Counting Processes and Survival Analysis.

Longitudinal features d=11:
  bili       — serum bilirubin (mg/dL)            → log-transform
  chol       — serum cholesterol (mg/dL)
  albumin    — serum albumin (g/dL)
  alk_phos   — alkaline phosphatase (U/L)         → log-transform
  ast        — aspartate aminotransferase (U/mL)  → log-transform
  platelet   — platelet count (per cubic mm/1000)
  protime    — prothrombin time (seconds)
  ascites    — ascites presence (0/1)
  hepato     — hepatomegaly (0/1)
  spiders    — blood vessel malformations (0/1)
  edema      — edema (0=none, 0.5=untreated, 1=treated)

Static features c=4: age, sex_male, trt (D-penicillamine=1/placebo=2), stage_bl

TTE: death (status==2); transplant (status==1) treated as censored
Timepoints: 0, 6, 12, 18, 24, 36 months (T=6)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# PBC2 is public and is downloaded on first run.
DATA_DIR   = os.environ.get("PBC2_DATA_DIR", "data")
DATA_FILE  = "pbcseq.csv"

# ── Feature definitions ────────────────────────────────────────────────────────

FEAT_COLS = [
    "bili",      "chol",    "albumin", "alk_phos",
    "ast",       "platelet","protime",
    "ascites",   "hepato",  "spiders", "edema",
]
d = len(FEAT_COLS)   # 11

STATIC_COLS = ["age", "sex_male", "trt_drug", "stage_bl"]
c_dim = len(STATIC_COLS)   # 4

TIMEPOINTS = [0, 6, 12, 18, 24, 36]   # months
T          = len(TIMEPOINTS)
S          = 60
BIN_HALF   = 3.0   # months tolerance for visit assignment

LOG_COLS = ["bili", "alk_phos", "ast"]


def _assign_tp(months):
    diffs = [abs(months - t) for t in TIMEPOINTS]
    idx   = int(np.argmin(diffs))
    return idx if diffs[idx] <= BIN_HALF else None


def load_pbc2(data_dir=DATA_DIR):
    """
    Load pbcseq.csv and build (X, M, C, ET, EI) tensors.

    Returns
    -------
    X, M, C, ET, EI, ids, scaler, meta
    """
    p    = Path(data_dir)
    df   = pd.read_csv(p / DATA_FILE)
    df   = df.rename(columns={"alk.phos": "alk_phos"})
    print(f"Loaded pbcseq: {len(df)} rows")

    # Days to months
    df["months"] = df["day"] / 30.44

    # Patient-level static features (from baseline row)
    bl = df.groupby("id").first().reset_index()
    bl["sex_male"] = (bl["sex"].str.strip().str.lower() == "m").astype(float)
    bl["trt_drug"] = (bl["trt"] == 1).astype(float)  # 1=D-penicillamine, 2=placebo
    bl["stage_bl"] = pd.to_numeric(bl["stage"], errors="coerce").fillna(0) / 4.0

    patients = sorted(df["id"].unique())
    N        = len(patients)
    pat2idx  = {p: i for i, p in enumerate(patients)}
    print(f"  Patients: {N}")

    # ── Build longitudinal tensors ─────────────────────────────────────────────
    X = np.full((N, T, d), np.nan, dtype=np.float32)
    M = np.zeros((N, T, d), dtype=np.float32)

    for _, row in df.iterrows():
        pid = row["id"]
        if pid not in pat2idx:
            continue
        ti = _assign_tp(row["months"])
        if ti is None:
            continue
        i = pat2idx[pid]
        for j, col in enumerate(FEAT_COLS):
            v = row.get(col, np.nan)
            if pd.notna(v):
                try:
                    X[i, ti, j] = float(v)
                    M[i, ti, j] = 1.0
                except (ValueError, TypeError):
                    pass

    # ── Static features ───────────────────────────────────────────────────────
    bl_idx = bl.set_index("id").reindex(patients)
    C      = np.zeros((N, c_dim), dtype=np.float32)
    for j, col in enumerate(STATIC_COLS):
        if col in bl_idx.columns:
            vals = pd.to_numeric(bl_idx[col], errors="coerce").fillna(0).values
            C[:, j] = vals.astype(np.float32)

    # ── TTE: death (status==2); transplant is censored ─────────────────────────
    # futime is the same for all rows of a patient (time to last event/censor)
    tte_df  = df.groupby("id")["futime"].max().reindex(patients)
    stat_df = df.groupby("id")["status"].max().reindex(patients)

    MAX_DAYS = 365.25 * 3   # 3-year horizon (months=36)
    ET_raw   = np.clip(tte_df.values / 30.44, 0, 36).astype(np.float32)
    EI       = (stat_df.values == 2).astype(np.float32)   # death only

    # Normalise event times to [0,1]
    ET       = (ET_raw / 36.0).astype(np.float32)
    events   = int(EI.sum())
    print(f"  Deaths (events): {events} / {N} ({events/N*100:.1f}%)")

    # ── Log-transform skewed biomarkers ───────────────────────────────────────
    for col in LOG_COLS:
        j = FEAT_COLS.index(col)
        obs = M[:, :, j].astype(bool)
        X[:, :, j][obs] = np.log1p(np.clip(X[:, :, j][obs], 0, None))

    # ── Scale (observed only) ─────────────────────────────────────────────────
    flat     = X.reshape(-1, d)
    obs_flat = M.reshape(-1, d).astype(bool)
    col_means = np.array([np.nanmean(flat[obs_flat[:, j], j]) if obs_flat[:, j].any() else 0.0
                           for j in range(d)], dtype=np.float32)
    col_stds  = np.array([max(np.nanstd(flat[obs_flat[:, j], j]), 1e-8) if obs_flat[:, j].any() else 1.0
                           for j in range(d)], dtype=np.float32)
    scaler        = StandardScaler()
    scaler.mean_  = col_means
    scaler.scale_ = col_stds

    X = np.where(obs_flat, (flat - col_means) / col_stds, 0.0).reshape(N, T, d).astype(np.float32)

    obs_rates = M.mean(axis=(0, 2))
    for ti, t in enumerate(TIMEPOINTS):
        print(f"    T={t:2d}m obs rate: {obs_rates[ti]:.2f}")
    print(f"  d={d}  c={c_dim}  T={T}")

    meta = {"n_patients": N, "n_events": events, "feat_cols": FEAT_COLS,
            "static_cols": STATIC_COLS, "timepoints": TIMEPOINTS,
            "dataset": "PBC2 (pbcseq)", "source": "R survival package"}
    return X, M, C, ET, EI, np.array(patients), scaler, meta


if __name__ == "__main__":
    X, M, C, ET, EI, ids, scaler, meta = load_pbc2()
    print(f"\nX shape: {X.shape}  C shape: {C.shape}")
    print(f"Event rate: {EI.mean():.3f}")