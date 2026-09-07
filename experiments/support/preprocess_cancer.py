"""
Preprocessing for SUPPORT + METABRIC datasets.

SUPPORT  : 8,873 critically ill patients — 7 physiological features, 7 static.
METABRIC : 1,904 breast-cancer patients — 4 gene-expression + 5 clinical.

We build a unified 3-timepoint tensor:
  T=0  (day   0) : observed baseline (all features present in SUPPORT)
  T=1  (day  90) : unobserved — model must generate
  T=2  (day 180) : unobserved — model must generate

This trains the DTG as a prospective generator:
  the NBM + PointPredictor learn to extrapolate future physiological
  states consistent with the observed survival outcome.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pycox.datasets import support as support_ds, metabric as metabric_ds


# ── Column mappings ────────────────────────────────────────────────────────────

# SUPPORT physiological measurements (dynamic — change over time in reality)
FEAT_COLS = [
    "map_bp",          # x7  mean arterial blood pressure (mmHg)
    "heart_rate",      # x8  heart rate (bpm)
    "resp_rate",       # x9  respiration rate (/min)
    "temperature",     # x10 body temperature (°C)
    "wbc",             # x11 white blood cell count
    "serum_sodium",    # x12 serum sodium
    "creatinine",      # x13 serum creatinine (mg/dL)
]

# Patient-level static features
STATIC_COLS = [
    "age",             # x0
    "sex",             # x1  0=female, 1=male
    "race",            # x2  0–9 categorical
    "n_comorbidities", # x3  0–5
    "diabetes",        # x4  binary
    "dementia",        # x5  binary
    "cancer_status",   # x6  0=none, 1=localised, 2=metastatic
]

d     = len(FEAT_COLS)    # 7
c_dim = len(STATIC_COLS)  # 7

# Three pseudo-timepoints (only T=0 is observed in SUPPORT)
TIMEPOINTS = [0, 90, 180]   # days
T = len(TIMEPOINTS)         # 3

# Survival bins (approx monthly over ~5.5 years)
MAX_DURATION_DAYS = 2029.0
S = 60


# ── Main loader ────────────────────────────────────────────────────────────────

def load_cancer(use_metabric: bool = False):
    """
    Returns
    -------
    X       : (N, T, d)   float32  longitudinal features  (masked where unobserved)
    M       : (N, T, d)   float32  observation mask
    C       : (N, c_dim)  float32  static context
    ET      : (N,)        float32  normalised event time in [0, 1]
    EI      : (N,)        float32  event indicator  {0, 1}
    ids     : (N,)        int      patient index
    scaler  : fitted StandardScaler on FEAT_COLS
    meta    : dict        dataset statistics
    """
    if use_metabric:
        return _load_metabric()
    return _load_support()


# ── SUPPORT loader ─────────────────────────────────────────────────────────────

def _load_support():
    df = support_ds.read_df()

    # Rename columns for clarity
    rename = {
        "x0": "age", "x1": "sex", "x2": "race",
        "x3": "n_comorbidities", "x4": "diabetes",
        "x5": "dementia", "x6": "cancer_status",
        "x7": "map_bp",   "x8": "heart_rate",  "x9": "resp_rate",
        "x10": "temperature", "x11": "wbc",
        "x12": "serum_sodium", "x13": "creatinine",
    }
    df = df.rename(columns=rename)

    N = len(df)
    print(f"SUPPORT dataset: {N} patients")
    print(f"  Death events : {df['event'].sum()} / {N}  ({df['event'].mean()*100:.1f}%)")
    print(f"  Duration     : {df['duration'].min():.0f} - {df['duration'].max():.0f} days")
    n_cancer = (df["cancer_status"] > 0).sum()
    print(f"  Cancer patients (any): {n_cancer} / {N}  ({n_cancer/N*100:.1f}%)")
    n_meta = (df["cancer_status"] == 2).sum()
    print(f"  Metastatic cancer    : {n_meta}")

    # ── Dynamic features (physiological) ──────────────────────────────────────
    feat_df = df[FEAT_COLS].copy()

    # clip extreme outliers at 1st/99th percentile
    for col in FEAT_COLS:
        lo, hi = feat_df[col].quantile(0.01), feat_df[col].quantile(0.99)
        feat_df[col] = feat_df[col].clip(lo, hi)

    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat_df.values.astype(np.float32))

    # Build (N, T=3, d=7) tensors — only T=0 is observed
    X = np.zeros((N, T, d), dtype=np.float32)
    M = np.zeros((N, T, d), dtype=np.float32)
    X[:, 0, :] = feat_scaled         # baseline observed
    M[:, 0, :] = 1.0                 # T=0 fully observed

    # ── Static features ────────────────────────────────────────────────────────
    C = df[STATIC_COLS].values.astype(np.float32)
    # normalise continuous statics
    C[:, 0] = (C[:, 0] - C[:, 0].mean()) / (C[:, 0].std() + 1e-8)   # age
    C[:, 2] = C[:, 2] / 9.0                                           # race → [0,1]
    C[:, 3] = C[:, 3] / 5.0                                           # n_comorbidities
    C[:, 6] = C[:, 6] / 2.0                                           # cancer_status → [0,1]

    # ── Survival outcome ───────────────────────────────────────────────────────
    ET = (df["duration"].values / MAX_DURATION_DAYS).astype(np.float32).clip(0, 1)
    EI = df["event"].values.astype(np.float32)

    ids = np.arange(N)

    meta = {
        "n_patients":   N,
        "n_events":     int(EI.sum()),
        "n_cancer":     int(n_cancer),
        "n_metastatic": int(n_meta),
        "feat_cols":    FEAT_COLS,
        "static_cols":  STATIC_COLS,
        "timepoints":   TIMEPOINTS,
        "dataset":      "SUPPORT",
    }

    print(f"  Tensor X=({N}, {T}, {d})  C=({N}, {c_dim})")
    print("Done.")
    return X, M, C, ET, EI, ids, scaler, meta


# ── METABRIC loader ────────────────────────────────────────────────────────────

METABRIC_FEAT  = ["x0", "x1", "x2", "x3"]   # gene expression: MKI67, EGFR, PGR, ERBB2
METABRIC_STATIC = ["x4", "x5", "x6", "x7", "x8"]  # hormone, radio, chemo, ER+, age

METABRIC_FEAT_NAMES   = ["MKI67", "EGFR", "PGR", "ERBB2"]
METABRIC_STATIC_NAMES = ["hormone_therapy", "radiotherapy", "chemotherapy", "ER_positive", "age"]


def _load_metabric():
    df = metabric_ds.read_df()
    N  = len(df)
    print(f"METABRIC dataset: {N} breast-cancer patients")
    print(f"  Death events : {df['event'].sum()} / {N}  ({df['event'].mean()*100:.1f}%)")
    print(f"  Duration     : {df['duration'].min():.1f} - {df['duration'].max():.1f} months")

    feat_df = df[METABRIC_FEAT].copy().astype(np.float32)
    d_m = len(METABRIC_FEAT)

    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat_df.values)

    X = np.zeros((N, T, d_m), dtype=np.float32)
    M = np.zeros((N, T, d_m), dtype=np.float32)
    X[:, 0, :] = feat_scaled
    M[:, 0, :] = 1.0

    C = df[METABRIC_STATIC].values.astype(np.float32)
    C[:, 4] = (C[:, 4] - C[:, 4].mean()) / (C[:, 4].std() + 1e-8)  # age

    MAX_DUR = df["duration"].max()
    ET = (df["duration"].values / MAX_DUR).astype(np.float32).clip(0, 1)
    EI = df["event"].values.astype(np.float32)
    ids = np.arange(N)

    meta = {
        "n_patients": N,
        "n_events":   int(EI.sum()),
        "feat_cols":  METABRIC_FEAT_NAMES,
        "static_cols": METABRIC_STATIC_NAMES,
        "timepoints": TIMEPOINTS,
        "dataset":    "METABRIC",
    }
    print(f"  Tensor X=({N}, {T}, {d_m})  C=({N}, {len(METABRIC_STATIC_NAMES)})")
    print("Done.")
    return X, M, C, ET, EI, ids, scaler, meta