"""
PPMI v2 Preprocessing — expanded to 59 individual UPDRS items + biomarkers.

Matches the unlearn.ai DTG paper (arXiv:2405.01488) which uses
"59 items of MDS-UPDRS Parts I, II, and III individually"
plus cognitive, CSF, and DaTscan biomarkers.

Data source: PPMI data folder (same files as v1)
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Point PPMI_DATA_DIR at your approved PPMI export. The data is not
# redistributable, so no default location inside the repository exists.
DATA_DIR = os.environ.get("PPMI_DATA_DIR", os.path.join("data", "ppmi"))

# ── Visit mapping ──────────────────────────────────────────────────────────────
VISIT_MAP    = {"BL": 0, "V04": 12, "V06": 24, "V08": 36, "V10": 48, "V12": 60}
TIMEPOINTS   = list(VISIT_MAP.values())   # [0,12,24,36,48,60]
T            = len(TIMEPOINTS)

# ── 59 Individual UPDRS items ──────────────────────────────────────────────────
NP1_CLI  = ["NP1COG","NP1HALL","NP1DPRS","NP1ANXS","NP1APAT","NP1DDS"]          # 6
NP1_PQ   = ["NP1CNST","NP1FATG","NP1LTHD","NP1PAIN","NP1SLPD","NP1SLPN","NP1URIN"]  # 7
NP2_ITEMS = ["NP2DRES","NP2EAT","NP2FREZ","NP2HOBB","NP2HWRT","NP2HYGN",
             "NP2RISE","NP2SALV","NP2SPCH","NP2SWAL","NP2TRMR","NP2TURN","NP2WALK"]  # 13
NP3_ITEMS = ["NP3SPCH","NP3FACXP","NP3RIGN","NP3RIGRU","NP3RIGLU","NP3RIGRL",
             "NP3RIGLL","NP3FTAPR","NP3FTAPL","NP3HMOVR","NP3HMOVL","NP3PRSPR",
             "NP3PRSPL","NP3TTAPR","NP3TTAPL","NP3LGAGR","NP3LGAGL","NP3RISNG",
             "NP3GAIT","NP3FRZGT","NP3PSTBL","NP3POSTR","NP3BRADY","NP3PTRMR",
             "NP3PTRML","NP3KTRMR","NP3KTRML","NP3RTARU","NP3RTALU","NP3RTARL",
             "NP3RTALL","NP3RTALJ","NP3RTCON"]  # 33

UPDRS_ITEMS = NP1_CLI + NP1_PQ + NP2_ITEMS + NP3_ITEMS   # 59 total

# ── Additional longitudinal biomarkers ─────────────────────────────────────────
COGNITIVE  = ["moca", "gds", "ess"]
CSF_COLS   = ["asyn", "tau", "ptau"]
DATSCAN    = ["MIA_CAUDATE_L", "MIA_CAUDATE_R", "MIA_PUTAMEN_L", "MIA_PUTAMEN_R"]

FEAT_COLS  = UPDRS_ITEMS + COGNITIVE + CSF_COLS + DATSCAN   # d = 59+3+3+4 = 69
d          = len(FEAT_COLS)

# ── Static patient features ────────────────────────────────────────────────────
STATIC_COLS = ["age", "SEX", "EDUCYRS", "duration_yrs",
               "APOE_e4", "LRRK2", "GBA", "SNCA", "PRKN"]
c_dim = len(STATIC_COLS)

# TTE bins
S = 60


# ── Genetic flags ──────────────────────────────────────────────────────────────
def _genetic_flags(df):
    sub = df["subgroup"].fillna("").astype(str)
    df["LRRK2"] = sub.str.contains(r"\bLRRK2\b", regex=True).astype(float)
    df["GBA"]   = sub.str.contains(r"\bGBA\b",   regex=True).astype(float)
    df["SNCA"]  = sub.str.contains(r"\bSNCA\b",  regex=True).astype(float)
    df["PRKN"]  = sub.str.contains(r"\bPRKN\b",  regex=True).astype(float)
    return df


# ── Main loader ────────────────────────────────────────────────────────────────

def load_ppmi_v2(data_dir=DATA_DIR):
    """
    Returns
    -------
    X      : (N, T, d)   float32
    M      : (N, T, d)   float32  mask
    C      : (N, c_dim)  float32
    ET     : (N,)        float32  normalised event time
    EI     : (N,)        float32  event indicator
    patnos : (N,)        int
    scaler : StandardScaler fitted on FEAT_COLS
    meta   : dict
    """
    print("Loading PPMI v2 — 59 UPDRS items + biomarkers...")

    # ── Curated base ──────────────────────────────────────────────────────────
    curated = pd.read_csv(f"{data_dir}/PPMI_Curated_Data_Cut_Public.csv",
                          low_memory=False)
    curated = curated[curated["COHORT"] == 1].copy()
    curated = curated[curated["EVENT_ID"].isin(VISIT_MAP.keys())].copy()
    curated = _genetic_flags(curated)
    curated["APOE_e4"] = (curated["APOE_e4"].fillna(0).astype(float) >= 1).astype(float)
    curated["visit_month"] = curated["EVENT_ID"].map(VISIT_MAP)

    # CSF log-transform
    for col in CSF_COLS:
        if col in curated.columns:
            curated[col] = np.log1p(curated[col].clip(lower=0))

    # ── Part I clinician ──────────────────────────────────────────────────────
    p1 = pd.read_csv(f"{data_dir}/MDS-UPDRS_Part_I_05May2026.csv", low_memory=False)
    p1 = p1[p1["EVENT_ID"].isin(VISIT_MAP.keys())][["PATNO","EVENT_ID"] + NP1_CLI]
    p1.columns = ["PATNO","EVENT_ID"] + NP1_CLI

    # ── Part I patient questionnaire ──────────────────────────────────────────
    p1q = pd.read_csv(f"{data_dir}/MDS-UPDRS_Part_I_Patient_Questionnaire_05May2026.csv",
                      low_memory=False)
    p1q = p1q[p1q["EVENT_ID"].isin(VISIT_MAP.keys())][["PATNO","EVENT_ID"] + NP1_PQ]

    # ── Part II patient questionnaire ─────────────────────────────────────────
    p2 = pd.read_csv(f"{data_dir}/MDS_UPDRS_Part_II__Patient_Questionnaire_05May2026.csv",
                     low_memory=False)
    p2 = p2[p2["EVENT_ID"].isin(VISIT_MAP.keys())][["PATNO","EVENT_ID"] + NP2_ITEMS]

    # ── Part III clinician ────────────────────────────────────────────────────
    p3 = pd.read_csv(f"{data_dir}/MDS-UPDRS_Part_III_05May2026.csv", low_memory=False)
    p3 = p3[p3["EVENT_ID"].isin(VISIT_MAP.keys())][["PATNO","EVENT_ID"] + NP3_ITEMS]

    # ── Merge everything ──────────────────────────────────────────────────────
    base_cols = ["PATNO","EVENT_ID","visit_month","age","SEX","EDUCYRS",
                 "duration_yrs","APOE_e4","LRRK2","GBA","SNCA","PRKN",
                 "subgroup"] + COGNITIVE + CSF_COLS + DATSCAN
    base_cols = [c for c in base_cols if c in curated.columns]
    df = curated[base_cols].copy()

    # keep only patients that appear in Part III (most restrictive)
    pd_patnos = set(p3["PATNO"].unique())
    df = df[df["PATNO"].isin(pd_patnos)]

    df = df.merge(p1,  on=["PATNO","EVENT_ID"], how="left")
    df = df.merge(p1q, on=["PATNO","EVENT_ID"], how="left")
    df = df.merge(p2,  on=["PATNO","EVENT_ID"], how="left")
    df = df.merge(p3,  on=["PATNO","EVENT_ID"], how="left")

    df = df.sort_values(["PATNO","visit_month"])
    patients = df["PATNO"].unique()
    N = len(patients)
    print(f"  PD patients : {N}   rows : {len(df)}")

    # ── Build tensors ─────────────────────────────────────────────────────────
    X = np.full((N, T, d), np.nan, dtype=np.float32)
    M = np.zeros((N, T, d), dtype=np.float32)

    pat2idx = {p: i for i, p in enumerate(patients)}
    vm_list = TIMEPOINTS

    for _, row in df.iterrows():
        i  = pat2idx[row["PATNO"]]
        ti = vm_list.index(row["visit_month"])
        for j, col in enumerate(FEAT_COLS):
            v = row.get(col, np.nan)
            if pd.notna(v):
                try:
                    X[i, ti, j] = float(v)
                    M[i, ti, j] = 1.0
                except (ValueError, TypeError):
                    pass

    # ── Static features ───────────────────────────────────────────────────────
    static_df = df.groupby("PATNO").first().reset_index()
    static_df = static_df.set_index("PATNO").reindex(patients)

    C = np.zeros((N, c_dim), dtype=np.float32)
    for j, col in enumerate(STATIC_COLS):
        if col in static_df.columns:
            vals = static_df[col].values.astype(float)
            C[:, j] = np.nan_to_num(vals, nan=0.0)

    # ── Scale features ────────────────────────────────────────────────────────
    flat = X.reshape(-1, d)
    obs_mask_flat = M.reshape(-1, d).astype(bool)
    scaler = StandardScaler()
    # fit only on observed values
    obs_vals = np.where(obs_mask_flat, flat, np.nan)
    col_means = np.nanmean(obs_vals, axis=0)
    col_stds  = np.nanstd(obs_vals,  axis=0) + 1e-8
    scaler.mean_  = col_means
    scaler.scale_ = col_stds
    X_scaled = np.where(obs_mask_flat, (flat - col_means) / col_stds, 0.0)
    X = X_scaled.reshape(N, T, d).astype(np.float32)

    # ── TTE: UPDRS-III >= 33 ─────────────────────────────────────────────────
    np3_idx = FEAT_COLS.index("NP3TOT") if "NP3TOT" in FEAT_COLS else None
    updrs3_idx = FEAT_COLS.index("NP3SPCH")   # first NP3 item; compute sum proxy

    # Use NP3 items to compute total (sum of 33 items; raw, before scaling)
    np3_raw  = np.full((N, T), np.nan)
    np3_cols_idx = [i for i, c in enumerate(FEAT_COLS) if c in NP3_ITEMS]
    raw_X = X * col_stds + col_means   # back to raw
    for ti in range(T):
        raw_X[:, ti, :]
        np3_sum = np.where(M[:, ti, :][:, np3_cols_idx].astype(bool),
                           raw_X[:, ti, :][:, np3_cols_idx], np.nan)
        np3_raw[:, ti] = np.nansum(np3_sum, axis=1)
        # set to nan where no NP3 items observed
        no_obs = (M[:, ti, :][:, np3_cols_idx].sum(axis=1) == 0)
        np3_raw[no_obs, ti] = np.nan

    EVENT_THRESH = 33.0
    ET_raw = np.full(N, 60.0, dtype=np.float32)
    EI     = np.zeros(N, dtype=np.float32)
    for i in range(N):
        for ti, t in enumerate(TIMEPOINTS):
            if np.isnan(np3_raw[i, ti]):
                continue
            if np3_raw[i, ti] >= EVENT_THRESH:
                ET_raw[i] = float(t)
                EI[i]     = 1.0
                break

    ET = (ET_raw / 60.0).astype(np.float32)

    events = int(EI.sum())
    print(f"  UPDRS-III >= 33 events : {events} / {N}  ({events/N*100:.1f}%)")

    # ── Genetic summary ───────────────────────────────────────────────────────
    g = C[:, 4:]  # APOE_e4, LRRK2, GBA, SNCA, PRKN
    print(f"  LRRK2+: {int(C[:,5].sum())}   GBA+: {int(C[:,6].sum())}   "
          f"SNCA+: {int(C[:,7].sum())}   PRKN+: {int(C[:,8].sum())}")

    obs_rates = M.mean(axis=(0, 2))
    for ti, t in enumerate(TIMEPOINTS):
        print(f"    T={t:3d}m obs rate: {obs_rates[ti]:.2f}")

    print(f"  Feature dim d={d}   Static dim c={c_dim}")
    print(f"  Tensor X=({N},{T},{d})   C=({N},{c_dim})")
    print("Done.")

    meta = {
        "n_patients": N,
        "n_events":   events,
        "n_lrrk2":    int(C[:,5].sum()),
        "n_gba":      int(C[:,6].sum()),
        "n_snca":     int(C[:,7].sum()),
        "n_prkn":     int(C[:,8].sum()),
        "feat_cols":  FEAT_COLS,
        "static_cols": STATIC_COLS,
        "timepoints":  TIMEPOINTS,
        "updrs_items": UPDRS_ITEMS,
    }
    return X, M, C, ET, EI, patients, scaler, meta