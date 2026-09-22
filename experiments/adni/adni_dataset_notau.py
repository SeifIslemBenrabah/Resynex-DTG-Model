"""
Variant of adni_dataset_altout.py with TAU removed from the outcome panel
entirely (not just log-transformed): six outcomes instead of seven --
CDRSB, MMSCORE, TOTAL13, ABETA, VENT_NORM, HIPPO_NORM. TAU is dropped from
the inputs too, not kept as an input-only column, since nothing asked for
that. VENT_NORM keeps its log1p transform (that gain was verified
independently of TAU and holds regardless of what else is in the panel).
"""

import os
import numpy as np
import pandas as pd
import pyreadr
import warnings

warnings.filterwarnings("ignore")

# ADNI is governed by its own Data Use Agreement and is never vendored here;
# place your own approved ADNIMERGE2 export under data/adni/ADNIMERGE2/data,
# or point ADNI_DATA_DIR at wherever you keep it.
ADNI_DIR = os.environ.get("ADNI_DATA_DIR", "data/adni/ADNIMERGE2/data")

OUTCOME_COLS = ["CDRSB", "MMSCORE", "TOTAL13", "ABETA", "VENT_NORM", "HIPPO_NORM"]

LOG_TRANSFORM_COLS = ["VENT_NORM"]

INPUT_ONLY_COLS = ["DX_NUM"]

STATIC_COLS = ["AGE_BL", "SEX", "EDUC", "APOE4_COUNT"]

MAX_DAYS = 7300.0

BASELINE_COLS = (STATIC_COLS
                 + [f"bl_{c}" for c in OUTCOME_COLS + INPUT_ONLY_COLS]
                 + [f"curr_{c}" for c in OUTCOME_COLS + INPUT_ONLY_COLS]
                 + ["curr_t_days_norm"])

_DX_MAP = {"CN": 0.0, "MCI": 1.0, "Dementia": 2.0}


def _load(name):
    return pyreadr.read_r(f"{ADNI_DIR}/{name}.rda")[name]


def _dedup_mean(df, value_cols):
    return df.groupby(["RID", "VISCODE"], as_index=False)[value_cols].mean()


def build(verbose=True):
    dx = _load("DXSUM")[["RID", "VISCODE", "EXAMDATE", "DIAGNOSIS"]].copy()
    dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    dx["DX_NUM"] = dx["DIAGNOSIS"].map(_DX_MAP)
    dx = dx.dropna(subset=["EXAMDATE"])
    dx = dx.drop_duplicates(subset=["RID", "VISCODE"])

    cdr = _load("CDR")[["RID", "VISCODE", "CDRSB"]]
    cdr = _dedup_mean(cdr, ["CDRSB"])

    mmse = _load("MMSE")[["RID", "VISCODE", "MMSCORE"]]
    mmse = _dedup_mean(mmse, ["MMSCORE"])

    adas = _load("ADAS")[["RID", "VISCODE", "TOTAL13"]]
    adas = _dedup_mean(adas, ["TOTAL13"])

    bio = _load("UPENNBIOMK_MASTER")[["RID", "VISCODE", "ABETA"]]
    bio["ABETA"] = pd.to_numeric(bio["ABETA"], errors="coerce")
    bio = _dedup_mean(bio, ["ABETA"])

    fsx = _load("UCSFFSX")[["RID", "VISCODE", "ST29SV", "ST88SV", "ST10CV",
                            "ST37SV", "ST96SV"]].copy()
    for c in ["ST29SV", "ST88SV", "ST10CV", "ST37SV", "ST96SV"]:
        fsx[c] = pd.to_numeric(fsx[c], errors="coerce")
    fsx["HIPPO_NORM"] = 1000.0 * (fsx["ST29SV"] + fsx["ST88SV"]) / 2.0 / fsx["ST10CV"]
    fsx["VENT_NORM"] = 1000.0 * (fsx["ST37SV"] + fsx["ST96SV"]) / 2.0 / fsx["ST10CV"]
    fsx = _dedup_mean(fsx, ["HIPPO_NORM", "VENT_NORM"])

    fsx["VENT_NORM"] = np.log1p(fsx["VENT_NORM"].clip(lower=0))

    long = dx
    for tbl in (cdr, mmse, adas, bio, fsx):
        long = long.merge(tbl, on=["RID", "VISCODE"], how="left")

    pt = _load("PTDEMOG")[["RID", "PTGENDER", "PTDOB", "PTEDUCAT"]].dropna(subset=["RID"])
    pt = pt.drop_duplicates(subset=["RID"], keep="first")
    pt["SEX"] = (pt["PTGENDER"] == "Male").astype(float)
    pt["EDUC"] = pd.to_numeric(pt["PTEDUCAT"], errors="coerce")
    pt["_DOB"] = pd.to_datetime(pt["PTDOB"], format="%m/%Y", errors="coerce")

    apoe = _load("APOERES")[["RID", "GENOTYPE"]].dropna(subset=["GENOTYPE"])
    apoe = apoe.drop_duplicates(subset=["RID"], keep="first")
    apoe["APOE4_COUNT"] = apoe["GENOTYPE"].astype(str).str.count("4")

    rows, ys, ts, pids = [], [], [], []
    for rid, g in long.groupby("RID"):
        g = g.sort_values("EXAMDATE").reset_index(drop=True)
        g = g.drop_duplicates(subset=["VISCODE"], keep="first")
        if len(g) < 2:
            continue

        prow = pt[pt.RID == rid]
        arow = apoe[apoe.RID == rid]
        if len(prow) == 0 or pd.isna(prow["_DOB"].iloc[0]):
            continue
        dob = prow["_DOB"].iloc[0]
        sex = prow["SEX"].iloc[0]
        educ = prow["EDUC"].iloc[0]
        apoe4 = float(arow["APOE4_COUNT"].iloc[0]) if len(arow) else np.nan

        bl = g.iloc[0]
        age_bl = (bl["EXAMDATE"] - dob).days / 365.25

        for i in range(len(g) - 1):
            cur, fut = g.iloc[i], g.iloc[i + 1]
            out = np.array([fut.get(c, np.nan) for c in OUTCOME_COLS],
                           dtype=np.float32)
            if np.isnan(out).all():
                continue
            rec = {"AGE_BL": age_bl, "SEX": sex, "EDUC": educ,
                   "APOE4_COUNT": apoe4}
            for c in OUTCOME_COLS + INPUT_ONLY_COLS:
                rec[f"bl_{c}"] = bl.get(c, np.nan)
                rec[f"curr_{c}"] = cur.get(c, np.nan)
            rec["curr_t_days_norm"] = (cur["EXAMDATE"] - bl["EXAMDATE"]).days / MAX_DAYS
            rows.append([rec[c] for c in BASELINE_COLS])
            ys.append(out)
            ts.append((fut["EXAMDATE"] - bl["EXAMDATE"]).days / MAX_DAYS)
            pids.append(rid)

    X = np.asarray(rows, dtype=np.float32)
    Y = np.asarray(ys, dtype=np.float32)
    Tf = np.asarray(ts, dtype=np.float32).reshape(-1, 1)
    PID = np.asarray(pids)

    et, ei = {}, {}
    for rid, g in long.groupby("RID"):
        g = g.sort_values("EXAMDATE")
        if g["EXAMDATE"].isna().all():
            continue
        bl_date = g["EXAMDATE"].iloc[0]
        conv = g[g["DX_NUM"] == 2.0]
        if len(conv):
            t = (conv["EXAMDATE"].iloc[0] - bl_date).days / MAX_DAYS
            et[rid], ei[rid] = t, 1.0
        else:
            t = (g["EXAMDATE"].iloc[-1] - bl_date).days / MAX_DAYS
            et[rid], ei[rid] = t, 0.0
    ET = np.array([et.get(p, 1.0) for p in PID], dtype=np.float32)
    EI = np.array([ei.get(p, 0.0) for p in PID], dtype=np.float32)

    if verbose:
        print(f"  ADNI (no-TAU): {len(X)} visit pairs from {len(np.unique(PID))} patients")
        print(f"  inputs {X.shape[1]}   outcomes {Y.shape[1]}")
        print(f"  events {int(EI.sum())} rows ({100*EI.mean():.1f}%)")
        print(f"  input missingness (mean): {np.isnan(X).mean():.3f}")
        print(f"  outcome missingness (mean): {np.isnan(Y).mean():.3f}")
    return X, Y, Tf, PID, ET, EI


if __name__ == "__main__":
    build()
