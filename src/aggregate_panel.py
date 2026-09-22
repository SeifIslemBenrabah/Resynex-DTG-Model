"""
Aggregated feature panel: clinical subscales instead of individual items.

Why this exists
---------------
The 69-feature panel predicts each MDS-UPDRS item separately. Thirty-three of
those items are clinician-assigned ordinal ratings on a five-level scale, and
the scale's own clinimetric evaluation reports inter-rater reliability as
adequate at the level of a summed score rather than item by item. Predicting
them individually therefore fights the measurement instrument, and it is not
what a clinical trial reads out: trial endpoints are summed scores.

This module rebuilds the target panel accordingly. The 59 UPDRS items become 9
clinically meaningful subscale sums; the 10 instrument-measured continuous
features (cognition, CSF, imaging) are carried through unchanged.

  59 UPDRS items + 10 continuous  ->  9 subscales + 10 continuous
           d = 69                            d = 19

The Part III decomposition follows the standard motor phenotype grouping
(tremor / rigidity / bradykinesia / axial / bulbar) so that the subscales mean
something clinically rather than being an arbitrary partition.

The time-to-event labels are deliberately NOT recomputed here. They are derived
from the original 69-item tensor by the existing, already-reported code path, so
that the event definition is byte-identical to the one behind the reported
concordance and the two panels remain comparable on discrimination.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from preprocess_v2 import (load_ppmi_v2, TIMEPOINTS, T, c_dim,
                           FEAT_COLS, NP1_CLI, NP1_PQ, NP2_ITEMS, NP3_ITEMS,
                           COGNITIVE, CSF_COLS, DATSCAN)

# ── Part III motor subscales (the 33 items, partitioned without overlap) ───────

NP3_TREMOR = ["NP3PTRMR", "NP3PTRML", "NP3KTRMR", "NP3KTRML", "NP3RTARU",
              "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ", "NP3RTCON"]   # 10
NP3_RIGID  = ["NP3RIGN", "NP3RIGRU", "NP3RIGLU", "NP3RIGRL", "NP3RIGLL"]    # 5
NP3_BRADY  = ["NP3FTAPR", "NP3FTAPL", "NP3HMOVR", "NP3HMOVL", "NP3PRSPR",
              "NP3PRSPL", "NP3TTAPR", "NP3TTAPL", "NP3LGAGR", "NP3LGAGL",
              "NP3BRADY"]                                                   # 11
NP3_AXIAL  = ["NP3RISNG", "NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR"]    # 5
NP3_BULBAR = ["NP3SPCH", "NP3FACXP"]                                        # 2

# name -> (source item list, minimum observed fraction to prorate)
AGGREGATES = [
    ("NP3_TOTAL",  NP3_ITEMS),
    ("NP3_TREMOR", NP3_TREMOR),
    ("NP3_RIGID",  NP3_RIGID),
    ("NP3_BRADY",  NP3_BRADY),
    ("NP3_AXIAL",  NP3_AXIAL),
    ("NP3_BULBAR", NP3_BULBAR),
    ("NP2_TOTAL",  NP2_ITEMS),
    ("NP1C_TOTAL", NP1_CLI),
    ("NP1P_TOTAL", NP1_PQ),
]

CONTINUOUS = COGNITIVE + CSF_COLS + DATSCAN          # 10, carried through

# Objective-endpoint panel. Restricting the prediction target to the
# instrument-measured endpoints -- striatal binding ratios and cerebrospinal
# biomarkers -- is a design choice rather than a convenience: these are the
# endpoints a trial adopts when it wants a reading that does not depend on a
# clinician's judgement, and they are the ones whose measurement noise does not
# dominate their signal. The clinician-rated scales remain in the INPUT and
# still define the clinical event, so the survival task is unchanged; they are
# simply not among the quantities the generator is asked to predict.
OBJECTIVE_COLS = CSF_COLS + DATSCAN                  # 7 targets

AGG_COLS = [name for name, _ in AGGREGATES] + CONTINUOUS
d_agg    = len(AGG_COLS)

MIN_OBS_FRAC = 0.6   # prorate a subscale only if this fraction of items is present


def _sanity_check_partition():
    """The five Part III subscales must partition the 33 items exactly once."""
    union = NP3_TREMOR + NP3_RIGID + NP3_BRADY + NP3_AXIAL + NP3_BULBAR
    assert len(union) == len(NP3_ITEMS) == 33, (len(union), len(NP3_ITEMS))
    assert set(union) == set(NP3_ITEMS), set(union) ^ set(NP3_ITEMS)
    assert len(set(union)) == 33, "an item appears in two subscales"


def build_aggregated(verbose=True):
    """
    Returns
    -------
    Xa      : (N, T, d_agg) standardized aggregated panel
    Ma      : (N, T, d_agg) observation mask
    C       : (N, c_dim)    static features, unchanged
    scaler  : the StandardScaler fitted on the aggregated panel
    X69, M69, scaler69 : the original panel, so the event labels can be
                         recomputed by the existing code path
    """
    _sanity_check_partition()

    X, M, C, ET0, EI0, patients, scaler, meta = load_ppmi_v2()
    N = X.shape[0]

    # back to raw clinical units before summing; summing standardized values
    # would produce a quantity with no clinical meaning
    raw = X * scaler.scale_ + scaler.mean_

    Xa_raw = np.full((N, T, d_agg), np.nan, dtype=np.float32)
    Ma     = np.zeros((N, T, d_agg), dtype=np.float32)

    for j, (name, cols) in enumerate(AGGREGATES):
        idx  = [FEAT_COLS.index(c) for c in cols]
        n_it = len(idx)
        for ti in range(T):
            obs  = M[:, ti, :][:, idx].astype(bool)          # (N, n_it)
            vals = np.where(obs, raw[:, ti, :][:, idx], np.nan)
            n_ok = obs.sum(axis=1)
            keep = n_ok >= max(1, int(np.ceil(MIN_OBS_FRAC * n_it)))
            # prorate: mean of the observed items scaled to the full item count,
            # which is the standard way a summed clinical scale handles a
            # missing item and is unbiased when items are missing at random
            with np.errstate(invalid="ignore"):
                prorated = np.nanmean(vals, axis=1) * n_it
            Xa_raw[keep, ti, j] = prorated[keep]
            Ma[keep, ti, j]     = 1.0

    for k, col in enumerate(CONTINUOUS):
        j = len(AGGREGATES) + k
        i = FEAT_COLS.index(col)
        Xa_raw[:, :, j] = raw[:, :, i]
        Ma[:, :, j]     = M[:, :, i]

    # standardize the aggregated panel on observed entries only
    flat = Xa_raw.reshape(-1, d_agg)
    mask = Ma.reshape(-1, d_agg).astype(bool)
    agg_scaler = StandardScaler()
    filled = np.where(mask, flat, np.nan)
    agg_scaler.mean_  = np.nanmean(filled, axis=0)
    agg_scaler.scale_ = np.nanstd(filled, axis=0)
    agg_scaler.scale_[agg_scaler.scale_ < 1e-8] = 1.0
    agg_scaler.var_ = agg_scaler.scale_ ** 2
    agg_scaler.n_features_in_ = d_agg

    Xa = (np.nan_to_num(Xa_raw, nan=0.0) - agg_scaler.mean_) / agg_scaler.scale_
    Xa = (Xa * Ma).astype(np.float32)      # unobserved entries carry no value

    if verbose:
        print(f"\n  Aggregated panel: d = {d_agg} "
              f"({len(AGGREGATES)} subscales + {len(CONTINUOUS)} continuous)")
        print(f"  {'feature':14s} {'items':>6s} {'obs rate':>9s} "
              f"{'mean(raw)':>10s} {'sd(raw)':>9s}")
        for j, name in enumerate(AGG_COLS):
            n_it = len(AGGREGATES[j][1]) if j < len(AGGREGATES) else 1
            print(f"  {name:14s} {n_it:6d} {Ma[:,:,j].mean():9.3f} "
                  f"{agg_scaler.mean_[j]:10.2f} {agg_scaler.scale_[j]:9.2f}")
        print(f"  overall observation rate: {Ma.mean():.3f} "
              f"(69-item panel: {M.mean():.3f})")

    return Xa, Ma, C, agg_scaler, X, M, scaler, patients


if __name__ == "__main__":
    Xa, Ma, C, sc, X69, M69, sc69, pats = build_aggregated()
    print(f"\n  Xa {Xa.shape}  Ma {Ma.shape}  C {C.shape}")
