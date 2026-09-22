# Resynex DTG — Digital Twin Generator for Clinical Trajectories

An energy-based Digital Twin Generator that forecasts a patient's untreated
clinical trajectory from a single baseline observation, together with the time
to a clinical milestone.

The architecture is the Neural Boltzmann Machine design of Lang (2023), coupled
to a DeepHit survival head through a differentiable trajectory-pooling layer.
This repository is the **research implementation and its evaluation**.
The platform that serves the trained model — six services, a dashboard and
a REST inference API — lives in
[Resynex-Platform](https://github.com/SeifIslemBenrabah/Resynex-Platform).

---

## What this is for

A Phase III trial commonly enrolls 500–2000 participants and runs for five to
eight years, and a large share of that cost buys the control arm rather than
knowledge about the drug. If a model can forecast the trajectory a participant
would have followed untreated, that forecast can absorb part of the statistical
load of a real control participant, and fewer need to be enrolled.

This repository trains and evaluates such a model, and reports honestly on where
it works and where it does not.

---

## Architecture

```
  baseline x₀ (incomplete) + mask m
              │
              ▼
   ┌──────────────────────┐
   │ Masked denoising     │  encoder 2d→256→128→128, decoder 128→128→256→d
   │ autoencoder          │  30% of observed entries dropped per pass
   └──────────┬───────────┘
              │  z ∈ ℝ¹²⁸
              ▼
   ┌──────────────────────┐   ξ = [z ‖ c ‖ t/t_max]
   │ Neural Boltzmann     │   bias μ(ξ), precision P(ξ), interaction W(ξ)
   │ Machine              │   Gaussian visible, Ising hidden
   └──────┬───────────┬───┘   Gibbs: K=4 training, K=32 generation
          │           │
     μ̄, Δμ         trajectory pool
          │
          ▼
   ┌──────────────────────┐
   │ DeepHit TTE head     │  [z ‖ x₀⊙m ‖ c ‖ μ̄ ‖ Δμ] → S=60 bins
   └──────────────────────┘  likelihood + pairwise ranking loss
```

Energy function:

```
U(y | ξ) = ½ (y − μ)ᵀ P (y − μ) − Σⱼ log cosh( Wⱼᵀ (y − μ) )
```

The hidden units are marginalised analytically, which is what keeps the model
tractable. `src/nbm.py` follows the reference implementation from unlearn.ai.

Three numerical safeguards proved necessary and are not optional:

- the precision network is clipped in log space before exponentiation;
- the interaction matrix is scaled by `1/√d` on output;
- the free energy uses `log cosh(u) = u − log 2 + softplus(−2u)`.

Without them, training diverges.

---

## Results

All figures are on held-out test splits, fixed before model development. The
headline numbers below are for the **objective-endpoint configuration**
(`src/train_objective.py` → `HybridDTG`/`ObjectiveDTG`): the same four-component
architecture, retargeted from the full 69-feature panel to seven
instrument-measured endpoints only (three CSF biomarkers, four DaTscan striatal
binding ratios), since clinician-rated scores turned out to consume model
capacity without adding learnable signal.

### Ensembling: a single seed is not a reliable measurement

Different training seeds of the *same* configuration converge to noticeably
different solutions, especially on the distributional metrics (PIT, MMD) —
enough that a single-seed comparison between two hyperparameter settings can
reverse when the seed changes. Averaging several independently seeded members
removes the part of that variance specific to one run while keeping the part
common to all of them, which is the actual signal. Point predictions and risk
scores are averaged across members; generative samples are a proper **mixture**
(pick a member uniformly, then sample from it) rather than an over-smoothed
average, so the ensemble's own predictive spread stays honest. See `ensemble.py`
and `metrics_objective_calibration_ensemble.py`.

### Parkinson's disease (PPMI, n = 1554, objective-endpoint panel)

| Metric | Single seed | **5-seed ensemble** |
|---|---|---|
| Concordance index | 0.8814 | **0.9097** |
| R² (variance explained) | 0.8309 | **0.8638** |
| PIT mean (ideal 0.5) | 0.611 | **0.527** |
| KS vs. uniform (ideal 0) | 0.231 | **0.076** |
| Calibration coverage @ 90% (ideal 0.90) | 0.912 | 0.909 |
| MMD², joint 7-outcome panel | — | **0.0011** (real-vs-real floor: −0.0004) |
| MMD², DaTscan panel | — | **−0.0002** (floor: −0.0002 — indistinguishable from real) |

Every metric improves or holds with the ensemble; none regresses. MMD² is
reported directly (not as a ratio to the real-vs-real floor): at these panel
sizes the floor's own expectation is ≈0, so dividing by it amplifies noise into
meaningless numbers — averaging over 20 posterior draws and 20 floor splits
(see `metrics_objective_calibration.py`) is what makes the raw MMD² trustworthy
instead.

### Cross-disease transfer: ADNI (Alzheimer's disease)

The same architecture family, trained **standalone from scratch on ADNI** (no
weights carried over from the PPMI run — see `experiments/adni/`), targeting
six outcomes (three cognitive scores, one CSF biomarker, two MRI volumetric
measures):

| Metric | Single seed | **5-seed ensemble** |
|---|---|---|
| Concordance index | 0.9405 | **0.9500** |
| R² (average) | 0.7817 | **0.8053** |
| PIT mean | 0.475 | **0.494** |
| KS vs. uniform | 0.050 | **0.040** |
| Calibration coverage @ 90% | 0.894 | **0.919** |
| MMD², 5 of 6 outcomes (excl. amyloid-beta) | — | **0.0028** (floor: 0.00001) |

The exception is amyloid-beta (ABETA): it is observed in only 95 of 1762 test
pairs (ADNI's lumbar-puncture sub-study is a small opt-in subset), and its own
calibration coverage stays weak (≈0.4–0.5 against a 0.90 target) even after
ensembling. That is a data-scarcity limit specific to one outcome, not a
property of the architecture — the other five outcomes ensemble to a
distribution statistically indistinguishable from real data.

### Cross-indication transfer, full panel

The earlier, full 69-feature configuration (`src/train_v3.py`), transferred
with no structural change to two further registries:

| Registry | Disease area | n | d | C-index |
|---|---|---|---|---|
| PPMI | Parkinson's disease | 1554 | 69 | **0.8749** |
| PBC2 | Primary biliary cholangitis | 312 | 11 | **0.8574** |
| SUPPORT | Critical care | 8873 | — | **0.6021** |

Transfer holds across chronic progressive disease and fails on critical care.
Cohort size is not the limiting factor — SUPPORT is the largest of the three.
What does not transfer is the architecture's core assumption: slow, monotone
progression from a stable baseline.

### Ablation: joint vs staged training — a negative result

The design hypothesis was that training the generative components jointly with
the survival head, through the pooling layer, beats training them in sequence.

**It does not, and the apparent evidence that it did was a confound.**

| Protocol | z | Event thr. | Test C-index |
|---|---|---|---|
| Staged, uncontrolled | 64 | 33 | 0.5016 |
| Curriculum (joint) | 128 | 25 | **0.8749** |
| **Staged, controlled** | **128** | **25** | **0.8760** |

The first two rows differ in *three* variables, not one: schedule, capacity, and
— decisively — the definition of the clinical event, at a threshold of 33
(40.7% event rate) versus 25 (63.1%). A concordance index computed under two
different event definitions is not one quantity measured twice.

Row three reverts only the schedule and holds everything else at row two. The
gap vanishes: 0.8760 staged against 0.8749 joint, a difference of 0.001 against
a confidence interval seventy times wider.

A result shaped like "chance versus strong" is the shape a confound produces.
Its size should be the first reason for suspicion, not for confidence.

Reproduce with `src/ablate_staged.py`.

---

## Repository layout

```
src/
  nbm.py                          Neural Boltzmann Machine (reference implementation)
  model_v3.py                     DTG_v3: imputer + NBM + pooled DeepHit head, full panel
  model_hybrid.py                 HybridDTG: DTG_v3 + Fourier-time flow predictor,
                                   Huber trajectory loss — the objective-endpoint model
  preprocess_v2.py                PPMI → tensors; 59 UPDRS items + biomarkers
  aggregate_panel.py              Full panel → 19-column aggregated + objective-endpoint views
  train_v3.py                     Curriculum joint training, full panel
  train_objective.py              Objective-endpoint training (the reported headline model)
  ensemble.py                     5-seed ensemble: point predictions averaged, risk averaged
  metrics_objective_calibration.py           Single-model PIT / MMD / calibration
  metrics_objective_calibration_ensemble.py  Ensemble PIT / MMD / calibration (mixture sampling)
  metrics_agg.py                  Shared RMSE/MAE/R² helpers, grouped by clinical procedure
  train_hybrid.py                 Training entry point ensemble.py drives per member
  ablate_staged.py                Controlled staged-vs-joint ablation (full panel)
  analyze_v4.py                   Bootstrap CIs + predicted/observed correlation

experiments/
  pbc2/                 Primary biliary cholangitis (public, cross-indication transfer)
  support/               Critical care (public, cross-indication transfer)
  adni/                  Alzheimer's disease (cross-disease transfer, standalone training)
    adni_dataset_notau.py                    ADNIMERGE2 -> tensors, 6-outcome panel
    model_platform_dtg.py                    Training wrapper importing PlatformDTG from
                                              the platform repo (see file docstring)
    train_adni_notau.py                      Standalone training entry point
    ensemble_adni_notau.py                   5-seed ensemble evaluation
    metrics_adni_calibration.py               Single-model PIT / MMD / calibration
    metrics_adni_calibration_ensemble.py      Ensemble PIT / MMD / calibration

results/                Committed run outputs — the evidence for the tables above
```

---

## Reproducing

```bash
pip install -r requirements.txt
```

### Data

**PPMI is not included and cannot be redistributed.** Access requires an
approved Data Use Agreement from [ppmi-info.org](https://www.ppmi-info.org/).
Place your own export under `data/ppmi/`, or point the `PPMI_DATA_DIR`
environment variable at wherever you keep it. With an approved export,
`src/preprocess_v2.py` derives the cohort:

- six protocol visits mapped to months 0, 12, 24, 36, 48, 60;
- 69 longitudinal features — 59 individual MDS-UPDRS items (Parts I, II, III),
  3 cognitive/mood scales, 3 CSF biomarkers, 4 striatal binding ratios;
- 9 static features — demographics, disease duration, 5 genetic carrier flags;
- event = first visit where the summed Part III score reaches the threshold.

This yields 1554 patients and 8983 visit records. Observation rates fall from
0.97 at baseline to 0.29 at month 60; that missingness is informative and is
carried as an explicit mask rather than imputed.

PBC2 and SUPPORT are public and downloaded by their own scripts.

**ADNI is not included and cannot be redistributed** either. Access requires
its own Data Use Agreement from [adni.loni.usc.edu](https://adni.loni.usc.edu/).
Place your own `ADNIMERGE2` export under `data/adni/ADNIMERGE2/data`, or point
`ADNI_DATA_DIR` at wherever you keep it.

### Training

```bash
# Objective-endpoint configuration — the reported headline model (7 targeted
# outcomes: 3 CSF biomarkers, 4 DaTscan ratios)
python src/train_objective.py \
    --out outputs_obj --epochs 70 --warmup 10 \
    --lam_var 1.0 --lam_cd 0.5 --nh 64 --seed 123

# 5-seed ensemble evaluation (point predictions averaged; generative samples
# are a mixture across members, not an average — see the Results section)
python src/metrics_objective_calibration_ensemble.py \
    --ckpts outputs_obj_seed123/model.pt,outputs_obj_seed42/model.pt,outputs_obj_seed456/model.pt,outputs_obj_seed789/model.pt,outputs_obj_seed1000/model.pt \
    --split_seed 42 --nh 64 --bins 60

# ADNI, standalone from scratch (no weights carried over from PPMI)
python experiments/adni/train_adni_notau.py \
    --epochs 60 --warmup 10 --lam_var 0.3 --lam_cd 0.1 --nh 32 --seed 123 \
    --out models_out/adni_notau_s123

# Full-panel configuration (69 features, all outcomes) — the earlier ablation record
python src/train_v3.py \
    --out outputs_v4 --epochs 300 --warmup 60 \
    --batch 16 --lr 3e-4 --nh 64 --z_dim 128 \
    --event_thresh 25 --seed 42

# Controlled staged ablation — only the schedule differs from the above
python src/ablate_staged.py \
    --out outputs_staged_z128 --phase1 60 --phase2 240 \
    --z_dim 128 --nh 64 --event_thresh 25 --seed 42

# Bootstrap CIs and the PROCOVA correlation
python src/analyze_v4.py \
    --ckpt outputs_v4/best_model.pt --out outputs_v4/analysis.json
```

A single objective-endpoint run (70 epochs) takes roughly 10 minutes; a
full-panel run (300 epochs) takes 20–30 minutes.

`--seed` governs parameter initialisation, data order, the artificial masking
in the autoencoder, **and the Gibbs chain** — the last is the one usually left
unseeded in energy-based implementations, and leaving it so makes two runs of
identical configuration report different numbers. That sensitivity is large
enough, in fact, that single-seed comparisons between hyperparameter settings
are not reliable — see the ensembling note in the Results section above.

### Weights

Trained checkpoints are attached to the GitHub Releases page rather than
committed. Each is about 4.5 MB.

---

## Known limitations

The first three points below are specific to the **full 69-feature panel**
(`train_v3.py`), not the objective-endpoint headline results:

- **Trajectory fidelity on the full panel is weak.** A normalised RMSE of 0.87
  means the model leaves most of the per-feature variance unexplained there.
  Nothing tried improved it: the error barely moves across a 2.4× change in
  parameter count and a complete change of schedule. This is exactly why the
  objective-endpoint configuration (targeting instrument-measured outcomes
  only) exists, and it reaches R² 0.83–0.86 instead.
- **Prognostic correlation decays fast (full panel).** Pearson r against
  observed values is 0.687 at baseline and about 0.32 by month 36.
- **The staged-vs-joint ablation is one pair of runs.** It bounds the schedule
  effect as small; it does not estimate it precisely.

The rest apply generally:

- **Retrospective only.** Evaluation is on held-out patients from the same
  registry, not a different site, protocol, or future trial population.
- **Amyloid-beta (ADNI) stays poorly calibrated.** See the Results section —
  ensembling does not fix a data-scarcity limit (95 of 1762 test pairs).
- **DaTscan MMD's real-vs-real floor is itself near zero at this sample size**,
  so it is reported as a raw MMD² rather than a ratio; see
  `metrics_objective_calibration.py`'s docstring for why the ratio form is
  unreliable here.

---

## Context

This repository builds an energy-based Digital Twin Generator and reports
what the measurements actually say, including where a hypothesis (joint vs.
staged training, a hyperparameter change, a larger NBM) did not hold up once
tested properly — see the ablation and the ensembling note above.

## References

- Lang, P. *Neural Boltzmann Machines.* arXiv, 2023.
- Alam, N. et al. *Digital Twin Generators for Disease Modeling.* arXiv:2405.01488, 2024.
- Lee, C. et al. *DeepHit.* AAAI, 2018.
- Marek, K. et al. *The Parkinson Progression Marker Initiative.* 2011.

## Licence

MIT for the code. The clinical registries carry their own terms; PPMI in
particular requires a Data Use Agreement and its data may not be redistributed.
