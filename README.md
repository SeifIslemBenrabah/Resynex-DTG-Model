# Resynex DTG — Digital Twin Generator for Clinical Trajectories

An energy-based Digital Twin Generator that forecasts a patient's untreated
clinical trajectory from a single baseline observation, together with the time
to a clinical milestone.

The architecture is the Neural Boltzmann Machine design of Lang (2023), coupled
to a DeepHit survival head through a differentiable trajectory-pooling layer.
This repository is the **research implementation and its evaluation**. The
platform that serves the trained model lives in a separate repository.

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

All figures are on held-out test splits, fixed before model development.

### Parkinson's disease (PPMI, n = 1554)

| Metric | Value | 95% CI |
|---|---|---|
| Concordance index | **0.8749** | [0.8394, 0.9084] |
| Trajectory RMSE (normalised) | 0.8724 | [0.7013, 1.0599] |

Intervals are percentile bootstrap over 2000 **patient-level** resamples. The
benchmark from published DTG deployments is 0.79; the lower bound clears it.

### Cross-indication transfer

The same architecture, retrained with no structural change:

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
  nbm.py                Neural Boltzmann Machine (reference implementation)
  model_v3.py           Full DTG: imputer + NBM + pooled DeepHit head
  model_v2.py           Earlier variant, kept for the ablation record
  preprocess_v2.py      PPMI → tensors; 59 UPDRS items + biomarkers
  train_v3.py           Curriculum joint training
  train_v2.py           Staged training (original)
  ablate_staged.py      Controlled staged run: only the schedule varies
  analyze_v4.py         Bootstrap CIs + predicted/observed correlation

experiments/
  pbc2/                 Primary biliary cholangitis
  support/              Critical care

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
With an approved export, `src/preprocess_v2.py` derives the cohort:

- six protocol visits mapped to months 0, 12, 24, 36, 48, 60;
- 69 longitudinal features — 59 individual MDS-UPDRS items (Parts I, II, III),
  3 cognitive/mood scales, 3 CSF biomarkers, 4 striatal binding ratios;
- 9 static features — demographics, disease duration, 5 genetic carrier flags;
- event = first visit where the summed Part III score reaches the threshold.

This yields 1554 patients and 8983 visit records. Observation rates fall from
0.97 at baseline to 0.29 at month 60; that missingness is informative and is
carried as an explicit mask rather than imputed.

PBC2 and SUPPORT are public and downloaded by their own scripts.

### Training

```bash
# Curriculum (joint) — the reported main run
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

A 300-epoch run takes roughly 20–30 minutes on CPU. No GPU is required, which
is deliberate: every result here is reproducible on ordinary hardware.

Seed 42 governs parameter initialisation, data order, the artificial masking in
the autoencoder, **and the Gibbs chain** — the last is the one usually left
unseeded in energy-based implementations, and leaving it so makes two runs of
identical configuration report different numbers.

### Weights

Trained checkpoints are attached to the GitHub Releases page rather than
committed. Each is about 4.5 MB.

---

## Known limitations

- **Trajectory fidelity is weak.** A normalised RMSE of 0.87 means the model
  leaves most of the per-feature variance unexplained. Nothing tried here
  improved it: the error barely moves across a 2.4× change in parameter count
  and a complete change of schedule.
- **Prognostic correlation decays fast.** Pearson r against observed values is
  0.687 at baseline and about 0.32 by month 36, so the ~20% control-arm
  reduction it implies applies to a near-term endpoint and overstates the
  benefit for a trial reading out later.
- **Single seed.** The ablation is one pair of runs. It bounds the schedule
  effect as small; it does not estimate it precisely.
- **Retrospective only.** Evaluation is on held-out patients from the same
  registry, not a different site, protocol, or future trial population.
- **Explainability not yet run.** SHAP attribution through the pooling stage is
  specified and implemented but the analysis has not been carried out.

---

## Context

This is the engineering follow-on to the Master's thesis *State of the Art of
Digital Twin for Clinical Trajectory Simulation*, which specified this
architecture and its evaluation plan without building it. The purpose of this
repository is to build it and report what the measurements actually say —
including where they contradict the specification.

## References

- Lang, P. *Neural Boltzmann Machines.* arXiv, 2023.
- Alam, N. et al. *Digital Twin Generators for Disease Modeling.* arXiv:2405.01488, 2024.
- Lee, C. et al. *DeepHit.* AAAI, 2018.
- Marek, K. et al. *The Parkinson Progression Marker Initiative.* 2011.

## Licence

MIT for the code. The clinical registries carry their own terms; PPMI in
particular requires a Data Use Agreement and its data may not be redistributed.
