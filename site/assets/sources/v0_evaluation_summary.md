# Historical V0 Evaluation Summary — Superseded

> **Do not use this file as the current V0 scientific record.** This 1,268-observation run inherited Awpy's 128 ticks/s default while the raw demo clock measured 64 ticks/s, so its nominal 10 / 20 / 30 / 40-second horizons were effectively about 20 / 40 / 60 / 80 seconds. It is preserved for auditability. The corrected record is documented in `docs/v0_explained.md` and the `artifacts/v0_tc_*.csv` tables.

## Task

V0 predicts the early-round T-side outcome on Mirage:

- A_PLANT
- B_PLANT
- NO_PLANT

Observations are taken at 10, 20, 30, and 40 seconds after freeze_end.

The output is probabilistic:

P(A_PLANT), P(B_PLANT), P(NO_PLANT)

---

## Dataset

- 20 Mirage demos
- 444 parsed rounds
- 443 valid-timing rounds
- 1 partial opening round excluded
- 1,268 eligible observations
- 0 missing feature values

Observation counts:

- 10s: 442
- 20s: 373
- 30s: 270
- 40s: 183

All evaluation uses frozen match-level grouped cross-validation.

Observations from the same match never appear in both train and test.

---

# Logistic Regression Ablation

| Stage | Features Added | Log Loss | Brier | Accuracy | Macro F1 |
|---|---|---:|---:|---:|---:|
| A0 | Class prior | 1.0240 | 0.6186 | 0.4913 | 0.2196 |
| A1 | Offensive geometry | 0.9161 | 0.5675 | 0.5142 | 0.4628 |
| A2 | Motion | 0.9119 | 0.5612 | 0.5347 | 0.4823 |
| A3 | Combat | 0.8359 | 0.4939 | 0.5994 | 0.5541 |
| A4 | Economy | 0.8509 | 0.5011 | 0.5986 | 0.5495 |
| A5 | Defense | 0.8408 | 0.4954 | 0.6151 | 0.5687 |

## Linear-model findings

### Offensive Geometry

Strong keep.

Geometry produced the first large improvement over the class-prior baseline.

It improved Log Loss from 1.0240 to 0.9161.

Geometry already provided predictive signal at the 10-second horizon.

### Motion

Weak positive.

Motion improved overall Log Loss only slightly:

0.9161 → 0.9119

Its value varied across horizons.

### Combat

Strong keep.

Combat produced the largest additional linear-model improvement:

0.9119 → 0.8359

Combat helped at every evaluated horizon.

### Economy

Drop for V0.

Adding current equipment value worsened probabilistic performance:

0.8359 → 0.8509

### Defense

Deferred after the linear experiment.

Defense improved classification metrics but slightly worsened Log Loss and Brier.

This motivated testing whether defensive information had nonlinear value.

---

# Why XGBoost?

Logistic Regression is a useful baseline because it is simple and interpretable, but it mainly represents linear relationships between features and the prediction score.

CS2 tactical states can contain interactions such as:

- T position × CT position
- player survival × spacing
- bomb location × defensive formation
- engagement distance × team geometry

These relationships may not be well represented by a linear model.

XGBoost was therefore introduced as the first nonlinear tabular baseline.

XGBoost builds many decision trees sequentially.

Each new tree focuses on reducing errors left by the previous ensemble.

The final prediction combines the contributions of many trees.

For V0, XGBoost is useful because:

- the dataset is tabular
- the dataset is relatively small
- nonlinear feature interactions are plausible
- tree models do not require neural-network-scale data
- it provides a strong comparison against Logistic Regression

No large hyperparameter search was performed.

A fixed conservative configuration was used to reduce the risk of tuning directly to the development OOF results.

---

# XGBoost Results

| Model | Features | Log Loss | Brier | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|
| XGB-A3 | Geometry + Motion + Combat | 0.8157 | 0.4880 | 0.6167 | 0.5876 |
| XGB-A4 | A3 + Economy | 0.8274 | 0.4958 | 0.6080 | 0.5786 |
| XGB-A5 | A3 + Defense | 0.8024 | 0.4785 | 0.6285 | 0.5996 |

XGB-A5 is the current best development model.

Compared with Logistic-A3:

Log Loss:

0.8359 → 0.8024

Brier:

0.4939 → 0.4785

Macro F1:

0.5541 → 0.5996

---

# Nonlinear Defense Finding

Defense did not improve probabilistic performance in Logistic Regression.

However:

XGB-A3 Log Loss:

0.8157

XGB-A5 Log Loss:

0.8024

Defense improved XGBoost at every tested horizon.

This supports the hypothesis that defensive information contains useful nonlinear or interaction-dependent predictive structure.

Economy did not recover under XGBoost and remains excluded from the current V0 feature set.

---

# Calibration

XGB-A5 top-label calibration:

ECE = 0.0454

Per-class ECE:

- A_PLANT: 0.0565
- B_PLANT: 0.0354
- NO_PLANT: 0.0432

Observed accuracy increased consistently with model confidence:

- confidence >= 0.50: 66.8%
- confidence >= 0.60: 72.7%
- confidence >= 0.70: 81.2%
- confidence >= 0.80: 88.6%
- confidence >= 0.90: 94.9%

The main calibration weakness was the 0.6–0.7 confidence range.

Mean confidence was approximately 0.646 while observed accuracy was 0.524.

The 30-second horizon also showed the largest calibration error.

V0 is reasonably calibrated as a research baseline, but it is not yet treated as deployment-calibrated.

---

# Error Structure

XGB-A5 confusion analysis showed:

- A → NO: 149
- B → NO: 81
- NO → A: 130
- NO → B: 44

Plant-vs-No-Plant errors:

404

A-vs-B site swaps:

67

Therefore:

85.8% of all classification errors crossed the Plant / No-Plant boundary.

Only 14.2% were direct A / B site swaps.

The dominant current problem is predicting whether a plant will occur.

---

# Hierarchical Task Analysis

## Plant vs No Plant

- Log Loss: 0.5740
- Brier: 0.1978
- Accuracy: 0.6909
- F1: 0.7066

## A vs B given True Plant

- Log Loss: 0.4490
- Brier: 0.1466
- Accuracy: 0.7845

Site choice becomes increasingly predictable at later eligible horizons:

- 10s: 76.5%
- 20s: 77.4%
- 30s: 80.5%
- 40s: 83.0%

Later-horizon populations are different because rounds that already planted or ended are excluded.

Therefore these values should not be interpreted as a causal longitudinal progression over an identical cohort.

---

# Group Permutation Importance

Destroying each feature family produced the following change in OOF Log Loss:

| Feature Group | Delta Log Loss |
|---|---:|
| Offensive Geometry | +0.2806 |
| Combat | +0.2315 |
| Defense | +0.0274 |
| Motion | +0.0112 |
| Horizon | -0.0002 |

Positive values mean model performance became worse after that information was destroyed.

These values are not percentages and should not be added together.

Feature groups can contain correlated or redundant information.

---

# Task-Specific Importance

## Plant vs No Plant

| Feature Group | Delta Log Loss |
|---|---:|
| Combat | +0.2287 |
| Offensive Geometry | +0.0287 |
| Defense | +0.0233 |
| Motion | +0.0084 |
| Horizon | ~0 |

Combat overwhelmingly dominates plant-occurrence prediction.

## A vs B Given True Plant

| Feature Group | Delta Log Loss |
|---|---:|
| Offensive Geometry | +0.4953 |
| Defense | +0.0081 |
| Combat | +0.0056 |
| Motion | +0.0054 |
| Horizon | ~0 |

Offensive Geometry overwhelmingly dominates site-choice prediction.

This supports treating plant feasibility and site selection as different tactical subproblems.

---

# Match-Level Robustness

XGB-A5 beat Logistic-A3 on:

- Log Loss: 13 / 20 held-out matches
- Brier: 13 / 20 held-out matches

Match-level Log Loss difference:

- mean: -0.0401
- median: -0.0285

Match-level Brier difference:

- mean: -0.0160
- median: -0.0141

The median improvement being negative indicates that the aggregate improvement is not explained only by a single unusually favorable match.

However, 7 of 20 matches still regressed.

Cross-match variability remains important.

---

# Match-Level Uncertainty

Paired match-level bootstrap with 100,000 resamples:

## Log Loss

Mean delta:

-0.0401

95% bootstrap interval:

[-0.0844, -0.0034]

Bootstrap fraction with mean delta below zero:

0.9855

## Brier

Mean delta:

-0.0160

95% bootstrap interval:

[-0.0355, +0.0024]

Bootstrap fraction with mean delta below zero:

0.9550

The Log Loss evidence is stronger because its bootstrap interval remains below zero.

The Brier interval narrowly crosses zero.

These are development-set uncertainty estimates and are not treated as final confirmatory test statistics.

---

# V0 Final Model

Current best development baseline:

XGB-A5

Features:

- Offensive Geometry
- Motion
- Combat
- Defense

Economy is excluded.

Metrics:

- Log Loss: 0.8024
- Brier: 0.4785
- Accuracy: 0.6285
- Macro F1: 0.5996
- A recall: 0.5711
- B recall: 0.4798
- No-Plant recall: 0.7207
- Top-label ECE: 0.0454

---

# V1 Research Hypothesis

V0 currently uses one three-class model:

State → A / B / No Plant

The V0 evidence suggests a candidate hierarchical architecture:

State
→ Plant feasibility
→ if Plant
→ Site choice

The motivation is evidence-based:

1. 85.8% of V0 errors cross the Plant / No-Plant boundary.
2. Plant occurrence is harder than conditional site prediction.
3. Combat dominates Plant / No-Plant prediction.
4. Offensive Geometry dominates A / B prediction.

This architecture is a V1 hypothesis.

It has not yet been demonstrated to outperform the V0 three-class model.

---

# Evaluation Status

V0 development evaluation is frozen.

Do not continue tuning XGB-A5 against the same 20-match OOF results and then treat those results as an unbiased final test.

Future model changes should ultimately be evaluated on additional untouched matches or an appropriately nested evaluation design.
