# V2 Independent Confirmation Model-Fitting Contract

Status: PRE-CONFIRMATION FREEZE

Purpose:
Define exactly how frozen V0 and frozen H2 are fitted on the legacy
20-match development corpus before either model is evaluated on D_confirm.

This contract is frozen before confirmation performance is inspected.

---

# 1. Scientific Question

The confirmation experiment asks:

Does the frozen H2 design generalize better than frozen V0 when both models
are fitted only on the legacy 20-match corpus and evaluated once on the
independent D_confirm corpus?

D_confirm must never participate in model fitting, calibration fitting,
feature selection, hyperparameter selection, or model redesign.

---

# 2. Training Corpus

Both V0 and H2 are fitted using only:

data/processed/v0_dataset_v2_timing_corrected.parquet

Frozen identity:

- 20 legacy Mirage matches
- 1,686 observations
- corrected 64 raw ticks / game-second timing
- true 10 / 20 / 30 / 40 second horizons

This corpus is D_legacy.

D_confirm is evaluation-only.

---

# 3. Frozen Labels

Multiclass target:

A_PLANT
B_PLANT
NO_PLANT

V0 directly predicts these three classes.

H2 decomposes the prediction into:

q = P(Plant | X)

r = P(A | Plant, X)

and composes:

P(A_PLANT)  = q * r
P(B_PLANT)  = q * (1 - r)
P(NO_PLANT) = 1 - q

---

# 4. Frozen V0 Feature Contract

V0 uses the existing XGB-A5 feature contract.

The feature set contains:

A1 + A2 + A3 + A5

A1 includes horizon_sec.

Total predictive features:

37

No feature may be added, removed, transformed, or retuned specifically for
D_confirm.

---

# 5. Frozen V0 Model Configuration

V0 uses XGBClassifier with:

n_estimators = 300
max_depth = 3
learning_rate = 0.03

min_child_weight = 5

subsample = 0.8
colsample_bytree = 0.8

reg_alpha = 0.5
reg_lambda = 5.0

objective = multi:softprob
num_class = 3
eval_metric = mlogloss

tree_method = hist

random_state = 42
n_jobs = -1

---

# 6. V0 Confirmation Fit

For the confirmation experiment:

1. Load all 1,686 D_legacy observations.
2. Select the frozen 37 V0 A5 features.
3. Fit one V0 multiclass XGBoost model on all D_legacy rows.
4. Do not use D_confirm for any fitting operation.
5. Apply the fitted model once to frozen D_confirm.

The previous V0 grouped OOF results remain development estimates.

The D_confirm result is the independent confirmation estimate.

---

# 7. Frozen H2 Plant Head

The H2 Plant Head inherits the V1-H1 Plant Head.

Target:

PLANT = 1 for A_PLANT or B_PLANT
NO_PLANT = 0

Frozen feature contract:

12 offensive geometry features
+ 4 motion features
+ 9 combat/state features
+ 11 defensive geometry features

horizon_sec is excluded.

Total Plant Head features:

36

---

# 8. Frozen H2 Site Head

The H2 Site Head is trained only on true-plant rows from D_legacy.

Target:

A_PLANT = 1
B_PLANT = 0

NO_PLANT rows do not participate in Site Head fitting.

Frozen feature contract:

12 offensive geometry features
+ 4 motion features
+ 11 defensive geometry features

Combat/state features are excluded.

horizon_sec is excluded.

Total Site Head features:

27

---

# 9. Frozen H2 Binary XGBoost Configuration

Both Plant Head and Site Head use:

n_estimators = 300
max_depth = 3
learning_rate = 0.03

min_child_weight = 5

subsample = 0.8
colsample_bytree = 0.8

reg_alpha = 0.5
reg_lambda = 5.0

objective = binary:logistic
eval_metric = logloss

tree_method = hist

random_state = 42
n_jobs = -1

No H2 hyperparameter may be changed after D_confirm is inspected.

---

# 10. Final Plant Head Fit

The final confirmation Plant Head is fitted on all D_legacy observations.

Training population:

all 1,686 legacy observations

Training target:

1 = eventual A_PLANT or B_PLANT
0 = NO_PLANT

The fitted model produces:

q = P(Plant | X)

for every D_confirm observation.

---

# 11. Site Calibration Training Population

Site calibration uses only true-plant D_legacy observations.

Legacy true-plant population:

A_PLANT + B_PLANT

The calibration procedure must never use:

- NO_PLANT rows as Site labels
- D_confirm features for fitting
- D_confirm labels
- D_confirm probabilities
- D_confirm model errors

---

# 12. Frozen H2 Calibration Procedure

The final confirmation calibrator is fitted using grouped out-of-fold
predictions generated entirely inside D_legacy.

Splitter:

StratifiedGroupKFold

Parameters:

n_splits = 4
shuffle = True
random_state = 42

Grouping unit:

demo_filename

Only true-plant legacy rows enter this calibration process.

For each inner fold:

1. train the frozen 27-feature Site Head on the other legacy groups
2. predict raw Site probabilities for the held-out legacy groups
3. store those predictions as calibration OOF predictions

After all four folds:

every calibration-training row must have a probability from a Site model
that did not train on that match.

---

# 13. Platt Calibration

Raw Site probability:

r_raw

First clip:

r_clip = clip(r_raw, 1e-6, 1 - 1e-6)

Then calculate:

z = log(r_clip / (1 - r_clip))

Fit:

LogisticRegression(
    max_iter = 2000,
    random_state = 42
)

Input:

z from grouped legacy OOF Site predictions

Target:

A_PLANT = 1
B_PLANT = 0

The fitted transform is conceptually:

r_calibrated = sigmoid(a * z + b)

The calibration coefficients are learned only from D_legacy.

---

# 14. Final Site Head Fit

After generating the grouped OOF predictions used for calibration:

fit one final Site Head on all true-plant D_legacy observations.

The final Site Head uses:

- frozen 27 features
- frozen binary XGBoost configuration

For D_confirm:

1. final Site Head produces r_raw
2. frozen Platt calibrator converts r_raw to r_calibrated

No calibration refitting occurs on D_confirm.

---

# 15. H2 Probability Composition

For each D_confirm observation:

q = final Plant Head probability

r = calibrated final Site Head probability

Then:

P(A_PLANT)  = q * r
P(B_PLANT)  = q * (1 - r)
P(NO_PLANT) = 1 - q

Required probability checks:

- every probability is finite
- every probability is between 0 and 1
- each probability row sums to 1 within numerical tolerance

---

# 16. D_confirm Feature Extraction

D_confirm must use the same frozen feature-generation implementation used by
the timing-corrected V0 pipeline.

The following may not change between D_legacy and D_confirm:

- timing contract
- observation horizons
- round eligibility logic
- snapshot resolution policy
- motion window
- bomb reconstruction logic
- geometric feature definitions
- combat/state feature definitions
- defensive feature definitions

Any incompatible new-demo parsing behavior must be treated as a data-quality
issue, not silently repaired after model performance is observed.

---

# 17. Model Artifact Freeze

The preferred workflow is:

D_legacy
→ fit V0
→ fit H2 Plant Head
→ generate H2 legacy grouped Site OOF probabilities
→ fit Platt calibrator
→ fit final H2 Site Head
→ save all model artifacts
→ record artifact hashes
→ freeze artifacts
→ evaluate D_confirm

This means confirmation models can be fitted before the full D_confirm corpus
is available.

That provides stronger separation between development and confirmation.

---

# 18. Confirmation Artifact Set

The final pre-confirmation artifact set should contain:

V0:
- frozen multiclass XGBoost model

H2:
- frozen Plant Head XGBoost model
- frozen Site Head XGBoost model
- frozen Platt calibration coefficients/model

Metadata:
- feature names
- model hyperparameters
- training row counts
- training match count
- package versions
- artifact SHA256 hashes
- training dataset identity/hash
- fitting timestamp
- git commit identity

These artifacts must not be regenerated because of D_confirm performance.

---

# 19. Confirmation Prediction Freeze

Once docs/v2_confirm_manifest.csv is frozen:

the prediction script must consume:

- frozen manifest
- frozen feature pipeline
- frozen V0 artifact
- frozen H2 artifacts

and produce one permanent prediction table.

No prediction row may be dropped because one model performs poorly on it.

Any technical exclusions must be defined independently of model outcome.

---

# 20. Confirmation Metrics

Primary:

Log Loss

Supporting probability metric:

Brier Score

Secondary diagnostics:

Accuracy
Macro F1
Macro Precision
Macro Recall
A_PLANT recall
B_PLANT recall
NO_PLANT recall

Both pooled and per-match analyses are required.

---

# 21. Paired Comparison

For each confirmation match m:

Delta_LL_m =
LL_H2,m - LL_V0,m

Delta_Brier_m =
Brier_H2,m - Brier_V0,m

Negative values favor H2.

Required match-level outputs:

match wins
equal-match mean delta
median delta
paired bootstrap interval

---

# 22. Forbidden Post-Confirmation Changes

After any V0-versus-H2 D_confirm score is inspected, the following may not
change while retaining the same confirmation claim:

- V0 features
- H2 Plant features
- H2 Site features
- XGBoost hyperparameters
- calibration type
- calibration folds
- calibration clipping
- target definition
- timing contract
- eligibility rules
- metric definitions
- D_confirm membership

A changed model becomes a new development candidate.

It requires future independent confirmation.

---

# 23. Interpretation

If H2 reproduces:

The result supports the claim that the H2 hierarchy/calibration design
generalizes beyond the legacy development corpus.

If H2 does not reproduce:

The result supports the conclusion that the V1 pooled improvement was not
sufficiently stable on independent matches.

Either outcome is scientifically useful.

---

# 24. Final Rule

D_confirm evaluates frozen models.

D_confirm does not help create them.
