# V1 Scientific Experiment Record

## Status

V1 scientific model-selection experiments are frozen.

Final decision:

**V0 XGB-A5 remains the selected development baseline.**

V1 produced useful structural findings, but no V1 candidate demonstrated a sufficiently robust improvement over V0 to justify replacing the frozen baseline.

All V1 results use the same 20 Mirage matches and therefore represent **development evidence**, not independent final generalization evidence.

---

# 1. Starting Point

The timing-corrected V0 baseline predicts:

- A_PLANT
- B_PLANT
- NO_PLANT

at true:

- 10 seconds
- 20 seconds
- 30 seconds
- 40 seconds

after `freeze_end`.

The frozen V0 XGB-A5 baseline uses 37 features:

- Offensive Geometry
- Motion
- Combat
- Defense

Economy is excluded.

Corrected V0 metrics:

| Metric | V0 XGB-A5 |
|---|---:|
| Log Loss | 0.838182 |
| Brier | 0.503190 |
| Accuracy | 0.594899 |
| Macro F1 | 0.566090 |
| A recall | 0.530435 |
| B recall | 0.434084 |
| NO recall | 0.703750 |

All comparisons in V1 reuse the frozen V0 match-to-fold assignment.

---

# 2. Why V1 Tested a Hierarchy

V0 diagnostics suggested that the three-class task contains two related but different prediction problems.

The decomposition is:

```text
State
  |
  +--> Will a plant happen?
  |
  +--> If a plant happens, A or B?
```

Define:

```text
q = P(Plant | X)

r = P(A | Plant, X)
```

Then the final three-class probabilities are:

```text
P(A)  = q * r

P(B)  = q * (1 - r)

P(NO) = 1 - q
```

The motivation came from the corrected V0 diagnostics.

For Plant vs No Plant:

* Combat was the strongest feature family.
* Geometry was useful but much weaker than Combat.

For A vs B among true plants:

* Offensive Geometry overwhelmingly dominated.
* Combat contributed approximately no useful site-choice signal.

This suggested that a single model may be asking one feature contract to solve two different tactical subproblems.

---

# 3. V1-H0 — Hierarchy Only

## Question

Does simply decomposing the three-class task into two separately trained heads improve prediction?

## Design

Two XGBoost heads were trained.

### Plant Head

Predict:

```text
Plant vs No Plant
```

Training population:

```text
all eligible observations
```

### Site Head

Predict:

```text
A vs B
```

Training population:

```text
true-plant training observations only
```

Both heads used the same 37-feature V0 XGB-A5 feature contract.

The XGBoost hyperparameters and frozen outer folds were unchanged.

## Result

Full H0:

| Metric    |       H0 |
| --------- | -------: |
| Log Loss  | 0.852546 |
| Brier     | 0.504516 |
| Accuracy  | 0.592527 |
| Macro F1  | 0.566455 |
| A recall  | 0.549565 |
| B recall  | 0.443730 |
| NO recall | 0.681250 |

Compared with V0:

```text
Log Loss:
0.838182 -> 0.852546
worse

Brier:
0.503190 -> 0.504516
worse
```

H0 did not improve the primary probabilistic metrics.

## Head diagnostics

Plant Head:

```text
Log Loss = 0.589191
Brier    = 0.204347
Accuracy = 0.672005
F1       = 0.699946
```

Site Head:

```text
Log Loss = 0.501147
Brier    = 0.163425
Accuracy = 0.762980
```

The dedicated Plant Head was competitive.

The main weakness was the probability quality of the Site Head.

## Decision

**Reject H0 as a V0 replacement.**

Do not conclude that hierarchy itself is invalid.

The architecture-only experiment showed that merely splitting the task was not enough.

---

# 4. V1-H1 — Feature-Specialized Hierarchy

## Question

Should the two heads receive different feature contracts based on the V0 task-specific importance evidence?

## Feature contracts

### Plant Head

36 features:

```text
Offensive Geometry
+ Motion
+ Combat
+ Defense
```

`horizon_sec` was removed.

### Site Head

27 features:

```text
Offensive Geometry
+ Motion
+ Defense
```

The Site Head omitted:

```text
Combat
horizon_sec
```

This was motivated by the finding that Combat strongly helped Plant vs No Plant but contributed almost no useful A/B signal.

## Result

Full H1:

| Metric    |       H1 |
| --------- | -------: |
| Log Loss  | 0.840649 |
| Brier     | 0.499879 |
| Accuracy  | 0.590154 |
| Macro F1  | 0.564673 |
| A recall  | 0.542609 |
| B recall  | 0.450161 |
| NO recall | 0.678750 |

Compared with H0:

```text
Log Loss:
0.852546 -> 0.840649

Brier:
0.504516 -> 0.499879
```

Feature specialization clearly improved the hierarchical model.

Compared with V0:

```text
Log Loss:
V0 = 0.838182
H1 = 0.840649
H1 slightly worse

Brier:
V0 = 0.503190
H1 = 0.499879
H1 better
```

H1 therefore produced a mixed result.

---

# 5. H1 Diagnosis

The hierarchical three-class Log Loss can be decomposed exactly as:

```text
LL_full
=
LL_plant
+
P(true plant) * LL_site_given_true_plant
```

For this dataset:

```text
n_total = 1686
n_plant = 886
```

For H1:

```text
0.589449
+
(886 / 1686) * 0.478017
=
0.840649
```

This identifies where the remaining Log Loss came from.

Plant Head:

```text
LL = 0.589449
```

Site Head:

```text
LL = 0.478017
Brier = 0.156244
Accuracy = 0.758465
```

The Site Head classification accuracy was already strong.

Its probability quality was weaker than desired.

This motivated one predeclared targeted experiment:

**calibrate Site Head probabilities without changing anything else.**

---

# 6. V1-H2 — Nested Site-Head Calibration

## Question

Can calibration improve the Site Head probability quality while preserving the same underlying Site classifier?

## Method

H2 changed only one component:

```text
Site Head probability calibration
```

Everything else remained unchanged:

* same 20 matches
* same frozen outer folds
* same Plant Head
* same 27 Site features
* same XGBoost configuration
* same training population

Nested Platt calibration was used.

Inside every outer fold:

```text
outer training matches
        |
        v
inner grouped CV
        |
        v
inner OOF Site probabilities
        |
        v
fit Platt calibrator
        |
        v
train Site XGB on all outer training plants
        |
        v
predict outer test
        |
        v
apply calibrator
```

The outer test fold never participated in calibration fitting.

## Isolation check

The uncalibrated H2 Site predictions reproduced H1 exactly:

```text
max difference = 0
```

Therefore calibration was the only experimental variable.

## Site Head result

Before calibration:

```text
Log Loss = 0.478017
Brier    = 0.156244
Accuracy = 0.758465
```

After calibration:

```text
Log Loss = 0.471829
Brier    = 0.154054
Accuracy = 0.758465
```

Calibration improved probability quality without changing the underlying Site classifier accuracy.

The learned calibration slopes were approximately:

```text
0.71 - 0.78
```

across folds.

Because these slopes were below one, the raw Site Head probabilities showed evidence of overconfidence.

---

# 7. H2 Full Three-Class Result

| Metric    |       V0 |       H2 |   H2 - V0 |
| --------- | -------: | -------: | --------: |
| Log Loss  | 0.838182 | 0.837397 | -0.000785 |
| Brier     | 0.503190 | 0.499781 | -0.003410 |
| Accuracy  | 0.594899 | 0.587189 | -0.007711 |
| Macro F1  | 0.566090 | 0.555824 | -0.010266 |
| A recall  | 0.530435 | 0.523478 | -0.006957 |
| B recall  | 0.434084 | 0.411576 | -0.022508 |
| NO recall | 0.703750 | 0.701250 | -0.002500 |

For the first time in V1, a hierarchical model slightly beat V0 on pooled OOF Log Loss.

It also improved Brier.

However, the Log Loss improvement was extremely small and classification metrics regressed.

Therefore H2 required match-level robustness analysis before it could be considered a new baseline.

---

# 8. H2 Match-Level Robustness

H2 vs V0 across 20 held-out matches:

```text
H2 better Log Loss:
9 / 20 matches

H2 better Brier:
12 / 20 matches
```

Equal-match statistics:

```text
Mean Δ Log Loss:
-0.001852

Median Δ Log Loss:
+0.007943
```

The negative mean indicates a small average H2 improvement.

The positive median means the typical match actually favored V0.

For Brier:

```text
Mean Δ Brier:
-0.004880

Median Δ Brier:
-0.007202
```

Brier was somewhat more favorable to H2.

---

# 9. H2 Paired Match Bootstrap

100,000 paired match-level bootstrap samples were used.

This gives every match equal weight.

It is not interpreted as a formal confirmatory p-value.

## Log Loss

```text
Mean delta:
-0.001852

95% bootstrap interval:
[-0.016685, +0.012170]

Bootstrap fraction with mean delta < 0:
0.5918
```

The interval crosses zero substantially.

Evidence for a stable H2 Log Loss improvement is weak.

## Brier

```text
Mean delta:
-0.004880

95% bootstrap interval:
[-0.013676, +0.003981]

Bootstrap fraction with mean delta < 0:
0.8621
```

Brier evidence is more favorable, but the interval still crosses zero.

## Hierarchy conclusion

H2 is a **promising development candidate**, but it does not provide sufficiently robust evidence to replace V0.

The hierarchical branch was frozen here.

No H3, H4, or repeated calibration variants were attempted.

This avoids repeatedly tuning against the same 20 development matches.

---

# 10. V1 Temporal Hypothesis

V0 contains current state plus a one-second motion estimate.

A separate V1 question was:

**Does longer temporal history provide useful information that is missing from the current-state representation?**

Before training a temporal model, the demo timing support was audited.

Candidate history windows were frozen as:

```text
1 second
3 seconds
5 seconds
```

---

# 11. Temporal Timing Audit

For all 1686 observations:

### 1-second history

```text
1686 / 1686 exact
```

### 3-second history

```text
1686 / 1686 exact
```

### 5-second history

```text
1685 / 1686 exact
```

One five-second history snapshot was missing at the exact requested tick.

The nearest available snapshots were:

```text
-1 tick
+1 tick
```

At 64 ticks per second:

```text
1 tick = 0.015625 seconds
```

The frozen V1 temporal history policy became:

```text
1. Prefer exact snapshot.
2. Otherwise permit at most 1 raw tick offset.
3. If both sides are equally close, prefer the earlier snapshot.
```

The single non-exact observation therefore used:

```text
5.015625 seconds
```

of history.

The evaluation population remained exactly 1686 observations.

---

# 12. V1-T0 Temporal Feature Dataset

The temporal dataset preserved all 37 frozen V0 features.

It added 34 temporal features:

```text
17 state variables
x
2 longer windows
=
34 temporal delta features
```

Windows:

```text
3 seconds
5 seconds
```

State variables included:

* T alive count
* CT alive count
* T health
* CT health
* T armor
* CT armor
* T centroid X/Y
* CT centroid X/Y
* T spread
* CT spread
* bomb X/Y
* bomb-to-T-centroid distance
* T-to-CT centroid distance
* minimum T-to-CT distance

Each delta had the form:

```text
delta_W
=
state_now
-
state_(W seconds ago)
```

The final T0 model therefore used:

```text
37 V0 features
+
34 temporal features
=
71 features
```

No hyperparameters were changed.

The same frozen outer folds were reused.

---

# 13. V1-T0 Temporal Baseline Result

| Metric    |       V0 |       T0 |   T0 - V0 |
| --------- | -------: | -------: | --------: |
| Log Loss  | 0.838182 | 0.844859 | +0.006677 |
| Brier     | 0.503190 | 0.507984 | +0.004793 |
| Accuracy  | 0.594899 | 0.593120 | -0.001779 |
| Macro F1  | 0.566090 | 0.563414 | -0.002676 |
| A recall  | 0.530435 | 0.549565 | +0.019130 |
| B recall  | 0.434084 | 0.421222 | -0.012862 |
| NO recall | 0.703750 | 0.691250 | -0.012500 |

T0 was worse than V0 on both primary probabilistic metrics.

## By true horizon

| Horizon |   N | Log Loss |  Brier | Accuracy | Macro F1 |
| ------- | --: | -------: | -----: | -------: | -------: |
| 10s     | 442 |   0.8915 | 0.5374 |   0.5928 |   0.5725 |
| 20s     | 442 |   0.8632 | 0.5201 |   0.5769 |   0.5415 |
| 30s     | 429 |   0.8234 | 0.4934 |   0.6037 |   0.5689 |
| 40s     | 373 |   0.7925 | 0.4756 |   0.6005 |   0.5725 |

The longer-history tabular representation did not produce a convincing predictive improvement.

---

# 14. Why V1 Did Not Proceed to a GRU

A GRU or another sequence neural network would introduce substantially more model complexity.

The predeclared gate was:

```text
First demonstrate that longer temporal information itself
adds useful predictive signal with a simple baseline.
```

T0 failed this gate.

Therefore V1 does not proceed to:

* GRU
* LSTM
* Transformer
* temporal convolution
* larger sequence architecture

This is intentional.

A more complicated model should not be introduced merely because it is more sophisticated.

The current evidence does not justify it.

---

# 15. Final V1 Model-Selection Decision

The principal candidates were:

| Model     |     Log Loss |        Brier |     Accuracy |     Macro F1 |
| --------- | -----------: | -----------: | -----------: | -----------: |
| V0 XGB-A5 | **0.838182** |     0.503190 | **0.594899** | **0.566090** |
| V1-H0     |     0.852546 |     0.504516 |     0.592527 |     0.566455 |
| V1-H1     |     0.840649 |     0.499879 |     0.590154 |     0.564673 |
| V1-H2     | **0.837397** | **0.499781** |     0.587189 |     0.555824 |
| V1-T0     |     0.844859 |     0.507984 |     0.593120 |     0.563414 |

H2 achieved the best pooled OOF Log Loss and Brier.

However:

* the Log Loss gain over V0 was only 0.000785
* H2 won Log Loss on only 9 of 20 matches
* median match-level ΔLL favored V0
* the 95% equal-match bootstrap interval crossed zero
* classification metrics regressed
* all H0/H1/H2 development decisions reused the same 20 matches

Therefore:

**V0 XGB-A5 remains the selected development baseline.**

H2 is preserved as a promising V1 research result, not promoted to the production/development baseline.

---

# 16. What V1 Taught Us

V1 produced several useful scientific findings even though it did not replace V0.

## Finding 1

The three-class task can meaningfully be decomposed into:

```text
Plant feasibility
+
conditional Site choice
```

The decomposition is mathematically clean and diagnostically useful.

## Finding 2

Feature specialization matters.

Removing Combat from the Site Head substantially improved the hierarchical Site prediction relative to H0.

## Finding 3

Site classification and Site probability quality are different problems.

The Site Head had strong accuracy while still producing suboptimal Log Loss.

## Finding 4

Nested calibration improved Site probability quality.

The improvement was real under leakage-safe outer evaluation.

## Finding 5

A tiny pooled OOF improvement is not sufficient evidence by itself.

Match-level robustness changed the interpretation of H2.

## Finding 6

Longer temporal history is not automatically useful.

Adding 3-second and 5-second hand-designed state deltas increased feature count from 37 to 71 but worsened the primary metrics.

## Finding 7

More model complexity is not automatically progress.

The temporal baseline result provided a principled reason not to escalate to a sequence neural network.

---

# 17. Current Scientific Status

```text
V0 timing-corrected baseline       FROZEN
V1 H0 hierarchy                    REJECTED
V1 H1 specialized hierarchy        PROMISING
V1 H2 calibrated hierarchy         PROMISING / NOT ROBUST ENOUGH
V1 T0 temporal baseline            REJECTED
GRU / sequence model               NOT JUSTIFIED
Selected development baseline      V0 XGB-A5
```

The next scientifically meaningful improvement should preferably involve new information or new data rather than repeated tuning against the same 20 matches.

Examples include:

* additional untouched Mirage matches
* richer tactical state representation
* map-aware semantic position features
* player-role information
* event-sequence information with a clear prior hypothesis
* eventually partial-observation / player-view modeling

---

# 18. Evidence Boundaries

These experiments support predictive development conclusions only.

They do not establish:

* causality
* optimal tactical recommendations
* player-side deployability
* generalization to all CS2 maps
* generalization to all competitive populations

The current data uses full observer / replay information on Mirage.

Later horizons are survivor-conditioned because rounds that have already planted or ended are not eligible.

Runtime artifacts are engineering products and are not independent scientific evidence.

Synthetic parity tests establish implementation consistency, not real-source correctness or generalization.

---

# 19. V1 Freeze Conclusion

V1 did not produce a sufficiently robust successor to V0.

That is a valid research outcome.

The project learned:

```text
what decomposition helps,
what feature specialization helps,
what calibration helps,
what temporal representation did not help,
and where the remaining uncertainty lies.
```

V0 remains the baseline.

V1 is preserved as an auditable sequence of controlled experiments rather than hidden failed tuning attempts.
