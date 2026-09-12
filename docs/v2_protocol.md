# V2 Scientific Protocol

## Status

**Pre-result experimental protocol.**

This document defines the V2 scientific rules before new-match model evaluation begins.

The purpose of V2 is not simply to obtain a lower score than V0 or V1.

The purpose is to answer two harder questions:

1. Which V1 findings reproduce on genuinely new matches?
2. Which new tactical information improves generalization beyond the frozen V0 baseline?

The central V2 research question is:

> **Can independently validated tactical information improve generalization beyond the frozen V0 model?**

This protocol must be frozen before V2 confirmation results are inspected.

If the protocol later needs to change, the change must be recorded explicitly with:

- date
- reason
- affected experiment
- whether any relevant evaluation result had already been viewed

A result-dependent rule change must never be silently presented as if it had been preregistered.

---

# 1. Scientific Starting Point

V2 begins from the completed V0 and V1 research record.

## 1.1 Selected baseline

The selected development baseline remains:

```text
V0 XGB-A5

Corrected V0 metrics on the legacy 20-match grouped-OOF development corpus:

Metric	V0 XGB-A5
Log Loss	0.838182
Brier	0.503190
Accuracy	0.594899
Macro F1	0.566090
A recall	0.530435
B recall	0.434084
NO recall	0.703750

The selected V0 feature contract contains 37 features:

Offensive Geometry
+ Motion
+ Combat
+ Defense

Economy remains excluded.

1.2 V1 findings carried into V2

V1 tested:

H0
hierarchy only

H1
feature-specialized hierarchy

H2
nested Site Head calibration

T0
3s / 5s temporal-delta baseline

The strongest V1 candidate was H2.

H2 pooled development metrics:

Metric	H2
Log Loss	0.837397
Brier	0.499781
Accuracy	0.587189
Macro F1	0.555824

However, H2 did not demonstrate sufficient match-level robustness:

H2 LL match wins:      9 / 20

Equal-match mean ΔLL:
-0.001852

Median ΔLL:
+0.007943

95% paired match bootstrap interval:
[-0.016685, +0.012170]

where:

ΔLL = LL(H2) - LL(V0)

Negative favors H2.

Therefore:

H2 is a promising development hypothesis, not a selected replacement for V0.

T0 also failed its temporal-information gate:

V0 LL = 0.838182
T0 LL = 0.844859

V0 Brier = 0.503190
T0 Brier = 0.507984

Therefore no sequence model was promoted in V1.

2. V2 Scientific Scope

V2 retains the existing prediction task.

Target:

Y ∈ {
    A_PLANT,
    B_PLANT,
    NO_PLANT
}

Map:

Mirage

Information scope:

full observer / replay information

V2 does not claim a hidden-information player-side predictor.

The prediction remains:

P(
    eventual plant outcome
    |
    state available at observation time,
    round still active,
    bomb not yet planted
)

Observation horizons remain:

10 seconds
20 seconds
30 seconds
40 seconds

after freeze_end.

V2 does not redefine the label semantics in order to improve scores.

3. V2 Research Questions

V2 is organized around four research questions.

RQ1 — Independent H2 confirmation

Does the frozen H2 hierarchical model reproduce its V1 probability-quality advantage on genuinely new matches?

This is the first V2 scientific question.

It must be answered before H2 is retuned using new confirmation data.

RQ2 — Map-semantic information

Does a map-aware tactical representation provide predictive information beyond the frozen V0 raw-state representation?

This produces the proposed new feature family:

A6 = Map Semantics

The first A6 experiment must isolate the value of new information rather than changing model architecture at the same time.

RQ3 — Utility and event semantics

After static map semantics are understood, do utility deployment and tactical events provide additional predictive information?

This branch is gated behind the earlier semantic-state experiment.

RQ4 — Path dependence and sequence modeling

Does round history contain useful information that is not recoverable from the current state and simple temporal summaries?

Sequence models are not assumed to be useful.

They must earn their place through an explicit temporal-information gate.

4. Evidence Hierarchy

V2 distinguishes three levels of evidence.

Level 1 — Development evidence

Used to:

generate hypotheses
design features
compare candidate representations
diagnose model failures
tune development models

Development evidence does not independently confirm a hypothesis that was created using the same data.

Level 2 — Independent confirmation

Used to test a candidate that was frozen before the confirmation data were evaluated.

The candidate may not be modified using the same confirmation result and then re-evaluated as if the test were still untouched.

Level 3 — External validation

Future work may evaluate:

different tournaments
different teams
different time periods
different CS2 patches
different maps
different player populations

External validation is outside the initial V2 scope but must remain distinct from ordinary cross-validation.

5. Data Roles

V2 will not treat all demos as interchangeable.

Every match must have an explicit scientific role.

5.1 Legacy Development Corpus

The existing 20 Mirage matches are designated:

D_legacy

Role:

legacy development corpus

These matches were already used to:

develop V0
diagnose V0
motivate hierarchy
evaluate H0
design H1
design H2
evaluate T0

Therefore they must not be described as an untouched final test set for V2.

They remain valid for:

implementation testing
feature development
exploratory diagnostics
reproducibility
model debugging
legacy comparisons
5.2 Batch A — Independent H2 Confirmation

The first new confirmation dataset is designated:

D_confirm_A

Its initial purpose is narrow:

Test frozen V0 versus frozen H2 without using Batch A to redesign either candidate.

Batch A must not be selected based on model predictions or final labels.

Inclusion and exclusion must follow predeclared technical rules.

5.3 V2 Development Pool

After the Batch A H2 confirmation result is frozen and reported, Batch A may later join a broader development pool.

That pool is designated:

D_v2_dev

It may be used for:

A6 development
semantic feature experiments
utility/event experiments
error analysis
future temporal diagnostics

Once Batch A has been inspected, it is no longer an untouched confirmation set for later V2 model-selection claims.

5.4 Final V2 Holdout

A later new-match batch must be reserved as:

D_v2_holdout

This dataset must remain untouched until:

the final V2 candidate is frozen
the baseline is frozen
the feature contract is frozen
all preprocessing is frozen
all model hyperparameters are frozen
the evaluation metrics are frozen
the promotion rule is frozen

If the holdout result is inspected and then causes the model or rules to change, that holdout becomes development evidence.

A new untouched holdout would then be required for independent confirmation.

6. New-Match Acquisition Policy

New data must be collected using a reproducible acquisition policy.

A match may not be included or excluded because its model result looks favorable or unfavorable.

Every acquired demo must be recorded in a manifest.

Required manifest fields include at least:

demo_id
filename
sha256
source
source_url_or_reference
download_date
match_date
event
map
team_a
team_b
final_score
parser_version
raw_file_size
round_count
parse_status
timing_status
scientific_role
exclusion_reason

Additional metadata may be added.

6.1 Duplicate prevention

Every demo must receive a SHA256 hash.

Identical hashes must not appear as independent matches.

Different filenames with identical content count as one demo.

6.2 Inclusion rules

Initial V2 confirmation data must satisfy:

CS2
Mirage
complete demo file
successful parser load
usable round metadata
usable player-state data
timing semantics auditable

Matches should be acquired according to the source-selection rule, not according to model behavior.

6.3 Exclusion rules

Technical exclusions may include:

corrupted demo
incomplete download
unsupported parser structure
missing required round metadata
timing semantics that cannot be validated
duplicate content
wrong map
non-CS2 demo

Every exclusion must be logged.

The following are not valid exclusion reasons:

model performs badly
unexpected class distribution
unusual strategy
unfamiliar team
H2 loses on this match
V0 loses on this match

Difficult matches are evidence.

They are not data errors.

7. Batch Size Freeze Rule

The final Batch A target size will be frozen after the V2-1 source-availability audit and before any Batch A model result is computed.

The source audit is allowed to inspect:

available match count
demo accessibility
file integrity
map identity
parser compatibility
match/date/team metadata

The source audit must not inspect:

V0 prediction performance
H2 prediction performance
H2 versus V0 deltas

The Batch A size decision must therefore be based on data availability and scientific practicality, not observed model results.

The protocol amendment that freezes the final Batch A size must be committed before H2 confirmation evaluation begins.

8. Timing Contract

The V0 timing correction remains scientifically binding.

The scientific unit is:

true game seconds

not parser constructor defaults.

Observation targets remain:

freeze_end + 10 seconds
freeze_end + 20 seconds
freeze_end + 30 seconds
freeze_end + 40 seconds
8.1 Tick-rate policy

Future demos must not blindly inherit a parser default tick rate.

Timing must be validated against demo time information.

The legacy corpus measured:

64 raw ticks / game-second

For future demos:

measure or validate timing semantics
record the measured result
use the validated mapping
do not silently force an incompatible demo into the 64-tick contract

If a future demo uses different timing semantics, an explicit timing adapter must be validated before scientific inclusion.

8.2 Observation resolver

The observation resolver must preserve the V0 scientific meaning:

Resolve the first valid state at or immediately after the requested true-time target according to the frozen lateness policy.

The resolved observation must occur before:

plant
round_end

The round must still be eligible at the resolved observation.

8.3 Temporal windows

Any future motion or history window must represent actual game seconds.

A requested:

1 second
3 seconds
5 seconds

must not be implemented using an unverified nominal tick conversion.

History-resolution deviations must be recorded explicitly.

9. Label Contract

The label remains determined by the eventual round outcome.

A plant  -> A_PLANT
B plant  -> B_PLANT
no plant -> NO_PLANT

A row may only exist when the prediction point occurs before plant and before round end.

No post-observation information may enter predictor features.

Future labels may be used for supervised training.

Future labels may not be used to route a live prediction path.

10. Frozen Models for Batch A

Batch A compares two candidates only:

Frozen V0
Frozen H2

No Batch A tuning is permitted before the primary comparison is frozen.

11. Frozen V0 Confirmation Model

For independent Batch A evaluation, V0 will be trained on all eligible D_legacy observations using the frozen V0 feature and model contract.

Training population:

1,686 legacy observations

Feature count:

37

Feature families:

Offensive Geometry
Motion
Combat
Defense

Frozen XGBoost configuration:

{
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}

The resulting model is evaluated directly on Batch A.

Batch A is not used to fit the model.

12. Frozen H2 Confirmation Model

H2 retains the V1 hierarchical structure.

Define:

q = P(Plant | X)

r = P(A | Plant, X)

Final probabilities:

P(A)  = q * r
P(B)  = q * (1 - r)
P(NO) = 1 - q
12.1 Plant Head

Training population:

all eligible D_legacy observations

Target:

Plant = 1
No Plant = 0

Feature count:

36

Feature families:

Offensive Geometry
Motion
Combat
Defense

horizon_sec remains excluded.

12.2 Site Head

Training population:

true-plant D_legacy observations only

Target:

A = 1
B = 0

Feature count:

27

Feature families:

Offensive Geometry
Motion
Defense

Excluded:

Combat
horizon_sec
12.3 H2 calibration fitting

The Batch A H2 calibrator must be fitted using legacy data only.

Procedure:

D_legacy true-plant rows
        ↓
frozen grouped folds
        ↓
generate legacy OOF raw Site probabilities
        ↓
convert probabilities to clipped logits
        ↓
fit Platt logistic calibrator
        ↓
refit base Site Head on all legacy true-plant rows
        ↓
refit Plant Head on all legacy rows
        ↓
evaluate frozen pipeline on Batch A

Batch A labels must not be used to fit:

Plant Head
Site Head
calibration slope
calibration intercept
13. Batch A Primary Evaluation

The scientific comparison is paired by match.

Every eligible Batch A observation receives:

V0 probability vector
H2 probability vector
true label
match_id
horizon

Primary probability metrics:

Log Loss
Brier Score

Secondary metrics:

Accuracy
Macro F1
Macro Precision
Macro Recall
A recall
B recall
NO recall

Supporting analyses:

per-horizon metrics
per-match metrics
paired match differences
bootstrap uncertainty
14. Two Primary Estimands

V2 explicitly distinguishes two scientific quantities.

14.1 Observation-weighted estimand

All eligible observations receive equal weight.

This answers:

For a randomly selected eligible observation from this evaluation population, which model gives better probability forecasts?

Reported as pooled:

Log Loss
Brier
14.2 Match-weighted estimand

Each match contributes one match-level score.

This answers:

For a randomly selected match in this evaluation population, which model tends to perform better when every match receives equal weight?

Reported using:

mean paired match delta
median paired match delta
match win count
paired match bootstrap interval

These estimands are related but not identical.

Neither should be silently substituted for the other.

15. Batch A Paired Differences

Define:

ΔLL_m =
LL(H2 on match m)
-
LL(V0 on match m)

and:

ΔBrier_m =
Brier(H2 on match m)
-
Brier(V0 on match m)

Interpretation:

negative -> H2 better
zero     -> tie
positive -> V0 better

The comparison must remain paired by match.

16. Batch A Bootstrap

Use paired match-level bootstrap.

Sampling unit:

match

not observation row.

For each bootstrap replicate:

sample Batch A matches with replacement
retain the paired H2 and V0 score difference for each sampled match
compute the equal-match mean difference

Default number of replicates:

100,000

Report:

mean observed delta
median observed delta
95% percentile bootstrap interval
fraction of bootstrap mean deltas < 0

The bootstrap fraction is not to be described as a formal p-value.

17. H2 Promotion Gate

H2 is not promoted merely because one pooled metric is numerically lower.

A Batch A promotion requires all of the following:

Required probability evidence
1. pooled Log Loss:
   H2 < V0

2. equal-match mean Log Loss:
   H2 < V0

3. Log Loss match win rate:
   H2 improves on more than 50% of Batch A matches

4. paired equal-match Log Loss bootstrap interval:
   upper bound <= 0

5. pooled Brier:
   H2 <= V0

6. equal-match mean Brier:
   H2 <= V0
Secondary classification safeguard

Accuracy and Macro F1 are not the primary optimization targets.

However, H2 must not show a large unexplained degradation in threshold classification behavior.

A provisional large-degradation flag is:

absolute Accuracy drop > 0.02
or
absolute Macro F1 drop > 0.02

Such a degradation blocks automatic promotion and requires scientific review.

This threshold must not be relaxed after viewing Batch A results merely to enable promotion.

Insufficient-evidence outcome

If the required H2 promotion gate is not met:

H2 is not promoted.

Possible interpretations include:

no reproducible H2 advantage
insufficient sample size
heterogeneous match behavior
development-specific V1 gain

The result must be recorded rather than tuned away.

18. Batch A Stop Rule

After the preregistered Batch A result is computed:

do not retune H2 on Batch A
and then call the same Batch A independent confirmation.

If H2 is modified after Batch A inspection:

Batch A becomes development evidence.

A new confirmation batch would be required for the modified H2.

19. V2 Map-Semantic Branch

After the Batch A confirmation record is frozen, V2 moves to new-information research.

The first new information family is:

A6 = Map Semantics

The hypothesis is:

Tactical map semantics may represent site intent and map control more directly than raw geometry alone.

20. A6 Version 0 Feature Classes

Initial A6 should remain interpretable.

It may contain five feature categories.

20.1 Zone occupancy

Examples:

t_a_ramp_count
t_palace_count
t_mid_count
t_connector_count
t_short_count
t_b_apps_count

ct_a_site_count
ct_connector_count
ct_mid_count
ct_short_count
ct_b_site_count
ct_market_count

Exact zone definitions must be versioned.

20.2 Bomb semantics

Examples:

bomb_zone
bomb_a_side_indicator
bomb_b_side_indicator
bomb_distance_to_a_route
bomb_distance_to_b_route

Bomb semantics must remain distinguishable from generic player geometry.

20.3 Route pressure

Initial route-pressure features must be explicit and interpretable.

Do not begin A6 with a separately learned hidden tactical score.

Possible components may include:

T count on A-access zones
T count on B-access zones
bomb route alignment
distance to route entry
20.4 Map control

Potential first-version indicators include:

t_mid_control
ct_mid_control
t_a_access_count
t_b_access_count
a_split_possible
b_split_possible

The exact definition of each indicator must be documented before model evaluation.

20.5 Semantic distribution

Potential features include:

number_of_t_zones_occupied
number_of_ct_zones_occupied
t_zone_entropy
ct_zone_entropy

Entropy represents semantic spatial dispersion rather than raw Euclidean spread.

21. A6 Annotation Rules

Map-zone definitions must be generated independently of outcome labels.

Zone boundaries may use:

map geometry
established location names
explicit coordinate regions
manually audited tactical regions

Zone definitions must not be adjusted after seeing which boundaries improve model performance unless the experiment is explicitly reclassified as development work.

Every semantic-map version must receive a version identifier.

Example:

mirage_semantic_map_v0
mirage_semantic_map_v1

Changes must be recorded.

22. First A6 Controlled Experiment

The first A6 experiment must compare:

V0 XGB-A5

against:

V0 XGB-(A5 + A6)

while holding constant:

data population
target
timing
folds
XGBoost configuration
evaluation metrics

The only intended experimental change is:

new semantic information

This experiment answers:

Does A6 itself provide predictive value?

It does not yet answer whether hierarchy should use A6 differently.

23. A6 Evaluation

Primary metrics:

Log Loss
Brier

Supporting:

Accuracy
Macro F1
per-horizon metrics
per-match metrics
feature-family permutation diagnostics

A6 should also be diagnosed separately for:

Plant vs No Plant

and:

A vs B among true plants

because V1 indicates that the two subproblems depend on different information.

24. Hierarchical Semantic Experiment

Only if A6 demonstrates useful signal should V2 test semantic specialization inside the hierarchy.

The first hierarchical semantic hypothesis should be:

Map semantics may benefit the Site Head more strongly than the Plant Head.

The initial comparison should preserve the H1/H2 contracts as much as possible.

For example:

Plant Head:
existing H1 feature contract

Site Head:
existing H1 Site contract
+ A6 semantics

This experiment must not simultaneously introduce:

a new neural architecture
new temporal windows
new utility features
new calibration family

unless explicitly defined as a later experiment.

25. Utility and Event Branch

Utility and event semantics are a separate information family.

Potential future inputs include:

smoke deployment
molotov deployment
flash deployment
HE usage
bomb drop
bomb pickup
kill events
damage events
utility commitment by site
recent rotation events

These features must not be mixed into the first A6 experiment.

The research sequence should remain:

raw state
    ↓
map semantics
    ↓
utility/event semantics
    ↓
temporal sequence representation

This preserves interpretability of improvements.

26. Temporal Branch

T0 already tested simple temporal deltas.

Its failure means:

Longer handcrafted 3s/5s deltas did not justify sequence-model escalation on the V1 development corpus.

This does not prove that all temporal information is useless.

However, V2 sequence modeling remains evidence-gated.

27. Path-Dependence Gate

A sequence model should be reopened only if V2 identifies evidence that:

P(Y | current state, history)

contains useful information beyond:

P(Y | current state)

Possible evidence includes:

event-order effects
rotations with similar final positions but different origins
fake-versus-direct-execute histories
utility commitment sequences
bomb-route history
state trajectories not captured by current features
28. Sequence Escalation Order

If the temporal gate is passed, complexity should increase gradually.

Preferred order:

simple temporal/event pooling
        ↓
GRU baseline
        ↓
TCN comparison
        ↓
Transformer only if justified

A Transformer is not the default next step.

Sequence complexity must be justified by:

information evidence
adequate data volume
adequate independent match diversity
29. Data Diversity

V2 must track more than row count.

At minimum report:

number of matches
number of rounds
number of eligible observations
unique teams
event/tournament count
date span
class distribution
horizon distribution

Where practical also report:

team overlap between development and confirmation
event overlap
time-period overlap

More observations from the same matches do not provide the same scientific value as more independent matches.

30. Effective Sample Size Principle

The nominal observation count is not the same as the number of independent experimental units.

Rows within one match share:

players
teams
strategies
opponent
match context
event environment

Therefore cross-match diversity is a central V2 resource.

V2 must avoid presenting:

N observations

as if all observations were statistically independent matches.

31. Multiple-Comparison Discipline

Every major V2 experiment must have:

experiment_id
question
hypothesis
single intended change
frozen comparison
result
interpretation
decision
next gate

Unsuccessful experiments must not be silently deleted.

Repeatedly testing many variants and reporting only the best one creates development-set optimism.

32. Experiment Ledger

V2 will maintain an experiment ledger.

Suggested schema:

ID	Question	Change	Data role	Primary metric	Result	Decision

Example IDs:

V2-C0
Frozen H2 independent confirmation

V2-S0
Map semantic audit

V2-S1
Flat A5 vs A5+A6

V2-S2
Semantic Site Head specialization

V2-E0
Utility/event representation audit

V2-T1
Path-dependence baseline

V2-Q1
GRU sequence baseline

IDs describe research sequence, not calendar deadlines.

33. Final V2 Holdout Policy

Before the final V2 holdout is opened, freeze:

selected baseline
selected V2 candidate
training population
feature contract
semantic-map version
temporal contract
hyperparameters
calibration procedure
preprocessing
evaluation metrics
promotion rule

The final holdout should be evaluated once under the frozen protocol.

Exploratory analysis may occur after the primary result is recorded, but it must be labeled exploratory.

34. Final V2 Promotion Logic

A new model is not selected because it has the lowest single development score.

Promotion requires a coherent evidence package.

The package must include:

Probability quality
Log Loss
Brier
Cross-match robustness
match wins
equal-match mean delta
median delta
paired bootstrap interval
Classification behavior
Accuracy
Macro F1
class recalls
confusion matrix
Scope validity
same scientific target
no leakage
valid timing
valid feature availability
Independent confirmation
untouched holdout result
35. Runtime Gate

Experimental models do not automatically receive serving artifacts.

Only a scientifically selected model proceeds to runtime work.

The runtime sequence remains:

selected scientific model
        ↓
train final development artifact
        ↓
save model + metadata
        ↓
offline/replay parity
        ↓
synthetic runtime validation
        ↓
synthetic GSI HTTP validation
        ↓
real-source validation when available

A runtime artifact trained on all development data is a serving artifact.

It is not independent scientific evidence.

36. Live-System Integrity

The project remains observer/replay oriented.

Future player-side research must respect competitive-integrity boundaries.

A V2 feature may only enter a live serving model if it is available from the declared live information source at prediction time.

No future label or inaccessible hidden information may be introduced merely because it exists in demo replay data.

37. Versioning Policy

Scientific artifacts must not overwrite earlier frozen evidence.

Use new versioned paths.

Examples:

data/processed/v2_*.parquet
data/interim/v2_*.csv

artifacts/v2_*.csv
artifacts/v2_*.json

docs/v2_*.md

Legacy V0/V1 results must remain available for audit.

38. Negative Results Policy

Negative results are scientific outputs.

Examples include:

H2 fails independent confirmation
A6 does not improve V0
utility semantics provide no additional signal
path-dependence baseline fails
GRU does not beat tabular baseline

These results must be retained and documented.

The project must not redefine failure as success after results are known.

39. Website Policy

The website must distinguish:

Scientific Record

from:

Explanation / Learning Layer

V2 pages should record:

Question
Hypothesis
Experimental contract
Result
Interpretation
Decision
Next gate

The website must distinguish:

development evidence
independent confirmation
engineering validation
real-source validation

These categories must not be collapsed into a single claim of "validated."

40. V2 Planned Stage Map
V2-0
Freeze V2 scientific protocol
        ↓
V2-1
Survey new demo sources
        ↓
V2-2
Acquire + audit new matches
        ↓
V2-3
Freeze Batch A size and manifest
        ↓
V2-C0
Frozen V0 vs frozen H2
independent confirmation
        ↓
     decision
        ↓
┌───────────────────────┐
│                       │
H2 confirms             H2 fails
│                       │
preserve as             close or defer
supported candidate     hierarchy branch
│                       │
└───────────┬───────────┘
            ↓
V2-S0
Build Mirage semantic map v0
            ↓
V2-S1
Controlled A5 vs A5+A6
            ↓
       evidence gate
            ↓
V2-S2
Hierarchical semantic specialization
only if justified
            ↓
V2-E0
Utility / event semantic baseline
            ↓
V2-T0
Path-dependence diagnostic
            ↓
       evidence gate
            ↓
V2-Q0
Sequence model baseline
only if justified
            ↓
V2-F0
Freeze final candidate
            ↓
V2-H0
Untouched final holdout
            ↓
final model-selection decision
            ↓
runtime/replay/live engineering
only if a new model is selected
            ↓
V2 scientific freeze
41. Immediate Next Step

After this protocol is committed, V2 begins with:

V2-1
New demo source and availability audit

This stage may inspect:

where demos can be obtained
number of available Mirage matches
dates
teams
events
file accessibility
parser compatibility

It must not yet use new-match V0/H2 performance to choose which matches enter Batch A.

The next artifact should be:

docs/v2_data_source_audit.md

and, once acquisition begins:

data/raw/v2_demo_manifest.csv
42. V2 Governing Principle

The V2 project follows this priority order:

Evidence quality
        ↓
Information quality
        ↓
Representation quality
        ↓
Model complexity

The scarce resource is not another architecture.

The scarce resource is:

new, independent, informative evidence.

