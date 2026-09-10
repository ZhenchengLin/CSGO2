# V0 Feature Candidate Matrix v1

## Purpose

This document defines the candidate feature space for the V0 Early-Round Site Outcome Forecaster.

Target:

P(A_PLANT, B_PLANT, NO_PLANT | observed game state up to time t)

Observation horizons:

- 10 seconds
- 20 seconds
- 30 seconds
- 40 seconds

The goal of this document is NOT to claim that every listed feature is useful.

A feature enters the candidate set only when it has at least one defensible reason:

1. Prior literature motivates the representation.
2. Counter-Strike game mechanics provide a clear domain hypothesis.
3. The feature can be reconstructed consistently in historical and online data.

Final inclusion is determined by ablation experiments.

---

# Evidence Standard

We use three evidence labels.

## LIT-CS

Direct evidence from Counter-Strike research.

This is the strongest domain-specific literature support.

## LIT-MA

Evidence from multi-agent / team positional tracking research.

These papers do not prove that the feature predicts CS2 bomb sites.

They show that the representation is useful for describing collective tactical structure.

## DOMAIN

Feature derived from Counter-Strike mechanics and our V0 prediction problem.

These are hypotheses that require empirical validation.

---

# Feature Families

## F0 — Temporal / Match Context

| Feature | Evidence | Reasoning | Demo | Live | Leakage | Status |
|---|---|---|---|---|---|---|
| horizon_sec | DOMAIN | Tactical intent should become more observable as the round develops | Yes | Yes | Low | TEST |
| score_difference | DOMAIN | Match context may influence aggression and strategic risk | Yes | Likely | Low | LATER |
| round_number | DOMAIN | May encode match phase but risks learning dataset-specific structure | Yes | Yes | Medium | DEFER |

Decision:

`horizon_sec` enters V0.

Round number is metadata first, not a model feature.

---

# F1 — Offensive Spatial Structure

These features describe the T-side formation.

## Team centroid

Features:

- t_centroid_x
- t_centroid_y
- t_centroid_z

Evidence:

LIT-MA

The geometric centroid is widely used to represent the collective location of a team.

CS2 hypothesis:

If the T-side collective position shifts toward one side of Mirage,
the final plant probability may change.

Status:

TEST

---

## Stretch / Spread

Feature:

- t_stretch_xy

Definition:

Average XY distance of alive T players from the T-team centroid.

Evidence:

LIT-MA

This corresponds closely to the tactical tracking concept commonly called
stretch index or team spread.

CS2 hypothesis:

A concentrated execute and a distributed default produce different spatial structures.

Status:

TEST

---

## Coordinate ranges

Features:

- t_range_x
- t_range_y

Definition:

max(X) - min(X)
max(Y) - min(Y)

Evidence:

LIT-MA

Tracking literature commonly uses team length and width.
For CS2 we initially keep map-coordinate-neutral names
`range_x` and `range_y`.

Status:

TEST

---

## Mean pairwise distance

Feature:

- t_mean_pairwise_distance

Evidence:

LIT-MA

Interpersonal distance is commonly used to describe team tactical organization.

CS2 hypothesis:

It provides information about team compactness that centroid alone cannot represent.

Status:

TEST

---

## Convex hull area

Feature:

- t_convex_hull_area

Evidence:

LIT-MA

Surface area enclosed by player locations is a common representation of occupied team space.

CS2 hypothesis:

A five-player execute, split attack, and map-wide default can have very different occupied areas.

Caution:

When fewer than three alive players remain,
the 2D convex hull area becomes degenerate.

Status:

TEST

---

# F2 — Defensive Spatial Structure

Full-observer features:

- ct_centroid_x
- ct_centroid_y
- ct_centroid_z
- ct_stretch_xy
- ct_range_x
- ct_range_y
- ct_mean_pairwise_distance
- ct_convex_hull_area

Evidence:

LIT-MA

Reasoning:

The defensive formation may influence whether an attacking plan continues,
rotates, aborts, or changes bomb-site outcome.

Important experiment:

Compare:

OFFENSIVE-ONLY
T + bomb + context

versus

FULL-OBSERVER
T + CT + bomb + context

This measures how much predictive information comes from offensive intent alone
versus attacker-defender interaction.

Status:

TEST AS SEPARATE ABLATION GROUP

---

# F3 — Inter-Team Spatial Interaction

Candidates:

- t_ct_centroid_distance
- minimum_t_ct_distance
- mean_nearest_opponent_distance

Evidence:

LIT-MA

Multi-agent tactical research studies inter-team distances,
opponent dyads, and relationships between collective formations.

CS2 hypothesis:

Contact pressure and defensive proximity may affect rotations,
site commitment, and aborted executes.

Status:

TEST

---

# F4 — Motion

Candidates:

- t_mean_speed
- ct_mean_speed
- t_centroid_velocity_x
- t_centroid_velocity_y
- ct_centroid_velocity_x
- ct_centroid_velocity_y
- bomb_speed

Evidence:

LIT-CS

Professional Counter-Strike trajectory research demonstrates
learnable temporal structure in player movement.

CS2 hypothesis:

Position tells us where a team is.

Velocity tells us where the team is going.

Example:

Two teams can have the same centroid position while one is rapidly executing
toward A and the other is rotating away.

Training-serving rule:

Do not depend exclusively on parser-specific velocity fields.

Velocity must also be reproducible from consecutive timestamped XYZ positions:

v = (position_t - position_t-dt) / dt

Status:

TEST

---

# F5 — Combat State

Candidates:

- t_alive
- ct_alive
- alive_difference
- t_health_sum
- ct_health_sum
- health_difference
- t_armor_sum
- ct_armor_sum

Evidence:

LIT-CS

Prior Counter-Strike predictive modeling has used health
and other player/team state variables.

CS2 hypothesis:

Player losses and damage change tactical options,
site commitment, rotations, and the probability of reaching a plant.

Status:

TEST

---

# F6 — Economy / Equipment

Candidates:

- t_equip_value_sum
- ct_equip_value_sum
- equip_value_difference

Evidence:

LIT-CS

Economy-related metrics have previously been used
for Counter-Strike predictive modeling.

CS2 hypothesis:

Available weapons and equipment constrain tactical possibilities.

Caution:

Equipment value may overlap strongly with weapon and armor information.

Ablation will determine whether it adds independent predictive value.

Status:

TEST

---

# F7 — Bomb / Objective Geometry

Candidates:

- bomb_x
- bomb_y
- bomb_z
- bomb_to_a_distance
- bomb_to_b_distance
- t_centroid_to_a_distance
- t_centroid_to_b_distance
- bomb_to_t_centroid_distance

Evidence:

DOMAIN

These features are not claimed to be established by the team-sports literature.

They arise directly from the V0 objective.

CS2 hypothesis:

The bomb is the object that ultimately determines the plant site.

Therefore its position relative to the two bomb sites
may contain strong predictive information.

Important caution:

Bomb-to-site distance may become an extremely strong late-round feature.

This is not automatically leakage because it is observable at time t,
but performance must be reported separately at 10 / 20 / 30 / 40 seconds
to show when the prediction becomes trivial.

Status:

TEST

---

# F8 — Map Region Representation

Candidates:

- number of T players in each tactical region
- number of CT players in each tactical region
- bomb region
- team-region occupancy vector

Evidence:

LIT-MA + DOMAIN

Spatial tactical research supports analyzing occupied regions.

However:

Awpy `place` names must NOT become our permanent representation.

We need our own map-region mapping:

XYZ
↓
Canonical Mirage Region Mapper
↓
Region ID

Reason:

The same mapping must be usable for demo and live data.

Status:

DEFER UNTIL MAP MAPPER EXISTS

---

# F9 — Weapon Composition

Possible candidates:

- rifle count
- AWP count
- SMG count
- pistol count
- utility count

Evidence:

LIT-CS + DOMAIN

Reasoning:

Weapon composition affects tactical options.

However this adds representation complexity
and may partially duplicate equipment-value information.

Status:

DEFER UNTIL SIMPLE BASELINE EXISTS

---

# F10 — Utility State

Possible candidates:

- T smoke inventory count
- T flash inventory count
- T molotov inventory count
- CT utility inventory
- active smoke regions
- active inferno regions

Evidence:

DOMAIN

Future utility semantics may also be supported by our planned Utility Knowledge Base.

Status:

DEFER FROM INITIAL BASELINE

Reason:

We first want to understand what geometry, motion,
combat state, and economy can predict without utility semantics.

---

# Explicitly Excluded From V0 Input

## Player identity

Do not use:

- name
- SteamID

Reason:

The model should learn tactical behavior,
not memorize that a particular player or team historically prefers a site.

Identity may later be introduced deliberately in the opponent-modeling version.

---

## Future information

Never use:

- final round winner
- round-end statistics
- plant event before prediction time
- future player states
- future utility events
- future kills or damage

---

## Transcript

Not a V0 model input.

Future roles:

- weak supervision
- tactical semantic labels
- explanation
- multimodal research

---

# Training-Serving Parity

Candidate features must eventually pass this rule:

Historical Demo
        ↓
Canonical State

Live Observer GSI
        ↓
Canonical State

        ↓

Same FeatureBuilder

        ↓

Same Model

Current public GSI capability suggests that observer mode can expose
all-player position, player state, weapons, equipment value,
bomb information, and grenade information.

However V0 does not consider live parity VERIFIED
until we capture and inspect our own real GSI payload.

Therefore the current live-availability status remains provisional.

---

# Initial Feature Ablation Plan

## A0

Class prior only.

No game-state features.

Purpose:

Trivial probability baseline.

---

## A1 — Offensive Geometry

- horizon_sec
- T centroid
- T stretch
- T coordinate ranges
- T mean pairwise distance
- T convex hull area
- bomb position / site distances

Question:

How much plant intent is visible from offensive geometry alone?

---

## A2 — Add Motion

A1 +

- T mean speed
- T centroid velocity
- bomb velocity

Question:

Does movement direction improve prediction beyond current position?

---

## A3 — Add Combat State

A2 +

- alive counts
- health
- armor

Question:

Does combat state explain tactical changes not visible from geometry?

---

## A4 — Add Economy

A3 +

- equipment value

Question:

Does resource state provide additional predictive information?

---

## A5 — Add Defense

A4 +

- CT spatial structure
- inter-team spatial interaction

Question:

How much additional predictive power comes from observing the defense?

---

# Evaluation Rule

Literature is used to justify TESTING a feature.

Literature is NOT treated as evidence that a feature improves our V0 target.

Final feature decisions require empirical evidence.

For every feature family:

Candidate
↓
Implement
↓
Match-level holdout
↓
Ablation
↓
Log Loss
Brier Score
Calibration
Macro F1
↓
KEEP / DROP

---

# References

1. Xia, X., Salinas, A., and Morstatter, F.
   "Precision Under Fire: Analysis and Predictive Modeling in Counter-Strike: Global Offensive."
   IEEE Conference on Games, 2025.
   DOI: 10.1109/COG64752.2025.11114164

2. Thai, D. T., and Scirea, M.
   "Trajectory Analysis and Prediction in Counter-Strike with LSTM Models."
   IEEE Conference on Games, 2025.
   DOI: 10.1109/COG64752.2025.11114397

3. Rico-González et al.
   "Identification, Computational Examination, Critical Assessment and Future Considerations
   of Distance Variables to Assess Collective Tactical Behaviour in Team Invasion Sports
   by Positional Data: A Systematic Review."
   2020.

4. Rico-González et al.
   "Identification, Computational Examination, Critical Assessment and Future Considerations
   of Spatial Tactical Variables to Assess the Use of Space in Team Sports by Positional Data:
   A Systematic Review."
   2021.

5. Zhang et al.
   "Navigating team tactical analysis in football:
   An analytical pipeline leveraging player tracking technology."
   2025.

---

# V1 Decision

Initial implementation will focus on:

F0 Temporal Context
F1 Offensive Spatial Structure
F3 Selected Spatial Interaction
F4 Motion
F5 Combat State
F6 Economy
F7 Bomb Geometry

CT information will be isolated so that offensive-only
and full-observer models can be compared directly.

Map-region, weapon-composition, and utility features
are deliberately deferred until the simpler baseline is evaluated.
