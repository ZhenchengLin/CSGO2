# V2 Confirmation Data-Integrity Repair Record

Status: **data/pipeline bug-fix phase complete; D_CONFIRM build + QA passed; corrected development baseline rebuild pending; no D_CONFIRM model predictions or metrics have been calculated.**

## Experiment boundary

- D_CONFIRM selection was frozen before scoring.
- First 30 eligible matches were selected chronologically.
- No reserve substitution was allowed after seeing data behavior.
- Freeze commit: `4614f91`.
- Map: Mirage.
- Raw timing contract: 64 ticks/sec.
- awpy: 2.0.2.
- demoparser2: 0.42.0.
- All repairs below happened before confirmation prediction.

## Debug path

### 1. Historical pickup / inventory gap

Nemesis vs SINNERS exposed a state where inventory did not show C4 even though the bomb event stream identified a pickup carrier.

Repair:

- use pickup only to identify WHO carries the bomb;
- use that player's CURRENT snapshot XYZ;
- never reuse stale pickup-event coordinates.

The timing-corrected legacy regression later passed all 20 matches and all 1686 observations.

### 2. Missing player snapshots

NiP vs FOKUS round 8 and ShindeN vs Fluxo round 19 exposed truncated player-state streams.

New rule:

- target snapshot must exist at the target tick or within +1 raw tick;
- otherwise exclude that observation;
- no interpolation;
- no carry-forward;
- no fabricated snapshot.

Reason code:

`MISSING_PLAYER_SNAPSHOT`

Commit:

`367bf2a` — `fix: exclude unavailable V2 player snapshots`

### 3. Wrong hypothesis: bomb round-number alignment

Leo vs UNiTY initially looked like a bomb-event round association problem.

We tested:

- event `round_num`
- independently derived round from the event tick interval

Result:

**0 round-number mismatches.**

So that hypothesis was rejected.

### 4. Wrong assumption: demo.rounds.bomb_site as truth

We then temporarily treated `demo.rounds.bomb_site` as the expected A/B label.

That assumption was wrong.

Across frozen D_CONFIRM:

- 298 valid plant events
- 188 event A ↔ planter place A
- 110 event B ↔ planter place B
- 0 event/place mismatch
- total agreement = **298 / 298**

XYZ positions also formed clean A-site and B-site clusters.

But `demo.rounds.bomb_site` reported B for:

- all 188 valid A-site plants
- all 110 valid B-site plants

Therefore:

`demo.rounds.bomb_site`

is NOT canonical A/B truth for this parser output.

### 5. Ghost plant events

We found plant events occurring after their associated round had already ended.

D_CONFIRM:

- 18 ghost plant events

Legacy 20-match corpus:

- 11 ghost plant events

Canonical rule:

A plant event is label-eligible only if:

`round.start <= plant_tick <= round.end`

Anything after round end is:

`GHOST_PLANT_EVENT`

and cannot create a label.

### 6. Legacy impact

The 11 ghost rounds contaminated the old timing-corrected development supervision.

Total observations:

1686

Affected labels:

44 / 1686 = 2.6097%

Transitions:

- 24 `A_PLANT → NO_PLANT`
- 20 `B_PLANT → NO_PLANT`

Corrected label totals:

- `NO_PLANT = 844`
- `A_PLANT = 551`
- `B_PLANT = 291`

Observation count remained exactly:

1686

So this defect was localized to y labels, not the frozen 37-feature X matrix.

### 7. Missing plant-event cases

Two D_CONFIRM rounds had metadata indicating a plant but no valid plant event:

- Dendele vs Turma do Pagode round 7
- NiP vs FOKUS round 8

Their player telemetry ended too early to reconstruct plant site reliably.

We explicitly refused to guess A/B from `demo.rounds.bomb_site`.

Reason:

`PLANT_LABEL_UNRESOLVED`

Action:

exclude the round.

### 8. Canonical plant-label contract

1. Plant event must occur inside the round interval.
2. Valid `BombsiteA → A_PLANT`.
3. Valid `BombsiteB → B_PLANT`.
4. Ghost plant cannot create a label.
5. Metadata plant with no valid event does not justify guessing A/B.
6. Unresolved site → `PLANT_LABEL_UNRESOLVED`.
7. `demo.rounds.bomb_site` is integrity evidence only.

Commit:

`13e35b7` — `fix: reject ghost plants in canonical labels`

### 9. Bomb-state availability

After Leo round 4's label was correctly changed to `NO_PLANT`, a separate feature problem remained.

Observation ticks:

- 10s = 30888
- 20s = 31528
- 30s = 32168
- 40s = 32808

First trustworthy pickup:

32231

Therefore the first three observations had no trustworthy bomb state available at or before their target time.

We explicitly forbid:

future pickup → earlier observation reconstruction

Instead:

`BOMB_STATE_UNRESOLVED`

excludes the observation.

Only missing historical evidence is excluded.

Real invariant failures such as:

- multiple C4 carriers
- missing pickup identity
- ambiguous pickup carrier
- invalid event state

still hard-fail.

Commit:

`383b678` — `fix: exclude unresolved V2 bomb states`

## Final D_CONFIRM build

Matches:

**30**

Feature-valid observations:

**2269**

Labels:

- A_PLANT = 693
- B_PLANT = 416
- NO_PLANT = 1160

Horizons:

- 10s = 600
- 20s = 596
- 30s = 577
- 40s = 496

Dataset SHA256:

`e781dbebe0956322476cd8841675ad9a22e9d680c6dec7a47f5d23c35829c367`

## Exclusion layers

Round-level:

- 5 × `MISSING_FREEZE_END`
- 2 × `PLANT_LABEL_UNRESOLVED`

Observation-level:

- 1 × `MISSING_PLAYER_SNAPSHOT`
- 3 × `BOMB_STATE_UNRESOLVED`

## Post-build QA

PASS:

- exactly 30 frozen matches
- exactly 2269 rows
- all 37 frozen features present
- no null
- no NaN
- no Inf
- exact label counts
- exact horizon counts
- zero duplicate observation keys
- exact label vocabulary

No model predictions or confirmation metrics were calculated during these repairs.

## Scientific consequence

The old V0 / H0 / H1 / H2 / T0 metrics remain useful historical evidence, but they are now classified as:

**pre-label-repair development results**

They are not the final canonical baseline because 44 / 1686 labels were wrong.

The next experiment is constrained:

1. rebuild corrected 20-match development data;
2. keep the same 1686 observations;
3. keep the same 37 features;
4. keep the same five match-level folds;
5. keep the same XGB-A5 hyperparameters;
6. keep the same H2 procedure;
7. DO NOT RETUNE;
8. freeze corrected V0/H2 artifacts;
9. then run one formal untouched D_CONFIRM evaluation.

## Permanent engineering lesson

Every data bug should leave a guardrail.

Debugging flow:

detect problem
→ determine root cause
→ fix current case
→ formulate invariant
→ add reason code / checker
→ regression test
→ prevent the same class of bug from silently returning.

Never repair an observation with future information.
