# V2 Independent Confirmation Selection Rule

Status: PRE-DATA SELECTION FREEZE

## Purpose

This document defines how the first V2 independent confirmation corpus,
D_confirm, is selected.

The rule is frozen before V0 or H2 performance is evaluated on any
new confirmation match.

The goal is to prevent cherry-picking and post-result dataset selection.

---

## Confirmation Question

The first V2 confirmation experiment asks:

Does the frozen H2 hierarchy reproduce its probability-quality advantage
over frozen V0 on genuinely new Mirage matches?

This is a confirmation experiment.

It is not a model-development experiment.

---

## Legacy Cutoff

The legacy development corpus contains Mirage demos through:

2026-09-08

Therefore D_confirm candidates must satisfy:

match_date > 2026-09-08

No match already used in D_legacy may enter D_confirm.

Exact file SHA256 duplication is prohibited.

---

## Target Confirmation Size

D_confirm target size:

30 unique Mirage map demos

The confirmation evaluation must not begin early because an interim subset
looks favorable or unfavorable.

The first V0-versus-H2 scientific comparison will be performed only after
the 30-demo confirmation manifest is frozen.

---

## Selection Order

Eligible demos are ordered by:

1. match_date ascending
2. demo_filename ascending as deterministic tie-breaker

The first 30 eligible demos become D_confirm.

There is no manual replacement of an eligible earlier demo with a later demo
based on team, tournament, result, class distribution, round count, or model
performance.

---

## Required Inclusion Criteria

A demo is eligible only if all of the following are true:

1. match_date is after 2026-09-08
2. map is de_mirage
3. the match is completed
4. a usable demo file is available
5. SHA256 does not duplicate D_legacy
6. SHA256 does not duplicate another incoming demo
7. measured raw demo clock is compatible with the frozen 64 ticks/sec contract
8. the demo parses successfully
9. required round timing information is available
10. required acquisition metadata is complete
11. scientific_role is confirm_candidate

---

## Technical Exclusion Criteria

A demo may be excluded before model prediction for:

- exact SHA256 duplication
- wrong map
- corrupted or unreadable demo
- parser failure
- timing-contract failure
- unrecoverable round-timing failure
- incomplete source identity
- missing required metadata

Every exclusion must be preserved in the intake audit.

---

## Forbidden Selection Criteria

The following must NOT be used to decide whether an otherwise valid demo
enters D_confirm:

- whether V0 performs well
- whether H2 performs well
- H2 minus V0 Log Loss
- H2 minus V0 Brier Score
- predicted probabilities
- match winner
- team ranking
- tournament prestige
- close versus one-sided score
- plant-site balance
- A_PLANT / B_PLANT / NO_PLANT distribution
- number of eligible prediction rows
- similarity to the legacy data
- subjective tactical interest

These variables may later be analyzed.

They may not determine confirmation-set membership.

---

## Team Overlap Policy

Teams that appeared in D_legacy are not automatically excluded.

The primary confirmation unit is a new match, not a new team.

However, overlap between legacy and confirmation teams must be reported.

Therefore the eventual scientific claim will distinguish:

match-level independence

from

team-level independence.

A future external-validation experiment may impose stronger team or event
separation.

---

## Tournament Overlap Policy

Tournament identity is not used as an inclusion or exclusion criterion.

Event diversity will be recorded and reported.

This avoids subjective event selection while still allowing later analysis
of distribution shift.

---

## Outcome-Blind Selection

Target labels may be generated as part of deterministic data-quality
processing.

However, target composition must not alter selection order.

D_confirm is selected using identity and technical-validity criteria only.

---

## Confirmation Manifest Freeze

After at least 30 eligible confirm_candidate demos exist,
scripts/freeze_v2_confirm_manifest.py will:

1. apply the frozen cutoff
2. verify technical eligibility
3. sort candidates deterministically
4. select the first 30
5. write docs/v2_confirm_manifest.csv

The generated manifest will include SHA256 identities.

Once committed, that manifest defines D_confirm.

The script refuses to overwrite an existing frozen manifest.

---

## No Sequential Peeking

Before docs/v2_confirm_manifest.csv is frozen:

- do not run V0 confirmation scores
- do not run H2 confirmation scores
- do not compare V0 versus H2
- do not inspect per-match model deltas

After the manifest is frozen, the confirmation evaluation is run as one
predeclared experiment.

---

## After Confirmation

Once the H2 confirmation result is calculated and permanently recorded,
D_confirm is scientifically consumed for that question.

It may later join a development pool for new hypotheses.

If it does, it can no longer serve as an untouched final V2 holdout.

A separate D_holdout will be required for final V2 confirmation.
