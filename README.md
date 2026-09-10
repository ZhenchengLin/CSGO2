# CS2 Tactical Intelligence

A machine learning engineering and tactical analytics project exploring whether historical Counter-Strike 2 game-state sequences can be used to recognize recurring team behavior and forecast tactical outcomes in real time.

The long-term goal is to build a tactical intelligence system that can:

**Observe → Understand → Predict → Retrieve → Model Opponents → Evaluate Responses**

The project is being developed incrementally, with each version evaluated before the next capability is added.

## Current Status

**Stage 0: Complete**

**Current Model: V0 — Early-Round Site Outcome Forecaster**

**V0 Status: Offline Development Evaluation Frozen**

V0 has a completed offline development evaluation on 20 Mirage matches and 1,268 eligible observations. XGB-A5 is the selected development baseline: log loss 0.8024, Brier 0.4785, accuracy 62.85%, and macro F1 0.5996. These are grouped out-of-fold development results, not untouched final-test or live-serving results.

Read the [complete V0 explanation](docs/v0_explained.md) or the [research website](site/index.html).

## V0 Research Question

Given only the observable state of a T-side round up to a particular point in time:

> Can we predict whether the round will eventually result in an A-site plant, B-site plant, or no plant?

Formally:

$$
P(Y \mid S_{0:t})
$$

where:

$$
Y \in \{A\ Plant,\ B\ Plant,\ No\ Plant\}
$$

Initial observation horizons:

* 10 seconds
* 20 seconds
* 30 seconds
* 40 seconds

Initial map scope:

* Mirage
* T-side rounds

## Why Start With This Problem?

The long-term project is intended to study tactical behavior, opponent tendencies, repeated patterns, and eventually tactical response evaluation.

However, those tasks introduce ambiguous concepts such as:

* executes
* defaults
* splits
* fakes
* rotations
* tactical intent

V0 deliberately begins with an objective prediction target.

Bombsite outcome can be derived directly from historical match telemetry, which allows the project to first validate the complete machine learning pipeline:

```text
Historical Demo
      ↓
Canonical Game State
      ↓
Feature Engineering
      ↓
Dataset
      ↓
Model Training
      ↓
Evaluation
      ↓
Replay / Live State
      ↓
Online Feature Generation
      ↓
Inference
      ↓
Web Interface
```

## Data Strategy

Different data sources serve different roles.

### Historical Training

CS2 demo files are the primary historical training source.

Initial parsing will use:

* Awpy
* demoparser2 when lower-level access is required

### Live Inference

CS2 Game State Integration in observer or spectator-compatible environments is the initial live data source.

### Replay

Historical states will eventually be replayed through the online inference pipeline to provide deterministic testing without requiring CS2 to run during every experiment.

### Controlled Experiments

A self-hosted CS2 server may later provide controlled telemetry and experimental scenarios.

A lightweight FPS or Pygame environment may also be explored for cheap synthetic system validation.

These environments complement real CS2 data rather than replace it.

## V0 Model Progression

The first prediction task will be approached with increasing levels of complexity:

```text
Class Prior
    ↓
Geometry Heuristic
    ↓
Logistic Regression
    ↓
XGBoost
    ↓
Temporal Model Candidate
GRU / TCN
```

More complex models must demonstrate measurable improvement over simpler baselines.

## Evaluation

Because V0 produces probabilities, the primary metrics are:

* Log Loss
* Brier Score
* Calibration

Secondary metrics include:

* Accuracy
* Macro F1
* Per-class Precision / Recall
* Confusion Matrix

Evaluation will be performed on held-out matches rather than randomly splitting rounds from the same match across training and test sets.

## Development Philosophy

Each version follows the same cycle:

```text
Freeze Specification
        ↓
Build
        ↓
Evaluate
        ↓
Collect Evidence
        ↓
Analyze Errors
        ↓
Document Limitations
        ↓
Design Next Version
```

New ideas are recorded rather than continuously expanding the current version.

The goal is to let observed problems and experimental evidence determine what the next version should change.

## Long-Term Research Directions

Future directions currently include:

* Near-future tactical state prediction
* Historical behavior retrieval
* Opponent-specific behavioral modeling
* Tactical concept recognition
* Transcript-assisted tactical labeling
* Utility and grenade semantic knowledge
* Self-hosted CS2 experimental environments
* Partial-observation modeling using legitimate player information and communication
* Tactical response and counter-strategy research

These are intentionally outside the V0 implementation scope.

## Project Structure

```text
CSGO2/
├── README.md
├── pyproject.toml
├── uv.lock
│
├── src/
│   └── cs2_tactical_intelligence/
│
├── tests/
├── data/
├── artifacts/
├── docs/
│
└── site/
    ├── index.html
    ├── pages/
    ├── assets/
    └── site-manifest.json
```

## Research Website

The project includes a dedicated research website under `site/`.

It records:

* Roadmap
* Evidence
* Data-source research
* Model versions
* Architecture decisions
* Research ideas
* Build history
* Engineering journey

Open locally with:

```bash
open site/index.html
```

The site will evolve alongside the project and eventually serve as the public evidence and research record.

## Current Next Step

Freeze the existing V0 evidence. The next research hypothesis is to separate plant feasibility from conditional site choice and compare it against XGB-A5 on untouched matches or an appropriate nested evaluation. Replay and live-source feature parity remain separate engineering milestones.
