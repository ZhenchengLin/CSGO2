# V0 Feature Engineering Evidence

## A1 — Offensive Geometry

**Numerical:** Passed. All 49 observations had complete values; distances, ranges, and areas were non-negative and within plausible Mirage scales.

**Tactical:** Passed. Round 1 geometry expanded during the mid-round and contracted as T players converged toward the eventual A plant.

**Finding:** Geometry is ready for ablation.

**Caveat:** Zero spread or hull area may indicate very few surviving players rather than a compact formation.


## A2 — Motion

**Numerical:** Passed. All 49 observations had complete position-derived motion features with plausible speed values.

**Tactical:** Passed. Round 1 showed fast early movement, slower mid-round positioning, and renewed bomb movement approaching the A plant.

**Finding:** Position-derived motion is ready for ablation.


## A3 — Combat

**Numerical:** Passed. All 49 observations had complete alive, health, and armor features.

**Tactical:** Passed. Every zero-spread observation occurred when only one T player remained alive.

**Finding:** Combat state resolves ambiguity in geometry-only features.


## A4 — Economy

**Numerical:** Passed. All 49 observations had complete equipment-value features.

**Tactical:** Passed. Equipment values changed consistently with the evolving surviving-player and resource state.

**Finding:** Economy is ready for ablation; independent predictive value remains unproven.


## A5 — Defense and Interaction

**Numerical:** Passed. All 49 observations had complete CT geometry and T–CT interaction features.

**Tactical:** Passed. Round 1 showed decreasing T–CT centroid distance as the two formations converged.

**Finding:** Full-observer defensive information is ready for comparison against offensive-only features.


# Current Conclusion

Feature Engineering v1 has passed implementation-level numerical and tactical sanity checks.

These checks validate feature construction, not predictive usefulness.

Predictive value will be determined using match-level holdout and controlled A0–A5 ablation.
