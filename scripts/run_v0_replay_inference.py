from pathlib import Path

import numpy as np
import polars as pl

from cs2_tactical_intelligence.v0.predictor import (
    V0Predictor,
)

from cs2_tactical_intelligence.v0.replay import (
    V0ReplayAdapter,
)


DEMO_PATH = Path(
    "data/raw/havu-vs-mellren-m1-mirage.dem"
)

DATASET_PATH = Path(
    "data/processed/v0_dataset_v1.parquet"
)

OUTPUT_PATH = Path(
    "data/interim/v0_havu_replay_predictions.parquet"
)


KEY_COLUMNS = [
    "demo_filename",
    "round_num",
    "horizon_sec",
    "target_tick",
]


PROBABILITY_COLUMNS = [
    "p_a_plant",
    "p_b_plant",
    "p_no_plant",
]


print("\n" + "=" * 100)
print("V0 — END-TO-END REPLAY INFERENCE")
print("=" * 100)


# ==================================================
# 1. Build replay features
# ==================================================

adapter = V0ReplayAdapter(
    DEMO_PATH
)

replay_features, invalid_rounds = (
    adapter.build_features()
)


assert "label" not in replay_features.columns


print(
    f"\nReplay observations: "
    f"{replay_features.height}"
)

print(
    f"Invalid rounds:      "
    f"{len(invalid_rounds)}"
)

print(
    "✅ Replay feature rows contain no future label"
)


# ==================================================
# 2. Load frozen runtime predictor
# ==================================================

predictor = V0Predictor()


print(
    "✅ Frozen V0 runtime predictor loaded"
)


# ==================================================
# Prediction helper
# ==================================================

def predict_frame(frame):

    rows = []

    for row in frame.iter_rows(
        named=True
    ):

        result = predictor.predict(
            row
        )

        rows.append({

            "demo_filename":
                row["demo_filename"],

            "round_num":
                row["round_num"],

            "horizon_sec":
                row["horizon_sec"],

            "target_tick":
                row["target_tick"],

            **result,
        })

    return pl.DataFrame(
        rows
    )


# ==================================================
# 3. Replay-path predictions
# ==================================================

replay_predictions = predict_frame(
    replay_features
)


print(
    "✅ Replay features successfully passed "
    "through V0Predictor"
)


# ==================================================
# 4. Probability validation
# ==================================================

probability_sums = (
    replay_predictions
    .select(
        pl.sum_horizontal(
            PROBABILITY_COLUMNS
        )
        .alias("probability_sum")
    )
)


max_sum_error = (
    probability_sums
    .select(
        (
            pl.col("probability_sum")
            - 1.0
        )
        .abs()
        .max()
    )
    .item()
)


print(
    "\nMaximum probability-sum error:"
)

print(
    max_sum_error
)


assert (
    max_sum_error
    < 1e-12
)


print(
    "✅ Every replay prediction sums to 1.0"
)


# ==================================================
# 5. End-to-end inference parity
#
# Compare predictions from:
#
# offline feature rows
# vs
# replay feature rows
#
# We intentionally do NOT use the true label for
# scoring here because this full-development runtime
# model was trained on the same development dataset.
# ==================================================

offline_features = (
    pl.read_parquet(
        DATASET_PATH
    )
    .filter(
        pl.col("demo_filename")
        == DEMO_PATH.name
    )
)


assert (
    offline_features
    .select(KEY_COLUMNS)
    .to_dicts()
    ==
    replay_features
    .select(KEY_COLUMNS)
    .to_dicts()
)


offline_predictions = predict_frame(
    offline_features
)


max_probability_difference = 0.0


for column in PROBABILITY_COLUMNS:

    offline_values = (
        offline_predictions[
            column
        ]
        .to_numpy()
    )

    replay_values = (
        replay_predictions[
            column
        ]
        .to_numpy()
    )

    difference = float(
        np.max(
            np.abs(
                offline_values
                - replay_values
            )
        )
    )

    max_probability_difference = max(
        max_probability_difference,
        difference,
    )


print(
    "\nMaximum offline/replay "
    "probability difference:"
)

print(
    max_probability_difference
)


assert (
    max_probability_difference
    < 1e-12
)


assert (
    offline_predictions[
        "prediction"
    ].to_list()
    ==
    replay_predictions[
        "prediction"
    ].to_list()
)


print(
    "✅ Offline and replay predictions match"
)


# ==================================================
# 6. Save replay prediction stream
# ==================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


replay_predictions.write_parquet(
    OUTPUT_PATH
)


print(
    f"\n✅ Saved replay predictions: "
    f"{OUTPUT_PATH}"
)


# ==================================================
# 7. Pick one round with the richest trajectory
# ==================================================

round_counts = (
    replay_predictions
    .group_by("round_num")
    .len()
    .sort(
        [
            "len",
            "round_num",
        ],
        descending=[
            True,
            False,
        ],
    )
)


target_round = int(
    round_counts[
        "round_num"
    ][0]
)


trajectory = (
    replay_predictions
    .filter(
        pl.col("round_num")
        == target_round
    )
    .sort("horizon_sec")
)


display = (
    trajectory
    .select([
        "horizon_sec",

        (
            pl.col("p_a_plant")
            * 100
        )
        .round(2)
        .alias("A_PLANT_%"),

        (
            pl.col("p_b_plant")
            * 100
        )
        .round(2)
        .alias("B_PLANT_%"),

        (
            pl.col("p_no_plant")
            * 100
        )
        .round(2)
        .alias("NO_PLANT_%"),

        "prediction",
        "confidence",
    ])
)


print(
    "\n" + "=" * 100
)

print(
    f"EXAMPLE PREDICTION TRAJECTORY "
    f"— ROUND {target_round}"
)

print(
    "=" * 100
)

print(
    display
)


print(
    "\nNOTE:"
)

print(
    "This is a runtime demonstration, "
    "NOT an independent evaluation."
)


print("\n" + "=" * 100)

print(
    "✅ V0 END-TO-END REPLAY "
    "INFERENCE PASSED"
)

print("=" * 100)
