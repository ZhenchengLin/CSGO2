from pathlib import Path

import polars as pl


# ============================================================
# Dataset paths
# ============================================================

OLD_CANDIDATES = [
    Path(
        "data/processed/"
        "v0_dataset_v1_frozen_reference.parquet"
    ),
    Path(
        "data/processed/"
        "v0_dataset_v1.parquet"
    ),
]

NEW_PATH = Path(
    "data/processed/"
    "v0_dataset_v2_timing_corrected.parquet"
)


old_path = next(
    (
        path
        for path in OLD_CANDIDATES
        if path.exists()
    ),
    None,
)

if old_path is None:
    raise FileNotFoundError(
        "Could not find old V0 dataset."
    )

if not NEW_PATH.exists():
    raise FileNotFoundError(
        NEW_PATH
    )


print("\n" + "=" * 100)
print("V0 TIMING REPAIR — FORENSIC COMPARISON")
print("=" * 100)

print(
    "\nOld dataset:",
    old_path,
)

print(
    "New dataset:",
    NEW_PATH,
)


old = pl.read_parquet(
    old_path
)

new = pl.read_parquet(
    NEW_PATH
)


print(
    "\nOld rows:",
    old.height,
)

print(
    "New rows:",
    new.height,
)


# ============================================================
# Map wrong old horizons onto their real-time equivalents.
#
# old nominal 10s:
#     10 * 128 = 1280 ticks
#     = true 20s at 64 ticks/sec
#
# old nominal 20s:
#     20 * 128 = 2560 ticks
#     = true 40s at 64 ticks/sec
# ============================================================

old_overlap = (
    old
    .filter(
        pl.col(
            "horizon_sec"
        ).is_in([
            10,
            20,
        ])
    )
    .with_columns(
        (
            pl.col("horizon_sec")
            * 2
        ).alias(
            "mapped_horizon_sec"
        )
    )
)


new_overlap = (
    new
    .filter(
        pl.col(
            "horizon_sec"
        ).is_in([
            20,
            40,
        ])
    )
    .with_columns(
        pl.col(
            "horizon_sec"
        ).alias(
            "mapped_horizon_sec"
        )
    )
)


print("\nOVERLAP COUNTS")

print(
    old_overlap
    .group_by(
        "horizon_sec"
    )
    .len()
    .sort(
        "horizon_sec"
    )
)

print(
    new_overlap
    .group_by(
        "horizon_sec"
    )
    .len()
    .sort(
        "horizon_sec"
    )
)


# ============================================================
# Join same real observation states
# ============================================================

joined = (
    old_overlap
    .join(
        new_overlap,
        on=[
            "demo_filename",
            "round_num",
            "mapped_horizon_sec",
        ],
        how="inner",
        suffix="_new",
    )
)


expected_overlap = (
    442 + 373
)

print(
    "\nJoined overlap rows:",
    joined.height,
)

print(
    "Expected overlap rows:",
    expected_overlap,
)

assert (
    joined.height
    == expected_overlap
)


# ============================================================
# Timing identity
# ============================================================

target_mismatches = (
    joined
    .filter(
        pl.col("target_tick")
        !=
        pl.col("target_tick_new")
    )
    .height
)

print(
    "\nTARGET TICK MISMATCHES:",
    target_mismatches,
)

assert (
    target_mismatches == 0
)


# ============================================================
# Label identity
# ============================================================

label_mismatches = (
    joined
    .filter(
        pl.col("label")
        !=
        pl.col("label_new")
    )
    .height
)

print(
    "LABEL MISMATCHES:",
    label_mismatches,
)

assert (
    label_mismatches == 0
)


# ============================================================
# Horizon semantic mismatch
# ============================================================

horizon_relation_bad = (
    joined
    .filter(
        pl.col(
            "horizon_sec_new"
        )
        !=
        (
            pl.col(
                "horizon_sec"
            )
            * 2
        )
    )
    .height
)

print(
    "\nHORIZON RELATION FAILURES:",
    horizon_relation_bad,
)

assert (
    horizon_relation_bad == 0
)

print(
    "\nConfirmed:"
)

print(
    "  old 10s == true 20s state"
)

print(
    "  old 20s == true 40s state"
)


# ============================================================
# Motion features
# ============================================================

MOTION_COLUMNS = [
    "t_mean_speed_1s",
    "t_centroid_velocity_x_1s",
    "t_centroid_velocity_y_1s",
    "bomb_speed_1s",
]


# ============================================================
# Columns dependent on the previous snapshot rather than
# purely current state.
# ============================================================

PREVIOUS_QA_COLUMNS = [
    "qa_motion_players_used",
    "qa_bomb_state_prev",
]


# ============================================================
# Metadata that is intentionally different or not a feature.
# ============================================================

EXCLUDED_COLUMNS = {
    "demo_filename",
    "round_num",
    "horizon_sec",
    "target_tick",
    "label",
    "mapped_horizon_sec",
    *MOTION_COLUMNS,
    *PREVIOUS_QA_COLUMNS,
}


# ============================================================
# Compare every current-state column.
#
# Since target_tick is identical, these should come from the
# exact same game state.
# ============================================================

current_state_columns = [
    column
    for column in old.columns
    if (
        column
        not in EXCLUDED_COLUMNS
        and (
            column + "_new"
            in joined.columns
        )
    )
]


print(
    "\nCURRENT-STATE FEATURE COMPARISON"
)

current_failures = []


for column in current_state_columns:

    dtype = old.schema[
        column
    ]

    if dtype.is_numeric():

        max_abs_diff = (
            joined
            .select(
                (
                    pl.col(column)
                    .cast(pl.Float64)
                    -
                    pl.col(
                        column + "_new"
                    )
                    .cast(pl.Float64)
                )
                .abs()
                .max()
                .alias(
                    "max_abs_diff"
                )
            )
            .item()
        )

        if max_abs_diff is None:
            max_abs_diff = 0.0

        print(
            f"{column:40s} "
            f"max_abs_diff="
            f"{max_abs_diff}"
        )

        if max_abs_diff != 0:
            current_failures.append(
                (
                    column,
                    max_abs_diff,
                )
            )

    else:

        mismatches = (
            joined
            .filter(
                pl.col(column)
                !=
                pl.col(
                    column + "_new"
                )
            )
            .height
        )

        print(
            f"{column:40s} "
            f"mismatches="
            f"{mismatches}"
        )

        if mismatches != 0:
            current_failures.append(
                (
                    column,
                    mismatches,
                )
            )


print(
    "\nCurrent-state failures:",
    current_failures,
)

assert (
    len(current_failures)
    == 0
)


# ============================================================
# Motion comparison
#
# These are EXPECTED to differ.
# ============================================================

print(
    "\nMOTION FEATURE DIFFERENCES"
)


motion_summary_rows = []


for column in MOTION_COLUMNS:

    diff = (
        joined
        .select(
            (
                pl.col(column)
                .cast(pl.Float64)
                -
                pl.col(
                    column + "_new"
                )
                .cast(pl.Float64)
            )
            .abs()
            .alias(
                "abs_diff"
            )
        )
    )

    mean_abs_diff = (
        diff[
            "abs_diff"
        ]
        .mean()
    )

    max_abs_diff = (
        diff[
            "abs_diff"
        ]
        .max()
    )

    equal_rows = (
        diff
        .filter(
            pl.col(
                "abs_diff"
            ) == 0
        )
        .height
    )

    motion_summary_rows.append({
        "feature":
            column,

        "mean_abs_diff":
            mean_abs_diff,

        "max_abs_diff":
            max_abs_diff,

        "equal_rows":
            equal_rows,

        "total_rows":
            joined.height,
    })


motion_summary = (
    pl.DataFrame(
        motion_summary_rows
    )
)

print(
    motion_summary
)


# ============================================================
# Final evidence
# ============================================================

print("\n" + "=" * 100)
print("FORENSIC FINDINGS")
print("=" * 100)

print(
    """
1. Old nominal 10s rows align exactly with new true 20s rows.
2. Old nominal 20s rows align exactly with new true 40s rows.
3. Target ticks are identical for all overlapping rows.
4. Labels are identical for all overlapping rows.
5. Current-state features are identical.
6. Motion features differ because the old pipeline used a
   128-tick history window, which is 2 real seconds.
7. The old horizon_sec feature itself was semantically mislabeled.
"""
)

print(
    "✅ V0 TIMING BUG CAUSAL CHAIN VERIFIED"
)
