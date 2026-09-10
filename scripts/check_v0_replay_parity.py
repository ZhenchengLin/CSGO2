from pathlib import Path

import numpy as np
import polars as pl

from cs2_tactical_intelligence.v0.replay import (
    V0ReplayAdapter,
)


DATASET_PATH = Path(
    "data/processed/v0_dataset_v1.parquet"
)

DEMO_PATH = Path(
    "data/raw/havu-vs-mellren-m1-mirage.dem"
)


print("\n" + "=" * 100)
print("V0 — OFFLINE / REPLAY FEATURE PARITY")
print("=" * 100)


# ==================================================
# Offline reference
# ==================================================

offline = (
    pl.read_parquet(
        DATASET_PATH
    )
    .filter(
        pl.col("demo_filename")
        == DEMO_PATH.name
    )
    .drop("label")
)


# ==================================================
# Replay path
# ==================================================

adapter = V0ReplayAdapter(
    DEMO_PATH
)


replay, invalid_rounds = (
    adapter.build_features()
)


print(
    f"\nOffline observations: "
    f"{offline.height}"
)

print(
    f"Replay observations:  "
    f"{replay.height}"
)


assert (
    offline.height
    == replay.height
)


# ==================================================
# Schema / ordering
# ==================================================

assert (
    offline.columns
    == replay.columns
), (
    "Offline/replay column order differs.\n\n"
    f"Offline:\n{offline.columns}\n\n"
    f"Replay:\n{replay.columns}"
)


assert (
    offline.schema
    == replay.schema
), (
    "Offline/replay schema differs.\n\n"
    f"Offline:\n{offline.schema}\n\n"
    f"Replay:\n{replay.schema}"
)


print(
    "✅ Column names, order, and dtypes match"
)


# ==================================================
# Observation identity
# ==================================================

KEY_COLUMNS = [
    "demo_filename",
    "round_num",
    "horizon_sec",
    "target_tick",
]


assert (
    offline
    .select(KEY_COLUMNS)
    .to_dicts()
    ==
    replay
    .select(KEY_COLUMNS)
    .to_dicts()
)


print(
    "✅ Observation identity matches"
)


# ==================================================
# String QA fields
# ==================================================

STRING_COLUMNS = [
    column
    for column, dtype
    in offline.schema.items()
    if dtype == pl.String
]


for column in STRING_COLUMNS:

    assert (
        offline[column].to_list()
        ==
        replay[column].to_list()
    ), (
        f"String mismatch: "
        f"{column}"
    )


print(
    "✅ Replay QA state matches"
)


# ==================================================
# Numeric parity
# ==================================================

NUMERIC_COLUMNS = [
    column
    for column, dtype
    in offline.schema.items()
    if dtype.is_numeric()
]


max_difference = 0.0
differences = []


for column in NUMERIC_COLUMNS:

    a = (
        offline[column]
        .to_numpy()
        .astype(float)
    )

    b = (
        replay[column]
        .to_numpy()
        .astype(float)
    )

    difference = float(
        np.max(
            np.abs(
                a - b
            )
        )
    )

    max_difference = max(
        max_difference,
        difference,
    )

    if difference > 1e-12:
        differences.append({
            "feature":
                column,

            "max_abs_diff":
                difference,
        })


print(
    "\nMaximum numeric difference:"
)

print(
    max_difference
)


if differences:

    print(
        "\n❌ DIFFERENCES"
    )

    print(
        pl.DataFrame(
            differences
        )
        .sort(
            "max_abs_diff",
            descending=True,
        )
    )

    raise AssertionError(
        "Replay parity failed"
    )


print(
    "✅ All numeric values match "
    "within 1e-12"
)


# ==================================================
# Exact equality
# ==================================================

exact_equal = (
    offline.equals(
        replay
    )
)


print(
    "\nExact DataFrame equality:",
    exact_equal,
)


assert exact_equal


print("\n" + "=" * 100)

print(
    "✅ V0 REPLAY PARITY PASSED"
)

print("=" * 100)
