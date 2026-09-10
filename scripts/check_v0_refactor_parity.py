from pathlib import Path

import numpy as np
import polars as pl


REFERENCE_PATH = Path(
    "data/processed/"
    "v0_dataset_v1_frozen_reference.parquet"
)

REFACTORED_PATH = Path(
    "data/processed/"
    "v0_dataset_v1.parquet"
)


reference = pl.read_parquet(
    REFERENCE_PATH
)

refactored = pl.read_parquet(
    REFACTORED_PATH
)


print(
    "\n"
    + "=" * 100
)

print(
    "V0 FEATURE REFACTOR — PARITY CHECK"
)

print(
    "=" * 100
)


# ==================================================
# Shape
# ==================================================

print(
    f"\nReference shape:  "
    f"{reference.shape}"
)

print(
    f"Refactored shape: "
    f"{refactored.shape}"
)


assert (
    reference.shape
    == refactored.shape
)


# ==================================================
# Column contract
# ==================================================

assert (
    reference.columns
    == refactored.columns
), (
    "Column order changed.\n"
    f"Reference:\n{reference.columns}\n\n"
    f"Refactored:\n{refactored.columns}"
)


assert (
    reference.schema
    == refactored.schema
), (
    "Schema changed.\n"
    f"Reference:\n{reference.schema}\n\n"
    f"Refactored:\n{refactored.schema}"
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
    "label",
]


assert (
    reference
    .select(KEY_COLUMNS)
    .to_dicts()
    ==
    refactored
    .select(KEY_COLUMNS)
    .to_dicts()
)


print(
    "✅ Observation identity and labels match"
)


# ==================================================
# String / categorical QA
# ==================================================

STRING_COLUMNS = [
    column
    for column, dtype
    in reference.schema.items()
    if dtype == pl.String
]


for column in STRING_COLUMNS:

    assert (
        reference[column]
        .to_list()
        ==
        refactored[column]
        .to_list()
    ), (
        f"String mismatch: {column}"
    )


print(
    "✅ String and QA fields match"
)


# ==================================================
# Numeric parity
# ==================================================

NUMERIC_COLUMNS = [
    column
    for column, dtype
    in reference.schema.items()
    if dtype.is_numeric()
]


rows = []


for column in NUMERIC_COLUMNS:

    old = (
        reference[column]
        .to_numpy()
    )

    new = (
        refactored[column]
        .to_numpy()
    )


    if np.issubdtype(
        old.dtype,
        np.integer,
    ):

        exact = np.array_equal(
            old,
            new,
        )

        max_abs_diff = (
            0.0
            if exact
            else float(
                np.max(
                    np.abs(
                        old - new
                    )
                )
            )
        )

    else:

        delta = np.abs(
            old.astype(float)
            - new.astype(float)
        )

        max_abs_diff = float(
            np.max(delta)
        )

        exact = np.array_equal(
            old,
            new,
        )


    rows.append({
        "feature":
            column,

        "exact_match":
            exact,

        "max_abs_diff":
            max_abs_diff,
    })


comparison = pl.DataFrame(
    rows
)


changed = comparison.filter(
    pl.col(
        "max_abs_diff"
    )
    > 1e-12
)


print(
    "\nMAXIMUM NUMERIC DIFFERENCE"
)

print(
    comparison[
        "max_abs_diff"
    ].max()
)


if changed.height > 0:

    print(
        "\n❌ FEATURES WITH DIFFERENCES"
    )

    print(
        changed.sort(
            "max_abs_diff",
            descending=True,
        )
    )

    raise AssertionError(
        "V0 refactor changed feature values"
    )


print(
    "✅ All numeric values match "
    "within 1e-12"
)


# ==================================================
# Exact dataframe equality
# ==================================================

exact_equal = (
    reference.equals(
        refactored
    )
)


print(
    f"\nExact DataFrame equality: "
    f"{exact_equal}"
)


print(
    "\n"
    + "=" * 100
)

print(
    "✅ V0 FEATURE REFACTOR PARITY PASSED"
)

print(
    "=" * 100
)
