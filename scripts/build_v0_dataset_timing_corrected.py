"""
Rebuild the V0 dataset after correcting the historical-demo time contract.

IMPORTANT
---------
This does NOT overwrite the historical 1268-row V0 dataset.

Historical artifact:
    data/processed/v0_dataset_v1.parquet

Corrected artifact:
    data/processed/v0_dataset_v2_timing_corrected.parquet

Correct timing:
    64 raw demo ticks / game second

Correct V0 observation horizons:
    10 / 20 / 30 / 40 real seconds after freeze_end

Correct motion window:
    1 real second
"""

from pathlib import Path

import polars as pl

from cs2_tactical_intelligence.v0.features import (
    build_features_for_demo,
)


RAW_DIR = Path("data/raw")

MANIFEST_PATH = (
    RAW_DIR
    / "demo_manifest.csv"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "v0_dataset_v2_timing_corrected.parquet"
)

AUDIT_PATH = Path(
    "data/interim/"
    "v0_feature_build_audit_timing_corrected.csv"
)


EXPECTED_HORIZON_COUNTS = {
    10: 442,
    20: 442,
    30: 429,
    40: 373,
}

EXPECTED_TOTAL = 1686
EXPECTED_MATCHES = 20


manifest = pl.read_csv(
    MANIFEST_PATH,
    null_values=[""],
    infer_schema_length=1000,
)


all_features = []
audit_rows = []


print("\n" + "=" * 100)
print("V0 — TIMING-CORRECTED FEATURE BUILD")
print("=" * 100)


for index, row in enumerate(
    manifest.iter_rows(named=True),
    start=1,
):

    filename = row["demo filename"]
    path = RAW_DIR / filename

    print(
        f"\n[{index:02d}/{manifest.height:02d}] "
        f"{filename}"
    )

    try:

        features, invalid_rounds = (
            build_features_for_demo(
                path
            )
        )

        print(
            f"  ✅ features={features.height}"
            f" | invalid rounds="
            f"{len(invalid_rounds)}"
        )

        all_features.append(
            features
        )

        audit_rows.append({
            "demo_filename":
                filename,

            "status":
                "built_ok",

            "n_features":
                features.height,

            "n_invalid_rounds":
                len(invalid_rounds),

            "error":
                None,
        })

    except Exception as exc:

        print(
            f"  ❌ FAILED: {exc}"
        )

        audit_rows.append({
            "demo_filename":
                filename,

            "status":
                "build_failed",

            "n_features":
                None,

            "n_invalid_rounds":
                None,

            "error":
                str(exc),
        })


audit = pl.DataFrame(
    audit_rows
)

successful = audit.filter(
    pl.col("status") == "built_ok"
)

failed = audit.filter(
    pl.col("status") != "built_ok"
)


if len(all_features) == 0:
    raise RuntimeError(
        "No demos produced features."
    )


dataset = pl.concat(
    all_features,
    how="vertical",
)


print("\n" + "=" * 100)
print("CORRECTED DATASET SUMMARY")
print("=" * 100)

print(
    f"Demos built:       "
    f"{successful.height}"
)

print(
    f"Demos failed:      "
    f"{failed.height}"
)

print(
    f"Observations:      "
    f"{dataset.height}"
)

print(
    f"Columns:           "
    f"{dataset.width}"
)


print("\nLABEL DISTRIBUTION")

label_summary = (
    dataset
    .group_by("label")
    .len()
    .sort("label")
)

print(
    label_summary
)


print("\nOBSERVATIONS BY HORIZON")

horizon_summary = (
    dataset
    .group_by("horizon_sec")
    .len()
    .sort("horizon_sec")
)

print(
    horizon_summary
)


actual_horizon_counts = dict(
    horizon_summary
    .select([
        "horizon_sec",
        "len",
    ])
    .iter_rows()
)


print("\nEXPECTED HORIZON COUNTS")
print(
    EXPECTED_HORIZON_COUNTS
)

print("\nACTUAL HORIZON COUNTS")
print(
    actual_horizon_counts
)


print("\nUNIQUE MATCHES")

n_matches = (
    dataset[
        "demo_filename"
    ]
    .n_unique()
)

print(
    n_matches
)


print("\nNULL COUNTS")

print(
    dataset.null_count()
)


# ==================================================
# Corrected-timing QA gates
# ==================================================

assert (
    failed.height == 0
), (
    f"{failed.height} demo builds failed"
)

assert (
    dataset.height
    == EXPECTED_TOTAL
), (
    f"Expected {EXPECTED_TOTAL} observations, "
    f"got {dataset.height}"
)

assert (
    n_matches
    == EXPECTED_MATCHES
)

assert (
    actual_horizon_counts
    == EXPECTED_HORIZON_COUNTS
), (
    f"Horizon counts differ: "
    f"{actual_horizon_counts}"
)


model_numeric_columns = [
    col
    for col, dtype
    in zip(
        dataset.columns,
        dataset.dtypes,
    )
    if dtype.is_numeric()
]

numeric_null_counts = (
    dataset
    .select(
        model_numeric_columns
    )
    .null_count()
    .row(0)
)

assert (
    numeric_null_counts
    == tuple(
        0
        for _ in model_numeric_columns
    )
)


# ==================================================
# Save ONLY after QA passes
# ==================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

AUDIT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

dataset.write_parquet(
    OUTPUT_PATH
)

audit.write_csv(
    AUDIT_PATH
)


print("\n" + "=" * 100)
print("✅ TIMING-CORRECTED V0 DATASET VERIFIED")
print("=" * 100)

print("\nSaved corrected dataset:")
print(
    OUTPUT_PATH
)

print("\nSaved corrected build audit:")
print(
    AUDIT_PATH
)
