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
    "data/processed/v0_dataset_v1.parquet"
)

AUDIT_PATH = Path(
    "data/interim/v0_feature_build_audit.csv"
)


manifest = pl.read_csv(
    MANIFEST_PATH,
    null_values=[""],
    infer_schema_length=1000,
)


all_features = []
audit_rows = []


print("\n" + "=" * 100)
print("V0 — BATCH FEATURE BUILD")
print("=" * 100)


for index, row in enumerate(
    manifest.iter_rows(named=True),
    start=1,
):

    filename = row[
        "demo filename"
    ]

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


        expected_obs = row.get(
            "n_observations"
        )


        if (
            expected_obs is not None
            and features.height
            != expected_obs
        ):
            raise ValueError(
                f"Observation mismatch: "
                f"manifest={expected_obs}, "
                f"features={features.height}"
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


# ==================================================
# Combine dataset
# ==================================================

if len(all_features) == 0:
    raise RuntimeError(
        "No demos produced features."
    )


dataset = pl.concat(
    all_features,
    how="vertical",
)


# ==================================================
# Save
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


# ==================================================
# QA summary
# ==================================================

print("\n" + "=" * 100)
print("DATASET SUMMARY")
print("=" * 100)


print(
    f"Demos built:        "
    f"{successful.height}"
)

print(
    f"Demos failed:       "
    f"{failed.height}"
)

print(
    f"Observations:       "
    f"{dataset.height}"
)

print(
    f"Columns:            "
    f"{dataset.width}"
)


print("\nLABEL DISTRIBUTION — OBSERVATION LEVEL")

print(
    dataset
    .group_by("label")
    .len()
    .sort("label")
)


print("\nOBSERVATIONS BY HORIZON")

print(
    dataset
    .group_by("horizon_sec")
    .len()
    .sort("horizon_sec")
)


print("\nNULL COUNTS")

nulls = dataset.null_count()

print(
    nulls
)


print("\nUNIQUE MATCHES")

print(
    dataset[
        "demo_filename"
    ].n_unique()
)


# ==================================================
# Important QA checks
# ==================================================

assert dataset.height == 1268

assert (
    dataset[
        "demo_filename"
    ].n_unique()
    == 20
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


assert (
    dataset
    .select(
        model_numeric_columns
    )
    .null_count()
    .row(0)
    == tuple(
        0
        for _ in model_numeric_columns
    )
)


if failed.height == 0:

    print(
        "\n✅ ALL 20 DEMOS BUILT SUCCESSFULLY"
    )

else:

    print(
        "\n❌ FEATURE BUILD FAILURES"
    )

    print(
        failed.select([
            "demo_filename",
            "error",
        ])
    )


print("\nSaved dataset:")
print(OUTPUT_PATH)

print("\nSaved build audit:")
print(AUDIT_PATH)
