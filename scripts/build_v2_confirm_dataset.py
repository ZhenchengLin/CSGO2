"""
Build the frozen V2 independent-confirmation dataset.

IMPORTANT
---------
This script may run ONLY AFTER:

1. docs/v2_confirm_manifest.csv exists
2. that manifest contains exactly 30 frozen matches
3. V2 frozen model verification has passed

This script performs feature extraction only.

It does NOT:
- train V0
- train H2
- load frozen models
- calculate predictions
- calculate scientific metrics
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from cs2_tactical_intelligence.v0.features import (
    build_features_for_demo,
)

from cs2_tactical_intelligence.v0.schema import (
    V0_MODEL_FEATURES,
)


MANIFEST_PATH = Path(
    "docs/v2_confirm_manifest.csv"
)

VERIFY_PATH = Path(
    "docs/v2_confirm_model_verification.json"
)

INCOMING_DIR = Path(
    "data/raw/v2_incoming"
)

OUTPUT_PATH = Path(
    "data/processed/v2_confirm_dataset.parquet"
)

AUDIT_PATH = Path(
    "data/interim/v2_confirm_dataset_audit.csv"
)

IDENTITY_PATH = Path(
    "docs/v2_confirm_dataset_identity.json"
)

EXPECTED_MATCHES = 30

EXPECTED_HORIZONS = {
    10,
    20,
    30,
    40,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    print(
        "\n"
        + "=" * 100
    )

    print(
        "V2 — BUILD FROZEN D_CONFIRM DATASET"
    )

    print(
        "=" * 100
    )

    # ---------------------------------------------------------
    # Guards
    # ---------------------------------------------------------

    require(
        MANIFEST_PATH.exists(),
        "Frozen D_confirm manifest does not exist.",
    )

    require(
        VERIFY_PATH.exists(),
        "Model-freeze verification record does not exist.",
    )

    require(
        not OUTPUT_PATH.exists(),
        f"Refusing to overwrite existing dataset: {OUTPUT_PATH}",
    )

    require(
        not IDENTITY_PATH.exists(),
        f"Refusing to overwrite identity record: {IDENTITY_PATH}",
    )

    verification = json.loads(
        VERIFY_PATH.read_text()
    )

    require(
        verification[
            "verification_status"
        ] == "PASS",
        "Frozen model verification did not PASS.",
    )

    require(
        verification[
            "scientific_metrics_calculated"
        ] is False,
        "Unexpected scientific metric state.",
    )

    manifest = pl.read_csv(
        MANIFEST_PATH,
        null_values=[""],
        infer_schema_length=1000,
    )

    require(
        manifest.height
        == EXPECTED_MATCHES,
        (
            "D_confirm manifest must contain exactly "
            f"{EXPECTED_MATCHES} matches. "
            f"Found {manifest.height}."
        ),
    )

    require(
        manifest[
            "demo_filename"
        ].n_unique()
        == EXPECTED_MATCHES,
        "Duplicate demo filenames in frozen manifest.",
    )

    require(
        manifest[
            "sha256"
        ].n_unique()
        == EXPECTED_MATCHES,
        "Duplicate SHA256 identities in frozen manifest.",
    )

    print(
        f"\nFrozen matches: {manifest.height}"
    )

    # ---------------------------------------------------------
    # Build each frozen match
    # ---------------------------------------------------------

    frames = []
    audit_rows = []

    for index, row in enumerate(
        manifest
        .sort(
            "selection_rank"
        )
        .iter_rows(
            named=True
        ),
        start=1,
    ):
        filename = row[
            "demo_filename"
        ]

        expected_sha = row[
            "sha256"
        ]

        demo_path = (
            INCOMING_DIR
            / filename
        )

        print(
            f"\n[{index:02d}/{EXPECTED_MATCHES:02d}] "
            f"{filename}"
        )

        require(
            demo_path.exists(),
            f"Missing frozen demo: {demo_path}",
        )

        actual_sha = (
            sha256_file(
                demo_path
            )
        )

        require(
            actual_sha
            == expected_sha,
            (
                "Demo SHA256 mismatch:\n"
                f"{filename}"
            ),
        )

        features, invalid_rounds = (
            build_features_for_demo(
                demo_path,
                allow_missing_snapshot_exclusions=True,
            )
        )

        require(
            features.height > 0,
            f"No features produced for {filename}",
        )

        require(
            set(
                features[
                    "horizon_sec"
                ].unique()
            )
            <= EXPECTED_HORIZONS,
            f"Unexpected horizon in {filename}",
        )

        missing_features = [
            feature
            for feature
            in V0_MODEL_FEATURES
            if feature
            not in features.columns
        ]

        require(
            not missing_features,
            (
                f"Missing frozen features in {filename}: "
                f"{missing_features}"
            ),
        )

        require(
            features[
                "demo_filename"
            ].n_unique()
            == 1,
            (
                "Unexpected demo identity count in "
                f"{filename}"
            ),
        )

        require(
            features[
                "demo_filename"
            ][0]
            == filename,
            (
                "Feature demo_filename does not match "
                f"manifest identity for {filename}"
            ),
        )

        null_counts = (
            features
            .select(
                V0_MODEL_FEATURES
            )
            .null_count()
            .row(0)
        )

        require(
            all(
                value == 0
                for value
                in null_counts
            ),
            f"Null predictive features in {filename}",
        )

        frames.append(
            features
        )

        audit_rows.append({
            "selection_rank":
                row["selection_rank"],

            "demo_filename":
                filename,

            "sha256":
                actual_sha,

            "n_observations":
                features.height,

            "n_rounds_with_observations":
                features[
                    "round_num"
                ].n_unique(),

            "n_invalid_rounds":
                len(
                    invalid_rounds
                ),

            "invalid_round_reasons":
                ";".join(
                    (
                        f"{item['round_num']}:"
                        f"{item['reason']}"
                    )
                    for item
                    in invalid_rounds
                ),

            "n_obs_10":
                features.filter(
                    pl.col(
                        "horizon_sec"
                    ) == 10
                ).height,

            "n_obs_20":
                features.filter(
                    pl.col(
                        "horizon_sec"
                    ) == 20
                ).height,

            "n_obs_30":
                features.filter(
                    pl.col(
                        "horizon_sec"
                    ) == 30
                ).height,

            "n_obs_40":
                features.filter(
                    pl.col(
                        "horizon_sec"
                    ) == 40
                ).height,
        })

        print(
            f"  ✅ observations={features.height}"
            f" | excluded rounds={len(invalid_rounds)}"
        )

    # ---------------------------------------------------------
    # Combine
    # ---------------------------------------------------------

    dataset = pl.concat(
        frames,
        how="vertical",
    )

    audit = pl.DataFrame(
        audit_rows
    )

    require(
        dataset[
            "demo_filename"
        ].n_unique()
        == EXPECTED_MATCHES,
        "Combined dataset does not contain 30 matches.",
    )

    key_columns = [
        "demo_filename",
        "round_num",
        "horizon_sec",
        "target_tick",
    ]

    duplicate_keys = (
        dataset
        .group_by(
            key_columns
        )
        .len()
        .filter(
            pl.col("len") > 1
        )
    )

    require(
        duplicate_keys.height == 0,
        "Duplicate observation keys detected.",
    )

    horizon_values = set(
        dataset[
            "horizon_sec"
        ].unique()
    )

    require(
        horizon_values
        == EXPECTED_HORIZONS,
        (
            "Expected all four frozen horizons. "
            f"Found {sorted(horizon_values)}"
        ),
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

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

    dataset_hash = (
        sha256_file(
            OUTPUT_PATH
        )
    )

    manifest_hash = (
        sha256_file(
            MANIFEST_PATH
        )
    )

    label_counts = (
        dataset
        .group_by(
            "label"
        )
        .len()
        .sort(
            "label"
        )
    )

    horizon_counts = (
        dataset
        .group_by(
            "horizon_sec"
        )
        .len()
        .sort(
            "horizon_sec"
        )
    )

    identity = {
        "role":
            "V2 independent confirmation dataset",

        "manifest":
            str(
                MANIFEST_PATH
            ),

        "manifest_sha256":
            manifest_hash,

        "dataset":
            str(
                OUTPUT_PATH
            ),

        "dataset_sha256":
            dataset_hash,

        "n_matches":
            int(
                dataset[
                    "demo_filename"
                ].n_unique()
            ),

        "n_observations":
            int(
                dataset.height
            ),

        "n_predictive_features":
            len(
                V0_MODEL_FEATURES
            ),

        "labels": {
            row["label"]:
                int(
                    row["len"]
                )

            for row
            in label_counts.iter_rows(
                named=True
            )
        },

        "horizons": {
            str(
                row["horizon_sec"]
            ):
                int(
                    row["len"]
                )

            for row
            in horizon_counts.iter_rows(
                named=True
            )
        },

        "scientific_metrics_calculated":
            False,

        "model_predictions_calculated":
            False,
    }

    IDENTITY_PATH.write_text(
        json.dumps(
            identity,
            indent=2,
        )
        + "\n"
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "D_CONFIRM DATASET SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        f"\nMatches:      "
        f"{identity['n_matches']}"
    )

    print(
        f"Observations: "
        f"{identity['n_observations']}"
    )

    print(
        "\nLabels:"
    )

    print(
        label_counts
    )

    print(
        "\nHorizons:"
    )

    print(
        horizon_counts
    )

    print(
        "\nDataset SHA256:"
    )

    print(
        dataset_hash
    )

    print(
        "\n✅ FROZEN D_CONFIRM DATASET BUILT"
    )

    print(
        "No model predictions or metrics were calculated."
    )


if __name__ == "__main__":
    main()
