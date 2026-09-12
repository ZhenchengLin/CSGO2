"""
Verify the frozen V2 confirmation model artifacts.

No model training.
No confirmation data access.
No scientific performance calculation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from xgboost import XGBClassifier

from cs2_tactical_intelligence.v0.schema import (
    V0_FEATURE_GROUPS,
    V0_MODEL_FEATURES,
)


FREEZE_PATH = Path(
    "docs/v2_confirm_model_freeze.json"
)

VERIFY_PATH = Path(
    "docs/v2_confirm_model_verification.json"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(
        path.read_bytes()
    )


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_xgb(path):
    model = XGBClassifier()
    model.load_model(path)
    return model


def logit(p):
    p = np.clip(
        p,
        1e-6,
        1.0 - 1e-6,
    )

    return np.log(
        p / (1.0 - p)
    )


def sigmoid(x):
    return 1.0 / (
        1.0 + np.exp(-x)
    )


def main():
    print(
        "\n"
        + "=" * 100
    )

    print(
        "V2 — VERIFY FROZEN CONFIRMATION MODELS"
    )

    print(
        "=" * 100
    )

    require(
        FREEZE_PATH.exists(),
        f"Missing freeze record: {FREEZE_PATH}",
    )

    record = json.loads(
        FREEZE_PATH.read_text()
    )

    require(
        record[
            "confirmation_data_accessed"
        ] is False,
        "Freeze record says confirmation data was accessed.",
    )

    # ---------------------------------------------------------
    # Training dataset identity
    # ---------------------------------------------------------

    dataset_path = Path(
        record[
            "training_dataset"
        ]
    )

    require(
        dataset_path.exists(),
        f"Missing dataset: {dataset_path}",
    )

    dataset_hash = sha256_file(
        dataset_path
    )

    require(
        dataset_hash
        == record[
            "training_dataset_sha256"
        ],
        "Training dataset SHA256 mismatch.",
    )

    print(
        "\n✅ Training dataset hash verified"
    )

    # ---------------------------------------------------------
    # Fit commit + exact committed fitting script
    # ---------------------------------------------------------

    fit_commit = record[
        "fit_git_commit"
    ]

    subprocess.run(
        [
            "git",
            "cat-file",
            "-e",
            f"{fit_commit}^{{commit}}",
        ],
        check=True,
    )

    committed_script = (
        subprocess.check_output([
            "git",
            "show",
            f"{fit_commit}:"
            f"{record['fit_script']}",
        ])
    )

    committed_script_hash = (
        sha256_bytes(
            committed_script
        )
    )

    require(
        committed_script_hash
        == record[
            "fit_script_sha256"
        ],
        "Committed fitting script hash mismatch.",
    )

    print(
        "✅ Fit commit and fitting script verified"
    )

    # ---------------------------------------------------------
    # Frozen artifact hashes
    # ---------------------------------------------------------

    artifact_paths = {}

    for name, info in (
        record[
            "artifacts"
        ].items()
    ):
        path = Path(
            info["path"]
        )

        require(
            path.exists(),
            f"Missing artifact: {path}",
        )

        actual_hash = (
            sha256_file(
                path
            )
        )

        require(
            actual_hash
            == info["sha256"],
            f"{name} SHA256 mismatch.",
        )

        artifact_paths[
            name
        ] = path

        print(
            f"✅ {name:12s} "
            "SHA256 verified"
        )

    # ---------------------------------------------------------
    # Feature contracts
    # ---------------------------------------------------------

    plant_features = (
        V0_FEATURE_GROUPS[
            "offensive_geometry"
        ]
        + V0_FEATURE_GROUPS[
            "motion"
        ]
        + V0_FEATURE_GROUPS[
            "combat"
        ]
        + V0_FEATURE_GROUPS[
            "defense"
        ]
    )

    site_features = (
        V0_FEATURE_GROUPS[
            "offensive_geometry"
        ]
        + V0_FEATURE_GROUPS[
            "motion"
        ]
        + V0_FEATURE_GROUPS[
            "defense"
        ]
    )

    require(
        record["v0"]["features"]
        == V0_MODEL_FEATURES,
        "Freeze record V0 feature contract changed.",
    )

    require(
        record["h2"][
            "plant_features"
        ]
        == plant_features,
        "Freeze record Plant feature contract changed.",
    )

    require(
        record["h2"][
            "site_features"
        ]
        == site_features,
        "Freeze record Site feature contract changed.",
    )

    # ---------------------------------------------------------
    # Reload models
    # ---------------------------------------------------------

    v0 = load_xgb(
        artifact_paths[
            "v0"
        ]
    )

    plant = load_xgb(
        artifact_paths[
            "h2_plant"
        ]
    )

    site = load_xgb(
        artifact_paths[
            "h2_site"
        ]
    )

    require(
        v0.get_booster().feature_names
        == V0_MODEL_FEATURES,
        "Reloaded V0 feature order mismatch.",
    )

    require(
        plant.get_booster().feature_names
        == plant_features,
        "Reloaded Plant Head feature order mismatch.",
    )

    require(
        site.get_booster().feature_names
        == site_features,
        "Reloaded Site Head feature order mismatch.",
    )

    print(
        "\n✅ Reloaded feature ordering verified"
    )

    # ---------------------------------------------------------
    # Platt identity
    # ---------------------------------------------------------

    platt = json.loads(
        artifact_paths[
            "h2_platt"
        ].read_text()
    )

    require(
        np.isclose(
            platt["slope"],
            record["h2"][
                "platt_slope"
            ],
            rtol=0.0,
            atol=1e-15,
        ),
        "Platt slope mismatch.",
    )

    require(
        np.isclose(
            platt["intercept"],
            record["h2"][
                "platt_intercept"
            ],
            rtol=0.0,
            atol=1e-15,
        ),
        "Platt intercept mismatch.",
    )

    require(
        platt[
            "n_calibration_rows"
        ]
        == 886,
        "Expected 886 Site calibration rows.",
    )

    require(
        platt[
            "n_calibration_matches"
        ]
        == 20,
        "Expected 20 calibration matches.",
    )

    print(
        "✅ Platt calibration identity verified"
    )

    # ---------------------------------------------------------
    # Reload prediction sanity check
    # Legacy rows only. NO scoring.
    # ---------------------------------------------------------

    df = pl.read_parquet(
        dataset_path
    )

    require(
        df.height
        == record[
            "training_rows"
        ],
        "Legacy row count mismatch.",
    )

    sample = df.head(
        min(
            64,
            df.height,
        )
    )

    X_v0 = (
        sample
        .select(
            V0_MODEL_FEATURES
        )
        .to_pandas()
    )

    X_plant = (
        sample
        .select(
            plant_features
        )
        .to_pandas()
    )

    X_site = (
        sample
        .select(
            site_features
        )
        .to_pandas()
    )

    p_v0 = (
        v0.predict_proba(
            X_v0
        )
        .astype(
            np.float64
        )
    )

    require(
        p_v0.shape[1] == 3,
        "V0 probability shape invalid.",
    )

    require(
        np.isfinite(
            p_v0
        ).all(),
        "V0 probabilities non-finite.",
    )

    require(
        np.allclose(
            p_v0.sum(
                axis=1
            ),
            1.0,
            atol=1e-8,
        ),
        "V0 probabilities do not sum to 1.",
    )

    q = (
        plant
        .predict_proba(
            X_plant
        )[:, 1]
        .astype(
            np.float64
        )
    )

    raw_r = (
        site
        .predict_proba(
            X_site
        )[:, 1]
        .astype(
            np.float64
        )
    )

    calibrated_r = sigmoid(
        platt["slope"]
        * logit(
            raw_r
        )
        + platt[
            "intercept"
        ]
    )

    p_h2 = np.column_stack([
        q * calibrated_r,
        q * (
            1.0
            - calibrated_r
        ),
        1.0 - q,
    ])

    require(
        np.isfinite(
            p_h2
        ).all(),
        "H2 probabilities non-finite.",
    )

    require(
        (
            p_h2 >= 0
        ).all()
        and (
            p_h2 <= 1
        ).all(),
        "H2 probability outside [0,1].",
    )

    require(
        np.allclose(
            p_h2.sum(
                axis=1
            ),
            1.0,
            atol=1e-12,
        ),
        "H2 probabilities do not sum to 1.",
    )

    print(
        "✅ Reloaded probability checks passed"
    )

    # ---------------------------------------------------------
    # Verification record
    # ---------------------------------------------------------

    verification = {
        "verification_status":
            "PASS",

        "verified_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "model_freeze_record":
            str(
                FREEZE_PATH
            ),

        "fit_git_commit":
            fit_commit,

        "training_dataset_sha256":
            dataset_hash,

        "artifact_sha256": {
            name:
                sha256_file(path)

            for name, path
            in artifact_paths.items()
        },

        "checks": {
            "confirmation_data_accessed":
                False,

            "training_dataset_hash":
                True,

            "fit_commit_exists":
                True,

            "committed_fit_script_hash":
                True,

            "artifact_hashes":
                True,

            "feature_contracts":
                True,

            "reloaded_feature_order":
                True,

            "platt_identity":
                True,

            "probability_sanity":
                True,
        },

        "scientific_metrics_calculated":
            False,
    }

    VERIFY_PATH.write_text(
        json.dumps(
            verification,
            indent=2,
        )
        + "\n"
    )

    print(
        "\nVerification record:"
    )

    print(
        VERIFY_PATH
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "✅ V2 MODEL FREEZE VERIFICATION PASSED"
    )

    print(
        "No confirmation metrics were calculated."
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
