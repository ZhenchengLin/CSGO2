"""
Fit and freeze the models used for V2 independent confirmation.

IMPORTANT
---------
This script uses ONLY the legacy 20-match development dataset.

It does NOT read:
- data/raw/v2_incoming
- data/interim/v2_intake_audit.csv
- docs/v2_confirm_manifest.csv

It does NOT calculate confirmation performance.

Outputs:
    artifacts/v2_confirm_frozen/
        v0_xgb_a5.json
        h2_plant_head.json
        h2_site_head.json
        h2_platt.json

    docs/v2_confirm_model_freeze.json
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import polars as pl

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from xgboost import XGBClassifier

from cs2_tactical_intelligence.v0.model import (
    V0_XGB_CONFIG,
    make_v0_model,
)

from cs2_tactical_intelligence.v0.schema import (
    V0_FEATURE_GROUPS,
    V0_LABELS,
    V0_MODEL_FEATURES,
)


DATASET_PATH = Path(
    "data/processed/v0_dataset_v2_timing_corrected.parquet"
)

SCRIPT_PATH = Path(
    "scripts/fit_v2_confirm_models.py"
)

ARTIFACT_DIR = Path(
    "artifacts/v2_confirm_frozen"
)

TEMP_DIR = Path(
    "artifacts/v2_confirm_frozen_tmp"
)

FREEZE_RECORD_PATH = Path(
    "docs/v2_confirm_model_freeze.json"
)

EXPECTED_ROWS = 1686
EXPECTED_MATCHES = 20

INNER_SPLITS = 4


PLANT_FEATURES = (
    V0_FEATURE_GROUPS["offensive_geometry"]
    + V0_FEATURE_GROUPS["motion"]
    + V0_FEATURE_GROUPS["combat"]
    + V0_FEATURE_GROUPS["defense"]
)

SITE_FEATURES = (
    V0_FEATURE_GROUPS["offensive_geometry"]
    + V0_FEATURE_GROUPS["motion"]
    + V0_FEATURE_GROUPS["defense"]
)


assert len(V0_MODEL_FEATURES) == 37
assert len(PLANT_FEATURES) == 36
assert len(SITE_FEATURES) == 27

assert len(set(V0_MODEL_FEATURES)) == 37
assert len(set(PLANT_FEATURES)) == 36
assert len(set(SITE_FEATURES)) == 27


H2_BINARY_CONFIG = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,

    "min_child_weight": 5,

    "subsample": 0.8,
    "colsample_bytree": 0.8,

    "reg_alpha": 0.5,
    "reg_lambda": 5.0,

    "objective": "binary:logistic",
    "eval_metric": "logloss",

    "tree_method": "hist",

    "random_state": 42,
    "n_jobs": -1,
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

            digest.update(
                block
            )

    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def git_head() -> str:
    return (
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            text=True,
        )
        .strip()
    )


def tracked_worktree_clean() -> bool:
    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return (
        result.stdout.strip()
        == ""
    )


def logit_feature(
    probabilities,
):
    p = np.clip(
        probabilities,
        1e-6,
        1.0 - 1e-6,
    )

    return np.log(
        p
        / (
            1.0
            - p
        )
    ).reshape(
        -1,
        1,
    )


def verify_feature_names(
    model,
    expected,
    name,
):
    actual = (
        model
        .get_booster()
        .feature_names
    )

    if actual != expected:
        raise RuntimeError(
            f"{name} feature ordering mismatch."
        )


def main():
    print(
        "\n"
        + "=" * 100
    )

    print(
        "V2 — FIT FROZEN CONFIRMATION MODELS"
    )

    print(
        "=" * 100
    )

    # =========================================================
    # Freeze safety
    # =========================================================

    if FREEZE_RECORD_PATH.exists():
        raise RuntimeError(
            "Freeze record already exists:\n"
            f"{FREEZE_RECORD_PATH}\n"
            "Refusing to regenerate frozen models."
        )

    if ARTIFACT_DIR.exists():
        raise RuntimeError(
            "Frozen artifact directory already exists:\n"
            f"{ARTIFACT_DIR}"
        )

    if TEMP_DIR.exists():
        shutil.rmtree(
            TEMP_DIR
        )

    if not tracked_worktree_clean():
        raise RuntimeError(
            "Tracked git working tree is not clean.\n"
            "Commit the fitting script before running it."
        )

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            DATASET_PATH
        )

    print(
        "\n✅ Git tracked working tree clean"
    )

    print(
        f"Git commit: {git_head()}"
    )

    # =========================================================
    # Load D_legacy
    # =========================================================

    df = pl.read_parquet(
        DATASET_PATH
    )

    assert (
        df.height
        == EXPECTED_ROWS
    )

    assert (
        df[
            "demo_filename"
        ].n_unique()
        == EXPECTED_MATCHES
    )

    required_features = sorted(
        set(
            V0_MODEL_FEATURES
            + PLANT_FEATURES
            + SITE_FEATURES
        )
    )

    missing = [
        feature
        for feature
        in required_features
        if feature
        not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing frozen features: "
            f"{missing}"
        )

    dataset_sha256 = (
        sha256_file(
            DATASET_PATH
        )
    )

    print(
        "\nLegacy dataset:"
    )

    print(
        f"  observations: {df.height}"
    )

    print(
        "  matches:      "
        f"{df['demo_filename'].n_unique()}"
    )

    print(
        "  SHA256:       "
        f"{dataset_sha256}"
    )

    y_labels = (
        df["label"]
        .to_numpy()
    )

    groups = (
        df["demo_filename"]
        .to_numpy()
    )

    label_to_id = {
        label: index
        for index, label
        in enumerate(
            V0_LABELS
        )
    }

    y_v0 = np.array([
        label_to_id[label]
        for label
        in y_labels
    ])

    y_plant = (
        y_labels
        != "NO_PLANT"
    ).astype(int)

    y_site_all = (
        y_labels
        == "A_PLANT"
    ).astype(int)

    plant_mask = (
        y_labels
        != "NO_PLANT"
    )

    # Pandas keeps feature names inside XGBoost artifacts.

    X_v0 = (
        df
        .select(
            V0_MODEL_FEATURES
        )
        .to_pandas()
    )

    X_plant = (
        df
        .select(
            PLANT_FEATURES
        )
        .to_pandas()
    )

    X_site_all = (
        df
        .select(
            SITE_FEATURES
        )
        .to_pandas()
    )

    if not np.isfinite(
        X_v0.to_numpy()
    ).all():
        raise RuntimeError(
            "Non-finite V0 features."
        )

    if not np.isfinite(
        X_plant.to_numpy()
    ).all():
        raise RuntimeError(
            "Non-finite H2 Plant features."
        )

    if not np.isfinite(
        X_site_all.to_numpy()
    ).all():
        raise RuntimeError(
            "Non-finite H2 Site features."
        )

    # =========================================================
    # Fit frozen V0
    # =========================================================

    print(
        "\n[1/4] Fitting frozen V0 XGB-A5..."
    )

    v0_model = (
        make_v0_model()
    )

    v0_model.fit(
        X_v0,
        y_v0,
    )

    verify_feature_names(
        v0_model,
        V0_MODEL_FEATURES,
        "V0",
    )

    print(
        "✅ V0 fitted"
    )

    # =========================================================
    # Fit frozen H2 Plant Head
    # =========================================================

    print(
        "\n[2/4] Fitting H2 Plant Head..."
    )

    plant_model = (
        XGBClassifier(
            **H2_BINARY_CONFIG
        )
    )

    plant_model.fit(
        X_plant,
        y_plant,
    )

    verify_feature_names(
        plant_model,
        PLANT_FEATURES,
        "H2 Plant Head",
    )

    print(
        "✅ H2 Plant Head fitted"
    )

    # =========================================================
    # Grouped OOF Site probabilities for Platt
    # =========================================================

    print(
        "\n[3/4] Building grouped Site OOF "
        "probabilities for Platt calibration..."
    )

    plant_indices = np.where(
        plant_mask
    )[0]

    X_site_plant = (
        X_site_all.iloc[
            plant_indices
        ]
        .reset_index(
            drop=True
        )
    )

    y_site = (
        y_site_all[
            plant_indices
        ]
    )

    site_groups = (
        groups[
            plant_indices
        ]
    )

    assert set(
        np.unique(
            y_site
        )
    ) == {
        0,
        1,
    }

    splitter = (
        StratifiedGroupKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=42,
        )
    )

    site_oof = np.full(
        len(
            plant_indices
        ),
        np.nan,
        dtype=np.float64,
    )

    split_rows = []

    for fold, (
        train_index,
        valid_index,
    ) in enumerate(
        splitter.split(
            X_site_plant,
            y_site,
            site_groups,
        ),
        start=1,
    ):
        train_groups = set(
            site_groups[
                train_index
            ]
        )

        valid_groups = set(
            site_groups[
                valid_index
            ]
        )

        assert not (
            train_groups
            & valid_groups
        )

        if len(
            np.unique(
                y_site[
                    train_index
                ]
            )
        ) != 2:
            raise RuntimeError(
                f"Site calibration fold {fold} "
                "training set lacks both classes."
            )

        inner_model = (
            XGBClassifier(
                **H2_BINARY_CONFIG
            )
        )

        inner_model.fit(
            X_site_plant.iloc[
                train_index
            ],
            y_site[
                train_index
            ],
        )

        predictions = (
            inner_model
            .predict_proba(
                X_site_plant.iloc[
                    valid_index
                ]
            )[:, 1]
            .astype(
                np.float64
            )
        )

        site_oof[
            valid_index
        ] = predictions

        split_rows.append({
            "fold":
                fold,

            "n_train_rows":
                int(
                    len(
                        train_index
                    )
                ),

            "n_valid_rows":
                int(
                    len(
                        valid_index
                    )
                ),

            "n_train_matches":
                len(
                    train_groups
                ),

            "n_valid_matches":
                len(
                    valid_groups
                ),
        })

        print(
            f"  fold {fold}: "
            f"train={len(train_index)}, "
            f"valid={len(valid_index)}, "
            f"valid matches={len(valid_groups)}"
        )

    if not np.isfinite(
        site_oof
    ).all():
        raise RuntimeError(
            "Incomplete Site OOF predictions."
        )

    # =========================================================
    # Fit final Platt calibrator
    # =========================================================

    calibrator = (
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        )
    )

    calibrator.fit(
        logit_feature(
            site_oof
        ),
        y_site,
    )

    platt_slope = float(
        calibrator.coef_[
            0,
            0,
        ]
    )

    platt_intercept = float(
        calibrator.intercept_[
            0
        ]
    )

    print(
        "\nPlatt parameters:"
    )

    print(
        f"  slope:     {platt_slope:.12f}"
    )

    print(
        f"  intercept: {platt_intercept:.12f}"
    )

    # =========================================================
    # Fit final H2 Site Head
    # =========================================================

    print(
        "\n[4/4] Fitting final H2 Site Head..."
    )

    site_model = (
        XGBClassifier(
            **H2_BINARY_CONFIG
        )
    )

    site_model.fit(
        X_site_plant,
        y_site,
    )

    verify_feature_names(
        site_model,
        SITE_FEATURES,
        "H2 Site Head",
    )

    print(
        "✅ H2 Site Head fitted"
    )

    # =========================================================
    # Basic prediction sanity checks only
    # No scientific performance metrics.
    # =========================================================

    sample_n = min(
        32,
        df.height,
    )

    q = (
        plant_model
        .predict_proba(
            X_plant.iloc[
                :sample_n
            ]
        )[:, 1]
        .astype(
            np.float64
        )
    )

    raw_r = (
        site_model
        .predict_proba(
            X_site_all.iloc[
                :sample_n
            ]
        )[:, 1]
        .astype(
            np.float64
        )
    )

    calibrated_r = (
        calibrator
        .predict_proba(
            logit_feature(
                raw_r
            )
        )[:, 1]
        .astype(
            np.float64
        )
    )

    composed = np.column_stack([
        q
        * calibrated_r,

        q
        * (
            1.0
            - calibrated_r
        ),

        1.0
        - q,
    ])

    assert np.isfinite(
        composed
    ).all()

    assert (
        composed >= 0.0
    ).all()

    assert (
        composed <= 1.0
    ).all()

    assert np.allclose(
        composed.sum(
            axis=1
        ),
        1.0,
        atol=1e-12,
    )

    print(
        "\n✅ H2 probability composition sanity check passed"
    )

    # =========================================================
    # Save through temporary directory
    # =========================================================

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    v0_path = (
        TEMP_DIR
        / "v0_xgb_a5.json"
    )

    plant_path = (
        TEMP_DIR
        / "h2_plant_head.json"
    )

    site_path = (
        TEMP_DIR
        / "h2_site_head.json"
    )

    platt_path = (
        TEMP_DIR
        / "h2_platt.json"
    )

    v0_model.save_model(
        v0_path
    )

    plant_model.save_model(
        plant_path
    )

    site_model.save_model(
        site_path
    )

    platt_record = {
        "method":
            "Platt calibration on Site Head logit",

        "input":
            "logit(raw Site probability)",

        "clip_epsilon":
            1e-6,

        "slope":
            platt_slope,

        "intercept":
            platt_intercept,

        "logistic_regression": {
            "max_iter":
                2000,

            "random_state":
                42,
        },

        "inner_grouped_cv": {
            "splitter":
                "StratifiedGroupKFold",

            "n_splits":
                INNER_SPLITS,

            "shuffle":
                True,

            "random_state":
                42,

            "group":
                "demo_filename",
        },

        "n_calibration_rows":
            int(
                len(
                    y_site
                )
            ),

        "n_calibration_matches":
            int(
                len(
                    set(
                        site_groups
                    )
                )
            ),

        "folds":
            split_rows,
    }

    platt_path.write_text(
        json.dumps(
            platt_record,
            indent=2,
        )
        + "\n"
    )

    TEMP_DIR.replace(
        ARTIFACT_DIR
    )

    # =========================================================
    # Freeze artifact hashes
    # =========================================================

    artifact_paths = {
        "v0":
            ARTIFACT_DIR
            / "v0_xgb_a5.json",

        "h2_plant":
            ARTIFACT_DIR
            / "h2_plant_head.json",

        "h2_site":
            ARTIFACT_DIR
            / "h2_site_head.json",

        "h2_platt":
            ARTIFACT_DIR
            / "h2_platt.json",
    }

    artifact_hashes = {
        name:
            sha256_file(path)

        for name, path
        in artifact_paths.items()
    }

    freeze_record = {
        "freeze_name":
            "V2 independent confirmation model freeze",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "fit_git_commit":
            git_head(),

        "fit_script":
            str(
                SCRIPT_PATH
            ),

        "fit_script_sha256":
            sha256_file(
                SCRIPT_PATH
            ),

        "training_dataset":
            str(
                DATASET_PATH
            ),

        "training_dataset_sha256":
            dataset_sha256,

        "training_rows":
            int(
                df.height
            ),

        "training_matches":
            int(
                df[
                    "demo_filename"
                ].n_unique()
            ),

        "labels":
            list(
                V0_LABELS
            ),

        "package_versions": {
            "numpy":
                package_version(
                    "numpy"
                ),

            "polars":
                package_version(
                    "polars"
                ),

            "scikit-learn":
                package_version(
                    "scikit-learn"
                ),

            "xgboost":
                package_version(
                    "xgboost"
                ),
        },

        "v0": {
            "features":
                V0_MODEL_FEATURES,

            "n_features":
                len(
                    V0_MODEL_FEATURES
                ),

            "config":
                V0_XGB_CONFIG,
        },

        "h2": {
            "plant_features":
                PLANT_FEATURES,

            "n_plant_features":
                len(
                    PLANT_FEATURES
                ),

            "site_features":
                SITE_FEATURES,

            "n_site_features":
                len(
                    SITE_FEATURES
                ),

            "binary_xgboost_config":
                H2_BINARY_CONFIG,

            "plant_training_rows":
                int(
                    df.height
                ),

            "site_training_rows":
                int(
                    plant_mask.sum()
                ),

            "platt_slope":
                platt_slope,

            "platt_intercept":
                platt_intercept,
        },

        "artifacts": {
            name: {
                "path":
                    str(
                        path
                    ),

                "sha256":
                    artifact_hashes[
                        name
                    ],
            }

            for name, path
            in artifact_paths.items()
        },

        "confirmation_data_accessed":
            False,

        "scientific_note":
            (
                "These models were fitted only on D_legacy. "
                "No V2 confirmation performance was calculated."
            ),
    }

    FREEZE_RECORD_PATH.write_text(
        json.dumps(
            freeze_record,
            indent=2,
        )
        + "\n"
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FROZEN ARTIFACT HASHES"
    )

    print(
        "=" * 100
    )

    for name, digest in (
        artifact_hashes.items()
    ):
        print(
            f"{name:12s} {digest}"
        )

    print(
        "\nFreeze record:"
    )

    print(
        FREEZE_RECORD_PATH
    )

    print(
        "\n✅ V2 CONFIRMATION MODELS FROZEN"
    )

    print(
        "\nNo D_confirm performance was calculated."
    )


if __name__ == "__main__":
    main()
