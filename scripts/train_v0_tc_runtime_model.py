from pathlib import Path
import hashlib
import json

import numpy as np
import polars as pl

from cs2_tactical_intelligence.v0.model import (
    V0_XGB_CONFIG,
    make_v0_model,
)

from cs2_tactical_intelligence.v0.schema import (
    V0_LABELS,
    V0_MODEL_FEATURES,
    V0_N_FEATURES,
)


DATASET_PATH = Path(
    "data/processed/v0_dataset_v2_timing_corrected.parquet"
)

MODEL_PATH = Path(
    "artifacts/v0_tc_xgb_a5_runtime.json"
)

METADATA_PATH = Path(
    "artifacts/v0_tc_xgb_a5_runtime_metadata.json"
)


LABEL_TO_ID = {
    label: index
    for index, label in enumerate(V0_LABELS)
}


print("\n" + "=" * 100)
print("V0 — TRAIN FULL-DEVELOPMENT RUNTIME MODEL")
print("=" * 100)


# ==================================================
# Load frozen V0 development dataset
# ==================================================

df = pl.read_parquet(
    DATASET_PATH
)


print(
    f"\nObservations: {df.height}"
)

print(
    f"Matches:      "
    f"{df['demo_filename'].n_unique()}"
)

print(
    f"Features:     {V0_N_FEATURES}"
)


assert df.height == 1686

assert (
    df["demo_filename"].n_unique()
    == 20
)

assert V0_N_FEATURES == 37


# ==================================================
# Validate feature contract
# ==================================================

missing_features = [
    feature
    for feature in V0_MODEL_FEATURES
    if feature not in df.columns
]


if missing_features:
    raise ValueError(
        f"Missing V0 features: "
        f"{missing_features}"
    )


print(
    "✅ Frozen 37-feature contract available"
)


# ==================================================
# Prepare X / y
# ==================================================

X = (
    df
    .select(
        V0_MODEL_FEATURES
    )
    .to_pandas()
)


y_labels = (
    df["label"]
    .to_list()
)


y = np.array([
    LABEL_TO_ID[label]
    for label in y_labels
])


assert sorted(
    np.unique(y).tolist()
) == [0, 1, 2]


print(
    "✅ Label encoding validated"
)


# ==================================================
# Train runtime model
# ==================================================

model = make_v0_model()


print(
    "\nTraining frozen XGB-A5 "
    "runtime configuration..."
)


model.fit(
    X,
    y,
)


print(
    "✅ Runtime model trained"
)


# ==================================================
# Validate model feature names
# ==================================================

booster_features = (
    model
    .get_booster()
    .feature_names
)


assert (
    booster_features
    == V0_MODEL_FEATURES
), (
    "Model feature ordering does not match "
    "the frozen V0 schema."
)


print(
    "✅ Model feature ordering matches schema"
)


# ==================================================
# Save model
# ==================================================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


model.save_model(
    MODEL_PATH
)


print(
    f"✅ Saved model: {MODEL_PATH}"
)


# ==================================================
# Dataset fingerprint
# ==================================================

dataset_sha256 = hashlib.sha256(
    DATASET_PATH.read_bytes()
).hexdigest()


# ==================================================
# Metadata
# ==================================================

label_counts = (
    df
    .group_by("label")
    .len()
    .sort("label")
)


metadata = {

    "version":
        "V0-TIMING-CORRECTED",

    "model_name":
        "XGB-A5",

    "role":
        "full-development runtime model",

    "evaluation_status":
        (
            "NOT an independent evaluation model. "
            "Reported V0 performance remains the "
            "frozen grouped OOF development result."
        ),

    "dataset":
        str(DATASET_PATH),

    "dataset_sha256":
        dataset_sha256,

    "n_observations":
        df.height,

    "n_matches":
        df[
            "demo_filename"
        ].n_unique(),

    "labels":
        V0_LABELS,

    "label_to_id":
        LABEL_TO_ID,

    "label_counts": {
        row["label"]:
            row["len"]
        for row
        in label_counts.iter_rows(
            named=True
        )
    },

    "n_features":
        V0_N_FEATURES,

    "features":
        V0_MODEL_FEATURES,

    "xgboost_config":
        V0_XGB_CONFIG,

    "frozen_development_metrics": {
        "log_loss":
            0.838182,

        "brier":
            0.503190,

        "accuracy":
            0.594899,

        "macro_f1":
            0.566090,
    },
}


METADATA_PATH.write_text(
    json.dumps(
        metadata,
        indent=2,
    )
)


print(
    f"✅ Saved metadata: "
    f"{METADATA_PATH}"
)


print(
    "\nDataset SHA256:"
)

print(
    dataset_sha256
)


print("\n" + "=" * 100)

print(
    "✅ V0 FULL-DEVELOPMENT "
    "RUNTIME MODEL READY"
)

print("=" * 100)
