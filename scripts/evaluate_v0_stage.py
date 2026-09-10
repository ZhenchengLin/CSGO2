import sys
from pathlib import Path

import numpy as np
import polars as pl

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path(
    "data/processed/v0_dataset_v1.parquet"
)

FOLD_PATH = Path(
    "data/processed/v0_a1_oof_predictions.parquet"
)

LABELS = [
    "A_PLANT",
    "B_PLANT",
    "NO_PLANT",
]


# ==================================================
# Feature families
# ==================================================

A1 = [
    "horizon_sec",

    "t_centroid_x",
    "t_centroid_y",
    "t_centroid_z",

    "t_stretch_xy",
    "t_range_x",
    "t_range_y",

    "t_mean_pairwise_distance",
    "t_convex_hull_area",

    "bomb_x",
    "bomb_y",
    "bomb_z",

    "bomb_to_t_centroid_distance",
]


A2 = [
    "t_mean_speed_1s",
    "t_centroid_velocity_x_1s",
    "t_centroid_velocity_y_1s",
    "bomb_speed_1s",
]


A3 = [
    "t_alive",
    "ct_alive",
    "alive_difference",

    "t_health_sum",
    "ct_health_sum",
    "health_difference",

    "t_armor_sum",
    "ct_armor_sum",
    "armor_difference",
]


A4 = [
    "t_equip_value_sum",
    "ct_equip_value_sum",
    "equip_value_difference",
]


A5 = [
    "ct_centroid_x",
    "ct_centroid_y",
    "ct_centroid_z",

    "ct_stretch_xy",

    "ct_range_x",
    "ct_range_y",

    "ct_mean_pairwise_distance",
    "ct_convex_hull_area",

    "t_ct_centroid_distance",
    "minimum_t_ct_distance",
    "mean_nearest_opponent_distance",
]


STAGES = {
    "A3": {
        "features": A1 + A2 + A3,
        "new_features": A3,
        "previous": "A2",
    },

    "A4": {
        "features": A1 + A2 + A3 + A4,
        "new_features": A4,
        "previous": "A3",
    },

    "A5": {
        # A4 Economy was dropped after ablation.
        # Compare Defense directly against the
        # current best feature set: A1 + A2 + A3.
        "features": A1 + A2 + A3 + A5,
        "new_features": A5,
        "previous": "A3",
    },
}


# ==================================================
# Helpers
# ==================================================

def multiclass_brier_score(
    y_true,
    probabilities,
):

    label_to_index = {
        label: i
        for i, label in enumerate(LABELS)
    }

    truth = np.zeros(
        (
            len(y_true),
            len(LABELS),
        ),
        dtype=float,
    )

    for i, label in enumerate(y_true):
        truth[
            i,
            label_to_index[label],
        ] = 1.0

    return np.mean(
        np.sum(
            (
                probabilities
                - truth
            ) ** 2,
            axis=1,
        )
    )


def reorder_probabilities(
    probabilities,
    model_classes,
):

    mapping = {
        label: i
        for i, label in enumerate(model_classes)
    }

    return np.column_stack([
        probabilities[
            :,
            mapping[label]
        ]
        for label in LABELS
    ])


def evaluate(
    y,
    probabilities,
):

    predictions = np.array([
        LABELS[i]
        for i in np.argmax(
            probabilities,
            axis=1,
        )
    ])

    recalls = recall_score(
        y,
        predictions,
        labels=LABELS,
        average=None,
        zero_division=0,
    )

    return {
        "log_loss":
            log_loss(
                y,
                probabilities,
                labels=LABELS,
            ),

        "brier_score":
            multiclass_brier_score(
                y,
                probabilities,
            ),

        "accuracy":
            accuracy_score(
                y,
                predictions,
            ),

        "macro_f1":
            f1_score(
                y,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),

        "macro_precision":
            precision_score(
                y,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),

        "macro_recall":
            recall_score(
                y,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),

        "recall_a":
            recalls[0],

        "recall_b":
            recalls[1],

        "recall_no":
            recalls[2],

        "predictions":
            predictions,
    }


# ==================================================
# Stage selection
# ==================================================

if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: uv run python "
        "scripts/evaluate_v0_stage.py A3"
    )


stage = sys.argv[1].upper()


if stage not in STAGES:
    raise SystemExit(
        f"Supported stages: "
        f"{list(STAGES.keys())}"
    )


config = STAGES[stage]

features = config[
    "features"
]

new_features = config[
    "new_features"
]

previous_stage = config[
    "previous"
]


previous_lower = (
    previous_stage.lower()
)

stage_lower = (
    stage.lower()
)


PREVIOUS_PREDICTION_PATH = Path(
    f"data/processed/"
    f"v0_{previous_lower}_oof_predictions.parquet"
)

PREVIOUS_METRIC_PATH = Path(
    f"artifacts/"
    f"v0_{previous_lower}_metrics.csv"
)

PREDICTION_PATH = Path(
    f"data/processed/"
    f"v0_{stage_lower}_oof_predictions.parquet"
)

METRIC_PATH = Path(
    f"artifacts/"
    f"v0_{stage_lower}_metrics.csv"
)


# ==================================================
# Load
# ==================================================

df = pl.read_parquet(
    DATASET_PATH
)

fold_reference = pl.read_parquet(
    FOLD_PATH
)

previous_predictions = pl.read_parquet(
    PREVIOUS_PREDICTION_PATH
)


KEYS = [
    "demo_filename",
    "round_num",
    "horizon_sec",
    "target_tick",
    "label",
]


assert (
    df.select(KEYS).to_dicts()
    ==
    fold_reference.select(KEYS).to_dicts()
)

assert (
    df.select(KEYS).to_dicts()
    ==
    previous_predictions.select(KEYS).to_dicts()
)


fold_ids = (
    fold_reference[
        "cv_fold"
    ].to_numpy()
)


print("\n" + "=" * 100)
print(
    f"V0 — {stage} LOGISTIC REGRESSION"
)
print("=" * 100)

print(
    f"Previous stage: {previous_stage}"
)

print(
    f"Observations:   {df.height}"
)

print(
    f"Matches:        "
    f"{df['demo_filename'].n_unique()}"
)

print(
    f"Features:       {len(features)}"
)


print(
    f"\nNEW {stage} FEATURES"
)

for feature in new_features:
    print(
        f"  {feature}"
    )


# ==================================================
# Arrays
# ==================================================

X = (
    df
    .select(features)
    .to_numpy()
)

y = (
    df["label"]
    .to_numpy()
)

groups = (
    df["demo_filename"]
    .to_numpy()
)


oof_probabilities = np.zeros(
    (
        df.height,
        len(LABELS),
    )
)


fold_rows = []


# ==================================================
# Exact same CV folds
# ==================================================

for fold in sorted(
    np.unique(fold_ids)
):

    test_index = np.where(
        fold_ids == fold
    )[0]

    train_index = np.where(
        fold_ids != fold
    )[0]


    train_groups = set(
        groups[train_index]
    )

    test_groups = set(
        groups[test_index]
    )


    assert not (
        train_groups
        & test_groups
    )


    X_train = X[
        train_index
    ]

    X_test = X[
        test_index
    ]

    y_train = y[
        train_index
    ]

    y_test = y[
        test_index
    ]


    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=3000,
                random_state=42,
            ),
        ),
    ])


    model.fit(
        X_train,
        y_train,
    )


    raw_probabilities = (
        model.predict_proba(
            X_test
        )
    )


    classes = (
        model
        .named_steps[
            "logistic_regression"
        ]
        .classes_
    )


    probabilities = (
        reorder_probabilities(
            raw_probabilities,
            classes,
        )
    )


    oof_probabilities[
        test_index
    ] = probabilities


    result = evaluate(
        y_test,
        probabilities,
    )


    print(
        f"\nFold {fold}"
    )

    print(
        f"  log loss: "
        f"{result['log_loss']:.4f}"
    )

    print(
        f"  brier:    "
        f"{result['brier_score']:.4f}"
    )

    print(
        f"  accuracy: "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"  macro F1: "
        f"{result['macro_f1']:.4f}"
    )


    fold_rows.append({
        "fold":
            int(fold),

        "n_train_matches":
            len(train_groups),

        "n_test_matches":
            len(test_groups),

        "n_train_observations":
            len(train_index),

        "n_test_observations":
            len(test_index),

        "log_loss":
            result["log_loss"],

        "brier_score":
            result["brier_score"],

        "accuracy":
            result["accuracy"],

        "macro_f1":
            result["macro_f1"],

        "macro_precision":
            result["macro_precision"],

        "macro_recall":
            result["macro_recall"],
    })


# ==================================================
# Overall
# ==================================================

overall = evaluate(
    y,
    oof_probabilities,
)


print("\n" + "=" * 100)
print(
    f"{stage} OUT-OF-FOLD RESULTS"
)
print("=" * 100)


print(
    f"Log Loss:        "
    f"{overall['log_loss']:.4f}"
)

print(
    f"Brier Score:     "
    f"{overall['brier_score']:.4f}"
)

print(
    f"Accuracy:        "
    f"{overall['accuracy']:.4f}"
)

print(
    f"Macro F1:        "
    f"{overall['macro_f1']:.4f}"
)

print(
    f"Macro Precision: "
    f"{overall['macro_precision']:.4f}"
)

print(
    f"Macro Recall:    "
    f"{overall['macro_recall']:.4f}"
)


print("\nPER-CLASS RECALL")

print(
    f"A_PLANT  : "
    f"{overall['recall_a']:.4f}"
)

print(
    f"B_PLANT  : "
    f"{overall['recall_b']:.4f}"
)

print(
    f"NO_PLANT : "
    f"{overall['recall_no']:.4f}"
)


# ==================================================
# Previous → current
# ==================================================

previous_metrics = pl.read_csv(
    PREVIOUS_METRIC_PATH
)


previous_overall = (
    previous_metrics
    .filter(
        pl.col("fold") == 0
    )
    .row(
        0,
        named=True,
    )
)


print("\n" + "=" * 100)
print(
    f"{previous_stage} → {stage} COMPARISON"
)
print("=" * 100)


for name, key in [
    (
        "Log Loss",
        "log_loss",
    ),
    (
        "Brier",
        "brier_score",
    ),
    (
        "Accuracy",
        "accuracy",
    ),
    (
        "Macro F1",
        "macro_f1",
    ),
]:

    old = previous_overall[
        key
    ]

    new = overall[
        key
    ]

    print(
        f"{name:10s}: "
        f"{old:.4f}"
        f" → {new:.4f}"
        f" | delta="
        f"{new - old:+.4f}"
    )


# ==================================================
# By horizon
# ==================================================

print("\n" + "=" * 100)
print(
    f"{previous_stage} → {stage} BY HORIZON"
)
print("=" * 100)


for horizon in [
    10,
    20,
    30,
    40,
]:

    mask = (
        df["horizon_sec"]
        .to_numpy()
        == horizon
    )

    y_h = y[
        mask
    ]

    current_prob = (
        oof_probabilities[
            mask
        ]
    )


    previous_h = (
        previous_predictions
        .filter(
            pl.col("horizon_sec")
            == horizon
        )
    )


    previous_prob = (
        previous_h
        .select([
            "p_a_plant",
            "p_b_plant",
            "p_no_plant",
        ])
        .to_numpy()
    )


    old = evaluate(
        y_h,
        previous_prob,
    )

    new = evaluate(
        y_h,
        current_prob,
    )


    print(
        f"\n{horizon}s "
        f"(n={len(y_h)})"
    )

    print(
        f"  Log Loss: "
        f"{old['log_loss']:.4f}"
        f" → {new['log_loss']:.4f}"
        f" | delta="
        f"{new['log_loss'] - old['log_loss']:+.4f}"
    )

    print(
        f"  Brier:    "
        f"{old['brier_score']:.4f}"
        f" → {new['brier_score']:.4f}"
        f" | delta="
        f"{new['brier_score'] - old['brier_score']:+.4f}"
    )

    print(
        f"  Macro F1: "
        f"{old['macro_f1']:.4f}"
        f" → {new['macro_f1']:.4f}"
    )


# ==================================================
# Save predictions
# ==================================================

prediction_df = (
    df
    .select(KEYS)
    .with_columns([
        pl.Series(
            "cv_fold",
            fold_ids,
        ),

        pl.Series(
            "p_a_plant",
            oof_probabilities[:, 0],
        ),

        pl.Series(
            "p_b_plant",
            oof_probabilities[:, 1],
        ),

        pl.Series(
            "p_no_plant",
            oof_probabilities[:, 2],
        ),

        pl.Series(
            "prediction",
            overall[
                "predictions"
            ],
        ),
    ])
)


prediction_df.write_parquet(
    PREDICTION_PATH
)


# ==================================================
# Save metrics
# ==================================================

metrics = pl.DataFrame(
    fold_rows
)


overall_row = pl.DataFrame({
    "fold": [0],

    "n_train_matches": [None],
    "n_test_matches": [20],

    "n_train_observations": [None],
    "n_test_observations": [
        df.height
    ],

    "log_loss": [
        overall["log_loss"]
    ],

    "brier_score": [
        overall["brier_score"]
    ],

    "accuracy": [
        overall["accuracy"]
    ],

    "macro_f1": [
        overall["macro_f1"]
    ],

    "macro_precision": [
        overall["macro_precision"]
    ],

    "macro_recall": [
        overall["macro_recall"]
    ],
})


metrics = pl.concat(
    [
        metrics,
        overall_row,
    ],
    how="diagonal_relaxed",
)


METRIC_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metrics.write_csv(
    METRIC_PATH
)


print("\nSaved predictions:")
print(PREDICTION_PATH)

print("\nSaved metrics:")
print(METRIC_PATH)

print(
    f"\n✅ {stage} ABLATION COMPLETE"
)
