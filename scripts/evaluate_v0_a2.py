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

A1_PREDICTION_PATH = Path(
    "data/processed/v0_a1_oof_predictions.parquet"
)

A1_METRIC_PATH = Path(
    "artifacts/v0_a1_metrics.csv"
)

PREDICTION_PATH = Path(
    "data/processed/v0_a2_oof_predictions.parquet"
)

METRIC_PATH = Path(
    "artifacts/v0_a2_metrics.csv"
)


LABELS = [
    "A_PLANT",
    "B_PLANT",
    "NO_PLANT",
]


# ==================================================
# A1 + A2 feature contract
# ==================================================

A1_FEATURES = [
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


A2_MOTION_FEATURES = [
    "t_mean_speed_1s",
    "t_centroid_velocity_x_1s",
    "t_centroid_velocity_y_1s",
    "bomb_speed_1s",
]


FEATURES = (
    A1_FEATURES
    + A2_MOTION_FEATURES
)


# ==================================================
# Metrics
# ==================================================

def multiclass_brier_score(
    y_true,
    probabilities,
):

    label_to_index = {
        label: index
        for index, label
        in enumerate(LABELS)
    }

    truth = np.zeros(
        (
            len(y_true),
            len(LABELS),
        ),
        dtype=float,
    )

    for row_index, label in enumerate(
        y_true
    ):
        truth[
            row_index,
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

    class_to_column = {
        label: index
        for index, label
        in enumerate(model_classes)
    }

    return np.column_stack([
        probabilities[
            :,
            class_to_column[label]
        ]
        for label in LABELS
    ])


def evaluate_predictions(
    y,
    probabilities,
):

    predictions = np.array([
        LABELS[index]
        for index
        in np.argmax(
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
# Load
# ==================================================

df = pl.read_parquet(
    DATASET_PATH
)

a1_predictions = pl.read_parquet(
    A1_PREDICTION_PATH
)


# ==================================================
# Verify that A1 predictions align exactly
# with the dataset before reusing folds.
# ==================================================

KEYS = [
    "demo_filename",
    "round_num",
    "horizon_sec",
    "target_tick",
    "label",
]


assert df.height == a1_predictions.height

assert (
    df.select(KEYS).to_dicts()
    ==
    a1_predictions.select(KEYS).to_dicts()
)


fold_ids = (
    a1_predictions[
        "cv_fold"
    ].to_numpy()
)


print("\n" + "=" * 100)
print("V0 — A2 LOGISTIC REGRESSION + MOTION")
print("=" * 100)

print(
    f"Observations: {df.height}"
)

print(
    f"Matches:      "
    f"{df['demo_filename'].n_unique()}"
)

print(
    f"Features:     {len(FEATURES)}"
)


print("\nNEW A2 MOTION FEATURES")

for feature in A2_MOTION_FEATURES:
    print(
        f"  {feature}"
    )


# ==================================================
# Arrays
# ==================================================

X = (
    df
    .select(FEATURES)
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
# Reuse EXACT A1 folds
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


    assert (
        len(
            train_groups
            & test_groups
        )
        == 0
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


    # ==============================================
    # Same model family and settings as A1
    # ==============================================

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


    model_classes = (
        model
        .named_steps[
            "logistic_regression"
        ]
        .classes_
    )


    probabilities = (
        reorder_probabilities(
            raw_probabilities,
            model_classes,
        )
    )


    oof_probabilities[
        test_index
    ] = probabilities


    result = evaluate_predictions(
        y_test,
        probabilities,
    )


    print(
        f"\nFold {fold}"
    )

    print(
        f"  train matches: "
        f"{len(train_groups)}"
    )

    print(
        f"  test matches:  "
        f"{len(test_groups)}"
    )

    print(
        f"  log loss:      "
        f"{result['log_loss']:.4f}"
    )

    print(
        f"  brier:         "
        f"{result['brier_score']:.4f}"
    )

    print(
        f"  accuracy:      "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"  macro F1:      "
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
# Overall A2 OOF
# ==================================================

overall = evaluate_predictions(
    y,
    oof_probabilities,
)


print("\n" + "=" * 100)
print("A2 OUT-OF-FOLD RESULTS")
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
    f"A_PLANT   : "
    f"{overall['recall_a']:.4f}"
)

print(
    f"B_PLANT   : "
    f"{overall['recall_b']:.4f}"
)

print(
    f"NO_PLANT  : "
    f"{overall['recall_no']:.4f}"
)


# ==================================================
# A1 → A2 comparison
# ==================================================

a1_metrics = pl.read_csv(
    A1_METRIC_PATH
)

a1_overall = (
    a1_metrics
    .filter(
        pl.col("fold") == 0
    )
    .row(
        0,
        named=True,
    )
)


print("\n" + "=" * 100)
print("A1 → A2 COMPARISON")
print("=" * 100)


print(
    f"Log Loss: "
    f"{a1_overall['log_loss']:.4f}"
    f" → {overall['log_loss']:.4f}"
    f" | delta="
    f"{overall['log_loss'] - a1_overall['log_loss']:+.4f}"
)

print(
    f"Brier:   "
    f"{a1_overall['brier_score']:.4f}"
    f" → {overall['brier_score']:.4f}"
    f" | delta="
    f"{overall['brier_score'] - a1_overall['brier_score']:+.4f}"
)

print(
    f"Accuracy:"
    f" {a1_overall['accuracy']:.4f}"
    f" → {overall['accuracy']:.4f}"
    f" | delta="
    f"{overall['accuracy'] - a1_overall['accuracy']:+.4f}"
)

print(
    f"Macro F1:"
    f" {a1_overall['macro_f1']:.4f}"
    f" → {overall['macro_f1']:.4f}"
    f" | delta="
    f"{overall['macro_f1'] - a1_overall['macro_f1']:+.4f}"
)


# ==================================================
# Save A2 OOF predictions
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
            overall["predictions"],
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


# ==================================================
# Horizon comparison
# ==================================================

print("\n" + "=" * 100)
print("A1 → A2 BY HORIZON")
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

    a2_prob_h = (
        oof_probabilities[
            mask
        ]
    )


    a1_h = (
        a1_predictions
        .filter(
            pl.col("horizon_sec")
            == horizon
        )
    )

    a1_prob_h = (
        a1_h
        .select([
            "p_a_plant",
            "p_b_plant",
            "p_no_plant",
        ])
        .to_numpy()
    )


    a1_result = (
        evaluate_predictions(
            y_h,
            a1_prob_h,
        )
    )

    a2_result = (
        evaluate_predictions(
            y_h,
            a2_prob_h,
        )
    )


    print(
        f"\n{horizon}s "
        f"(n={len(y_h)})"
    )

    print(
        f"  Log Loss: "
        f"{a1_result['log_loss']:.4f}"
        f" → {a2_result['log_loss']:.4f}"
        f" | delta="
        f"{a2_result['log_loss'] - a1_result['log_loss']:+.4f}"
    )

    print(
        f"  Brier:    "
        f"{a1_result['brier_score']:.4f}"
        f" → {a2_result['brier_score']:.4f}"
        f" | delta="
        f"{a2_result['brier_score'] - a1_result['brier_score']:+.4f}"
    )

    print(
        f"  Macro F1: "
        f"{a1_result['macro_f1']:.4f}"
        f" → {a2_result['macro_f1']:.4f}"
    )


print("\nSaved predictions:")
print(PREDICTION_PATH)

print("\nSaved metrics:")
print(METRIC_PATH)

print(
    "\n✅ A2 MOTION ABLATION COMPLETE"
)
