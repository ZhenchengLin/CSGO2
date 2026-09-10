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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path(
    "data/processed/v0_dataset_v1.parquet"
)

A0_METRIC_PATH = Path(
    "artifacts/v0_a0_metrics.csv"
)

PREDICTION_PATH = Path(
    "data/processed/v0_a1_oof_predictions.parquet"
)

METRIC_PATH = Path(
    "artifacts/v0_a1_metrics.csv"
)


LABELS = [
    "A_PLANT",
    "B_PLANT",
    "NO_PLANT",
]

N_SPLITS = 5
RANDOM_STATE = 42


# ==================================================
# A1 feature contract
# ==================================================

FEATURES = [

    # Context
    "horizon_sec",

    # Offensive geometry
    "t_centroid_x",
    "t_centroid_y",
    "t_centroid_z",

    "t_stretch_xy",

    "t_range_x",
    "t_range_y",

    "t_mean_pairwise_distance",
    "t_convex_hull_area",

    # Bomb geometry
    "bomb_x",
    "bomb_y",
    "bomb_z",

    "bomb_to_t_centroid_distance",
]


# ==================================================
# Multiclass Brier score
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


# ==================================================
# Probability-column safety
# ==================================================

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


# ==================================================
# Load dataset
# ==================================================

df = pl.read_parquet(
    DATASET_PATH
)


print("\n" + "=" * 100)
print("V0 — A1 LOGISTIC REGRESSION")
print("=" * 100)

print(
    f"Observations: {df.height}"
)

print(
    f"Matches:      "
    f"{df['demo_filename'].n_unique()}"
)

print(
    f"Features:     "
    f"{len(FEATURES)}"
)


print("\nA1 FEATURES")

for feature in FEATURES:
    print(
        f"  {feature}"
    )


# ==================================================
# sklearn arrays
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


# ==================================================
# Match-level grouped CV
# ==================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


oof_probabilities = np.zeros(
    (
        df.height,
        len(LABELS),
    )
)

fold_ids = np.zeros(
    df.height,
    dtype=int,
)

fold_rows = []


for fold, (
    train_index,
    test_index,
) in enumerate(
    cv.split(
        X,
        y,
        groups,
    ),
    start=1,
):

    train_groups = set(
        groups[train_index]
    )

    test_groups = set(
        groups[test_index]
    )


    # ----------------------------------------------
    # Match leakage protection
    # ----------------------------------------------

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


    # ----------------------------------------------
    # Model
    #
    # StandardScaler is fit ONLY on training data
    # because it lives inside the sklearn pipeline.
    # ----------------------------------------------

    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE,
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

    fold_ids[
        test_index
    ] = fold


    predictions = np.array([
        LABELS[index]
        for index
        in np.argmax(
            probabilities,
            axis=1,
        )
    ])


    # ----------------------------------------------
    # Metrics
    # ----------------------------------------------

    fold_log_loss = log_loss(
        y_test,
        probabilities,
        labels=LABELS,
    )


    fold_brier = (
        multiclass_brier_score(
            y_test,
            probabilities,
        )
    )


    fold_accuracy = (
        accuracy_score(
            y_test,
            predictions,
        )
    )


    fold_macro_f1 = (
        f1_score(
            y_test,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
    )


    fold_macro_precision = (
        precision_score(
            y_test,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
    )


    fold_macro_recall = (
        recall_score(
            y_test,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
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
        f"{fold_log_loss:.4f}"
    )

    print(
        f"  brier:         "
        f"{fold_brier:.4f}"
    )

    print(
        f"  accuracy:      "
        f"{fold_accuracy:.4f}"
    )

    print(
        f"  macro F1:      "
        f"{fold_macro_f1:.4f}"
    )


    fold_rows.append({

        "fold":
            fold,

        "n_train_matches":
            len(train_groups),

        "n_test_matches":
            len(test_groups),

        "n_train_observations":
            len(train_index),

        "n_test_observations":
            len(test_index),

        "log_loss":
            fold_log_loss,

        "brier_score":
            fold_brier,

        "accuracy":
            fold_accuracy,

        "macro_f1":
            fold_macro_f1,

        "macro_precision":
            fold_macro_precision,

        "macro_recall":
            fold_macro_recall,
    })


# ==================================================
# Overall OOF evaluation
# ==================================================

oof_predictions = np.array([
    LABELS[index]
    for index
    in np.argmax(
        oof_probabilities,
        axis=1,
    )
])


overall_log_loss = (
    log_loss(
        y,
        oof_probabilities,
        labels=LABELS,
    )
)


overall_brier = (
    multiclass_brier_score(
        y,
        oof_probabilities,
    )
)


overall_accuracy = (
    accuracy_score(
        y,
        oof_predictions,
    )
)


overall_macro_f1 = (
    f1_score(
        y,
        oof_predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
)


overall_macro_precision = (
    precision_score(
        y,
        oof_predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
)


overall_macro_recall = (
    recall_score(
        y,
        oof_predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
)


print("\n" + "=" * 100)
print("A1 OUT-OF-FOLD RESULTS")
print("=" * 100)

print(
    f"Log Loss:       "
    f"{overall_log_loss:.4f}"
)

print(
    f"Brier Score:    "
    f"{overall_brier:.4f}"
)

print(
    f"Accuracy:       "
    f"{overall_accuracy:.4f}"
)

print(
    f"Macro F1:       "
    f"{overall_macro_f1:.4f}"
)

print(
    f"Macro Precision:"
    f" {overall_macro_precision:.4f}"
)

print(
    f"Macro Recall:   "
    f"{overall_macro_recall:.4f}"
)


# ==================================================
# Per-class recall
# ==================================================

class_recall = recall_score(
    y,
    oof_predictions,
    labels=LABELS,
    average=None,
    zero_division=0,
)


print("\nPER-CLASS RECALL")

for label, value in zip(
    LABELS,
    class_recall,
):

    print(
        f"{label:10s}: "
        f"{value:.4f}"
    )


# ==================================================
# Compare directly against A0
# ==================================================

a0_metrics = pl.read_csv(
    A0_METRIC_PATH
)

a0 = (
    a0_metrics
    .filter(
        pl.col("fold") == 0
    )
    .row(
        0,
        named=True,
    )
)


print("\n" + "=" * 100)
print("A0 → A1 COMPARISON")
print("=" * 100)


print(
    f"Log Loss: "
    f"{a0['log_loss']:.4f}"
    f" → {overall_log_loss:.4f}"
    f" | delta="
    f"{overall_log_loss - a0['log_loss']:+.4f}"
)

print(
    f"Brier:   "
    f"{a0['brier_score']:.4f}"
    f" → {overall_brier:.4f}"
    f" | delta="
    f"{overall_brier - a0['brier_score']:+.4f}"
)

print(
    f"Accuracy:"
    f" {a0['accuracy']:.4f}"
    f" → {overall_accuracy:.4f}"
    f" | delta="
    f"{overall_accuracy - a0['accuracy']:+.4f}"
)

print(
    f"Macro F1:"
    f" {a0['macro_f1']:.4f}"
    f" → {overall_macro_f1:.4f}"
    f" | delta="
    f"{overall_macro_f1 - a0['macro_f1']:+.4f}"
)


# ==================================================
# Save OOF predictions
# ==================================================

prediction_df = (
    df
    .select([
        "demo_filename",
        "round_num",
        "horizon_sec",
        "target_tick",
        "label",
    ])
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
            oof_predictions,
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
        overall_log_loss
    ],

    "brier_score": [
        overall_brier
    ],

    "accuracy": [
        overall_accuracy
    ],

    "macro_f1": [
        overall_macro_f1
    ],

    "macro_precision": [
        overall_macro_precision
    ],

    "macro_recall": [
        overall_macro_recall
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
    "\n✅ A1 LOGISTIC REGRESSION COMPLETE"
)
