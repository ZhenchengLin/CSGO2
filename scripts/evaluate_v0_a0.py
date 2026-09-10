from pathlib import Path

import numpy as np
import polars as pl

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
)


DATASET_PATH = Path(
    "data/processed/v0_dataset_v1.parquet"
)

PREDICTION_PATH = Path(
    "data/processed/v0_a0_oof_predictions.parquet"
)

METRIC_PATH = Path(
    "artifacts/v0_a0_metrics.csv"
)


LABELS = [
    "A_PLANT",
    "B_PLANT",
    "NO_PLANT",
]

N_SPLITS = 5
RANDOM_STATE = 42


# ==================================================
# Helpers
# ==================================================

def multiclass_brier_score(
    y_true,
    probabilities,
):
    """
    Multiclass Brier score:

    mean over observations of the summed
    squared difference between predicted
    probabilities and one-hot truth.

    Lower is better.
    """

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
# Load dataset
# ==================================================

df = pl.read_parquet(
    DATASET_PATH
)


print("\n" + "=" * 100)
print("V0 — A0 CLASS PRIOR BASELINE")
print("=" * 100)

print(
    f"Observations: {df.height}"
)

print(
    f"Matches:      "
    f"{df['demo_filename'].n_unique()}"
)


# ==================================================
# Arrays used by sklearn
# ==================================================

y = df["label"].to_numpy()

groups = (
    df["demo_filename"]
    .to_numpy()
)


# Dummy X:
# A0 intentionally uses no features.
#
# StratifiedGroupKFold only needs X to know
# the number of rows.
X_dummy = np.zeros(
    (
        df.height,
        1,
    )
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
        X_dummy,
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
    # Leakage check
    # ----------------------------------------------

    overlap = (
        train_groups
        & test_groups
    )

    assert len(overlap) == 0


    y_train = y[
        train_index
    ]

    y_test = y[
        test_index
    ]


    # ----------------------------------------------
    # Learn class prior from TRAIN only
    # ----------------------------------------------

    counts = {
        label: int(
            np.sum(
                y_train == label
            )
        )
        for label in LABELS
    }


    priors = np.array([
        counts[label]
        / len(y_train)
        for label in LABELS
    ])


    # Every test observation receives
    # exactly the same A0 probabilities.
    probabilities = np.tile(
        priors,
        (
            len(test_index),
            1,
        ),
    )


    oof_probabilities[
        test_index
    ] = probabilities

    fold_ids[
        test_index
    ] = fold


    predictions = np.array(
        [
            LABELS[index]
            for index
            in np.argmax(
                probabilities,
                axis=1,
            )
        ]
    )


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


    fold_accuracy = accuracy_score(
        y_test,
        predictions,
    )


    fold_macro_f1 = f1_score(
        y_test,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
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
        "  train priors:  "
        f"A={priors[0]:.3f} "
        f"B={priors[1]:.3f} "
        f"N={priors[2]:.3f}"
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
        "fold": fold,

        "n_train_matches":
            len(train_groups),

        "n_test_matches":
            len(test_groups),

        "n_train_observations":
            len(train_index),

        "n_test_observations":
            len(test_index),

        "prior_a":
            priors[0],

        "prior_b":
            priors[1],

        "prior_no_plant":
            priors[2],

        "log_loss":
            fold_log_loss,

        "brier_score":
            fold_brier,

        "accuracy":
            fold_accuracy,

        "macro_f1":
            fold_macro_f1,
    })


# ==================================================
# Overall OOF evaluation
# ==================================================

oof_predictions = np.array(
    [
        LABELS[index]
        for index
        in np.argmax(
            oof_probabilities,
            axis=1,
        )
    ]
)


overall_log_loss = log_loss(
    y,
    oof_probabilities,
    labels=LABELS,
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


print("\n" + "=" * 100)
print("A0 OUT-OF-FOLD RESULTS")
print("=" * 100)


print(
    f"Log Loss:    "
    f"{overall_log_loss:.4f}"
)

print(
    f"Brier Score: "
    f"{overall_brier:.4f}"
)

print(
    f"Accuracy:    "
    f"{overall_accuracy:.4f}"
)

print(
    f"Macro F1:    "
    f"{overall_macro_f1:.4f}"
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

    "prior_a": [None],
    "prior_b": [None],
    "prior_no_plant": [None],

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
    "\n✅ A0 BASELINE COMPLETE"
)
