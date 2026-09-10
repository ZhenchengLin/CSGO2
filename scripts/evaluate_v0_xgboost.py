from pathlib import Path

import numpy as np
import polars as pl
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


DATASET_PATH = Path(
    "data/processed/v0_dataset_v1.parquet"
)

FOLD_PATH = Path(
    "data/processed/v0_a1_oof_predictions.parquet"
)

OUTPUT_DIR = Path(
    "artifacts"
)

PREDICTION_DIR = Path(
    "data/processed"
)


LABELS = [
    "A_PLANT",
    "B_PLANT",
    "NO_PLANT",
]

LABEL_TO_ID = {
    label: i
    for i, label in enumerate(LABELS)
}

ID_TO_LABEL = {
    i: label
    for label, i in LABEL_TO_ID.items()
}


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


FEATURE_SETS = {

    "XGB_A3": (
        A1
        + A2
        + A3
    ),

    "XGB_A4": (
        A1
        + A2
        + A3
        + A4
    ),

    "XGB_A5": (
        A1
        + A2
        + A3
        + A5
    ),
}


# ==================================================
# Fixed XGBoost config
#
# IMPORTANT:
# These are NOT tuned against OOF results.
# They are a conservative first nonlinear baseline.
# ==================================================

XGB_CONFIG = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,

    "min_child_weight": 5,

    "subsample": 0.8,
    "colsample_bytree": 0.8,

    "reg_alpha": 0.5,
    "reg_lambda": 5.0,

    "objective": "multi:softprob",
    "num_class": 3,

    "eval_metric": "mlogloss",

    "tree_method": "hist",

    "random_state": 42,
    "n_jobs": -1,
}


# ==================================================
# Metrics
# ==================================================

def multiclass_brier_score(
    y_true,
    probabilities,
):

    truth = np.zeros(
        (
            len(y_true),
            len(LABELS),
        ),
        dtype=float,
    )

    truth[
        np.arange(len(y_true)),
        y_true,
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


def evaluate(
    y_true,
    probabilities,
):

    pred_id = np.argmax(
        probabilities,
        axis=1,
    )

    pred_label = np.array([
        ID_TO_LABEL[i]
        for i in pred_id
    ])

    true_label = np.array([
        ID_TO_LABEL[i]
        for i in y_true
    ])

    recalls = recall_score(
        true_label,
        pred_label,
        labels=LABELS,
        average=None,
        zero_division=0,
    )

    return {
        "log_loss":
            log_loss(
                y_true,
                probabilities,
                labels=[
                    0,
                    1,
                    2,
                ],
            ),

        "brier_score":
            multiclass_brier_score(
                y_true,
                probabilities,
            ),

        "accuracy":
            accuracy_score(
                true_label,
                pred_label,
            ),

        "macro_f1":
            f1_score(
                true_label,
                pred_label,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),

        "macro_precision":
            precision_score(
                true_label,
                pred_label,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),

        "macro_recall":
            recall_score(
                true_label,
                pred_label,
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
            pred_label,
    }


# ==================================================
# Load data + frozen folds
# ==================================================

df = pl.read_parquet(
    DATASET_PATH
)

fold_reference = pl.read_parquet(
    FOLD_PATH
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


fold_ids = (
    fold_reference[
        "cv_fold"
    ].to_numpy()
)


groups = (
    df[
        "demo_filename"
    ].to_numpy()
)


y_labels = (
    df[
        "label"
    ].to_numpy()
)


y = np.array([
    LABEL_TO_ID[label]
    for label in y_labels
])


print("\n" + "=" * 105)
print("V0 — XGBOOST NONLINEAR RETEST")
print("=" * 105)

print(
    f"Observations: {df.height}"
)

print(
    f"Matches:      "
    f"{df['demo_filename'].n_unique()}"
)

print(
    f"Frozen folds: "
    f"{len(np.unique(fold_ids))}"
)


# ==================================================
# Run every feature set
# ==================================================

summary_rows = []


for stage, features in (
    FEATURE_SETS.items()
):

    print("\n" + "=" * 105)
    print(
        f"{stage} | "
        f"{len(features)} features"
    )
    print("=" * 105)


    X = (
        df
        .select(features)
        .to_numpy()
    )


    oof_probabilities = np.zeros(
        (
            df.height,
            len(LABELS),
        ),
        dtype=float,
    )


    fold_rows = []


    for fold in sorted(
        np.unique(fold_ids)
    ):

        test_index = np.where(
            fold_ids == fold
        )[0]

        train_index = np.where(
            fold_ids != fold
        )[0]


        train_matches = set(
            groups[
                train_index
            ]
        )

        test_matches = set(
            groups[
                test_index
            ]
        )


        # Match leakage check
        assert not (
            train_matches
            & test_matches
        )


        model = XGBClassifier(
            **XGB_CONFIG
        )


        model.fit(
            X[
                train_index
            ],
            y[
                train_index
            ],
        )


        probabilities = (
            model.predict_proba(
                X[
                    test_index
                ]
            )
        )

        # XGBoost returns Float32 probabilities.
        # Convert to Float64 BEFORE normalization so
        # sklearn probability checks are not affected
        # by Float32 rounding error.
        probabilities = probabilities.astype(
            np.float64
        )

        probabilities = (
            probabilities
            / probabilities.sum(
                axis=1,
                keepdims=True,
            )
        )


        assert (
            probabilities.shape[1]
            == 3
        )


        oof_probabilities[
            test_index
        ] = probabilities


        fold_result = evaluate(
            y[
                test_index
            ],
            probabilities,
        )


        print(
            f"\nFold {fold}"
        )

        print(
            f"  log loss: "
            f"{fold_result['log_loss']:.4f}"
        )

        print(
            f"  brier:    "
            f"{fold_result['brier_score']:.4f}"
        )

        print(
            f"  accuracy: "
            f"{fold_result['accuracy']:.4f}"
        )

        print(
            f"  macro F1: "
            f"{fold_result['macro_f1']:.4f}"
        )


        fold_rows.append({
            "fold":
                int(fold),

            "log_loss":
                fold_result[
                    "log_loss"
                ],

            "brier_score":
                fold_result[
                    "brier_score"
                ],

            "accuracy":
                fold_result[
                    "accuracy"
                ],

            "macro_f1":
                fold_result[
                    "macro_f1"
                ],
        })


    # ==============================================
    # Overall OOF
    # ==============================================

    result = evaluate(
        y,
        oof_probabilities,
    )


    print("\n" + "-" * 105)
    print(
        f"{stage} OUT-OF-FOLD"
    )
    print("-" * 105)


    print(
        f"Log Loss:        "
        f"{result['log_loss']:.4f}"
    )

    print(
        f"Brier Score:     "
        f"{result['brier_score']:.4f}"
    )

    print(
        f"Accuracy:        "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Macro F1:        "
        f"{result['macro_f1']:.4f}"
    )

    print(
        f"Macro Precision: "
        f"{result['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall:    "
        f"{result['macro_recall']:.4f}"
    )


    print("\nPER-CLASS RECALL")

    print(
        f"A_PLANT  : "
        f"{result['recall_a']:.4f}"
    )

    print(
        f"B_PLANT  : "
        f"{result['recall_b']:.4f}"
    )

    print(
        f"NO_PLANT : "
        f"{result['recall_no']:.4f}"
    )


    # ==============================================
    # Horizon breakdown
    # ==============================================

    print("\nBY HORIZON")

    for horizon in [
        10,
        20,
        30,
        40,
    ]:

        mask = (
            df[
                "horizon_sec"
            ].to_numpy()
            == horizon
        )

        horizon_result = evaluate(
            y[
                mask
            ],
            oof_probabilities[
                mask
            ],
        )


        print(
            f"  {horizon:>2}s"
            f" | n={mask.sum():>3}"
            f" | LL="
            f"{horizon_result['log_loss']:.4f}"
            f" | Brier="
            f"{horizon_result['brier_score']:.4f}"
            f" | F1="
            f"{horizon_result['macro_f1']:.4f}"
        )


    # ==============================================
    # Save predictions
    # ==============================================

    stage_lower = stage.lower()


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
                result[
                    "predictions"
                ],
            ),
        ])
    )


    prediction_df.write_parquet(
        PREDICTION_DIR
        / f"v0_{stage_lower}"
          f"_oof_predictions.parquet"
    )


    # ==============================================
    # Summary
    # ==============================================

    summary_rows.append({
        "stage":
            stage,

        "n_features":
            len(features),

        "log_loss":
            result[
                "log_loss"
            ],

        "brier":
            result[
                "brier_score"
            ],

        "accuracy":
            result[
                "accuracy"
            ],

        "macro_f1":
            result[
                "macro_f1"
            ],

        "macro_precision":
            result[
                "macro_precision"
            ],

        "macro_recall":
            result[
                "macro_recall"
            ],

        "recall_a":
            result[
                "recall_a"
            ],

        "recall_b":
            result[
                "recall_b"
            ],

        "recall_no":
            result[
                "recall_no"
            ],
    })


# ==================================================
# Final comparison
# ==================================================

summary = pl.DataFrame(
    summary_rows
)


print("\n" + "=" * 105)
print("XGBOOST SUMMARY")
print("=" * 105)

print(
    summary.select([
        "stage",
        "n_features",
        "log_loss",
        "brier",
        "accuracy",
        "macro_f1",
    ])
)


# ==================================================
# Compare against best Logistic Regression
# ==================================================

lr_a3 = (
    pl.read_csv(
        "artifacts/v0_a3_metrics.csv"
    )
    .filter(
        pl.col("fold") == 0
    )
    .row(
        0,
        named=True,
    )
)


print("\n" + "=" * 105)
print("BEST LOGISTIC A3 vs XGBOOST")
print("=" * 105)


for row in (
    summary.iter_rows(
        named=True
    )
):

    print(
        f"\n{row['stage']}"
    )

    print(
        f"  Log Loss: "
        f"{lr_a3['log_loss']:.4f}"
        f" → {row['log_loss']:.4f}"
        f" | delta="
        f"{row['log_loss'] - lr_a3['log_loss']:+.4f}"
    )

    print(
        f"  Brier:    "
        f"{lr_a3['brier_score']:.4f}"
        f" → {row['brier']:.4f}"
        f" | delta="
        f"{row['brier'] - lr_a3['brier_score']:+.4f}"
    )

    print(
        f"  Macro F1: "
        f"{lr_a3['macro_f1']:.4f}"
        f" → {row['macro_f1']:.4f}"
        f" | delta="
        f"{row['macro_f1'] - lr_a3['macro_f1']:+.4f}"
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


summary.write_csv(
    OUTPUT_DIR
    / "v0_xgboost_summary.csv"
)


print(
    "\nSaved: "
    "artifacts/v0_xgboost_summary.csv"
)

print(
    "\n✅ XGBOOST NONLINEAR RETEST COMPLETE"
)
