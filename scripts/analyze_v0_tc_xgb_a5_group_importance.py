from pathlib import Path

import numpy as np
import polars as pl
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
)


DATASET_PATH = Path(
    "data/processed/v0_dataset_v2_timing_corrected.parquet"
)

FOLD_PATH = Path(
    "data/processed/v0_tc_a1_oof_predictions.parquet"
)

SUMMARY_PATH = Path(
    "artifacts/v0_tc_xgb_a5_group_permutation_importance.csv"
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


# ==================================================
# Feature groups
# ==================================================

HORIZON = [
    "horizon_sec",
]


OFFENSIVE_GEOMETRY = [
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


MOTION = [
    "t_mean_speed_1s",
    "t_centroid_velocity_x_1s",
    "t_centroid_velocity_y_1s",
    "bomb_speed_1s",
]


COMBAT = [
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


DEFENSE = [
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


FEATURE_GROUPS = {
    "Horizon":
        HORIZON,

    "Offensive Geometry":
        OFFENSIVE_GEOMETRY,

    "Motion":
        MOTION,

    "Combat":
        COMBAT,

    "Defense":
        DEFENSE,
}


FEATURES = (
    HORIZON
    + OFFENSIVE_GEOMETRY
    + MOTION
    + COMBAT
    + DEFENSE
)


# ==================================================
# Same fixed XGB config
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


N_REPEATS = 20


# ==================================================
# Metrics
# ==================================================

def normalize_probabilities(
    p,
):

    p = p.astype(
        np.float64
    )

    return (
        p
        / p.sum(
            axis=1,
            keepdims=True,
        )
    )


def multiclass_brier(
    y,
    p,
):

    truth = np.zeros(
        (
            len(y),
            3,
        ),
        dtype=float,
    )

    truth[
        np.arange(
            len(y)
        ),
        y,
    ] = 1.0

    return np.mean(
        np.sum(
            (
                p
                - truth
            ) ** 2,
            axis=1,
        )
    )


def evaluate(
    y,
    p,
):

    pred = np.argmax(
        p,
        axis=1,
    )

    return {
        "log_loss":
            log_loss(
                y,
                p,
                labels=[
                    0,
                    1,
                    2,
                ],
            ),

        "brier":
            multiclass_brier(
                y,
                p,
            ),

        "accuracy":
            accuracy_score(
                y,
                pred,
            ),

        "macro_f1":
            f1_score(
                y,
                pred,
                average="macro",
                zero_division=0,
            ),
    }


# ==================================================
# Load
# ==================================================

df = pl.read_parquet(
    DATASET_PATH
)

fold_df = pl.read_parquet(
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
    fold_df.select(KEYS).to_dicts()
)


fold_ids = (
    fold_df[
        "cv_fold"
    ].to_numpy()
)


y = np.array([
    LABEL_TO_ID[label]
    for label
    in df[
        "label"
    ].to_numpy()
])


X = (
    df
    .select(FEATURES)
    .to_numpy()
)


# ==================================================
# Column indices
# ==================================================

feature_to_index = {
    feature: i
    for i, feature
    in enumerate(FEATURES)
}


group_indices = {
    group: np.array([
        feature_to_index[
            feature
        ]
        for feature in features
    ])
    for group, features
    in FEATURE_GROUPS.items()
}


# ==================================================
# Train one model per frozen fold
# ==================================================

fold_models = []


baseline_probabilities = np.zeros(
    (
        len(y),
        3,
    ),
    dtype=float,
)


print("\n" + "=" * 105)
print("XGB_A5 — GROUP PERMUTATION IMPORTANCE")
print("=" * 105)


for fold in sorted(
    np.unique(fold_ids)
):

    train_index = np.where(
        fold_ids != fold
    )[0]

    test_index = np.where(
        fold_ids == fold
    )[0]


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


    p = normalize_probabilities(
        model.predict_proba(
            X[
                test_index
            ]
        )
    )


    baseline_probabilities[
        test_index
    ] = p


    fold_models.append(
        (
            fold,
            model,
            test_index,
        )
    )


# ==================================================
# Baseline reproduction
# ==================================================

baseline = evaluate(
    y,
    baseline_probabilities,
)


print("\nBASELINE XGB_A5")

print(
    f"Log Loss: "
    f"{baseline['log_loss']:.4f}"
)

print(
    f"Brier:    "
    f"{baseline['brier']:.4f}"
)

print(
    f"Accuracy: "
    f"{baseline['accuracy']:.4f}"
)

print(
    f"Macro F1: "
    f"{baseline['macro_f1']:.4f}"
)


# ==================================================
# Group permutation
#
# A whole feature family receives the SAME row
# permutation. This preserves relationships inside
# that family while breaking its relationship with
# the outcome and the other feature groups.
# ==================================================

rows = []


for group_name, columns in (
    group_indices.items()
):

    print("\n" + "-" * 105)
    print(group_name)
    print("-" * 105)


    delta_ll = []
    delta_brier = []
    delta_accuracy = []
    delta_f1 = []


    for repeat in range(
        N_REPEATS
    ):

        permuted_probabilities = np.zeros(
            (
                len(y),
                3,
            ),
            dtype=float,
        )


        for fold, model, test_index in (
            fold_models
        ):

            X_test = (
                X[
                    test_index
                ]
                .copy()
            )


            rng = np.random.default_rng(
                42
                + repeat * 100
                + int(fold)
            )


            permutation = rng.permutation(
                len(
                    test_index
                )
            )


            # Permute the group as one block.
            X_test[
                :,
                columns
            ] = (
                X_test[
                    permutation
                ][
                    :,
                    columns
                ]
            )


            p = normalize_probabilities(
                model.predict_proba(
                    X_test
                )
            )


            permuted_probabilities[
                test_index
            ] = p


        permuted = evaluate(
            y,
            permuted_probabilities,
        )


        # Positive = performance became worse
        # after destroying this feature group.
        delta_ll.append(
            permuted[
                "log_loss"
            ]
            - baseline[
                "log_loss"
            ]
        )

        delta_brier.append(
            permuted[
                "brier"
            ]
            - baseline[
                "brier"
            ]
        )

        delta_accuracy.append(
            baseline[
                "accuracy"
            ]
            - permuted[
                "accuracy"
            ]
        )

        delta_f1.append(
            baseline[
                "macro_f1"
            ]
            - permuted[
                "macro_f1"
            ]
        )


    mean_ll = float(
        np.mean(
            delta_ll
        )
    )

    std_ll = float(
        np.std(
            delta_ll
        )
    )

    mean_brier = float(
        np.mean(
            delta_brier
        )
    )

    mean_acc = float(
        np.mean(
            delta_accuracy
        )
    )

    mean_f1 = float(
        np.mean(
            delta_f1
        )
    )


    print(
        f"Δ Log Loss: "
        f"{mean_ll:+.4f}"
        f" ± {std_ll:.4f}"
    )

    print(
        f"Δ Brier:    "
        f"{mean_brier:+.4f}"
    )

    print(
        f"Δ Accuracy: "
        f"{mean_acc:+.4f}"
    )

    print(
        f"Δ Macro F1: "
        f"{mean_f1:+.4f}"
    )


    rows.append({
        "feature_group":
            group_name,

        "n_features":
            len(
                columns
            ),

        "delta_log_loss_mean":
            mean_ll,

        "delta_log_loss_std":
            std_ll,

        "delta_brier_mean":
            mean_brier,

        "delta_accuracy_mean":
            mean_acc,

        "delta_macro_f1_mean":
            mean_f1,
    })


# ==================================================
# Ranking
# ==================================================

summary = (
    pl.DataFrame(
        rows
    )
    .sort(
        "delta_log_loss_mean",
        descending=True,
    )
)


print("\n" + "=" * 105)
print("GROUP IMPORTANCE RANKING")
print("=" * 105)


print(
    summary
)


SUMMARY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


summary.write_csv(
    SUMMARY_PATH
)


print("\nSaved:")
print(
    SUMMARY_PATH
)

print(
    "\n✅ GROUP PERMUTATION IMPORTANCE COMPLETE"
)
