from pathlib import Path

import numpy as np
import polars as pl
from xgboost import XGBClassifier
from sklearn.metrics import log_loss


DATASET_PATH = Path(
    "data/processed/v0_dataset_v1.parquet"
)

FOLD_PATH = Path(
    "data/processed/v0_a1_oof_predictions.parquet"
)

OUTPUT_PATH = Path(
    "artifacts/v0_xgb_a5_task_group_importance.csv"
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
# XGBoost config
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
# Helpers
# ==================================================

def normalize(
    probabilities,
):

    probabilities = (
        probabilities
        .astype(
            np.float64
        )
    )

    return (
        probabilities
        / probabilities.sum(
            axis=1,
            keepdims=True,
        )
    )


def evaluate_tasks(
    labels,
    probabilities,
):

    # ----------------------------------------------
    # Full 3-class task
    # ----------------------------------------------

    y_three = np.array([
        LABEL_TO_ID[label]
        for label in labels
    ])


    three_class_ll = log_loss(
        y_three,
        probabilities,
        labels=[
            0,
            1,
            2,
        ],
    )


    # ----------------------------------------------
    # Task 1:
    # Plant vs No Plant
    # ----------------------------------------------

    y_plant = (
        labels
        != "NO_PLANT"
    ).astype(int)


    p_plant = (
        probabilities[:, 0]
        + probabilities[:, 1]
    )


    plant_probabilities = (
        np.column_stack([
            1.0 - p_plant,
            p_plant,
        ])
    )


    plant_ll = log_loss(
        y_plant,
        plant_probabilities,
        labels=[
            0,
            1,
        ],
    )


    # ----------------------------------------------
    # Task 2:
    # A vs B given TRUE plant
    # ----------------------------------------------

    true_plant_mask = (
        labels
        != "NO_PLANT"
    )


    plant_prob_mass = (
        probabilities[
            true_plant_mask,
            0
        ]
        +
        probabilities[
            true_plant_mask,
            1
        ]
    )


    # Safe numerical floor
    plant_prob_mass = np.maximum(
        plant_prob_mass,
        1e-12,
    )


    p_a_given_plant = (
        probabilities[
            true_plant_mask,
            0
        ]
        /
        plant_prob_mass
    )


    p_b_given_plant = (
        probabilities[
            true_plant_mask,
            1
        ]
        /
        plant_prob_mass
    )


    site_probabilities = np.column_stack([
        p_a_given_plant,
        p_b_given_plant,
    ])


    site_labels = labels[
        true_plant_mask
    ]


    y_site = np.array([
        0 if label == "A_PLANT"
        else 1
        for label in site_labels
    ])


    site_ll = log_loss(
        y_site,
        site_probabilities,
        labels=[
            0,
            1,
        ],
    )


    return {
        "three_class_log_loss":
            three_class_ll,

        "plant_log_loss":
            plant_ll,

        "site_log_loss":
            site_ll,
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
    ]
    .to_numpy()
)


labels = (
    df[
        "label"
    ]
    .to_numpy()
)


y = np.array([
    LABEL_TO_ID[label]
    for label in labels
])


X = (
    df
    .select(
        FEATURES
    )
    .to_numpy()
)


feature_to_index = {
    feature: i
    for i, feature
    in enumerate(FEATURES)
}


group_indices = {
    group_name: np.array([
        feature_to_index[
            feature
        ]
        for feature
        in features
    ])

    for group_name, features
    in FEATURE_GROUPS.items()
}


# ==================================================
# Train models on frozen folds
# ==================================================

fold_models = []


baseline_probabilities = np.zeros(
    (
        len(df),
        3,
    ),
    dtype=float,
)


print("\n" + "=" * 110)
print("XGB_A5 — TASK-SPECIFIC GROUP IMPORTANCE")
print("=" * 110)


for fold in sorted(
    np.unique(
        fold_ids
    )
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


    probabilities = normalize(
        model.predict_proba(
            X[
                test_index
            ]
        )
    )


    baseline_probabilities[
        test_index
    ] = probabilities


    fold_models.append(
        (
            int(fold),
            model,
            test_index,
        )
    )


# ==================================================
# Baseline
# ==================================================

baseline = evaluate_tasks(
    labels,
    baseline_probabilities,
)


print("\nBASELINE")

print(
    f"3-class LL:       "
    f"{baseline['three_class_log_loss']:.4f}"
)

print(
    f"Plant vs No LL:   "
    f"{baseline['plant_log_loss']:.4f}"
)

print(
    f"A vs B | Plant LL:"
    f" {baseline['site_log_loss']:.4f}"
)


# ==================================================
# Permutation experiment
# ==================================================

rows = []


for group_name, columns in (
    group_indices.items()
):

    delta_three = []
    delta_plant = []
    delta_site = []


    for repeat in range(
        N_REPEATS
    ):

        permuted_probabilities = np.zeros(
            (
                len(df),
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
                + fold
            )


            permutation = (
                rng.permutation(
                    len(
                        test_index
                    )
                )
            )


            # Preserve relationships inside
            # the feature family.
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


            permuted_probabilities[
                test_index
            ] = normalize(
                model.predict_proba(
                    X_test
                )
            )


        result = evaluate_tasks(
            labels,
            permuted_probabilities,
        )


        delta_three.append(
            result[
                "three_class_log_loss"
            ]
            -
            baseline[
                "three_class_log_loss"
            ]
        )


        delta_plant.append(
            result[
                "plant_log_loss"
            ]
            -
            baseline[
                "plant_log_loss"
            ]
        )


        delta_site.append(
            result[
                "site_log_loss"
            ]
            -
            baseline[
                "site_log_loss"
            ]
        )


    row = {
        "feature_group":
            group_name,

        "n_features":
            len(
                columns
            ),

        "delta_three_class_ll":
            float(
                np.mean(
                    delta_three
                )
            ),

        "std_three_class_ll":
            float(
                np.std(
                    delta_three
                )
            ),

        "delta_plant_ll":
            float(
                np.mean(
                    delta_plant
                )
            ),

        "std_plant_ll":
            float(
                np.std(
                    delta_plant
                )
            ),

        "delta_site_ll":
            float(
                np.mean(
                    delta_site
                )
            ),

        "std_site_ll":
            float(
                np.std(
                    delta_site
                )
            ),
    }


    rows.append(
        row
    )


    print("\n" + "-" * 110)
    print(
        group_name
    )
    print("-" * 110)


    print(
        "Δ 3-class LL:     "
        f"{row['delta_three_class_ll']:+.4f}"
        " ± "
        f"{row['std_three_class_ll']:.4f}"
    )


    print(
        "Δ Plant/No LL:    "
        f"{row['delta_plant_ll']:+.4f}"
        " ± "
        f"{row['std_plant_ll']:.4f}"
    )


    print(
        "Δ A/B | Plant LL: "
        f"{row['delta_site_ll']:+.4f}"
        " ± "
        f"{row['std_site_ll']:.4f}"
    )


# ==================================================
# Results
# ==================================================

summary = pl.DataFrame(
    rows
)


print("\n" + "=" * 110)
print("IMPORTANCE BY SUBTASK")
print("=" * 110)


print(
    summary
    .select([
        "feature_group",
        "delta_three_class_ll",
        "delta_plant_ll",
        "delta_site_ll",
    ])
    .sort(
        "delta_three_class_ll",
        descending=True,
    )
)


print("\nPLANT vs NO — RANKING")

print(
    summary
    .select([
        "feature_group",
        "delta_plant_ll",
    ])
    .sort(
        "delta_plant_ll",
        descending=True,
    )
)


print("\nA vs B | TRUE PLANT — RANKING")

print(
    summary
    .select([
        "feature_group",
        "delta_site_ll",
    ])
    .sort(
        "delta_site_ll",
        descending=True,
    )
)


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


summary.write_csv(
    OUTPUT_PATH
)


print("\nSaved:")
print(
    OUTPUT_PATH
)

print(
    "\n✅ TASK-SPECIFIC GROUP IMPORTANCE COMPLETE"
)
