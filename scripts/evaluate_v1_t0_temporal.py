from pathlib import Path

import numpy as np
import polars as pl

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from xgboost import XGBClassifier


# ============================================================
# Paths
# ============================================================

DATASET_PATH = Path(
    "data/processed/"
    "v1_t0_temporal_dataset.parquet"
)

FOLD_MAP_PATH = Path(
    "data/interim/"
    "v0_timing_corrected_fold_map.csv"
)

V0_OOF_PATH = Path(
    "data/processed/"
    "v0_tc_xgb_a5_oof_predictions.parquet"
)

OUTPUT_OOF_PATH = Path(
    "data/processed/"
    "v1_t0_temporal_oof_predictions.parquet"
)

METRICS_PATH = Path(
    "artifacts/"
    "v1_t0_temporal_metrics.csv"
)

COMPARISON_PATH = Path(
    "artifacts/"
    "v1_t0_vs_v0_comparison.csv"
)


# ============================================================
# Labels
# ============================================================

LABELS = [
    "A_PLANT",
    "B_PLANT",
    "NO_PLANT",
]

LABEL_TO_ID = {
    label: i
    for i, label
    in enumerate(LABELS)
}


# ============================================================
# Frozen V0 37-feature contract
# ============================================================

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


V0_FEATURES = (
    A1
    + A2
    + A3
    + A5
)


assert len(V0_FEATURES) == 37
assert len(set(V0_FEATURES)) == 37


# ============================================================
# Temporal feature contract
# ============================================================

TEMPORAL_STATE_FIELDS = [
    "t_alive",
    "ct_alive",

    "t_health_sum",
    "ct_health_sum",

    "t_armor_sum",
    "ct_armor_sum",

    "t_centroid_x",
    "t_centroid_y",

    "ct_centroid_x",
    "ct_centroid_y",

    "t_stretch_xy",
    "ct_stretch_xy",

    "bomb_x",
    "bomb_y",

    "bomb_to_t_centroid_distance",
    "t_ct_centroid_distance",
    "minimum_t_ct_distance",
]


TEMPORAL_WINDOWS = [
    3,
    5,
]


TEMPORAL_FEATURES = [
    f"{field}_delta_{window}s"

    for window in TEMPORAL_WINDOWS

    for field in TEMPORAL_STATE_FIELDS
]


assert len(TEMPORAL_FEATURES) == 34
assert len(set(TEMPORAL_FEATURES)) == 34


FEATURES = (
    V0_FEATURES
    + TEMPORAL_FEATURES
)


assert len(FEATURES) == 71
assert len(set(FEATURES)) == 71


# ============================================================
# Frozen XGB config
# ============================================================

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


# ============================================================
# Helpers
# ============================================================

def multiclass_brier(
    y_true_id,
    probabilities,
):

    truth = np.zeros(
        (
            len(y_true_id),
            3,
        ),
        dtype=float,
    )

    truth[
        np.arange(
            len(y_true_id)
        ),
        y_true_id,
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
    y_true_labels,
    probabilities,
):

    y_true_id = np.array([
        LABEL_TO_ID[label]
        for label in y_true_labels
    ])


    prediction_id = np.argmax(
        probabilities,
        axis=1,
    )


    prediction_labels = np.array([
        LABELS[i]
        for i in prediction_id
    ])


    recalls = recall_score(
        y_true_labels,
        prediction_labels,
        labels=LABELS,
        average=None,
        zero_division=0,
    )


    return {
        "log_loss":
            log_loss(
                y_true_id,
                probabilities,
                labels=[
                    0,
                    1,
                    2,
                ],
            ),

        "brier_score":
            multiclass_brier(
                y_true_id,
                probabilities,
            ),

        "accuracy":
            accuracy_score(
                y_true_labels,
                prediction_labels,
            ),

        "macro_f1":
            f1_score(
                y_true_labels,
                prediction_labels,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),

        "macro_precision":
            precision_score(
                y_true_labels,
                prediction_labels,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),

        "macro_recall":
            recall_score(
                y_true_labels,
                prediction_labels,
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

        "prediction":
            prediction_labels,
    }


def probabilities_from_frame(
    frame,
):

    return (
        frame
        .select([
            "p_a_plant",
            "p_b_plant",
            "p_no_plant",
        ])
        .to_numpy()
        .astype(
            np.float64
        )
    )


# ============================================================
# Load
# ============================================================

df = pl.read_parquet(
    DATASET_PATH
)

fold_map = pl.read_csv(
    FOLD_MAP_PATH
)

v0 = pl.read_parquet(
    V0_OOF_PATH
)


assert df.height == 1686
assert v0.height == 1686

assert (
    df["demo_filename"]
    .n_unique()
    == 20
)


KEYS = [
    "demo_filename",
    "round_num",
    "horizon_sec",
    "target_tick",
    "label",
]


assert (
    df
    .select(
        KEYS
    )
    .equals(
        v0.select(
            KEYS
        )
    )
)


missing = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]


assert not missing, (
    f"Missing features: {missing}"
)


X = (
    df
    .select(
        FEATURES
    )
    .to_numpy()
    .astype(
        np.float64
    )
)


assert np.isfinite(
    X
).all()


y_labels = (
    df[
        "label"
    ]
    .to_numpy()
)


y = np.array([
    LABEL_TO_ID[label]
    for label in y_labels
])


fold_lookup = {
    row["demo_filename"]:
        int(
            row["cv_fold"]
        )

    for row in (
        fold_map
        .iter_rows(
            named=True
        )
    )
}


fold_ids = np.array([
    fold_lookup[
        filename
    ]

    for filename
    in df[
        "demo_filename"
    ].to_list()
])


assert set(
    np.unique(
        fold_ids
    )
) == {
    1,
    2,
    3,
    4,
    5,
}


# ============================================================
# OOF training
# ============================================================

oof = np.full(
    (
        df.height,
        3,
    ),
    np.nan,
    dtype=np.float64,
)


fold_rows = []


print(
    "\n"
    + "=" * 100
)

print(
    "V1-T0 — TABULAR TEMPORAL BASELINE"
)

print(
    "=" * 100
)

print(
    f"\nObservations:     "
    f"{df.height}"
)

print(
    f"Matches:          "
    f"{df['demo_filename'].n_unique()}"
)

print(
    f"V0 features:      "
    f"{len(V0_FEATURES)}"
)

print(
    f"Temporal features:"
    f" {len(TEMPORAL_FEATURES)}"
)

print(
    f"Total features:   "
    f"{len(FEATURES)}"
)

print(
    f"Frozen folds:     "
    f"{len(np.unique(fold_ids))}"
)


for fold in sorted(
    np.unique(
        fold_ids
    )
):

    train_index = np.where(
        fold_ids
        != fold
    )[0]

    test_index = np.where(
        fold_ids
        == fold
    )[0]


    train_groups = set(
        df[
            "demo_filename"
        ]
        .to_numpy()[
            train_index
        ]
    )

    test_groups = set(
        df[
            "demo_filename"
        ]
        .to_numpy()[
            test_index
        ]
    )


    assert not (
        train_groups
        & test_groups
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
        model
        .predict_proba(
            X[
                test_index
            ]
        )
        .astype(
            np.float64
        )
    )


    assert probabilities.shape == (
        len(test_index),
        3,
    )

    assert np.allclose(
        probabilities.sum(
            axis=1
        ),
        1.0,
        atol=1e-6,
    )


    oof[
        test_index
    ] = probabilities


    result = evaluate(
        y_labels[
            test_index
        ],
        probabilities,
    )


    fold_rows.append({
        "fold":
            int(fold),

        "n_train":
            len(
                train_index
            ),

        "n_test":
            len(
                test_index
            ),

        "log_loss":
            result[
                "log_loss"
            ],

        "brier_score":
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
    })


    print(
        f"\nFold {fold}"
    )

    print(
        f"  train rows: "
        f"{len(train_index)}"
    )

    print(
        f"  test rows:  "
        f"{len(test_index)}"
    )

    print(
        f"  LL="
        f"{result['log_loss']:.4f}"
        f" | Brier="
        f"{result['brier_score']:.4f}"
        f" | Acc="
        f"{result['accuracy']:.4f}"
        f" | F1="
        f"{result['macro_f1']:.4f}"
    )


assert np.isfinite(
    oof
).all()


# ============================================================
# Full OOF
# ============================================================

t0_result = evaluate(
    y_labels,
    oof,
)


print(
    "\n"
    + "-" * 100
)

print(
    "V1-T0 FULL THREE-CLASS OOF"
)

print(
    "-" * 100
)


for metric in [
    "log_loss",
    "brier_score",
    "accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "recall_a",
    "recall_b",
    "recall_no",
]:

    print(
        f"{metric:16s}: "
        f"{t0_result[metric]:.6f}"
    )


# ============================================================
# By horizon
# ============================================================

print(
    "\nBY TRUE HORIZON"
)


horizons = (
    df[
        "horizon_sec"
    ]
    .to_numpy()
)


for horizon in [
    10,
    20,
    30,
    40,
]:

    mask = (
        horizons
        == horizon
    )


    result = evaluate(
        y_labels[
            mask
        ],
        oof[
            mask
        ],
    )


    print(
        f"  {horizon:>2}s"
        f" | n={mask.sum():>3}"
        f" | LL="
        f"{result['log_loss']:.4f}"
        f" | Brier="
        f"{result['brier_score']:.4f}"
        f" | Acc="
        f"{result['accuracy']:.4f}"
        f" | F1="
        f"{result['macro_f1']:.4f}"
    )


# ============================================================
# Compare frozen V0
# ============================================================

v0_probabilities = (
    probabilities_from_frame(
        v0
    )
)


v0_result = evaluate(
    y_labels,
    v0_probabilities,
)


comparison_rows = []


for metric in [
    "log_loss",
    "brier_score",
    "accuracy",
    "macro_f1",
    "recall_a",
    "recall_b",
    "recall_no",
]:

    comparison_rows.append({
        "metric":
            metric,

        "v0_xgb_a5":
            v0_result[
                metric
            ],

        "v1_t0_temporal":
            t0_result[
                metric
            ],

        "delta_t0_minus_v0":
            t0_result[
                metric
            ]
            -
            v0_result[
                metric
            ],
    })


comparison = pl.DataFrame(
    comparison_rows
)


comparison.write_csv(
    COMPARISON_PATH
)


print(
    "\n"
    + "=" * 100
)

print(
    "V1-T0 VS FROZEN V0 XGB-A5"
)

print(
    "=" * 100
)


for row in comparison_rows:

    print(
        f"{row['metric']:14s}"
        f" | V0="
        f"{row['v0_xgb_a5']:.6f}"
        f" | T0="
        f"{row['v1_t0_temporal']:.6f}"
        f" | Δ="
        f"{row['delta_t0_minus_v0']:+.6f}"
    )


# ============================================================
# Save OOF
# ============================================================

prediction_df = (
    df
    .select(
        KEYS
    )
    .with_columns([
        pl.Series(
            "cv_fold",
            fold_ids,
        ),

        pl.Series(
            "p_a_plant",
            oof[
                :, 0
            ],
        ),

        pl.Series(
            "p_b_plant",
            oof[
                :, 1
            ],
        ),

        pl.Series(
            "p_no_plant",
            oof[
                :, 2
            ],
        ),

        pl.Series(
            "prediction",
            t0_result[
                "prediction"
            ],
        ),
    ])
)


prediction_df.write_parquet(
    OUTPUT_OOF_PATH
)


overall_row = {
    "fold":
        0,

    "n_train":
        None,

    "n_test":
        df.height,

    "log_loss":
        t0_result[
            "log_loss"
        ],

    "brier_score":
        t0_result[
            "brier_score"
        ],

    "accuracy":
        t0_result[
            "accuracy"
        ],

    "macro_f1":
        t0_result[
            "macro_f1"
        ],
}


pl.DataFrame(
    fold_rows
    + [
        overall_row
    ]
).write_csv(
    METRICS_PATH
)


print(
    "\nSaved:"
)

print(
    f"  {OUTPUT_OOF_PATH}"
)

print(
    f"  {METRICS_PATH}"
)

print(
    f"  {COMPARISON_PATH}"
)


print(
    "\n✅ V1-T0 TEMPORAL BASELINE COMPLETE"
)
