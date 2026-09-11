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

from xgboost import XGBClassifier


# ============================================================
# Paths
# ============================================================

DATASET_PATH = Path(
    "data/processed/"
    "v0_dataset_v2_timing_corrected.parquet"
)

FOLD_MAP_PATH = Path(
    "data/interim/"
    "v0_timing_corrected_fold_map.csv"
)

OUTPUT_DIR = Path("artifacts")

PREDICTION_DIR = Path(
    "data/processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
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
    for i, label in enumerate(LABELS)
}

ID_TO_LABEL = {
    i: label
    for label, i in LABEL_TO_ID.items()
}


# ============================================================
# Feature families
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


LOGISTIC_STAGES = {
    "A1": A1,
    "A2": A1 + A2,
    "A3": A1 + A2 + A3,
    "A4": A1 + A2 + A3 + A4,

    # Preserve original V0 design:
    # A4 economy was dropped.
    "A5": A1 + A2 + A3 + A5,
}


XGB_STAGES = {
    "XGB_A3":
        A1 + A2 + A3,

    "XGB_A4":
        A1 + A2 + A3 + A4,

    "XGB_A5":
        A1 + A2 + A3 + A5,
}


# ============================================================
# Fixed XGBoost config
#
# EXACTLY preserve the original V0 configuration.
# No tuning after timing correction.
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
# Metrics
# ============================================================

def multiclass_brier_score(
    y_true,
    probabilities,
):

    label_to_index = {
        label: i
        for i, label
        in enumerate(LABELS)
    }

    truth = np.zeros(
        (
            len(y_true),
            len(LABELS),
        ),
        dtype=float,
    )

    for i, label in enumerate(
        y_true
    ):
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
        for i, label
        in enumerate(model_classes)
    }

    return np.column_stack([
        probabilities[
            :,
            mapping[label],
        ]
        for label in LABELS
    ])


def evaluate(
    y_true,
    probabilities,
):

    prediction_ids = np.argmax(
        probabilities,
        axis=1,
    )

    predictions = np.array([
        LABELS[i]
        for i in prediction_ids
    ])

    recalls = recall_score(
        y_true,
        predictions,
        labels=LABELS,
        average=None,
        zero_division=0,
    )

    return {
        "log_loss":
            log_loss(
                y_true,
                probabilities,
                labels=LABELS,
            ),

        "brier_score":
            multiclass_brier_score(
                y_true,
                probabilities,
            ),

        "accuracy":
            accuracy_score(
                y_true,
                predictions,
            ),

        "macro_f1":
            f1_score(
                y_true,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),

        "macro_precision":
            precision_score(
                y_true,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            ),

        "macro_recall":
            recall_score(
                y_true,
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


# ============================================================
# Load corrected dataset + frozen match folds
# ============================================================

df = pl.read_parquet(
    DATASET_PATH
)

fold_map = pl.read_csv(
    FOLD_MAP_PATH
)


assert df.height == 1686

assert (
    df["demo_filename"]
    .n_unique()
    == 20
)


fold_lookup = {
    row["demo_filename"]:
        int(row["cv_fold"])

    for row in fold_map.iter_rows(
        named=True
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


assert (
    len(
        np.unique(
            fold_ids
        )
    )
    == 5
)


groups = (
    df[
        "demo_filename"
    ]
    .to_numpy()
)

y = (
    df["label"]
    .to_numpy()
)


print("\n" + "=" * 110)
print("V0 — TIMING-CORRECTED SCIENTIFIC EVALUATION")
print("=" * 110)

print(
    f"Dataset:      {DATASET_PATH}"
)

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


# ============================================================
# Validate fold leakage
# ============================================================

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

    assert not (
        train_matches
        & test_matches
    )


print(
    "\n✅ Frozen match-level CV leakage check passed"
)


# ============================================================
# Shared save helpers
# ============================================================

KEYS = [
    "demo_filename",
    "round_num",
    "horizon_sec",
    "target_tick",
    "label",
]


def save_predictions(
    name,
    probabilities,
    predictions,
):

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
                probabilities[:, 0],
            ),

            pl.Series(
                "p_b_plant",
                probabilities[:, 1],
            ),

            pl.Series(
                "p_no_plant",
                probabilities[:, 2],
            ),

            pl.Series(
                "prediction",
                predictions,
            ),
        ])
    )

    path = (
        PREDICTION_DIR
        / (
            f"v0_tc_{name.lower()}"
            "_oof_predictions.parquet"
        )
    )

    prediction_df.write_parquet(
        path
    )

    return path


def save_metrics(
    name,
    fold_rows,
    overall,
):

    overall_row = {
        "fold": 0,

        "n_train_matches":
            None,

        "n_test_matches":
            20,

        "n_train_observations":
            None,

        "n_test_observations":
            df.height,

        "log_loss":
            overall["log_loss"],

        "brier_score":
            overall["brier_score"],

        "accuracy":
            overall["accuracy"],

        "macro_f1":
            overall["macro_f1"],

        "macro_precision":
            overall["macro_precision"],

        "macro_recall":
            overall["macro_recall"],

        "recall_a":
            overall["recall_a"],

        "recall_b":
            overall["recall_b"],

        "recall_no":
            overall["recall_no"],
    }

    metrics = pl.DataFrame(
        fold_rows
        + [overall_row]
    )

    path = (
        OUTPUT_DIR
        / (
            f"v0_tc_{name.lower()}"
            "_metrics.csv"
        )
    )

    metrics.write_csv(
        path
    )

    return path


def print_overall(
    name,
    result,
):

    print(
        "\n" + "-" * 110
    )

    print(
        f"{name} OUT-OF-FOLD"
    )

    print(
        "-" * 110
    )

    print(
        f"Log Loss:        "
        f"{result['log_loss']:.6f}"
    )

    print(
        f"Brier Score:     "
        f"{result['brier_score']:.6f}"
    )

    print(
        f"Accuracy:        "
        f"{result['accuracy']:.6f}"
    )

    print(
        f"Macro F1:        "
        f"{result['macro_f1']:.6f}"
    )

    print(
        f"Macro Precision: "
        f"{result['macro_precision']:.6f}"
    )

    print(
        f"Macro Recall:    "
        f"{result['macro_recall']:.6f}"
    )

    print(
        "\nPer-class recall:"
    )

    print(
        f"  A_PLANT:  "
        f"{result['recall_a']:.6f}"
    )

    print(
        f"  B_PLANT:  "
        f"{result['recall_b']:.6f}"
    )

    print(
        f"  NO_PLANT: "
        f"{result['recall_no']:.6f}"
    )


def print_by_horizon(
    probabilities,
):

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
            y[
                mask
            ],
            probabilities[
                mask
            ],
        )

        print(
            f"  {horizon:>2}s"
            f" | n={mask.sum():>3}"
            f" | LL={result['log_loss']:.4f}"
            f" | Brier={result['brier_score']:.4f}"
            f" | Acc={result['accuracy']:.4f}"
            f" | F1={result['macro_f1']:.4f}"
        )


# ============================================================
# A0 — training-fold class prior
# ============================================================

print("\n" + "=" * 110)
print("A0 — CLASS PRIOR BASELINE")
print("=" * 110)


a0_probabilities = np.zeros(
    (
        df.height,
        len(LABELS),
    ),
    dtype=float,
)

a0_fold_rows = []


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

    y_train = y[
        train_index
    ]

    priors = np.array([
        np.mean(
            y_train
            == label
        )
        for label in LABELS
    ])

    probabilities = np.tile(
        priors,
        (
            len(test_index),
            1,
        ),
    )

    a0_probabilities[
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
        f" | train={len(train_index)}"
        f" | test={len(test_index)}"
        f" | priors="
        f"{priors.round(4)}"
        f" | LL="
        f"{fold_result['log_loss']:.4f}"
    )

    a0_fold_rows.append({
        "fold":
            int(fold),

        "n_train_matches":
            len(
                set(
                    groups[
                        train_index
                    ]
                )
            ),

        "n_test_matches":
            len(
                set(
                    groups[
                        test_index
                    ]
                )
            ),

        "n_train_observations":
            len(
                train_index
            ),

        "n_test_observations":
            len(
                test_index
            ),

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

        "macro_precision":
            fold_result[
                "macro_precision"
            ],

        "macro_recall":
            fold_result[
                "macro_recall"
            ],

        "recall_a":
            fold_result[
                "recall_a"
            ],

        "recall_b":
            fold_result[
                "recall_b"
            ],

        "recall_no":
            fold_result[
                "recall_no"
            ],
    })


a0_result = evaluate(
    y,
    a0_probabilities,
)

print_overall(
    "A0",
    a0_result,
)

print_by_horizon(
    a0_probabilities
)

save_predictions(
    "a0",
    a0_probabilities,
    a0_result[
        "predictions"
    ],
)

save_metrics(
    "a0",
    a0_fold_rows,
    a0_result,
)


# ============================================================
# Logistic regression A1 → A5
# ============================================================

summary_rows = [{
    "model":
        "A0",

    "model_family":
        "PRIOR",

    "n_features":
        0,

    "log_loss":
        a0_result[
            "log_loss"
        ],

    "brier_score":
        a0_result[
            "brier_score"
        ],

    "accuracy":
        a0_result[
            "accuracy"
        ],

    "macro_f1":
        a0_result[
            "macro_f1"
        ],

    "recall_a":
        a0_result[
            "recall_a"
        ],

    "recall_b":
        a0_result[
            "recall_b"
        ],

    "recall_no":
        a0_result[
            "recall_no"
        ],
}]


logistic_results = {}


for stage, features in (
    LOGISTIC_STAGES.items()
):

    print(
        "\n" + "=" * 110
    )

    print(
        f"{stage} — LOGISTIC REGRESSION"
        f" | {len(features)} features"
    )

    print(
        "=" * 110
    )

    X = (
        df
        .select(
            features
        )
        .to_numpy()
    )

    probabilities_all = np.zeros(
        (
            df.height,
            len(LABELS),
        ),
        dtype=float,
    )

    fold_rows = []


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
            X[
                train_index
            ],
            y[
                train_index
            ],
        )

        raw_probabilities = (
            model
            .predict_proba(
                X[
                    test_index
                ]
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

        probabilities_all[
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
            f" | LL="
            f"{fold_result['log_loss']:.4f}"
            f" | Brier="
            f"{fold_result['brier_score']:.4f}"
            f" | Acc="
            f"{fold_result['accuracy']:.4f}"
            f" | F1="
            f"{fold_result['macro_f1']:.4f}"
        )

        fold_rows.append({
            "fold":
                int(fold),

            "n_train_matches":
                len(
                    set(
                        groups[
                            train_index
                        ]
                    )
                ),

            "n_test_matches":
                len(
                    set(
                        groups[
                            test_index
                        ]
                    )
                ),

            "n_train_observations":
                len(
                    train_index
                ),

            "n_test_observations":
                len(
                    test_index
                ),

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

            "macro_precision":
                fold_result[
                    "macro_precision"
                ],

            "macro_recall":
                fold_result[
                    "macro_recall"
                ],

            "recall_a":
                fold_result[
                    "recall_a"
                ],

            "recall_b":
                fold_result[
                    "recall_b"
                ],

            "recall_no":
                fold_result[
                    "recall_no"
                ],
        })


    result = evaluate(
        y,
        probabilities_all,
    )

    logistic_results[
        stage
    ] = result

    print_overall(
        stage,
        result,
    )

    print_by_horizon(
        probabilities_all
    )

    save_predictions(
        stage,
        probabilities_all,
        result[
            "predictions"
        ],
    )

    save_metrics(
        stage,
        fold_rows,
        result,
    )

    summary_rows.append({
        "model":
            stage,

        "model_family":
            "LOGISTIC",

        "n_features":
            len(features),

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


# ============================================================
# Logistic ablation summary
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "TIMING-CORRECTED LOGISTIC ABLATION"
)

print(
    "=" * 110
)


previous_name = "A0"
previous_result = a0_result


for stage in [
    "A1",
    "A2",
    "A3",
    "A4",
]:

    current = (
        logistic_results[
            stage
        ]
    )

    print(
        f"\n{previous_name} → {stage}"
    )

    print(
        f"  LL:    "
        f"{previous_result['log_loss']:.4f}"
        f" → {current['log_loss']:.4f}"
        f" | Δ="
        f"{current['log_loss'] - previous_result['log_loss']:+.4f}"
    )

    print(
        f"  Brier: "
        f"{previous_result['brier_score']:.4f}"
        f" → {current['brier_score']:.4f}"
        f" | Δ="
        f"{current['brier_score'] - previous_result['brier_score']:+.4f}"
    )

    print(
        f"  F1:    "
        f"{previous_result['macro_f1']:.4f}"
        f" → {current['macro_f1']:.4f}"
        f" | Δ="
        f"{current['macro_f1'] - previous_result['macro_f1']:+.4f}"
    )

    previous_name = stage
    previous_result = current


# A5 is intentionally compared directly with A3.
a3 = logistic_results[
    "A3"
]

a5 = logistic_results[
    "A5"
]

print(
    "\nA3 → A5"
    " (A4 economy excluded)"
)

print(
    f"  LL:    "
    f"{a3['log_loss']:.4f}"
    f" → {a5['log_loss']:.4f}"
    f" | Δ="
    f"{a5['log_loss'] - a3['log_loss']:+.4f}"
)

print(
    f"  Brier: "
    f"{a3['brier_score']:.4f}"
    f" → {a5['brier_score']:.4f}"
    f" | Δ="
    f"{a5['brier_score'] - a3['brier_score']:+.4f}"
)

print(
    f"  F1:    "
    f"{a3['macro_f1']:.4f}"
    f" → {a5['macro_f1']:.4f}"
    f" | Δ="
    f"{a5['macro_f1'] - a3['macro_f1']:+.4f}"
)


# ============================================================
# XGBoost
# ============================================================

y_ids = np.array([
    LABEL_TO_ID[
        label
    ]
    for label in y
])


for stage, features in (
    XGB_STAGES.items()
):

    print(
        "\n" + "=" * 110
    )

    print(
        f"{stage} — XGBOOST"
        f" | {len(features)} features"
    )

    print(
        "=" * 110
    )

    X = (
        df
        .select(
            features
        )
        .to_numpy()
    )

    probabilities_all = np.zeros(
        (
            df.height,
            len(LABELS),
        ),
        dtype=float,
    )

    fold_rows = []


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
            y_ids[
                train_index
            ],
        )

        probabilities = (
            model.predict_proba(
                X[
                    test_index
                ]
            )
            .astype(
                np.float64
            )
        )

        probabilities = (
            probabilities
            / probabilities.sum(
                axis=1,
                keepdims=True,
            )
        )

        probabilities_all[
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
            f" | LL="
            f"{fold_result['log_loss']:.4f}"
            f" | Brier="
            f"{fold_result['brier_score']:.4f}"
            f" | Acc="
            f"{fold_result['accuracy']:.4f}"
            f" | F1="
            f"{fold_result['macro_f1']:.4f}"
        )

        fold_rows.append({
            "fold":
                int(fold),

            "n_train_matches":
                len(
                    set(
                        groups[
                            train_index
                        ]
                    )
                ),

            "n_test_matches":
                len(
                    set(
                        groups[
                            test_index
                        ]
                    )
                ),

            "n_train_observations":
                len(
                    train_index
                ),

            "n_test_observations":
                len(
                    test_index
                ),

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

            "macro_precision":
                fold_result[
                    "macro_precision"
                ],

            "macro_recall":
                fold_result[
                    "macro_recall"
                ],

            "recall_a":
                fold_result[
                    "recall_a"
                ],

            "recall_b":
                fold_result[
                    "recall_b"
                ],

            "recall_no":
                fold_result[
                    "recall_no"
                ],
        })


    result = evaluate(
        y,
        probabilities_all,
    )

    print_overall(
        stage,
        result,
    )

    print_by_horizon(
        probabilities_all
    )

    save_predictions(
        stage,
        probabilities_all,
        result[
            "predictions"
        ],
    )

    save_metrics(
        stage,
        fold_rows,
        result,
    )

    summary_rows.append({
        "model":
            stage,

        "model_family":
            "XGBOOST",

        "n_features":
            len(features),

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


# ============================================================
# Final summary
# ============================================================

summary = (
    pl.DataFrame(
        summary_rows
    )
)


summary_path = Path(
    "artifacts/"
    "v0_tc_model_summary.csv"
)

summary.write_csv(
    summary_path
)


print(
    "\n" + "=" * 110
)

print(
    "TIMING-CORRECTED V0 MODEL SUMMARY"
)

print(
    "=" * 110
)


print(
    summary.select([
        "model",
        "model_family",
        "n_features",
        "log_loss",
        "brier_score",
        "accuracy",
        "macro_f1",
    ])
)


print(
    "\nSaved summary:"
)

print(
    summary_path
)


print(
    "\n✅ TIMING-CORRECTED V0 "
    "SCIENTIFIC EVALUATION COMPLETE"
)
