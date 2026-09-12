from pathlib import Path

import numpy as np
import polars as pl

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
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
    "v0_dataset_v2_timing_corrected.parquet"
)

FOLD_MAP_PATH = Path(
    "data/interim/"
    "v0_timing_corrected_fold_map.csv"
)

V0_OOF_PATH = Path(
    "data/processed/"
    "v0_tc_xgb_a5_oof_predictions.parquet"
)

V1_OOF_PATH = Path(
    "data/processed/"
    "v1_h0_oof_predictions.parquet"
)

METRICS_PATH = Path(
    "artifacts/"
    "v1_h0_metrics.csv"
)

COMPARISON_PATH = Path(
    "artifacts/"
    "v1_h0_vs_v0_comparison.csv"
)


# ============================================================
# Frozen labels
# ============================================================

LABELS = [
    "A_PLANT",
    "B_PLANT",
    "NO_PLANT",
]


# ============================================================
# Frozen V0 A5 feature contract
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


FEATURES = (
    A1
    + A2
    + A3
    + A5
)


assert len(FEATURES) == 37
assert len(set(FEATURES)) == 37


# ============================================================
# Frozen V0 XGBoost configuration
#
# V1-H0 changes architecture only.
# It does NOT tune hyperparameters.
# ============================================================

XGB_BINARY_CONFIG = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,

    "min_child_weight": 5,

    "subsample": 0.8,
    "colsample_bytree": 0.8,

    "reg_alpha": 0.5,
    "reg_lambda": 5.0,

    "objective": "binary:logistic",
    "eval_metric": "logloss",

    "tree_method": "hist",

    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# Helpers
# ============================================================

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

    label_to_index = {
        label: i
        for i, label
        in enumerate(LABELS)
    }

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


def evaluate_multiclass(
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

    precisions = precision_score(
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

        "precision_a":
            precisions[0],

        "precision_b":
            precisions[1],

        "precision_no":
            precisions[2],

        "predictions":
            predictions,
    }


def evaluate_binary(
    y_true,
    probability_positive,
):

    predictions = (
        probability_positive
        >= 0.5
    ).astype(int)

    return {
        "log_loss":
            log_loss(
                y_true,
                np.column_stack([
                    1.0
                    - probability_positive,
                    probability_positive,
                ]),
                labels=[0, 1],
            ),

        "brier_score":
            brier_score_loss(
                y_true,
                probability_positive,
            ),

        "accuracy":
            accuracy_score(
                y_true,
                predictions,
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),
    }


def metric_row(
    fold,
    train_index,
    test_index,
    groups,
    result,
    n_head2_train,
):

    return {
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
            len(train_index),

        "n_head2_train_observations":
            int(n_head2_train),

        "n_test_observations":
            len(test_index),

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
    }


def print_multiclass_result(
    name,
    result,
):

    print(
        "\n"
        + "-" * 100
    )

    print(name)

    print(
        "-" * 100
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


# ============================================================
# Load corrected data
# ============================================================

df = pl.read_parquet(
    DATASET_PATH
)

fold_map = pl.read_csv(
    FOLD_MAP_PATH
)


assert df.height == 1686

assert (
    df[
        "demo_filename"
    ]
    .n_unique()
    == 20
)


missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

assert not missing_features, (
    "Missing frozen V0 features: "
    f"{missing_features}"
)


fold_lookup = {
    row[
        "demo_filename"
    ]:
        int(
            row[
                "cv_fold"
            ]
        )

    for row
    in fold_map.iter_rows(
        named=True
    )
}


assert (
    set(
        df[
            "demo_filename"
        ].unique()
    )
    ==
    set(
        fold_lookup.keys()
    )
)


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
    set(
        np.unique(
            fold_ids
        )
    )
    == {
        1,
        2,
        3,
        4,
        5,
    }
)


groups = (
    df[
        "demo_filename"
    ]
    .to_numpy()
)


y = (
    df[
        "label"
    ]
    .to_numpy()
)


X = (
    df
    .select(
        FEATURES
    )
    .to_numpy()
)


assert np.isfinite(X).all()


# Head 1:
# A/B plant = 1
# No plant = 0

y_plant = (
    y
    != "NO_PLANT"
).astype(int)


# Head 2:
# A plant = 1
# B plant = 0
#
# Only true-plant TRAINING rows are
# allowed into Head 2 fitting.

y_site_a = (
    y
    == "A_PLANT"
).astype(int)


print(
    "\n"
    + "=" * 100
)

print(
    "V1-H0 — HIERARCHICAL ARCHITECTURE-ONLY EVALUATION"
)

print(
    "=" * 100
)


print(
    f"\nDataset:      "
    f"{DATASET_PATH}"
)

print(
    f"Observations: "
    f"{df.height}"
)

print(
    f"Matches:      "
    f"{df['demo_filename'].n_unique()}"
)

print(
    f"Features:     "
    f"{len(FEATURES)}"
)

print(
    f"Frozen folds: "
    f"{len(np.unique(fold_ids))}"
)


print(
    "\nClass counts:"
)

for label in LABELS:

    print(
        f"  {label:10s}: "
        f"{np.sum(y == label)}"
    )


print(
    "\nHead 1 training target:"
)

print(
    f"  PLANT:    "
    f"{np.sum(y_plant == 1)}"
)

print(
    f"  NO_PLANT: "
    f"{np.sum(y_plant == 0)}"
)


print(
    "\nHead 2 population:"
)

print(
    f"  A_PLANT: "
    f"{np.sum(y == 'A_PLANT')}"
)

print(
    f"  B_PLANT: "
    f"{np.sum(y == 'B_PLANT')}"
)


# ============================================================
# Frozen-fold leakage audit
# ============================================================

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
    "\n✅ Frozen match-level leakage check passed"
)


# ============================================================
# OOF storage
# ============================================================

q_plant_all = np.full(
    df.height,
    np.nan,
    dtype=np.float64,
)

r_a_given_plant_all = np.full(
    df.height,
    np.nan,
    dtype=np.float64,
)

probabilities_all = np.full(
    (
        df.height,
        3,
    ),
    np.nan,
    dtype=np.float64,
)

fold_rows = []


# ============================================================
# Train V1-H0
# ============================================================

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


    # --------------------------------------------------------
    # Head 1
    #
    # q = P(plant | X)
    # --------------------------------------------------------

    head1 = XGBClassifier(
        **XGB_BINARY_CONFIG
    )

    assert (
        set(
            np.unique(
                y_plant[
                    train_index
                ]
            )
        )
        == {
            0,
            1,
        }
    )

    head1.fit(
        X[
            train_index
        ],
        y_plant[
            train_index
        ],
    )

    q = (
        head1
        .predict_proba(
            X[
                test_index
            ]
        )[:, 1]
        .astype(
            np.float64
        )
    )


    # --------------------------------------------------------
    # Head 2
    #
    # r = P(A | plant, X)
    #
    # Train ONLY on true-plant rows
    # from training matches.
    # --------------------------------------------------------

    head2_train_index = (
        train_index[
            y_plant[
                train_index
            ]
            == 1
        ]
    )


    assert (
        set(
            np.unique(
                y_site_a[
                    head2_train_index
                ]
            )
        )
        == {
            0,
            1,
        }
    )


    head2 = XGBClassifier(
        **XGB_BINARY_CONFIG
    )

    head2.fit(
        X[
            head2_train_index
        ],
        y_site_a[
            head2_train_index
        ],
    )

    r = (
        head2
        .predict_proba(
            X[
                test_index
            ]
        )[:, 1]
        .astype(
            np.float64
        )
    )


    # --------------------------------------------------------
    # Compose full three-class probabilities
    #
    # P(A)  = q * r
    # P(B)  = q * (1-r)
    # P(NO) = 1-q
    # --------------------------------------------------------

    probabilities = np.column_stack([
        q * r,
        q * (
            1.0
            - r
        ),
        1.0
        - q,
    ])


    assert np.isfinite(
        probabilities
    ).all()

    assert (
        probabilities.min()
        >= 0.0
    )

    assert (
        probabilities.max()
        <= 1.0
    )


    probability_sum_error = np.max(
        np.abs(
            probabilities.sum(
                axis=1
            )
            - 1.0
        )
    )


    assert (
        probability_sum_error
        < 1e-10
    ), (
        "Probability composition "
        "does not sum to one: "
        f"{probability_sum_error}"
    )


    q_plant_all[
        test_index
    ] = q

    r_a_given_plant_all[
        test_index
    ] = r

    probabilities_all[
        test_index
    ] = probabilities


    fold_result = (
        evaluate_multiclass(
            y[
                test_index
            ],
            probabilities,
        )
    )


    print(
        f"\nFold {fold}"
    )

    print(
        f"  train rows:       "
        f"{len(train_index)}"
    )

    print(
        f"  Head 2 plant rows:"
        f" {len(head2_train_index)}"
    )

    print(
        f"  test rows:        "
        f"{len(test_index)}"
    )

    print(
        f"  LL={fold_result['log_loss']:.4f}"
        f" | Brier={fold_result['brier_score']:.4f}"
        f" | Acc={fold_result['accuracy']:.4f}"
        f" | F1={fold_result['macro_f1']:.4f}"
    )


    fold_rows.append(
        metric_row(
            fold,
            train_index,
            test_index,
            groups,
            fold_result,
            len(
                head2_train_index
            ),
        )
    )


# ============================================================
# Sanity checks
# ============================================================

assert np.isfinite(
    q_plant_all
).all()

assert np.isfinite(
    r_a_given_plant_all
).all()

assert np.isfinite(
    probabilities_all
).all()


final_sum_error = np.max(
    np.abs(
        probabilities_all.sum(
            axis=1
        )
        - 1.0
    )
)


assert (
    final_sum_error
    < 1e-10
)


# ============================================================
# Full three-class evaluation
# ============================================================

v1_result = evaluate_multiclass(
    y,
    probabilities_all,
)


print_multiclass_result(
    "V1-H0 FULL THREE-CLASS OOF",
    v1_result,
)


# ============================================================
# Head-level diagnostics
#
# These are diagnostics.
# Main model selection remains the composed
# three-class output above.
# ============================================================

head1_result = evaluate_binary(
    y_plant,
    q_plant_all,
)


true_plant_mask = (
    y_plant
    == 1
)


head2_result = evaluate_binary(
    y_site_a[
        true_plant_mask
    ],
    r_a_given_plant_all[
        true_plant_mask
    ],
)


print(
    "\nHEAD 1 DIAGNOSTIC"
    " — PLANT VS NO PLANT"
)

print(
    f"  LL={head1_result['log_loss']:.6f}"
    f" | Brier={head1_result['brier_score']:.6f}"
    f" | Acc={head1_result['accuracy']:.6f}"
    f" | F1={head1_result['f1']:.6f}"
)


print(
    "\nHEAD 2 DIAGNOSTIC"
    " — A VS B | TRUE PLANT"
)

print(
    f"  n={true_plant_mask.sum()}"
    f" | LL={head2_result['log_loss']:.6f}"
    f" | Brier={head2_result['brier_score']:.6f}"
    f" | Acc={head2_result['accuracy']:.6f}"
    f" | F1={head2_result['f1']:.6f}"
)


# ============================================================
# By-horizon evaluation
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

    result = evaluate_multiclass(
        y[
            mask
        ],
        probabilities_all[
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
# Save V1 OOF predictions
# ============================================================

KEYS = [
    "demo_filename",
    "round_num",
    "horizon_sec",
    "target_tick",
    "label",
]


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
            "q_plant",
            q_plant_all,
        ),

        pl.Series(
            "r_a_given_plant",
            r_a_given_plant_all,
        ),

        pl.Series(
            "p_a_plant",
            probabilities_all[
                :, 0
            ],
        ),

        pl.Series(
            "p_b_plant",
            probabilities_all[
                :, 1
            ],
        ),

        pl.Series(
            "p_no_plant",
            probabilities_all[
                :, 2
            ],
        ),

        pl.Series(
            "prediction",
            v1_result[
                "predictions"
            ],
        ),
    ])
)


prediction_df.write_parquet(
    V1_OOF_PATH
)


# ============================================================
# Save fold + overall metrics
# ============================================================

overall_row = {
    "fold":
        0,

    "n_train_matches":
        None,

    "n_test_matches":
        20,

    "n_train_observations":
        None,

    "n_head2_train_observations":
        None,

    "n_test_observations":
        df.height,

    "log_loss":
        v1_result[
            "log_loss"
        ],

    "brier_score":
        v1_result[
            "brier_score"
        ],

    "accuracy":
        v1_result[
            "accuracy"
        ],

    "macro_f1":
        v1_result[
            "macro_f1"
        ],

    "macro_precision":
        v1_result[
            "macro_precision"
        ],

    "macro_recall":
        v1_result[
            "macro_recall"
        ],

    "recall_a":
        v1_result[
            "recall_a"
        ],

    "recall_b":
        v1_result[
            "recall_b"
        ],

    "recall_no":
        v1_result[
            "recall_no"
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


# ============================================================
# Verify and compare with frozen V0 OOF
# ============================================================

v0 = pl.read_parquet(
    V0_OOF_PATH
)


assert v0.height == df.height


v0_keys = (
    v0
    .select(
        KEYS
    )
)

current_keys = (
    df
    .select(
        KEYS
    )
)


assert (
    v0_keys
    .equals(
        current_keys
    )
), (
    "V0 OOF ordering / keys do not "
    "match corrected dataset."
)


v0_probabilities = np.column_stack([
    v0[
        "p_a_plant"
    ].to_numpy(),

    v0[
        "p_b_plant"
    ].to_numpy(),

    v0[
        "p_no_plant"
    ].to_numpy(),
]).astype(
    np.float64
)


v0_result = evaluate_multiclass(
    y,
    v0_probabilities,
)


# Frozen corrected baseline assertion.
assert abs(
    v0_result[
        "log_loss"
    ]
    - 0.838182
) < 1e-5

assert abs(
    v0_result[
        "brier_score"
    ]
    - 0.503190
) < 1e-5

assert abs(
    v0_result[
        "accuracy"
    ]
    - 0.594899
) < 1e-5

assert abs(
    v0_result[
        "macro_f1"
    ]
    - 0.566090
) < 1e-5


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

    v0_value = (
        v0_result[
            metric
        ]
    )

    v1_value = (
        v1_result[
            metric
        ]
    )

    comparison_rows.append({
        "metric":
            metric,

        "v0_xgb_a5":
            v0_value,

        "v1_h0":
            v1_value,

        "delta_v1_minus_v0":
            v1_value
            - v0_value,
    })


comparison_df = pl.DataFrame(
    comparison_rows
)


comparison_df.write_csv(
    COMPARISON_PATH
)


print(
    "\n"
    + "=" * 100
)

print(
    "V1-H0 VS CORRECTED V0 XGB-A5"
)

print(
    "=" * 100
)


for row in comparison_rows:

    metric = row[
        "metric"
    ]

    print(
        f"{metric:14s}"
        f" | V0="
        f"{row['v0_xgb_a5']:.6f}"
        f" | V1="
        f"{row['v1_h0']:.6f}"
        f" | Δ="
        f"{row['delta_v1_minus_v0']:+.6f}"
    )


print(
    "\nInterpretation:"
)

print(
    "  For Log Loss and Brier: "
    "negative Δ is better."
)

print(
    "  For Accuracy, F1, and Recall: "
    "positive Δ is better."
)


print(
    "\nSaved:"
)

print(
    f"  {V1_OOF_PATH}"
)

print(
    f"  {METRICS_PATH}"
)

print(
    f"  {COMPARISON_PATH}"
)


print(
    "\n✅ V1-H0 HIERARCHICAL "
    "OOF EVALUATION COMPLETE"
)
