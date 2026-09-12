from pathlib import Path

import numpy as np
import polars as pl

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from xgboost import XGBClassifier


# ============================================================
# Paths
# ============================================================

DATASET_PATH = Path(
    "data/processed/v0_dataset_v2_timing_corrected.parquet"
)

FOLD_MAP_PATH = Path(
    "data/interim/v0_timing_corrected_fold_map.csv"
)

V0_OOF_PATH = Path(
    "data/processed/v0_tc_xgb_a5_oof_predictions.parquet"
)

H1_OOF_PATH = Path(
    "data/processed/v1_h1_oof_predictions.parquet"
)

H2_OOF_PATH = Path(
    "data/processed/v1_h2_oof_predictions.parquet"
)

METRICS_PATH = Path(
    "artifacts/v1_h2_metrics.csv"
)

COMPARISON_PATH = Path(
    "artifacts/v1_h2_vs_h1_v0_comparison.csv"
)


# ============================================================
# Labels
# ============================================================

LABELS = [
    "A_PLANT",
    "B_PLANT",
    "NO_PLANT",
]


# ============================================================
# H1 Site Head feature contract
#
# 12 offensive geometry
# + 4 motion
# + 11 defense
# = 27
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


OFFENSIVE_GEOMETRY = A1[1:]


SITE_FEATURES = (
    OFFENSIVE_GEOMETRY
    + A2
    + A5
)


assert len(SITE_FEATURES) == 27
assert len(set(SITE_FEATURES)) == 27


# ============================================================
# Frozen XGBoost configuration
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


INNER_SPLITS = 4


# ============================================================
# Helpers
# ============================================================

def multiclass_brier(
    y_true,
    probabilities,
):

    truth = np.zeros(
        (
            len(y_true),
            3,
        ),
        dtype=float,
    )

    mapping = {
        label: i
        for i, label
        in enumerate(LABELS)
    }

    for i, label in enumerate(y_true):
        truth[
            i,
            mapping[label],
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

    ids = np.argmax(
        probabilities,
        axis=1,
    )

    predictions = np.array([
        LABELS[i]
        for i in ids
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
            multiclass_brier(
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


def evaluate_binary(
    y_true,
    p_positive,
):

    pred = (
        p_positive
        >= 0.5
    ).astype(int)

    return {
        "log_loss":
            log_loss(
                y_true,
                np.column_stack([
                    1.0 - p_positive,
                    p_positive,
                ]),
                labels=[0, 1],
            ),

        "brier_score":
            brier_score_loss(
                y_true,
                p_positive,
            ),

        "accuracy":
            accuracy_score(
                y_true,
                pred,
            ),

        "f1":
            f1_score(
                y_true,
                pred,
                zero_division=0,
            ),
    }


def logit_feature(
    probabilities,
):

    p = np.clip(
        probabilities,
        1e-6,
        1.0 - 1e-6,
    )

    return np.log(
        p
        / (
            1.0
            - p
        )
    ).reshape(
        -1,
        1,
    )


def probabilities_from_frame(
    frame,
):

    return np.column_stack([
        frame[
            "p_a_plant"
        ].to_numpy(),

        frame[
            "p_b_plant"
        ].to_numpy(),

        frame[
            "p_no_plant"
        ].to_numpy(),
    ]).astype(
        np.float64
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

h1 = pl.read_parquet(
    H1_OOF_PATH
)

v0 = pl.read_parquet(
    V0_OOF_PATH
)


assert df.height == 1686
assert h1.height == 1686
assert v0.height == 1686


KEYS = [
    "demo_filename",
    "round_num",
    "horizon_sec",
    "target_tick",
    "label",
]


assert (
    df
    .select(KEYS)
    .equals(
        h1.select(KEYS)
    )
)

assert (
    df
    .select(KEYS)
    .equals(
        v0.select(KEYS)
    )
)


fold_lookup = {
    row["demo_filename"]:
        int(row["cv_fold"])

    for row
    in fold_map.iter_rows(
        named=True
    )
}


fold_ids = np.array([
    fold_lookup[x]

    for x
    in df[
        "demo_filename"
    ].to_list()
])


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


X_site = (
    df
    .select(
        SITE_FEATURES
    )
    .to_numpy()
)


assert np.isfinite(
    X_site
).all()


# ============================================================
# Freeze H1 Plant Head
#
# H2 does NOT retrain or change q.
# ============================================================

q_plant = (
    h1[
        "q_plant"
    ]
    .to_numpy()
    .astype(
        np.float64
    )
)


h1_raw_r = (
    h1[
        "r_a_given_plant"
    ]
    .to_numpy()
    .astype(
        np.float64
    )
)


# ============================================================
# Site labels
# ============================================================

is_plant = (
    y
    != "NO_PLANT"
)


y_site_a = (
    y
    == "A_PLANT"
).astype(int)


# ============================================================
# Storage
# ============================================================

raw_r_all = np.full(
    df.height,
    np.nan,
    dtype=np.float64,
)

calibrated_r_all = np.full(
    df.height,
    np.nan,
    dtype=np.float64,
)

h2_probabilities = np.full(
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
    "V1-H2 — NESTED SITE-HEAD PLATT CALIBRATION"
)

print(
    "=" * 100
)

print(
    f"\nObservations:   {df.height}"
)

print(
    f"Matches:        "
    f"{df['demo_filename'].n_unique()}"
)

print(
    f"Site features:  "
    f"{len(SITE_FEATURES)}"
)

print(
    f"Outer folds:    "
    f"{len(np.unique(fold_ids))}"
)

print(
    f"Inner folds:    "
    f"{INNER_SPLITS}"
)

print(
    "\nPlant Head: frozen from V1-H1"
)

print(
    "Site Head: same H1 27-feature XGBoost"
)

print(
    "Only change: nested Platt calibration"
)


# ============================================================
# Outer CV
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


    # Site Head only trains on
    # true-plant rows from outer training matches.

    site_train_index = (
        train_index[
            is_plant[
                train_index
            ]
        ]
    )


    X_train_site = (
        X_site[
            site_train_index
        ]
    )

    y_train_site = (
        y_site_a[
            site_train_index
        ]
    )

    groups_train_site = (
        groups[
            site_train_index
        ]
    )


    assert set(
        np.unique(
            y_train_site
        )
    ) == {
        0,
        1,
    }


    # ========================================================
    # Inner grouped OOF predictions
    #
    # These are used ONLY to train the calibrator.
    # ========================================================

    inner_splitter = (
        StratifiedGroupKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=42,
        )
    )


    inner_oof = np.full(
        len(
            site_train_index
        ),
        np.nan,
        dtype=np.float64,
    )


    for (
        inner_train,
        inner_valid,
    ) in inner_splitter.split(
        X_train_site,
        y_train_site,
        groups_train_site,
    ):

        inner_train_groups = set(
            groups_train_site[
                inner_train
            ]
        )

        inner_valid_groups = set(
            groups_train_site[
                inner_valid
            ]
        )

        assert not (
            inner_train_groups
            & inner_valid_groups
        )


        inner_model = XGBClassifier(
            **XGB_BINARY_CONFIG
        )

        inner_model.fit(
            X_train_site[
                inner_train
            ],
            y_train_site[
                inner_train
            ],
        )


        inner_oof[
            inner_valid
        ] = (
            inner_model
            .predict_proba(
                X_train_site[
                    inner_valid
                ]
            )[:, 1]
            .astype(
                np.float64
            )
        )


    assert np.isfinite(
        inner_oof
    ).all()


    # ========================================================
    # Platt calibrator
    #
    # logistic(
    #     a * logit(raw_probability)
    #     + b
    # )
    # ========================================================

    calibrator = (
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        )
    )


    calibrator.fit(
        logit_feature(
            inner_oof
        ),
        y_train_site,
    )


    calibration_slope = float(
        calibrator.coef_[
            0,
            0,
        ]
    )

    calibration_intercept = float(
        calibrator.intercept_[
            0
        ]
    )


    # ========================================================
    # Final Site Head for this outer fold
    #
    # Same model as H1.
    # ========================================================

    site_model = XGBClassifier(
        **XGB_BINARY_CONFIG
    )


    site_model.fit(
        X_train_site,
        y_train_site,
    )


    raw_r = (
        site_model
        .predict_proba(
            X_site[
                test_index
            ]
        )[:, 1]
        .astype(
            np.float64
        )
    )


    calibrated_r = (
        calibrator
        .predict_proba(
            logit_feature(
                raw_r
            )
        )[:, 1]
        .astype(
            np.float64
        )
    )


    raw_r_all[
        test_index
    ] = raw_r

    calibrated_r_all[
        test_index
    ] = calibrated_r


    # ========================================================
    # Compose full 3-class probability
    #
    # Plant Head q is exactly H1.
    # Only r has changed.
    # ========================================================

    q = (
        q_plant[
            test_index
        ]
    )


    probabilities = np.column_stack([
        q
        * calibrated_r,

        q
        * (
            1.0
            - calibrated_r
        ),

        1.0
        - q,
    ])


    assert np.allclose(
        probabilities.sum(
            axis=1
        ),
        1.0,
        atol=1e-12,
    )


    h2_probabilities[
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


    fold_rows.append({
        "fold":
            int(fold),

        "n_train_observations":
            len(train_index),

        "n_site_train_observations":
            len(site_train_index),

        "n_test_observations":
            len(test_index),

        "calibration_slope":
            calibration_slope,

        "calibration_intercept":
            calibration_intercept,

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


    print(
        f"\nFold {fold}"
        f" | Site train={len(site_train_index)}"
        f" | Test={len(test_index)}"
    )

    print(
        f"  calibration slope="
        f"{calibration_slope:.4f}"
        f" | intercept="
        f"{calibration_intercept:.4f}"
    )

    print(
        f"  LL={fold_result['log_loss']:.4f}"
        f" | Brier={fold_result['brier_score']:.4f}"
        f" | Acc={fold_result['accuracy']:.4f}"
        f" | F1={fold_result['macro_f1']:.4f}"
    )


# ============================================================
# Important isolation check
#
# Raw H2 Site Head MUST reproduce
# H1 Site Head before calibration.
# ============================================================

max_raw_site_difference = float(
    np.max(
        np.abs(
            raw_r_all
            - h1_raw_r
        )
    )
)


print(
    "\nRaw H1/H2 Site Head max difference:"
)

print(
    f"  {max_raw_site_difference:.12g}"
)


assert (
    max_raw_site_difference
    < 1e-7
), (
    "H2 raw Site Head no longer "
    "matches H1. More than calibration "
    "changed."
)


print(
    "✅ H2 isolation check passed"
)


# ============================================================
# Site Head diagnostic
# ============================================================

plant_mask = (
    is_plant
)


raw_site_result = (
    evaluate_binary(
        y_site_a[
            plant_mask
        ],
        raw_r_all[
            plant_mask
        ],
    )
)


calibrated_site_result = (
    evaluate_binary(
        y_site_a[
            plant_mask
        ],
        calibrated_r_all[
            plant_mask
        ],
    )
)


print(
    "\n"
    + "-" * 100
)

print(
    "SITE HEAD — BEFORE VS AFTER CALIBRATION"
)

print(
    "-" * 100
)


print(
    "RAW H1 SITE"
)

print(
    f"  LL="
    f"{raw_site_result['log_loss']:.6f}"
    f" | Brier="
    f"{raw_site_result['brier_score']:.6f}"
    f" | Acc="
    f"{raw_site_result['accuracy']:.6f}"
    f" | F1="
    f"{raw_site_result['f1']:.6f}"
)


print(
    "CALIBRATED H2 SITE"
)

print(
    f"  LL="
    f"{calibrated_site_result['log_loss']:.6f}"
    f" | Brier="
    f"{calibrated_site_result['brier_score']:.6f}"
    f" | Acc="
    f"{calibrated_site_result['accuracy']:.6f}"
    f" | F1="
    f"{calibrated_site_result['f1']:.6f}"
)


# ============================================================
# Full H2
# ============================================================

h2_result = evaluate_multiclass(
    y,
    h2_probabilities,
)


print(
    "\n"
    + "-" * 100
)

print(
    "V1-H2 FULL THREE-CLASS OOF"
)

print(
    "-" * 100
)


for name in [
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
        f"{name:16s}: "
        f"{h2_result[name]:.6f}"
    )


# ============================================================
# Horizon
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

    result = (
        evaluate_multiclass(
            y[
                mask
            ],
            h2_probabilities[
                mask
            ],
        )
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
# Compare V0 / H1 / H2
# ============================================================

v0_result = evaluate_multiclass(
    y,
    probabilities_from_frame(
        v0
    ),
)


h1_result = evaluate_multiclass(
    y,
    probabilities_from_frame(
        h1
    ),
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

        "v1_h1":
            h1_result[
                metric
            ],

        "v1_h2":
            h2_result[
                metric
            ],

        "delta_h2_minus_h1":
            h2_result[
                metric
            ]
            - h1_result[
                metric
            ],

        "delta_h2_minus_v0":
            h2_result[
                metric
            ]
            - v0_result[
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
    "V0 VS H1 VS H2"
)

print(
    "=" * 100
)


for row in comparison_rows:

    print(
        f"{row['metric']:14s}"
        f" | V0="
        f"{row['v0_xgb_a5']:.6f}"
        f" | H1="
        f"{row['v1_h1']:.6f}"
        f" | H2="
        f"{row['v1_h2']:.6f}"
        f" | H2-H1="
        f"{row['delta_h2_minus_h1']:+.6f}"
        f" | H2-V0="
        f"{row['delta_h2_minus_v0']:+.6f}"
    )


# ============================================================
# Save H2 OOF
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
            "q_plant",
            q_plant,
        ),

        pl.Series(
            "r_a_given_plant_raw",
            raw_r_all,
        ),

        pl.Series(
            "r_a_given_plant",
            calibrated_r_all,
        ),

        pl.Series(
            "p_a_plant",
            h2_probabilities[
                :, 0
            ],
        ),

        pl.Series(
            "p_b_plant",
            h2_probabilities[
                :, 1
            ],
        ),

        pl.Series(
            "p_no_plant",
            h2_probabilities[
                :, 2
            ],
        ),

        pl.Series(
            "prediction",
            h2_result[
                "predictions"
            ],
        ),
    ])
)


prediction_df.write_parquet(
    H2_OOF_PATH
)


overall_row = {
    "fold":
        0,

    "n_train_observations":
        None,

    "n_site_train_observations":
        None,

    "n_test_observations":
        df.height,

    "calibration_slope":
        None,

    "calibration_intercept":
        None,

    "log_loss":
        h2_result[
            "log_loss"
        ],

    "brier_score":
        h2_result[
            "brier_score"
        ],

    "accuracy":
        h2_result[
            "accuracy"
        ],

    "macro_f1":
        h2_result[
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
    f"  {H2_OOF_PATH}"
)

print(
    f"  {METRICS_PATH}"
)

print(
    f"  {COMPARISON_PATH}"
)


print(
    "\n✅ V1-H2 NESTED SITE CALIBRATION COMPLETE"
)
