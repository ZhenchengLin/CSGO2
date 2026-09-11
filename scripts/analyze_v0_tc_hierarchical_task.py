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


PREDICTION_PATH = Path(
    "data/processed/v0_tc_xgb_a5_oof_predictions.parquet"
)

OUTPUT_PATH = Path(
    "artifacts/v0_tc_xgb_a5_hierarchical_analysis.csv"
)


df = pl.read_parquet(
    PREDICTION_PATH
)


p_a = (
    df["p_a_plant"]
    .to_numpy()
    .astype(float)
)

p_b = (
    df["p_b_plant"]
    .to_numpy()
    .astype(float)
)

p_no = (
    df["p_no_plant"]
    .to_numpy()
    .astype(float)
)

labels = (
    df["label"]
    .to_numpy()
)

horizons = (
    df["horizon_sec"]
    .to_numpy()
)


# ==================================================
# TASK 1
# PLANT vs NO_PLANT
# ==================================================

y_plant = (
    labels != "NO_PLANT"
).astype(int)


p_plant = (
    p_a + p_b
)


pred_plant = (
    p_plant >= 0.5
).astype(int)


def evaluate_plant(
    mask,
):

    y = y_plant[
        mask
    ]

    p = p_plant[
        mask
    ]

    pred = pred_plant[
        mask
    ]


    return {
        "n":
            len(y),

        "log_loss":
            log_loss(
                y,
                np.column_stack([
                    1.0 - p,
                    p,
                ]),
                labels=[
                    0,
                    1,
                ],
            ),

        "brier":
            brier_score_loss(
                y,
                p,
            ),

        "accuracy":
            accuracy_score(
                y,
                pred,
            ),

        "precision":
            precision_score(
                y,
                pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y,
                pred,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y,
                pred,
                zero_division=0,
            ),
    }


print("\n" + "=" * 100)
print("TASK 1 — PLANT vs NO_PLANT")
print("=" * 100)


all_mask = np.ones(
    len(df),
    dtype=bool,
)


plant_overall = evaluate_plant(
    all_mask
)


for key, value in (
    plant_overall.items()
):

    if key == "n":
        print(
            f"{key:10s}: {value}"
        )

    else:
        print(
            f"{key:10s}: {value:.4f}"
        )


# ==================================================
# TASK 2
# A vs B GIVEN TRUE PLANT
# ==================================================

plant_mask = (
    labels != "NO_PLANT"
)


site_labels = labels[
    plant_mask
]


site_p_a_raw = p_a[
    plant_mask
]

site_p_b_raw = p_b[
    plant_mask
]


plant_mass = (
    site_p_a_raw
    + site_p_b_raw
)


assert np.all(
    plant_mass > 0
)


p_a_given_plant = (
    site_p_a_raw
    / plant_mass
)

p_b_given_plant = (
    site_p_b_raw
    / plant_mass
)


site_probabilities = np.column_stack([
    p_a_given_plant,
    p_b_given_plant,
])


site_true = np.array([
    0 if label == "A_PLANT"
    else 1
    for label in site_labels
])


site_pred = np.argmax(
    site_probabilities,
    axis=1,
)


site_accuracy = accuracy_score(
    site_true,
    site_pred,
)


site_log_loss = log_loss(
    site_true,
    site_probabilities,
    labels=[
        0,
        1,
    ],
)


site_brier = np.mean(
    (
        p_b_given_plant
        - site_true
    ) ** 2
)


print("\n" + "=" * 100)
print("TASK 2 — A vs B | TRUE PLANT")
print("=" * 100)


print(
    f"n         : "
    f"{len(site_true)}"
)

print(
    f"Log Loss  : "
    f"{site_log_loss:.4f}"
)

print(
    f"Brier     : "
    f"{site_brier:.4f}"
)

print(
    f"Accuracy  : "
    f"{site_accuracy:.4f}"
)


# ==================================================
# Confusion-derived diagnostic
#
# Among TRUE plant observations where the model also
# predicts A or B, how often is the site correct?
# ==================================================

predicted_labels = (
    df["prediction"]
    .to_numpy()
)


true_plant_and_pred_plant = (
    (labels != "NO_PLANT")
    &
    (predicted_labels != "NO_PLANT")
)


n_true_and_pred_plant = int(
    true_plant_and_pred_plant.sum()
)


site_correct_when_pred_plant = int(
    (
        labels[
            true_plant_and_pred_plant
        ]
        ==
        predicted_labels[
            true_plant_and_pred_plant
        ]
    ).sum()
)


conditional_site_accuracy = (
    site_correct_when_pred_plant
    / n_true_and_pred_plant
)


print("\n" + "=" * 100)
print("SITE ACCURACY WHEN TRUE=PLANT AND PREDICTION=PLANT")
print("=" * 100)


print(
    f"Observations: "
    f"{n_true_and_pred_plant}"
)

print(
    f"Correct site: "
    f"{site_correct_when_pred_plant}"
)

print(
    f"Accuracy:     "
    f"{conditional_site_accuracy:.4f}"
)


# ==================================================
# By horizon
# ==================================================

print("\n" + "=" * 100)
print("HIERARCHICAL PERFORMANCE BY HORIZON")
print("=" * 100)


print(
    f"{'H':>4}"
    f"{'N':>6}"
    f"{'PLANT LL':>12}"
    f"{'PLANT ACC':>12}"
    f"{'PLANT F1':>11}"
    f"{'SITE N':>9}"
    f"{'SITE LL':>11}"
    f"{'SITE ACC':>11}"
)


rows = []


for horizon in [
    10,
    20,
    30,
    40,
]:

    mask = (
        horizons == horizon
    )


    plant_result = (
        evaluate_plant(
            mask
        )
    )


    # ----------------------------------------------
    # True planted observations at this horizon
    # ----------------------------------------------

    site_mask = (
        mask
        &
        (
            labels
            != "NO_PLANT"
        )
    )


    y_site_labels = labels[
        site_mask
    ]


    p_a_h = p_a[
        site_mask
    ]

    p_b_h = p_b[
        site_mask
    ]


    mass_h = (
        p_a_h
        + p_b_h
    )


    p_a_cond = (
        p_a_h
        / mass_h
    )

    p_b_cond = (
        p_b_h
        / mass_h
    )


    probs_h = np.column_stack([
        p_a_cond,
        p_b_cond,
    ])


    y_site_h = np.array([
        0 if label == "A_PLANT"
        else 1
        for label in y_site_labels
    ])


    pred_site_h = np.argmax(
        probs_h,
        axis=1,
    )


    site_ll_h = log_loss(
        y_site_h,
        probs_h,
        labels=[
            0,
            1,
        ],
    )


    site_acc_h = accuracy_score(
        y_site_h,
        pred_site_h,
    )


    print(
        f"{horizon:>4}"
        f"{plant_result['n']:>6}"
        f"{plant_result['log_loss']:>12.4f}"
        f"{plant_result['accuracy']:>12.4f}"
        f"{plant_result['f1']:>11.4f}"
        f"{len(y_site_h):>9}"
        f"{site_ll_h:>11.4f}"
        f"{site_acc_h:>11.4f}"
    )


    rows.append({
        "horizon_sec":
            horizon,

        "n":
            plant_result[
                "n"
            ],

        "plant_log_loss":
            plant_result[
                "log_loss"
            ],

        "plant_brier":
            plant_result[
                "brier"
            ],

        "plant_accuracy":
            plant_result[
                "accuracy"
            ],

        "plant_precision":
            plant_result[
                "precision"
            ],

        "plant_recall":
            plant_result[
                "recall"
            ],

        "plant_f1":
            plant_result[
                "f1"
            ],

        "site_n":
            len(
                y_site_h
            ),

        "site_log_loss":
            site_ll_h,

        "site_accuracy":
            site_acc_h,
    })


# ==================================================
# Error composition
# ==================================================

true = labels
pred = predicted_labels


plant_no_errors = (
    (
        (true != "NO_PLANT")
        &
        (pred == "NO_PLANT")
    )
    |
    (
        (true == "NO_PLANT")
        &
        (pred != "NO_PLANT")
    )
)


site_swap_errors = (
    (
        (true == "A_PLANT")
        &
        (pred == "B_PLANT")
    )
    |
    (
        (true == "B_PLANT")
        &
        (pred == "A_PLANT")
    )
)


total_errors = (
    true != pred
)


n_errors = int(
    total_errors.sum()
)

n_plant_no = int(
    plant_no_errors.sum()
)

n_site_swap = int(
    site_swap_errors.sum()
)


print("\n" + "=" * 100)
print("ERROR COMPOSITION")
print("=" * 100)


print(
    f"Total errors:           "
    f"{n_errors}"
)

print(
    f"Plant ↔ No Plant:       "
    f"{n_plant_no}"
    f" "
    f"({n_plant_no / n_errors:.1%})"
)

print(
    f"A ↔ B site swaps:       "
    f"{n_site_swap}"
    f" "
    f"({n_site_swap / n_errors:.1%})"
)


# ==================================================
# Save
# ==================================================

summary = pl.DataFrame(
    rows
)


summary.write_csv(
    OUTPUT_PATH
)


print("\nSaved:")
print(
    OUTPUT_PATH
)

print(
    "\n✅ HIERARCHICAL TASK ANALYSIS COMPLETE"
)
