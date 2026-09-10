from pathlib import Path

import numpy as np
import polars as pl

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)


PREDICTION_PATH = Path(
    "data/processed/v0_xgb_a5_oof_predictions.parquet"
)

CONFUSION_PATH = Path(
    "artifacts/v0_xgb_a5_confusion_matrix.csv"
)

ERROR_PATH = Path(
    "data/interim/v0_xgb_a5_high_confidence_errors.csv"
)


LABELS = [
    "A_PLANT",
    "B_PLANT",
    "NO_PLANT",
]


df = pl.read_parquet(
    PREDICTION_PATH
)


y_true = df["label"].to_numpy()
y_pred = df["prediction"].to_numpy()


# ==================================================
# 1. Confusion matrix
# ==================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=LABELS,
)


cm_normalized = confusion_matrix(
    y_true,
    y_pred,
    labels=LABELS,
    normalize="true",
)


print("\n" + "=" * 100)
print("XGB_A5 CONFUSION MATRIX — COUNTS")
print("=" * 100)

print("Rows = TRUE")
print("Columns = PREDICTED\n")

print(
    f"{'':12s}"
    f"{'A_PLANT':>12}"
    f"{'B_PLANT':>12}"
    f"{'NO_PLANT':>12}"
)


for label, row in zip(
    LABELS,
    cm,
):

    print(
        f"{label:12s}"
        f"{row[0]:>12}"
        f"{row[1]:>12}"
        f"{row[2]:>12}"
    )


print("\n" + "=" * 100)
print("XGB_A5 CONFUSION MATRIX — TRUE-CLASS NORMALIZED")
print("=" * 100)

print(
    f"{'':12s}"
    f"{'A_PLANT':>12}"
    f"{'B_PLANT':>12}"
    f"{'NO_PLANT':>12}"
)


for label, row in zip(
    LABELS,
    cm_normalized,
):

    print(
        f"{label:12s}"
        f"{row[0]:>12.3f}"
        f"{row[1]:>12.3f}"
        f"{row[2]:>12.3f}"
    )


# ==================================================
# 2. Per-class metrics
# ==================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )
)


print("\n" + "=" * 100)
print("PER-CLASS ERROR PROFILE")
print("=" * 100)


for i, label in enumerate(
    LABELS
):

    print(
        f"{label:10s}"
        f" | support={support[i]:4d}"
        f" | precision={precision[i]:.3f}"
        f" | recall={recall[i]:.3f}"
        f" | F1={f1[i]:.3f}"
    )


# ==================================================
# 3. Accuracy / confidence by horizon
# ==================================================

probabilities = (
    df
    .select([
        "p_a_plant",
        "p_b_plant",
        "p_no_plant",
    ])
    .to_numpy()
)


confidence = (
    probabilities.max(
        axis=1
    )
)


correct = (
    y_true == y_pred
)


horizons = (
    df[
        "horizon_sec"
    ]
    .to_numpy()
)


print("\n" + "=" * 100)
print("ERROR PROFILE BY HORIZON")
print("=" * 100)


for horizon in [
    10,
    20,
    30,
    40,
]:

    mask = (
        horizons == horizon
    )

    accuracy = (
        correct[
            mask
        ].mean()
    )

    mean_confidence = (
        confidence[
            mask
        ].mean()
    )

    error_confidence = (
        confidence[
            mask
            & (~correct)
        ]
    )


    if len(
        error_confidence
    ) > 0:

        mean_error_confidence = (
            error_confidence.mean()
        )

    else:

        mean_error_confidence = (
            float("nan")
        )


    print(
        f"{horizon:>2}s"
        f" | n={mask.sum():3d}"
        f" | accuracy={accuracy:.3f}"
        f" | mean confidence={mean_confidence:.3f}"
        f" | error confidence={mean_error_confidence:.3f}"
    )


# ==================================================
# 4. High-confidence errors
# ==================================================

analysis = (
    df
    .with_columns([
        pl.Series(
            "confidence",
            confidence,
        ),

        pl.Series(
            "correct",
            correct,
        ),
    ])
)


high_confidence_errors = (
    analysis
    .filter(
        ~pl.col("correct")
    )
    .sort(
        "confidence",
        descending=True,
    )
)


print("\n" + "=" * 100)
print("TOP 20 HIGHEST-CONFIDENCE ERRORS")
print("=" * 100)


print(
    high_confidence_errors
    .select([
        "demo_filename",
        "round_num",
        "horizon_sec",

        "label",
        "prediction",

        "confidence",

        "p_a_plant",
        "p_b_plant",
        "p_no_plant",
    ])
    .head(20)
)


# ==================================================
# 5. Error pairs
# ==================================================

print("\n" + "=" * 100)
print("MOST COMMON ERROR TYPES")
print("=" * 100)


print(
    analysis
    .filter(
        ~pl.col("correct")
    )
    .group_by([
        "label",
        "prediction",
    ])
    .len()
    .sort(
        "len",
        descending=True,
    )
)


# ==================================================
# Save
# ==================================================

confusion_df = pl.DataFrame({

    "true_label":
        LABELS,

    "pred_A_PLANT":
        cm[:, 0],

    "pred_B_PLANT":
        cm[:, 1],

    "pred_NO_PLANT":
        cm[:, 2],
})


confusion_df.write_csv(
    CONFUSION_PATH
)


high_confidence_errors.write_csv(
    ERROR_PATH
)


print("\nSaved confusion matrix:")
print(CONFUSION_PATH)

print("\nSaved error cases:")
print(ERROR_PATH)

print(
    "\n✅ XGB_A5 ERROR ANALYSIS COMPLETE"
)
