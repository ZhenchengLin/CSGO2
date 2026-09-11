from pathlib import Path

import numpy as np
import polars as pl


PREDICTION_PATH = Path(
    "data/processed/v0_tc_xgb_a5_oof_predictions.parquet"
)

SUMMARY_PATH = Path(
    "artifacts/v0_tc_xgb_a5_calibration_summary.csv"
)

BIN_PATH = Path(
    "artifacts/v0_tc_xgb_a5_calibration_bins.csv"
)


LABELS = [
    "A_PLANT",
    "B_PLANT",
    "NO_PLANT",
]


PROBABILITY_COLUMNS = [
    "p_a_plant",
    "p_b_plant",
    "p_no_plant",
]


# ==================================================
# Calibration helper
# ==================================================

def calibration_bins(
    probabilities,
    outcomes,
    n_bins,
):
    """
    Equal-width calibration bins.

    probabilities:
        predicted probability in [0, 1]

    outcomes:
        binary outcome:
        1 = event happened
        0 = event did not happen
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    outcomes = np.asarray(
        outcomes,
        dtype=float,
    )

    # Map:
    # [0.0, 0.1) -> bin 0
    # ...
    # [0.9, 1.0] -> final bin
    bin_ids = np.minimum(
        (
            probabilities
            * n_bins
        ).astype(int),
        n_bins - 1,
    )

    rows = []

    total = len(
        probabilities
    )

    ece = 0.0
    max_gap = 0.0


    for bin_id in range(
        n_bins
    ):

        mask = (
            bin_ids == bin_id
        )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue


        mean_probability = float(
            probabilities[
                mask
            ].mean()
        )

        observed_rate = float(
            outcomes[
                mask
            ].mean()
        )

        gap = abs(
            mean_probability
            - observed_rate
        )


        weight = (
            count
            / total
        )


        ece += (
            weight
            * gap
        )

        max_gap = max(
            max_gap,
            gap,
        )


        rows.append({
            "bin":
                bin_id + 1,

            "lower":
                bin_id / n_bins,

            "upper":
                (bin_id + 1)
                / n_bins,

            "n":
                count,

            "mean_probability":
                mean_probability,

            "observed_rate":
                observed_rate,

            "absolute_gap":
                gap,
        })


    return (
        float(ece),
        float(max_gap),
        rows,
    )


# ==================================================
# Load
# ==================================================

df = pl.read_parquet(
    PREDICTION_PATH
)


probabilities = (
    df
    .select(
        PROBABILITY_COLUMNS
    )
    .to_numpy()
    .astype(
        np.float64
    )
)


y = (
    df[
        "label"
    ]
    .to_numpy()
)


# ==================================================
# Probability integrity
# ==================================================

probability_sums = (
    probabilities.sum(
        axis=1
    )
)


max_probability_error = (
    np.abs(
        probability_sums
        - 1.0
    )
    .max()
)


assert (
    max_probability_error
    < 1e-10
)


print("\n" + "=" * 105)
print("V0 — XGB_A5 CALIBRATION ANALYSIS")
print("=" * 105)

print(
    f"Observations: "
    f"{df.height}"
)

print(
    f"Matches:      "
    f"{df['demo_filename'].n_unique()}"
)

print(
    f"Max probability-sum error: "
    f"{max_probability_error:.3e}"
)


summary_rows = []
all_bin_rows = []


# ==================================================
# 1. Top-label confidence calibration
# ==================================================

predicted_index = np.argmax(
    probabilities,
    axis=1,
)


confidence = np.max(
    probabilities,
    axis=1,
)


predicted_labels = np.array([
    LABELS[index]
    for index
    in predicted_index
])


correct = (
    predicted_labels
    == y
).astype(float)


top_ece, top_mce, top_bins = (
    calibration_bins(
        confidence,
        correct,
        n_bins=10,
    )
)


print("\n" + "=" * 105)
print("TOP-LABEL CONFIDENCE CALIBRATION")
print("=" * 105)


print(
    f"ECE: "
    f"{top_ece:.4f}"
)

print(
    f"Maximum bin gap: "
    f"{top_mce:.4f}"
)


print(
    "\n"
    f"{'BIN':<10}"
    f"{'N':>6}"
    f"{'MEAN CONF':>14}"
    f"{'ACCURACY':>12}"
    f"{'GAP':>10}"
)


for row in top_bins:

    print(
        f"{row['lower']:.1f}-"
        f"{row['upper']:.1f}"
        f"{row['n']:>8}"
        f"{row['mean_probability']:>14.3f}"
        f"{row['observed_rate']:>12.3f}"
        f"{row['absolute_gap']:>10.3f}"
    )

    all_bin_rows.append({
        "scope":
            "overall",

        "target":
            "TOP_LABEL",

        **row,
    })


summary_rows.append({
    "scope":
        "overall",

    "target":
        "TOP_LABEL",

    "n":
        df.height,

    "ece":
        top_ece,

    "max_bin_gap":
        top_mce,
})


# ==================================================
# 2. Per-class calibration
# ==================================================

print("\n" + "=" * 105)
print("PER-CLASS CALIBRATION")
print("=" * 105)


for class_index, label in enumerate(
    LABELS
):

    class_probability = (
        probabilities[
            :,
            class_index
        ]
    )

    class_truth = (
        y == label
    ).astype(float)


    ece, mce, bins = (
        calibration_bins(
            class_probability,
            class_truth,
            n_bins=10,
        )
    )


    print(
        f"{label:10s}"
        f" | ECE={ece:.4f}"
        f" | max gap={mce:.4f}"
    )


    summary_rows.append({
        "scope":
            "overall",

        "target":
            label,

        "n":
            df.height,

        "ece":
            ece,

        "max_bin_gap":
            mce,
    })


    for row in bins:

        all_bin_rows.append({
            "scope":
                "overall",

            "target":
                label,

            **row,
        })


# ==================================================
# 3. Calibration by horizon
#
# Use 5 bins because later horizons have fewer
# observations, especially 40 sec.
# ==================================================

print("\n" + "=" * 105)
print("CALIBRATION BY HORIZON")
print("=" * 105)


horizons = (
    df[
        "horizon_sec"
    ]
    .to_numpy()
)


print(
    f"\n"
    f"{'HORIZON':<10}"
    f"{'N':>6}"
    f"{'TOP ECE':>12}"
    f"{'A ECE':>10}"
    f"{'B ECE':>10}"
    f"{'NO ECE':>10}"
)


for horizon in [
    10,
    20,
    30,
    40,
]:

    mask = (
        horizons == horizon
    )

    p_h = probabilities[
        mask
    ]

    y_h = y[
        mask
    ]


    # ----------------------------------------------
    # Top-label ECE
    # ----------------------------------------------

    pred_index_h = np.argmax(
        p_h,
        axis=1,
    )

    pred_labels_h = np.array([
        LABELS[index]
        for index
        in pred_index_h
    ])

    confidence_h = np.max(
        p_h,
        axis=1,
    )

    correct_h = (
        pred_labels_h
        == y_h
    ).astype(float)


    top_ece_h, top_mce_h, bins_h = (
        calibration_bins(
            confidence_h,
            correct_h,
            n_bins=5,
        )
    )


    class_eces = {}


    for class_index, label in enumerate(
        LABELS
    ):

        truth_h = (
            y_h == label
        ).astype(float)

        ece_h, mce_h, _ = (
            calibration_bins(
                p_h[
                    :,
                    class_index
                ],
                truth_h,
                n_bins=5,
            )
        )

        class_eces[
            label
        ] = ece_h


    print(
        f"{horizon:<10}"
        f"{mask.sum():>6}"
        f"{top_ece_h:>12.4f}"
        f"{class_eces['A_PLANT']:>10.4f}"
        f"{class_eces['B_PLANT']:>10.4f}"
        f"{class_eces['NO_PLANT']:>10.4f}"
    )


    summary_rows.append({
        "scope":
            f"{horizon}s",

        "target":
            "TOP_LABEL",

        "n":
            int(
                mask.sum()
            ),

        "ece":
            top_ece_h,

        "max_bin_gap":
            top_mce_h,
    })


    for class_label in LABELS:

        summary_rows.append({
            "scope":
                f"{horizon}s",

            "target":
                class_label,

            "n":
                int(
                    mask.sum()
                ),

            "ece":
                class_eces[
                    class_label
                ],

            "max_bin_gap":
                None,
        })


# ==================================================
# 4. Confidence distribution
# ==================================================

print("\n" + "=" * 105)
print("CONFIDENCE DISTRIBUTION")
print("=" * 105)


for threshold in [
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]:

    mask = (
        confidence
        >= threshold
    )

    count = int(
        mask.sum()
    )


    if count == 0:

        accuracy = float("nan")

    else:

        accuracy = float(
            correct[
                mask
            ].mean()
        )


    print(
        f"P(max) >= "
        f"{threshold:.2f}"
        f" | n={count:4d}"
        f" | observed accuracy="
        f"{accuracy:.3f}"
    )


# ==================================================
# Save artifacts
# ==================================================

summary = pl.DataFrame(
    summary_rows
)

bins = pl.DataFrame(
    all_bin_rows
)


SUMMARY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


summary.write_csv(
    SUMMARY_PATH
)

bins.write_csv(
    BIN_PATH
)


print("\nSaved calibration summary:")
print(
    SUMMARY_PATH
)

print("\nSaved calibration bins:")
print(
    BIN_PATH
)

print(
    "\n✅ XGB_A5 CALIBRATION ANALYSIS COMPLETE"
)
