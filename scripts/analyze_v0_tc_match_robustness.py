from pathlib import Path

import numpy as np
import polars as pl

from sklearn.metrics import log_loss


LR_PATH = Path(
    "data/processed/v0_tc_a3_oof_predictions.parquet"
)

XGB_PATH = Path(
    "data/processed/v0_tc_xgb_a5_oof_predictions.parquet"
)

OUTPUT_PATH = Path(
    "artifacts/v0_tc_match_robustness.csv"
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


PROB_COLS = [
    "p_a_plant",
    "p_b_plant",
    "p_no_plant",
]


# ==================================================
# Load
# ==================================================

lr = pl.read_parquet(
    LR_PATH
)

xgb = pl.read_parquet(
    XGB_PATH
)


KEYS = [
    "demo_filename",
    "round_num",
    "horizon_sec",
    "target_tick",
    "label",
]


assert (
    lr.select(KEYS).to_dicts()
    ==
    xgb.select(KEYS).to_dicts()
)


# ==================================================
# Helpers
# ==================================================

def multiclass_brier(
    y,
    probabilities,
):

    truth = np.zeros(
        (
            len(y),
            3,
        ),
        dtype=float,
    )

    truth[
        np.arange(len(y)),
        y,
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
    labels,
    probabilities,
):

    y = np.array([
        LABEL_TO_ID[label]
        for label in labels
    ])


    return {
        "log_loss":
            log_loss(
                y,
                probabilities,
                labels=[
                    0,
                    1,
                    2,
                ],
            ),

        "brier":
            multiclass_brier(
                y,
                probabilities,
            ),
    }


# ==================================================
# Per-match comparison
# ==================================================

matches = (
    lr[
        "demo_filename"
    ]
    .unique()
    .sort()
    .to_list()
)


rows = []


for demo in matches:

    lr_match = (
        lr
        .filter(
            pl.col(
                "demo_filename"
            )
            == demo
        )
    )

    xgb_match = (
        xgb
        .filter(
            pl.col(
                "demo_filename"
            )
            == demo
        )
    )


    labels = (
        lr_match[
            "label"
        ]
        .to_numpy()
    )


    lr_prob = (
        lr_match
        .select(
            PROB_COLS
        )
        .to_numpy()
        .astype(float)
    )


    xgb_prob = (
        xgb_match
        .select(
            PROB_COLS
        )
        .to_numpy()
        .astype(float)
    )


    lr_result = evaluate(
        labels,
        lr_prob,
    )


    xgb_result = evaluate(
        labels,
        xgb_prob,
    )


    rows.append({
        "demo_filename":
            demo,

        "n":
            len(labels),

        "lr_a3_log_loss":
            lr_result[
                "log_loss"
            ],

        "xgb_a5_log_loss":
            xgb_result[
                "log_loss"
            ],

        "delta_log_loss":
            xgb_result[
                "log_loss"
            ]
            -
            lr_result[
                "log_loss"
            ],

        "lr_a3_brier":
            lr_result[
                "brier"
            ],

        "xgb_a5_brier":
            xgb_result[
                "brier"
            ],

        "delta_brier":
            xgb_result[
                "brier"
            ]
            -
            lr_result[
                "brier"
            ],
    })


result = (
    pl.DataFrame(
        rows
    )
    .sort(
        "delta_log_loss"
    )
)


# ==================================================
# Summary
# ==================================================

delta_ll = (
    result[
        "delta_log_loss"
    ]
    .to_numpy()
)


delta_brier = (
    result[
        "delta_brier"
    ]
    .to_numpy()
)


xgb_wins_ll = int(
    (
        delta_ll < 0
    ).sum()
)


xgb_wins_brier = int(
    (
        delta_brier < 0
    ).sum()
)


print("\n" + "=" * 110)
print("V0 — MATCH-LEVEL ROBUSTNESS")
print("=" * 110)


print(
    f"Matches: "
    f"{len(result)}"
)

print(
    f"XGB_A5 better Log Loss: "
    f"{xgb_wins_ll}/{len(result)}"
)

print(
    f"XGB_A5 better Brier:    "
    f"{xgb_wins_brier}/{len(result)}"
)


print(
    f"\nMedian Δ Log Loss: "
    f"{np.median(delta_ll):+.4f}"
)

print(
    f"Mean Δ Log Loss:   "
    f"{np.mean(delta_ll):+.4f}"
)


print(
    f"Median Δ Brier:    "
    f"{np.median(delta_brier):+.4f}"
)

print(
    f"Mean Δ Brier:      "
    f"{np.mean(delta_brier):+.4f}"
)


print("\n" + "=" * 110)
print("PER-MATCH RESULTS")
print("=" * 110)


print(
    result.select([
        "demo_filename",
        "n",
        "lr_a3_log_loss",
        "xgb_a5_log_loss",
        "delta_log_loss",
    ])
)


print("\n" + "=" * 110)
print("BEST XGB IMPROVEMENTS")
print("=" * 110)


print(
    result
    .select([
        "demo_filename",
        "n",
        "delta_log_loss",
    ])
    .head(5)
)


print("\n" + "=" * 110)
print("WORST XGB REGRESSIONS")
print("=" * 110)


print(
    result
    .select([
        "demo_filename",
        "n",
        "delta_log_loss",
    ])
    .tail(5)
)


result.write_csv(
    OUTPUT_PATH
)


print("\nSaved:")
print(
    OUTPUT_PATH
)

print(
    "\n✅ MATCH ROBUSTNESS ANALYSIS COMPLETE"
)
