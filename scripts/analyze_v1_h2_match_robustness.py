from pathlib import Path

import numpy as np
import polars as pl

from sklearn.metrics import log_loss


V0_PATH = Path(
    "data/processed/v0_tc_xgb_a5_oof_predictions.parquet"
)

H2_PATH = Path(
    "data/processed/v1_h2_oof_predictions.parquet"
)

MATCH_OUTPUT_PATH = Path(
    "artifacts/v1_h2_match_robustness.csv"
)

BOOTSTRAP_OUTPUT_PATH = Path(
    "artifacts/v1_h2_match_uncertainty.csv"
)


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


PROB_COLS = [
    "p_a_plant",
    "p_b_plant",
    "p_no_plant",
]


KEYS = [
    "demo_filename",
    "round_num",
    "horizon_sec",
    "target_tick",
    "label",
]


# ============================================================
# Helpers
# ============================================================

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
        np.arange(
            len(y)
        ),
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


def summarize_bootstrap(
    name,
    values,
    bootstrap_values,
):

    mean = float(
        np.mean(
            values
        )
    )

    median = float(
        np.median(
            values
        )
    )

    ci_low = float(
        np.percentile(
            bootstrap_values,
            2.5,
        )
    )

    ci_high = float(
        np.percentile(
            bootstrap_values,
            97.5,
        )
    )

    probability_better = float(
        np.mean(
            bootstrap_values
            < 0
        )
    )

    print(
        f"\n{name}"
    )

    print(
        f"Mean delta:       "
        f"{mean:+.6f}"
    )

    print(
        f"Median delta:     "
        f"{median:+.6f}"
    )

    print(
        f"95% bootstrap CI: "
        f"[{ci_low:+.6f}, "
        f"{ci_high:+.6f}]"
    )

    print(
        f"P(mean delta < 0): "
        f"{probability_better:.4f}"
    )

    return {
        "metric":
            name,

        "mean_delta":
            mean,

        "median_delta":
            median,

        "ci_2.5":
            ci_low,

        "ci_97.5":
            ci_high,

        "bootstrap_probability_delta_below_zero":
            probability_better,
    }


# ============================================================
# Load
# ============================================================

v0 = pl.read_parquet(
    V0_PATH
)

h2 = pl.read_parquet(
    H2_PATH
)


assert (
    v0.height
    == h2.height
    == 1686
)

assert (
    v0
    .select(
        KEYS
    )
    .equals(
        h2.select(
            KEYS
        )
    )
)


# ============================================================
# Per-match comparison
# ============================================================

matches = (
    v0[
        "demo_filename"
    ]
    .unique()
    .sort()
    .to_list()
)


assert len(
    matches
) == 20


rows = []


for demo in matches:

    v0_match = (
        v0
        .filter(
            pl.col(
                "demo_filename"
            )
            == demo
        )
    )

    h2_match = (
        h2
        .filter(
            pl.col(
                "demo_filename"
            )
            == demo
        )
    )


    labels = (
        v0_match[
            "label"
        ]
        .to_numpy()
    )


    v0_prob = (
        v0_match
        .select(
            PROB_COLS
        )
        .to_numpy()
        .astype(
            np.float64
        )
    )


    h2_prob = (
        h2_match
        .select(
            PROB_COLS
        )
        .to_numpy()
        .astype(
            np.float64
        )
    )


    v0_result = evaluate(
        labels,
        v0_prob,
    )

    h2_result = evaluate(
        labels,
        h2_prob,
    )


    rows.append({
        "demo_filename":
            demo,

        "n":
            len(labels),

        "v0_log_loss":
            v0_result[
                "log_loss"
            ],

        "h2_log_loss":
            h2_result[
                "log_loss"
            ],

        "delta_log_loss":
            h2_result[
                "log_loss"
            ]
            -
            v0_result[
                "log_loss"
            ],

        "v0_brier":
            v0_result[
                "brier"
            ],

        "h2_brier":
            h2_result[
                "brier"
            ],

        "delta_brier":
            h2_result[
                "brier"
            ]
            -
            v0_result[
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


result.write_csv(
    MATCH_OUTPUT_PATH
)


# ============================================================
# Match-level summary
# ============================================================

delta_ll = (
    result[
        "delta_log_loss"
    ]
    .to_numpy()
    .astype(float)
)

delta_brier = (
    result[
        "delta_brier"
    ]
    .to_numpy()
    .astype(float)
)


h2_wins_ll = int(
    (
        delta_ll
        < 0
    ).sum()
)

h2_wins_brier = int(
    (
        delta_brier
        < 0
    ).sum()
)


print(
    "\n"
    + "=" * 100
)

print(
    "V1-H2 — MATCH-LEVEL ROBUSTNESS VS V0"
)

print(
    "=" * 100
)

print(
    f"\nMatches: "
    f"{len(result)}"
)

print(
    f"H2 better Log Loss: "
    f"{h2_wins_ll}/{len(result)}"
)

print(
    f"H2 better Brier:    "
    f"{h2_wins_brier}/{len(result)}"
)

print(
    f"\nMean Δ Log Loss:   "
    f"{np.mean(delta_ll):+.6f}"
)

print(
    f"Median Δ Log Loss: "
    f"{np.median(delta_ll):+.6f}"
)

print(
    f"Mean Δ Brier:      "
    f"{np.mean(delta_brier):+.6f}"
)

print(
    f"Median Δ Brier:    "
    f"{np.median(delta_brier):+.6f}"
)


print(
    "\nBEST H2 MATCH IMPROVEMENTS"
)

print(
    result
    .select([
        "demo_filename",
        "n",
        "delta_log_loss",
        "delta_brier",
    ])
    .head(5)
)


print(
    "\nWORST H2 MATCH REGRESSIONS"
)

print(
    result
    .select([
        "demo_filename",
        "n",
        "delta_log_loss",
        "delta_brier",
    ])
    .tail(5)
)


# ============================================================
# Paired match bootstrap
# ============================================================

N_BOOTSTRAP = 100_000

rng = np.random.default_rng(
    42
)

n_matches = len(
    delta_ll
)


indices = rng.integers(
    0,
    n_matches,
    size=(
        N_BOOTSTRAP,
        n_matches,
    ),
)


boot_ll = (
    delta_ll[
        indices
    ]
    .mean(
        axis=1
    )
)


boot_brier = (
    delta_brier[
        indices
    ]
    .mean(
        axis=1
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "PAIRED MATCH BOOTSTRAP"
)

print(
    "=" * 100
)

print(
    f"\nBootstrap samples: "
    f"{N_BOOTSTRAP:,}"
)


bootstrap_rows = [

    summarize_bootstrap(
        "Log Loss",
        delta_ll,
        boot_ll,
    ),

    summarize_bootstrap(
        "Brier",
        delta_brier,
        boot_brier,
    ),
]


pl.DataFrame(
    bootstrap_rows
).write_csv(
    BOOTSTRAP_OUTPUT_PATH
)


print(
    "\nSaved:"
)

print(
    f"  {MATCH_OUTPUT_PATH}"
)

print(
    f"  {BOOTSTRAP_OUTPUT_PATH}"
)


print(
    "\n✅ V1-H2 MATCH ROBUSTNESS ANALYSIS COMPLETE"
)
