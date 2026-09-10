from pathlib import Path

import numpy as np
import polars as pl


INPUT_PATH = Path(
    "artifacts/v0_match_robustness.csv"
)

OUTPUT_PATH = Path(
    "artifacts/v0_match_uncertainty.csv"
)


df = pl.read_csv(
    INPUT_PATH
)


delta_ll = (
    df["delta_log_loss"]
    .to_numpy()
    .astype(float)
)

delta_brier = (
    df["delta_brier"]
    .to_numpy()
    .astype(float)
)


N_BOOTSTRAP = 100_000

rng = np.random.default_rng(
    42
)


n_matches = len(
    delta_ll
)


# ==================================================
# Match-level paired bootstrap
# ==================================================

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


def summarize(
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
        f"Mean delta:      "
        f"{mean:+.4f}"
    )

    print(
        f"Median delta:    "
        f"{median:+.4f}"
    )

    print(
        f"95% bootstrap CI:"
        f" [{ci_low:+.4f}, "
        f"{ci_high:+.4f}]"
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


print(
    "\n"
    + "=" * 100
)

print(
    "V0 — MATCH-LEVEL UNCERTAINTY"
)

print(
    "=" * 100
)

print(
    f"Matches: "
    f"{n_matches}"
)

print(
    f"Bootstrap samples: "
    f"{N_BOOTSTRAP:,}"
)


rows = [

    summarize(
        "Log Loss",
        delta_ll,
        boot_ll,
    ),

    summarize(
        "Brier",
        delta_brier,
        boot_brier,
    ),
]


result = pl.DataFrame(
    rows
)


result.write_csv(
    OUTPUT_PATH
)


print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)

print(
    "\n✅ MATCH UNCERTAINTY ANALYSIS COMPLETE"
)
