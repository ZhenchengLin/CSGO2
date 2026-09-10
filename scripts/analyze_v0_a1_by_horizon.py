import numpy as np
import polars as pl

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    recall_score,
)


A0_PATH = "data/processed/v0_a0_oof_predictions.parquet"
A1_PATH = "data/processed/v0_a1_oof_predictions.parquet"

LABELS = [
    "A_PLANT",
    "B_PLANT",
    "NO_PLANT",
]


def brier_score(df):

    probabilities = df.select([
        "p_a_plant",
        "p_b_plant",
        "p_no_plant",
    ]).to_numpy()

    y = df["label"].to_list()

    label_to_index = {
        label: i
        for i, label in enumerate(LABELS)
    }

    truth = np.zeros(
        (
            len(y),
            len(LABELS),
        )
    )

    for i, label in enumerate(y):
        truth[
            i,
            label_to_index[label],
        ] = 1.0

    return np.mean(
        np.sum(
            (probabilities - truth) ** 2,
            axis=1,
        )
    )


def evaluate(df):

    y = df["label"].to_numpy()

    probabilities = df.select([
        "p_a_plant",
        "p_b_plant",
        "p_no_plant",
    ]).to_numpy()

    prediction = df[
        "prediction"
    ].to_numpy()

    recalls = recall_score(
        y,
        prediction,
        labels=LABELS,
        average=None,
        zero_division=0,
    )

    return {
        "n": df.height,

        "log_loss": log_loss(
            y,
            probabilities,
            labels=LABELS,
        ),

        "brier": brier_score(df),

        "accuracy": accuracy_score(
            y,
            prediction,
        ),

        "macro_f1": f1_score(
            y,
            prediction,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),

        "recall_a": recalls[0],
        "recall_b": recalls[1],
        "recall_no": recalls[2],
    }


a0 = pl.read_parquet(A0_PATH)
a1 = pl.read_parquet(A1_PATH)


print("\n" + "=" * 110)
print("A0 → A1 BY HORIZON")
print("=" * 110)


for horizon in [10, 20, 30, 40]:

    a0_h = a0.filter(
        pl.col("horizon_sec") == horizon
    )

    a1_h = a1.filter(
        pl.col("horizon_sec") == horizon
    )

    m0 = evaluate(a0_h)
    m1 = evaluate(a1_h)

    print(
        f"\n{horizon}s  "
        f"(n={m1['n']})"
    )

    print(
        f"  Log Loss: "
        f"{m0['log_loss']:.4f}"
        f" → {m1['log_loss']:.4f}"
        f"  delta={m1['log_loss'] - m0['log_loss']:+.4f}"
    )

    print(
        f"  Brier:    "
        f"{m0['brier']:.4f}"
        f" → {m1['brier']:.4f}"
        f"  delta={m1['brier'] - m0['brier']:+.4f}"
    )

    print(
        f"  Accuracy: "
        f"{m0['accuracy']:.4f}"
        f" → {m1['accuracy']:.4f}"
    )

    print(
        f"  Macro F1: "
        f"{m0['macro_f1']:.4f}"
        f" → {m1['macro_f1']:.4f}"
    )

    print(
        "  A1 Recall:"
        f" A={m1['recall_a']:.3f}"
        f" B={m1['recall_b']:.3f}"
        f" N={m1['recall_no']:.3f}"
    )


print("\n" + "=" * 110)
print("LABEL DISTRIBUTION BY HORIZON")
print("=" * 110)

print(
    a1
    .group_by([
        "horizon_sec",
        "label",
    ])
    .len()
    .sort([
        "horizon_sec",
        "label",
    ])
)
