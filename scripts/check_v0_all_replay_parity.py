from pathlib import Path

import polars as pl

from cs2_tactical_intelligence.v0.replay import (
    V0ReplayAdapter,
)

from cs2_tactical_intelligence.v0.predictor import (
    V0Predictor,
)


DATASET_PATH = Path(
    "data/processed/v0_dataset_v1.parquet"
)

RAW_DIR = Path(
    "data/raw"
)


dataset = pl.read_parquet(
    DATASET_PATH
)


demo_names = (
    dataset["demo_filename"]
    .unique()
    .sort()
    .to_list()
)


predictor = V0Predictor()


print("\n" + "=" * 100)
print("V0 — ALL-MATCH REPLAY / OFFLINE PARITY")
print("=" * 100)


total_offline = 0
total_replay = 0


for index, demo_name in enumerate(
    demo_names,
    start=1,
):

    demo_path = (
        RAW_DIR
        / demo_name
    )

    print(
        f"\n[{index:02d}/{len(demo_names):02d}] "
        f"{demo_name}"
    )

    if not demo_path.exists():
        raise FileNotFoundError(
            demo_path
        )

    # ==================================================
    # Offline reference
    # ==================================================

    offline = (
        dataset
        .filter(
            pl.col("demo_filename")
            == demo_name
        )
        .drop("label")
    )

    # ==================================================
    # Replay
    # ==================================================

    adapter = V0ReplayAdapter(
        demo_path
    )

    replay, invalid_rounds = (
        adapter.build_features()
    )

    print(
        f"  offline={offline.height}"
        f" | replay={replay.height}"
        f" | invalid_rounds="
        f"{len(invalid_rounds)}"
    )

    # ==================================================
    # Exact feature parity
    # ==================================================

    assert (
        offline.columns
        == replay.columns
    ), (
        f"Column mismatch: "
        f"{demo_name}"
    )

    assert (
        offline.schema
        == replay.schema
    ), (
        f"Schema mismatch: "
        f"{demo_name}"
    )

    assert (
        offline.equals(
            replay
        )
    ), (
        f"Feature parity failed: "
        f"{demo_name}"
    )

    # ==================================================
    # Runtime inference validation
    # ==================================================

    for row in replay.iter_rows(
        named=True
    ):

        result = predictor.predict(
            row
        )

        probability_sum = (
            result["p_a_plant"]
            + result["p_b_plant"]
            + result["p_no_plant"]
        )

        assert (
            abs(
                probability_sum
                - 1.0
            )
            < 1e-12
        )

    total_offline += (
        offline.height
    )

    total_replay += (
        replay.height
    )

    print(
        "  ✅ exact feature parity "
        "+ inference valid"
    )


print("\n" + "=" * 100)

print(
    f"Matches validated:      "
    f"{len(demo_names)}"
)

print(
    f"Offline observations:   "
    f"{total_offline}"
)

print(
    f"Replay observations:    "
    f"{total_replay}"
)


assert (
    len(demo_names)
    == 20
)

assert (
    total_offline
    == 1268
)

assert (
    total_replay
    == 1268
)


print(
    "\n✅ ALL 20 MATCHES HAVE "
    "EXACT OFFLINE / REPLAY PARITY"
)

print(
    "✅ ALL 1268 REPLAY OBSERVATIONS "
    "PASS RUNTIME INFERENCE"
)

print("=" * 100)
