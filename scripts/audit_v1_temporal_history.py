from pathlib import Path

import polars as pl

from cs2_tactical_intelligence.v0.features import (
    V0_PLAYER_PROPS,
)

from cs2_tactical_intelligence.v0.timing import (
    open_v0_demo,
    seconds_to_demo_ticks,
)


DATASET_PATH = Path(
    "data/processed/"
    "v0_dataset_v2_timing_corrected.parquet"
)

RAW_DIR = Path("data/raw")

OUTPUT_PATH = Path(
    "data/interim/"
    "v1_temporal_history_audit.csv"
)

WINDOWS_SEC = [
    1,
    3,
    5,
]


# ============================================================
# Load corrected observations
# ============================================================

df = pl.read_parquet(
    DATASET_PATH
)

assert df.height == 1686

assert (
    df["demo_filename"]
    .n_unique()
    == 20
)


rows = []


print(
    "\n"
    + "=" * 100
)

print(
    "V1-T0-A — TEMPORAL HISTORY AVAILABILITY AUDIT"
)

print(
    "=" * 100
)

print(
    f"\nObservations: "
    f"{df.height}"
)

print(
    f"Matches:      "
    f"{df['demo_filename'].n_unique()}"
)

print(
    f"Windows:      "
    f"{WINDOWS_SEC}"
)


# ============================================================
# Audit every demo
# ============================================================

demo_names = (
    df["demo_filename"]
    .unique()
    .sort()
    .to_list()
)


for i, filename in enumerate(
    demo_names,
    start=1,
):

    print(
        f"\n[{i:02d}/{len(demo_names):02d}] "
        f"{filename}"
    )

    path = (
        RAW_DIR
        / filename
    )

    demo = open_v0_demo(
        path,
        verbose=False,
    )

    demo.parse(
        player_props=
            V0_PLAYER_PROPS
    )


    # --------------------------------------------------------
    # Exact available player snapshot ticks
    # --------------------------------------------------------

    ticks_by_round = {}

    tick_rows = (
        demo.ticks
        .select([
            "round_num",
            "tick",
        ])
        .unique()
    )

    for row in tick_rows.iter_rows(
        named=True
    ):

        round_num = row[
            "round_num"
        ]

        tick = row[
            "tick"
        ]

        ticks_by_round.setdefault(
            round_num,
            set(),
        ).add(
            tick
        )


    # --------------------------------------------------------
    # freeze_end lookup
    # --------------------------------------------------------

    freeze_end_lookup = {
        row["round_num"]:
            row["freeze_end"]

        for row
        in demo.rounds.iter_rows(
            named=True
        )
    }


    observations = (
        df
        .filter(
            pl.col(
                "demo_filename"
            )
            == filename
        )
    )


    demo_ok = 0
    demo_checks = 0


    for obs in observations.iter_rows(
        named=True
    ):

        round_num = int(
            obs["round_num"]
        )

        target_tick = int(
            obs["target_tick"]
        )

        available = (
            ticks_by_round[
                round_num
            ]
        )

        freeze_end = (
            freeze_end_lookup[
                round_num
            ]
        )


        for window_sec in WINDOWS_SEC:

            lag_ticks = (
                seconds_to_demo_ticks(
                    window_sec
                )
            )

            history_tick = (
                target_tick
                - lag_ticks
            )

            exact_exists = (
                history_tick
                in available
            )

            before_freeze = (
                freeze_end is not None
                and history_tick
                < freeze_end
            )

            valid = (
                exact_exists
                and not before_freeze
            )


            demo_checks += 1

            if valid:
                demo_ok += 1


            rows.append({
                "demo_filename":
                    filename,

                "round_num":
                    round_num,

                "horizon_sec":
                    int(
                        obs[
                            "horizon_sec"
                        ]
                    ),

                "target_tick":
                    target_tick,

                "window_sec":
                    window_sec,

                "history_tick":
                    history_tick,

                "exact_snapshot_exists":
                    exact_exists,

                "before_freeze_end":
                    before_freeze,

                "valid_history":
                    valid,
            })


    print(
        f"  valid checks: "
        f"{demo_ok}/{demo_checks}"
    )


# ============================================================
# Save
# ============================================================

audit = pl.DataFrame(
    rows
)

audit.write_csv(
    OUTPUT_PATH
)


# ============================================================
# Summary by window
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "SUMMARY BY HISTORY WINDOW"
)

print(
    "=" * 100
)


for window_sec in WINDOWS_SEC:

    subset = (
        audit
        .filter(
            pl.col(
                "window_sec"
            )
            == window_sec
        )
    )

    exact_count = (
        subset
        .filter(
            pl.col(
                "exact_snapshot_exists"
            )
        )
        .height
    )

    valid_count = (
        subset
        .filter(
            pl.col(
                "valid_history"
            )
        )
        .height
    )

    before_freeze_count = (
        subset
        .filter(
            pl.col(
                "before_freeze_end"
            )
        )
        .height
    )


    print(
        f"\n{window_sec}s history"
    )

    print(
        f"  exact snapshot: "
        f"{exact_count}/{df.height}"
    )

    print(
        f"  valid history:  "
        f"{valid_count}/{df.height}"
    )

    print(
        f"  before freeze:  "
        f"{before_freeze_count}"
    )


# ============================================================
# Observation-level full-window eligibility
# ============================================================

eligibility = (
    audit
    .group_by([
        "demo_filename",
        "round_num",
        "horizon_sec",
        "target_tick",
    ])
    .agg(
        pl.col(
            "valid_history"
        )
        .all()
        .alias(
            "all_windows_valid"
        )
    )
)


all_valid = (
    eligibility
    .filter(
        pl.col(
            "all_windows_valid"
        )
    )
    .height
)


print(
    "\n"
    + "=" * 100
)

print(
    "T0 ELIGIBILITY"
)

print(
    "=" * 100
)

print(
    f"\nAll 1/3/5s histories valid: "
    f"{all_valid}/{df.height}"
)


invalid = (
    audit
    .filter(
        ~pl.col(
            "valid_history"
        )
    )
)


if invalid.height == 0:

    print(
        "\n✅ ALL TEMPORAL HISTORY "
        "CHECKS PASSED"
    )

else:

    print(
        f"\n⚠️ Invalid temporal checks: "
        f"{invalid.height}"
    )

    print(
        invalid.head(20)
    )


print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)

print(
    "\n✅ V1-T0-A TEMPORAL AUDIT COMPLETE"
)
