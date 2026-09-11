"""
Audit historical V0 snapshot timing after correcting the demo clock.

For each corrected V0 observation:

1. Nominal target:
       freeze_end + horizon

2. Current snapshot:
       first available player snapshot AT OR AFTER target

3. Desired previous time:
       actual_current_tick - 1 real second

4. Previous snapshot:
       latest available player snapshot AT OR BEFORE desired previous time

This mirrors the causal timing semantics of the live engine.
"""

from bisect import bisect_left, bisect_right
from pathlib import Path

import polars as pl

from cs2_tactical_intelligence.v0.features import (
    build_observations,
    V0_PLAYER_PROPS,
    MOTION_WINDOW_SEC,
)

from cs2_tactical_intelligence.v0.timing import (
    open_v0_demo,
    seconds_to_demo_ticks,
    V0_DEMO_TICKS_PER_SECOND,
)


RAW_DIR = Path("data/raw")

manifest = pl.read_csv(
    RAW_DIR / "demo_manifest.csv",
    null_values=[""],
    infer_schema_length=1000,
)

motion_ticks = seconds_to_demo_ticks(
    MOTION_WINDOW_SEC
)

rows = []


print("\n" + "=" * 100)
print("V0 HISTORICAL SNAPSHOT TIMING AUDIT")
print("=" * 100)


for i, manifest_row in enumerate(
    manifest.iter_rows(named=True),
    start=1,
):

    filename = manifest_row["demo filename"]

    print(
        f"[{i:02d}/{manifest.height:02d}] "
        f"{filename}"
    )

    demo = open_v0_demo(
        RAW_DIR / filename,
        verbose=False,
    )

    demo.parse(
        player_props=V0_PLAYER_PROPS
    )

    observations, invalid = (
        build_observations(demo)
    )

    # Available player snapshot ticks by round.
    ticks_by_round = {}

    for round_num in (
        demo.ticks[
            "round_num"
        ]
        .unique()
        .drop_nulls()
        .to_list()
    ):

        ticks = sorted(
            set(
                demo.ticks
                .filter(
                    pl.col("round_num")
                    == round_num
                )["tick"]
                .to_list()
            )
        )

        ticks_by_round[
            round_num
        ] = ticks


    for obs in observations:

        round_num = obs["round_num"]
        nominal_target = obs["target_tick"]

        round_ticks = (
            ticks_by_round[
                round_num
            ]
        )

        # ==========================================
        # Current:
        # first available snapshot >= target
        # ==========================================

        current_pos = bisect_left(
            round_ticks,
            nominal_target,
        )

        if current_pos >= len(
            round_ticks
        ):
            raise RuntimeError(
                f"No snapshot at/after target: "
                f"{filename}, "
                f"round={round_num}, "
                f"target={nominal_target}"
            )

        actual_current = (
            round_ticks[
                current_pos
            ]
        )

        current_lateness_ticks = (
            actual_current
            - nominal_target
        )

        # ==========================================
        # Previous:
        # latest snapshot <= current - 1 sec
        # ==========================================

        desired_previous = (
            actual_current
            - motion_ticks
        )

        previous_pos = (
            bisect_right(
                round_ticks,
                desired_previous,
            )
            - 1
        )

        if previous_pos < 0:
            raise RuntimeError(
                f"No historical snapshot: "
                f"{filename}, "
                f"round={round_num}, "
                f"desired={desired_previous}"
            )

        actual_previous = (
            round_ticks[
                previous_pos
            ]
        )

        previous_staleness_ticks = (
            desired_previous
            - actual_previous
        )

        actual_motion_ticks = (
            actual_current
            - actual_previous
        )

        actual_motion_sec = (
            actual_motion_ticks
            / V0_DEMO_TICKS_PER_SECOND
        )

        motion_error_sec = abs(
            actual_motion_sec
            - MOTION_WINDOW_SEC
        )

        rows.append({
            "demo_filename":
                filename,

            "round_num":
                round_num,

            "horizon_sec":
                obs["horizon_sec"],

            "nominal_target_tick":
                nominal_target,

            "actual_current_tick":
                actual_current,

            "current_lateness_ticks":
                current_lateness_ticks,

            "current_lateness_sec":
                current_lateness_ticks
                / V0_DEMO_TICKS_PER_SECOND,

            "desired_previous_tick":
                desired_previous,

            "actual_previous_tick":
                actual_previous,

            "previous_staleness_ticks":
                previous_staleness_ticks,

            "previous_staleness_sec":
                previous_staleness_ticks
                / V0_DEMO_TICKS_PER_SECOND,

            "actual_motion_sec":
                actual_motion_sec,

            "motion_error_sec":
                motion_error_sec,
        })


df = pl.DataFrame(
    rows
)


print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    "Observations:",
    df.height,
)

print(
    "Exact current snapshots:",
    df.filter(
        pl.col(
            "current_lateness_ticks"
        ) == 0
    ).height,
)

print(
    "Non-exact current snapshots:",
    df.filter(
        pl.col(
            "current_lateness_ticks"
        ) > 0
    ).height,
)


print(
    "\nCURRENT LATENESS TICKS"
)

print(
    df.group_by(
        "current_lateness_ticks"
    )
    .len()
    .sort(
        "current_lateness_ticks"
    )
)


print(
    "\nPREVIOUS STALENESS TICKS"
)

print(
    df.group_by(
        "previous_staleness_ticks"
    )
    .len()
    .sort(
        "previous_staleness_ticks"
    )
)


print(
    "\nMAX CURRENT LATENESS"
)

print(
    df.select([
        pl.col(
            "current_lateness_ticks"
        ).max(),

        pl.col(
            "current_lateness_sec"
        ).max(),
    ])
)


print(
    "\nMAX PREVIOUS STALENESS"
)

print(
    df.select([
        pl.col(
            "previous_staleness_ticks"
        ).max(),

        pl.col(
            "previous_staleness_sec"
        ).max(),
    ])
)


print(
    "\nMAX MOTION WINDOW ERROR"
)

print(
    df.select(
        pl.col(
            "motion_error_sec"
        ).max()
    )
)


print(
    "\nTOP 20 TIMING OFFENDERS"
)

print(
    df.sort([
        "current_lateness_ticks",
        "previous_staleness_ticks",
    ], descending=True)
    .head(20)
)


output = Path(
    "data/interim/"
    "v0_snapshot_timing_audit.csv"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.write_csv(
    output
)

print(
    f"\nSaved: {output}"
)

print(
    "\n✅ V0 SNAPSHOT TIMING AUDIT COMPLETE"
)
