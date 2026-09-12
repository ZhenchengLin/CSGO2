from bisect import bisect_left
from pathlib import Path

import numpy as np
import polars as pl

from cs2_tactical_intelligence.v0.features import (
    V0_PLAYER_PROPS,
    centroid,
    distance_xy,
    get_bomb_position,
    stretch_xy,
)

from cs2_tactical_intelligence.v0.timing import (
    V0_DEMO_TICKS_PER_SECOND,
    open_v0_demo,
    seconds_to_demo_ticks,
)


BASE_PATH = Path(
    "data/processed/"
    "v0_dataset_v2_timing_corrected.parquet"
)

RAW_DIR = Path(
    "data/raw"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "v1_t0_temporal_dataset.parquet"
)

AUDIT_PATH = Path(
    "data/interim/"
    "v1_t0_temporal_build_audit.csv"
)


WINDOWS_SEC = [
    3,
    5,
]


MAX_HISTORY_OFFSET_TICKS = 1


STATE_FIELDS = [
    "t_alive",
    "ct_alive",

    "t_health_sum",
    "ct_health_sum",

    "t_armor_sum",
    "ct_armor_sum",

    "t_centroid_x",
    "t_centroid_y",

    "ct_centroid_x",
    "ct_centroid_y",

    "t_stretch_xy",
    "ct_stretch_xy",

    "bomb_x",
    "bomb_y",

    "bomb_to_t_centroid_distance",
    "t_ct_centroid_distance",
    "minimum_t_ct_distance",
]


# ============================================================
# Resolve historical snapshot
# ============================================================

def resolve_history_tick(
    available_ticks,
    desired_tick,
):

    position = bisect_left(
        available_ticks,
        desired_tick,
    )


    # Exact
    if (
        position
        < len(available_ticks)
        and
        available_ticks[position]
        == desired_tick
    ):
        return (
            desired_tick,
            0,
        )


    candidates = []


    # Earlier
    if position > 0:

        earlier = (
            available_ticks[
                position - 1
            ]
        )

        candidates.append(
            (
                abs(
                    earlier
                    - desired_tick
                ),
                0,   # tie-break: prefer earlier
                earlier,
            )
        )


    # Later
    if position < len(
        available_ticks
    ):

        later = (
            available_ticks[
                position
            ]
        )

        candidates.append(
            (
                abs(
                    later
                    - desired_tick
                ),
                1,
                later,
            )
        )


    if not candidates:

        raise ValueError(
            "No historical snapshot candidates"
        )


    candidates.sort()

    distance, _, resolved_tick = (
        candidates[0]
    )


    if (
        distance
        > MAX_HISTORY_OFFSET_TICKS
    ):

        raise ValueError(
            "Historical snapshot offset "
            f"too large: desired={desired_tick}, "
            f"resolved={resolved_tick}, "
            f"distance={distance}"
        )


    return (
        resolved_tick,
        resolved_tick
        - desired_tick,
    )


# ============================================================
# Historical state summary
# ============================================================

def summarize_state(
    demo,
    snapshot,
    round_num,
    tick,
):

    t_alive_df = (
        snapshot
        .filter(
            (pl.col("side") == "t")
            &
            (pl.col("health") > 0)
        )
    )

    ct_alive_df = (
        snapshot
        .filter(
            (pl.col("side") == "ct")
            &
            (pl.col("health") > 0)
        )
    )


    if t_alive_df.height == 0:

        raise ValueError(
            "Historical state has zero "
            "alive T players"
        )


    # --------------------------------------------------------
    # T state
    # --------------------------------------------------------

    t_cx, t_cy, _ = centroid(
        t_alive_df
    )

    t_stretch = stretch_xy(
        t_alive_df,
        t_cx,
        t_cy,
    )


    # --------------------------------------------------------
    # CT state
    # --------------------------------------------------------

    if ct_alive_df.height > 0:

        ct_cx, ct_cy, _ = centroid(
            ct_alive_df
        )

        ct_stretch = stretch_xy(
            ct_alive_df,
            ct_cx,
            ct_cy,
        )

        t_ct_centroid_distance = (
            distance_xy(
                t_cx,
                t_cy,
                ct_cx,
                ct_cy,
            )
        )


        cross_distances = []

        for t_player in (
            t_alive_df
            .iter_rows(
                named=True
            )
        ):

            for ct_player in (
                ct_alive_df
                .iter_rows(
                    named=True
                )
            ):

                cross_distances.append(
                    distance_xy(
                        t_player["X"],
                        t_player["Y"],
                        ct_player["X"],
                        ct_player["Y"],
                    )
                )


        minimum_t_ct_distance = min(
            cross_distances
        )


    else:

        ct_cx = 0.0
        ct_cy = 0.0
        ct_stretch = 0.0

        t_ct_centroid_distance = 0.0
        minimum_t_ct_distance = 0.0


    # --------------------------------------------------------
    # Bomb state
    # --------------------------------------------------------

    (
        bomb_x,
        bomb_y,
        _,
        _,
    ) = get_bomb_position(
        demo,
        snapshot,
        round_num,
        tick,
    )


    bomb_to_t_centroid_distance = (
        distance_xy(
            bomb_x,
            bomb_y,
            t_cx,
            t_cy,
        )
    )


    return {
        "t_alive":
            float(
                t_alive_df.height
            ),

        "ct_alive":
            float(
                ct_alive_df.height
            ),

        "t_health_sum":
            float(
                t_alive_df[
                    "health"
                ].sum()
            ),

        "ct_health_sum":
            float(
                ct_alive_df[
                    "health"
                ].sum()
            )
            if ct_alive_df.height > 0
            else 0.0,

        "t_armor_sum":
            float(
                t_alive_df[
                    "armor"
                ].sum()
            ),

        "ct_armor_sum":
            float(
                ct_alive_df[
                    "armor"
                ].sum()
            )
            if ct_alive_df.height > 0
            else 0.0,

        "t_centroid_x":
            float(t_cx),

        "t_centroid_y":
            float(t_cy),

        "ct_centroid_x":
            float(ct_cx),

        "ct_centroid_y":
            float(ct_cy),

        "t_stretch_xy":
            float(t_stretch),

        "ct_stretch_xy":
            float(ct_stretch),

        "bomb_x":
            float(bomb_x),

        "bomb_y":
            float(bomb_y),

        "bomb_to_t_centroid_distance":
            float(
                bomb_to_t_centroid_distance
            ),

        "t_ct_centroid_distance":
            float(
                t_ct_centroid_distance
            ),

        "minimum_t_ct_distance":
            float(
                minimum_t_ct_distance
            ),
    }


# ============================================================
# Load base
# ============================================================

base = (
    pl.read_parquet(
        BASE_PATH
    )
    .with_row_index(
        "_row_id"
    )
)


assert base.height == 1686
assert (
    base["demo_filename"]
    .n_unique()
    == 20
)


temporal_rows = []
audit_rows = []


print(
    "\n"
    + "=" * 100
)

print(
    "V1-T0 — TEMPORAL FEATURE DATASET BUILD"
)

print(
    "=" * 100
)

print(
    f"\nBase rows: "
    f"{base.height}"
)

print(
    f"Windows:   "
    f"{WINDOWS_SEC}"
)

print(
    f"Fields/window: "
    f"{len(STATE_FIELDS)}"
)

print(
    f"New temporal features: "
    f"{len(STATE_FIELDS) * len(WINDOWS_SEC)}"
)


# ============================================================
# Process each demo once
# ============================================================

demo_names = (
    base[
        "demo_filename"
    ]
    .unique()
    .sort()
    .to_list()
)


for demo_i, filename in enumerate(
    demo_names,
    start=1,
):

    print(
        f"\n[{demo_i:02d}/"
        f"{len(demo_names):02d}] "
        f"{filename}"
    )


    demo = open_v0_demo(
        RAW_DIR / filename,
        verbose=False,
    )

    demo.parse(
        player_props=
            V0_PLAYER_PROPS
    )


    # --------------------------------------------------------
    # Available ticks by round
    # --------------------------------------------------------

    ticks_by_round = {}

    for row in (
        demo.ticks
        .select([
            "round_num",
            "tick",
        ])
        .unique()
        .sort([
            "round_num",
            "tick",
        ])
        .iter_rows(
            named=True
        )
    ):

        ticks_by_round.setdefault(
            row["round_num"],
            [],
        ).append(
            row["tick"]
        )


    freeze_end_lookup = {
        row["round_num"]:
            row["freeze_end"]

        for row in (
            demo.rounds
            .iter_rows(
                named=True
            )
        )
    }


    demo_rows = (
        base
        .filter(
            pl.col(
                "demo_filename"
            )
            == filename
        )
    )


    # --------------------------------------------------------
    # Resolve all needed history ticks
    # --------------------------------------------------------

    resolved_requests = {}

    needed_pairs = set()


    for obs in demo_rows.iter_rows(
        named=True
    ):

        round_num = int(
            obs["round_num"]
        )

        target_tick = int(
            obs["target_tick"]
        )


        for window_sec in WINDOWS_SEC:

            desired_tick = (
                target_tick
                - seconds_to_demo_ticks(
                    window_sec
                )
            )


            (
                resolved_tick,
                offset_ticks,
            ) = resolve_history_tick(
                ticks_by_round[
                    round_num
                ],
                desired_tick,
            )


            freeze_end = (
                freeze_end_lookup[
                    round_num
                ]
            )


            if (
                freeze_end is not None
                and
                resolved_tick
                < freeze_end
            ):

                raise ValueError(
                    "Resolved historical snapshot "
                    "crossed before freeze_end"
                )


            key = (
                int(obs["_row_id"]),
                window_sec,
            )


            resolved_requests[key] = {
                "round_num":
                    round_num,

                "desired_tick":
                    desired_tick,

                "resolved_tick":
                    resolved_tick,

                "offset_ticks":
                    offset_ticks,
            }


            needed_pairs.add(
                (
                    round_num,
                    resolved_tick,
                )
            )


    # --------------------------------------------------------
    # Snapshot cache
    # --------------------------------------------------------

    needed_ticks = {
        tick
        for _, tick
        in needed_pairs
    }


    filtered = (
        demo.ticks
        .filter(
            pl.col(
                "tick"
            )
            .is_in(
                list(
                    needed_ticks
                )
            )
        )
    )


    snapshot_index = {}


    for group in (
        filtered
        .partition_by(
            [
                "round_num",
                "tick",
            ],
            maintain_order=True,
        )
    ):

        snapshot_index[
            (
                int(
                    group[
                        "round_num"
                    ][0]
                ),
                int(
                    group[
                        "tick"
                    ][0]
                ),
            )
        ] = group


    # --------------------------------------------------------
    # Build temporal deltas
    # --------------------------------------------------------

    demo_offset_count = 0


    for obs in demo_rows.iter_rows(
        named=True
    ):

        output = {
            "_row_id":
                int(
                    obs["_row_id"]
                ),
        }


        for window_sec in WINDOWS_SEC:

            request = (
                resolved_requests[
                    (
                        int(
                            obs["_row_id"]
                        ),
                        window_sec,
                    )
                ]
            )


            round_num = (
                request[
                    "round_num"
                ]
            )

            resolved_tick = (
                request[
                    "resolved_tick"
                ]
            )

            offset_ticks = (
                request[
                    "offset_ticks"
                ]
            )


            if offset_ticks != 0:
                demo_offset_count += 1


            snapshot = (
                snapshot_index[
                    (
                        round_num,
                        resolved_tick,
                    )
                ]
            )


            past = summarize_state(
                demo,
                snapshot,
                round_num,
                resolved_tick,
            )


            for field in STATE_FIELDS:

                current_value = float(
                    obs[field]
                )

                history_value = float(
                    past[field]
                )


                output[
                    f"{field}_delta_"
                    f"{window_sec}s"
                ] = (
                    current_value
                    - history_value
                )


            actual_window_sec = (
                (
                    int(
                        obs["target_tick"]
                    )
                    - resolved_tick
                )
                /
                V0_DEMO_TICKS_PER_SECOND
            )


            output[
                f"qa_history_"
                f"{window_sec}s_offset_ticks"
            ] = (
                offset_ticks
            )


            output[
                f"qa_history_"
                f"{window_sec}s_actual_sec"
            ] = (
                actual_window_sec
            )


            audit_rows.append({
                "demo_filename":
                    filename,

                "round_num":
                    int(
                        obs[
                            "round_num"
                        ]
                    ),

                "horizon_sec":
                    int(
                        obs[
                            "horizon_sec"
                        ]
                    ),

                "target_tick":
                    int(
                        obs[
                            "target_tick"
                        ]
                    ),

                "window_sec":
                    window_sec,

                "desired_history_tick":
                    request[
                        "desired_tick"
                    ],

                "resolved_history_tick":
                    resolved_tick,

                "offset_ticks":
                    offset_ticks,

                "actual_window_sec":
                    actual_window_sec,
            })


        temporal_rows.append(
            output
        )


    print(
        f"  observations="
        f"{demo_rows.height}"
        f" | nonexact history="
        f"{demo_offset_count}"
    )


# ============================================================
# Merge back into frozen V0 population
# ============================================================

temporal = pl.DataFrame(
    temporal_rows
)


assert temporal.height == 1686

assert (
    temporal[
        "_row_id"
    ]
    .n_unique()
    == 1686
)


dataset = (
    base
    .join(
        temporal,
        on="_row_id",
        how="left",
    )
    .sort(
        "_row_id"
    )
    .drop(
        "_row_id"
    )
)


assert dataset.height == 1686


# ============================================================
# QA
# ============================================================

TEMPORAL_FEATURES = [
    f"{field}_delta_{window}s"

    for window in WINDOWS_SEC

    for field in STATE_FIELDS
]


assert (
    len(
        TEMPORAL_FEATURES
    )
    == 34
)


matrix = (
    dataset
    .select(
        TEMPORAL_FEATURES
    )
    .to_numpy()
)


assert np.isfinite(
    matrix
).all()


audit = pl.DataFrame(
    audit_rows
)


assert audit.height == (
    1686
    * len(WINDOWS_SEC)
)


print(
    "\n"
    + "=" * 100
)

print(
    "TEMPORAL BUILD QA"
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


    nonexact = (
        subset
        .filter(
            pl.col(
                "offset_ticks"
            )
            != 0
        )
    )


    print(
        f"\n{window_sec}s"
    )

    print(
        f"  rows: "
        f"{subset.height}"
    )

    print(
        f"  exact history: "
        f"{subset.height - nonexact.height}"
        f"/{subset.height}"
    )

    print(
        f"  nonexact: "
        f"{nonexact.height}"
    )

    print(
        f"  actual window range: "
        f"{subset['actual_window_sec'].min():.6f}"
        f" .. "
        f"{subset['actual_window_sec'].max():.6f}"
    )


print(
    f"\nFinal dataset rows: "
    f"{dataset.height}"
)

print(
    f"New temporal features: "
    f"{len(TEMPORAL_FEATURES)}"
)

print(
    "NaN/Inf check: PASS"
)


# ============================================================
# Save
# ============================================================

dataset.write_parquet(
    OUTPUT_PATH
)

audit.write_csv(
    AUDIT_PATH
)


print(
    "\nSaved:"
)

print(
    f"  {OUTPUT_PATH}"
)

print(
    f"  {AUDIT_PATH}"
)

print(
    "\n✅ V1-T0 TEMPORAL DATASET BUILD COMPLETE"
)
