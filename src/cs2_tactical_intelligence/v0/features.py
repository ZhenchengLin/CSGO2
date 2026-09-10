import math
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from awpy import Demo
from scipy.spatial import ConvexHull, QhullError


HORIZONS_SEC = [10, 20, 30, 40]
MOTION_WINDOW_SEC = 1.0

V0_PLAYER_PROPS = [
    "armor_value",
    "current_equip_value",
    "inventory",
]


# ==================================================
# Basic helpers
# ==================================================

def has_c4(inventory):
    if inventory is None:
        return False

    return any(
        "C4" in item
        for item in inventory
    )


def distance_xy(x1, y1, x2, y2):
    return math.sqrt(
        (x1 - x2) ** 2
        + (y1 - y2) ** 2
    )


def centroid(team):
    return (
        team["X"].mean(),
        team["Y"].mean(),
        team["Z"].mean(),
    )


def stretch_xy(team, cx, cy):
    distances = [
        distance_xy(
            row["X"],
            row["Y"],
            cx,
            cy,
        )
        for row in team.iter_rows(named=True)
    ]

    return sum(distances) / len(distances)


def mean_pairwise_distance(team):
    points = [
        (row["X"], row["Y"])
        for row in team.iter_rows(named=True)
    ]

    if len(points) < 2:
        return 0.0

    distances = [
        distance_xy(
            p1[0],
            p1[1],
            p2[0],
            p2[1],
        )
        for p1, p2 in combinations(points, 2)
    ]

    return sum(distances) / len(distances)


def convex_hull_area_xy(team):
    if team.height < 3:
        return 0.0

    points = np.array([
        [row["X"], row["Y"]]
        for row in team.iter_rows(named=True)
    ])

    try:
        hull = ConvexHull(points)

        # In scipy 2D ConvexHull:
        # volume = enclosed polygon area
        return float(hull.volume)

    except QhullError:
        return 0.0


# ==================================================
# Labels
# ==================================================

def build_plant_lookup(bomb):
    lookup = {}

    plants = (
        bomb
        .filter(pl.col("event") == "plant")
        .select([
            "round_num",
            "tick",
            "bombsite",
        ])
        .sort([
            "round_num",
            "tick",
        ])
    )

    for row in plants.iter_rows(named=True):

        round_num = row["round_num"]

        if round_num in lookup:
            raise ValueError(
                f"Multiple plants in round {round_num}"
            )

        if row["bombsite"] == "BombsiteA":
            label = "A_PLANT"

        elif row["bombsite"] == "BombsiteB":
            label = "B_PLANT"

        else:
            raise ValueError(
                f"Unknown bombsite: {row['bombsite']}"
            )

        lookup[round_num] = {
            "plant_tick": row["tick"],
            "label": label,
        }

    return lookup


# ==================================================
# Observation generation
# ==================================================

def build_observations(demo):
    plant_lookup = build_plant_lookup(
        demo.bomb
    )

    observations = []
    invalid_rounds = []

    for round_row in (
        demo.rounds
        .sort("round_num")
        .iter_rows(named=True)
    ):

        round_num = round_row["round_num"]
        freeze_end = round_row["freeze_end"]
        round_end = round_row["end"]

        # ------------------------------------------
        # Round-level data validation
        # ------------------------------------------

        if freeze_end is None:
            invalid_rounds.append({
                "round_num": round_num,
                "reason": "MISSING_FREEZE_END",
            })
            continue

        if round_end is None:
            invalid_rounds.append({
                "round_num": round_num,
                "reason": "MISSING_ROUND_END",
            })
            continue

        if freeze_end >= round_end:
            invalid_rounds.append({
                "round_num": round_num,
                "reason": "INVALID_TIMING_ORDER",
            })
            continue

        plant_info = plant_lookup.get(
            round_num
        )

        if plant_info is None:
            plant_tick = None
            label = "NO_PLANT"

        else:
            plant_tick = plant_info[
                "plant_tick"
            ]
            label = plant_info["label"]

        # ------------------------------------------
        # Horizon observations
        # ------------------------------------------

        for horizon_sec in HORIZONS_SEC:

            target_tick = (
                freeze_end
                + int(
                    round(
                        horizon_sec
                        * demo.tickrate
                    )
                )
            )

            before_round_end = (
                target_tick < round_end
            )

            before_plant = (
                plant_tick is None
                or target_tick < plant_tick
            )

            if (
                before_round_end
                and before_plant
            ):

                observations.append({
                    "round_num": round_num,
                    "horizon_sec": horizon_sec,
                    "target_tick": target_tick,
                    "label": label,
                })

    return observations, invalid_rounds


# ==================================================
# Bomb state reconstruction
# ==================================================

def get_bomb_position(
    demo,
    snapshot,
    round_num,
    tick,
):
    carriers = []

    for player in snapshot.iter_rows(
        named=True
    ):
        if (
            player["side"] == "t"
            and has_c4(
                player["inventory"]
            )
        ):
            carriers.append(player)

    # ----------------------------------------------
    # Inventory has priority
    # ----------------------------------------------

    if len(carriers) == 1:
        carrier = carriers[0]

        return (
            carrier["X"],
            carrier["Y"],
            carrier["Z"],
            "CARRIED",
        )

    if len(carriers) > 1:
        raise ValueError(
            f"Multiple C4 carriers at "
            f"round={round_num}, tick={tick}"
        )

    # ----------------------------------------------
    # Nobody carries C4:
    # use latest bomb event
    # ----------------------------------------------

    events = (
        demo.bomb
        .filter(
            (pl.col("round_num") == round_num)
            & (pl.col("tick") <= tick)
        )
        .sort("tick")
        .tail(1)
    )

    if events.height == 0:
        raise ValueError(
            f"No bomb state available at "
            f"round={round_num}, tick={tick}"
        )

    event = events.row(
        0,
        named=True,
    )

    if event["event"] == "drop":
        return (
            event["X"],
            event["Y"],
            event["Z"],
            "DROPPED",
        )

    raise ValueError(
        f"Unexpected bomb state "
        f"round={round_num}, tick={tick}: "
        f"{event['event']}"
    )


# ==================================================
# Snapshot cache
# ==================================================

def build_snapshot_index(
    demo,
    observations,
):
    """
    Filter demo.ticks ONCE to only the ticks
    needed by V0.

    This avoids scanning ~1M player rows
    separately for every feature calculation.
    """

    lag_ticks = int(
        round(
            MOTION_WINDOW_SEC
            * demo.tickrate
        )
    )

    needed = set()

    for obs in observations:
        needed.add(
            (
                obs["round_num"],
                obs["target_tick"],
            )
        )

        needed.add(
            (
                obs["round_num"],
                obs["target_tick"]
                - lag_ticks,
            )
        )

    needed_ticks = {
        tick
        for _, tick in needed
    }

    filtered = (
        demo.ticks
        .filter(
            pl.col("tick").is_in(
                list(needed_ticks)
            )
        )
    )

    index = {}

    groups = filtered.partition_by(
        [
            "round_num",
            "tick",
        ],
        maintain_order=True,
    )

    for group in groups:
        key = (
            group["round_num"][0],
            group["tick"][0],
        )

        index[key] = group

    return index


def require_snapshot(
    snapshot_index,
    round_num,
    tick,
):
    key = (
        round_num,
        tick,
    )

    if key not in snapshot_index:
        raise ValueError(
            f"Missing exact player snapshot: "
            f"round={round_num}, tick={tick}"
        )

    return snapshot_index[key]


# ==================================================
# Pure V0 feature calculation
# ==================================================

def build_feature_row(
    *,
    horizon_sec,
    current,
    previous,
    bomb_now,
    bomb_previous,
):
    """
    Build one V0 feature row using only information
    available at the observation time.

    This function must NOT use:

    - final round outcome
    - future plant events
    - round winner
    - future player state
    - demo filename
    - training label

    Parameters
    ----------
    horizon_sec:
        Seconds since freeze_end for this observation.

    current:
        Player snapshot at the observation time.

    previous:
        Player snapshot one motion window earlier.

    bomb_now:
        (x, y, z) bomb position at the observation time.

    bomb_previous:
        (x, y, z) bomb position one motion window earlier.

    Returns
    -------
    dict
        Feature values plus runtime QA metadata.
    """

    # ==============================================
    # Current alive teams
    # ==============================================

    t_alive_df = current.filter(
        (pl.col("side") == "t")
        & (pl.col("health") > 0)
    )

    ct_alive_df = current.filter(
        (pl.col("side") == "ct")
        & (pl.col("health") > 0)
    )

    if t_alive_df.height == 0:
        raise ValueError(
            "Cannot build V0 features with "
            "zero alive T players"
        )

    # ==============================================
    # A1 — Offensive Geometry
    # ==============================================

    t_cx, t_cy, t_cz = centroid(
        t_alive_df
    )

    t_stretch = stretch_xy(
        t_alive_df,
        t_cx,
        t_cy,
    )

    t_range_x = (
        t_alive_df["X"].max()
        - t_alive_df["X"].min()
    )

    t_range_y = (
        t_alive_df["Y"].max()
        - t_alive_df["Y"].min()
    )

    t_pairwise = mean_pairwise_distance(
        t_alive_df
    )

    t_hull = convex_hull_area_xy(
        t_alive_df
    )

    (
        bomb_x,
        bomb_y,
        bomb_z,
    ) = bomb_now

    bomb_to_t_centroid = distance_xy(
        bomb_x,
        bomb_y,
        t_cx,
        t_cy,
    )

    # ==============================================
    # A2 — Motion
    #
    # Only identities alive NOW are compared against
    # their positions one second earlier.
    #
    # This exactly preserves the frozen V0 design.
    # ==============================================

    current_t = (
        t_alive_df
        .select([
            "steamid",
            "X",
            "Y",
            "Z",
        ])
        .rename({
            "X": "current_X",
            "Y": "current_Y",
            "Z": "current_Z",
        })
    )

    previous_t = (
        previous
        .select([
            "steamid",
            "X",
            "Y",
            "Z",
        ])
        .rename({
            "X": "previous_X",
            "Y": "previous_Y",
            "Z": "previous_Z",
        })
    )

    movement = (
        current_t
        .join(
            previous_t,
            on="steamid",
            how="inner",
        )
    )

    if movement.height == 0:
        raise ValueError(
            "No comparable T players "
            "for motion features"
        )

    movement = (
        movement
        .with_columns([
            (
                (
                    pl.col("current_X")
                    - pl.col("previous_X")
                )
                / MOTION_WINDOW_SEC
            ).alias("vx"),

            (
                (
                    pl.col("current_Y")
                    - pl.col("previous_Y")
                )
                / MOTION_WINDOW_SEC
            ).alias("vy"),
        ])
        .with_columns(
            (
                pl.col("vx") ** 2
                + pl.col("vy") ** 2
            )
            .sqrt()
            .alias("speed")
        )
    )

    t_mean_speed = (
        movement["speed"].mean()
    )

    centroid_vx = (
        movement["current_X"].mean()
        - movement["previous_X"].mean()
    ) / MOTION_WINDOW_SEC

    centroid_vy = (
        movement["current_Y"].mean()
        - movement["previous_Y"].mean()
    ) / MOTION_WINDOW_SEC

    (
        bomb_x_previous,
        bomb_y_previous,
        _,
    ) = bomb_previous

    bomb_speed = (
        distance_xy(
            bomb_x,
            bomb_y,
            bomb_x_previous,
            bomb_y_previous,
        )
        / MOTION_WINDOW_SEC
    )

    # ==============================================
    # A3 — Combat
    # ==============================================

    t_alive = t_alive_df.height
    ct_alive = ct_alive_df.height

    t_health_sum = (
        t_alive_df["health"].sum()
    )

    ct_health_sum = (
        ct_alive_df["health"].sum()
    )

    t_armor_sum = (
        t_alive_df["armor"].sum()
    )

    ct_armor_sum = (
        ct_alive_df["armor"].sum()
    )

    # ==============================================
    # A4 — Economy
    #
    # Retained in the canonical dataset so the
    # historical ablation remains reproducible.
    #
    # It is NOT part of V0_MODEL_FEATURES.
    # ==============================================

    t_equip_value_sum = (
        t_alive_df[
            "current_equip_value"
        ].sum()
    )

    ct_equip_value_sum = (
        ct_alive_df[
            "current_equip_value"
        ].sum()
    )

    # ==============================================
    # A5 — Defense / Interaction
    # ==============================================

    if ct_alive_df.height > 0:

        (
            ct_cx,
            ct_cy,
            ct_cz,
        ) = centroid(
            ct_alive_df
        )

        ct_stretch = stretch_xy(
            ct_alive_df,
            ct_cx,
            ct_cy,
        )

        ct_range_x = (
            ct_alive_df["X"].max()
            - ct_alive_df["X"].min()
        )

        ct_range_y = (
            ct_alive_df["Y"].max()
            - ct_alive_df["Y"].min()
        )

        ct_pairwise = (
            mean_pairwise_distance(
                ct_alive_df
            )
        )

        ct_hull = (
            convex_hull_area_xy(
                ct_alive_df
            )
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
        nearest_distances = []

        for t_player in (
            t_alive_df
            .iter_rows(named=True)
        ):

            distances_to_ct = []

            for ct_player in (
                ct_alive_df
                .iter_rows(named=True)
            ):

                d = distance_xy(
                    t_player["X"],
                    t_player["Y"],
                    ct_player["X"],
                    ct_player["Y"],
                )

                cross_distances.append(
                    d
                )

                distances_to_ct.append(
                    d
                )

            nearest_distances.append(
                min(distances_to_ct)
            )

        minimum_t_ct_distance = min(
            cross_distances
        )

        mean_nearest_opponent_distance = (
            sum(nearest_distances)
            / len(nearest_distances)
        )

    else:

        ct_cx = 0.0
        ct_cy = 0.0
        ct_cz = 0.0

        ct_stretch = 0.0
        ct_range_x = 0.0
        ct_range_y = 0.0

        ct_pairwise = 0.0
        ct_hull = 0.0

        t_ct_centroid_distance = 0.0
        minimum_t_ct_distance = 0.0
        mean_nearest_opponent_distance = 0.0

    # ==============================================
    # Pure feature output
    # ==============================================

    return {

        # Model context
        "horizon_sec":
            horizon_sec,

        # A1
        "t_centroid_x":
            t_cx,

        "t_centroid_y":
            t_cy,

        "t_centroid_z":
            t_cz,

        "t_stretch_xy":
            t_stretch,

        "t_range_x":
            t_range_x,

        "t_range_y":
            t_range_y,

        "t_mean_pairwise_distance":
            t_pairwise,

        "t_convex_hull_area":
            t_hull,

        "bomb_x":
            bomb_x,

        "bomb_y":
            bomb_y,

        "bomb_z":
            bomb_z,

        "bomb_to_t_centroid_distance":
            bomb_to_t_centroid,

        # A2
        "t_mean_speed_1s":
            t_mean_speed,

        "t_centroid_velocity_x_1s":
            centroid_vx,

        "t_centroid_velocity_y_1s":
            centroid_vy,

        "bomb_speed_1s":
            bomb_speed,

        # A3
        "t_alive":
            t_alive,

        "ct_alive":
            ct_alive,

        "alive_difference":
            t_alive
            - ct_alive,

        "t_health_sum":
            t_health_sum,

        "ct_health_sum":
            ct_health_sum,

        "health_difference":
            t_health_sum
            - ct_health_sum,

        "t_armor_sum":
            t_armor_sum,

        "ct_armor_sum":
            ct_armor_sum,

        "armor_difference":
            t_armor_sum
            - ct_armor_sum,

        # A4
        "t_equip_value_sum":
            t_equip_value_sum,

        "ct_equip_value_sum":
            ct_equip_value_sum,

        "equip_value_difference":
            t_equip_value_sum
            - ct_equip_value_sum,

        # A5
        "ct_centroid_x":
            ct_cx,

        "ct_centroid_y":
            ct_cy,

        "ct_centroid_z":
            ct_cz,

        "ct_stretch_xy":
            ct_stretch,

        "ct_range_x":
            ct_range_x,

        "ct_range_y":
            ct_range_y,

        "ct_mean_pairwise_distance":
            ct_pairwise,

        "ct_convex_hull_area":
            ct_hull,

        "t_ct_centroid_distance":
            t_ct_centroid_distance,

        "minimum_t_ct_distance":
            minimum_t_ct_distance,

        "mean_nearest_opponent_distance":
            mean_nearest_opponent_distance,

        # Runtime QA
        "qa_motion_players_used":
            movement.height,
    }



# ==================================================
# Main FeatureBuilder
# ==================================================

def build_features_for_demo(
    demo_path,
):
    demo_path = Path(demo_path)

    demo = Demo(
        str(demo_path),
        verbose=False,
    )

    demo.parse(
        player_props=V0_PLAYER_PROPS
    )

    map_name = demo.header.get(
        "map_name"
    )

    if map_name != "de_mirage":
        raise ValueError(
            f"Expected de_mirage, got {map_name}"
        )

    observations, invalid_rounds = (
        build_observations(demo)
    )

    snapshot_index = (
        build_snapshot_index(
            demo,
            observations,
        )
    )

    lag_ticks = int(
        round(
            MOTION_WINDOW_SEC
            * demo.tickrate
        )
    )

    rows = []

    for obs in observations:

        round_num = obs["round_num"]
        horizon_sec = obs["horizon_sec"]
        target_tick = obs["target_tick"]
        label = obs["label"]

        previous_tick = (
            target_tick
            - lag_ticks
        )

        current = require_snapshot(
            snapshot_index,
            round_num,
            target_tick,
        )

        previous = require_snapshot(
            snapshot_index,
            round_num,
            previous_tick,
        )

        # ==========================================
        # Adapter-specific bomb reconstruction
        #
        # Historical demos reconstruct bomb state
        # from inventory + bomb events.
        #
        # The pure FeatureBuilder only receives the
        # resulting bomb positions.
        # ==========================================

        (
            bomb_x,
            bomb_y,
            bomb_z,
            bomb_state_now,
        ) = get_bomb_position(
            demo,
            current,
            round_num,
            target_tick,
        )

        (
            bomb_x_prev,
            bomb_y_prev,
            bomb_z_prev,
            bomb_state_prev,
        ) = get_bomb_position(
            demo,
            previous,
            round_num,
            previous_tick,
        )

        # ==========================================
        # Shared feature calculation
        #
        # No future label or round outcome enters
        # build_feature_row().
        # ==========================================

        feature_row = build_feature_row(
            horizon_sec=horizon_sec,
            current=current,
            previous=previous,
            bomb_now=(
                bomb_x,
                bomb_y,
                bomb_z,
            ),
            bomb_previous=(
                bomb_x_prev,
                bomb_y_prev,
                bomb_z_prev,
            ),
        )

        # ==========================================
        # Offline-only metadata + target
        # ==========================================

        rows.append({

            "demo_filename":
                demo_path.name,

            "round_num":
                round_num,

            "horizon_sec":
                horizon_sec,

            "target_tick":
                target_tick,

            **feature_row,

            "qa_bomb_state_prev":
                bomb_state_prev,

            "qa_bomb_state_now":
                bomb_state_now,

            "label":
                label,
        })


    features = pl.DataFrame(rows)

    # ==========================================
    # Canonical V0 feature schema
    #
    # Different demos/parser outputs may infer
    # different numeric dtypes for the same
    # semantic field (e.g. Int64 vs Float64).
    #
    # Normalize here so every demo produces
    # exactly the same schema before batching.
    # ==========================================

    integer_columns = [
        "round_num",
        "horizon_sec",
        "target_tick",
        "t_alive",
        "ct_alive",
        "alive_difference",
        "qa_motion_players_used",
    ]

    float_columns = [
        # A1
        "t_centroid_x",
        "t_centroid_y",
        "t_centroid_z",
        "t_stretch_xy",
        "t_range_x",
        "t_range_y",
        "t_mean_pairwise_distance",
        "t_convex_hull_area",
        "bomb_x",
        "bomb_y",
        "bomb_z",
        "bomb_to_t_centroid_distance",

        # A2
        "t_mean_speed_1s",
        "t_centroid_velocity_x_1s",
        "t_centroid_velocity_y_1s",
        "bomb_speed_1s",

        # A3
        "t_health_sum",
        "ct_health_sum",
        "health_difference",
        "t_armor_sum",
        "ct_armor_sum",
        "armor_difference",

        # A4
        "t_equip_value_sum",
        "ct_equip_value_sum",
        "equip_value_difference",

        # A5
        "ct_centroid_x",
        "ct_centroid_y",
        "ct_centroid_z",
        "ct_stretch_xy",
        "ct_range_x",
        "ct_range_y",
        "ct_mean_pairwise_distance",
        "ct_convex_hull_area",
        "t_ct_centroid_distance",
        "minimum_t_ct_distance",
        "mean_nearest_opponent_distance",
    ]

    string_columns = [
        "demo_filename",
        "qa_bomb_state_prev",
        "qa_bomb_state_now",
        "label",
    ]

    features = features.with_columns(
        [
            pl.col(column).cast(pl.Int64)
            for column in integer_columns
        ]
        + [
            pl.col(column).cast(pl.Float64)
            for column in float_columns
        ]
        + [
            pl.col(column).cast(pl.Utf8)
            for column in string_columns
        ]
    )

    return (
        features,
        invalid_rounds,
    )
