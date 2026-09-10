import math

import polars as pl
from awpy import Demo


DEMO_PATH = "data/raw/havu-vs-mellren-m1-mirage.dem"

A1_PATH = "data/processed/v0_features_a1.parquet"
OUTPUT_PATH = "data/processed/v0_features_a2.parquet"

MOTION_WINDOW_SEC = 1.0


# --------------------------------------------------
# 1. Load A1 features
# --------------------------------------------------

a1 = pl.read_parquet(A1_PATH)


# --------------------------------------------------
# 2. Parse demo
#
# We need inventory so we can reconstruct
# bomb position at arbitrary ticks.
# --------------------------------------------------

demo = Demo(DEMO_PATH, verbose=True)

demo.parse(
    player_props=[
        "inventory",
    ]
)

tickrate = demo.tickrate

lag_ticks = int(
    MOTION_WINDOW_SEC * tickrate
)


# --------------------------------------------------
# Helper:
# Does inventory contain C4?
# --------------------------------------------------

def has_c4(inventory):

    if inventory is None:
        return False

    return any(
        "C4" in item
        for item in inventory
    )


# --------------------------------------------------
# Helper:
# Get player state at exact tick
# --------------------------------------------------

def get_tick_players(round_num, tick):

    return (
        demo.ticks
        .filter(
            (pl.col("round_num") == round_num)
            & (pl.col("tick") == tick)
        )
    )


# --------------------------------------------------
# Helper:
# Reconstruct bomb position at arbitrary tick
# --------------------------------------------------

def get_bomb_position(round_num, tick):

    snapshot = get_tick_players(
        round_num,
        tick,
    )


    # ----------------------------------------------
    # First priority:
    # player inventory
    # ----------------------------------------------

    carriers = []

    for player in snapshot.iter_rows(named=True):

        if (
            player["side"] == "t"
            and has_c4(player["inventory"])
        ):
            carriers.append(player)


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
            f"Multiple bomb carriers at "
            f"round={round_num}, tick={tick}"
        )


    # ----------------------------------------------
    # Nobody carries it.
    # Check latest bomb event.
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
            f"Could not reconstruct bomb position at "
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
        f"Unexpected bomb state at "
        f"round={round_num}, tick={tick}: "
        f"{event['event']}"
    )


# --------------------------------------------------
# 3. Build A2 motion features
# --------------------------------------------------

motion_rows = []


for observation in a1.iter_rows(named=True):

    round_num = observation["round_num"]
    horizon_sec = observation["horizon_sec"]
    target_tick = observation["target_tick"]

    previous_tick = (
        target_tick - lag_ticks
    )


    # --------------------------------------------------
    # Current and previous player states
    # --------------------------------------------------

    current = get_tick_players(
        round_num,
        target_tick,
    )

    previous = get_tick_players(
        round_num,
        previous_tick,
    )


    # --------------------------------------------------
    # Current alive T players
    #
    # We use the SAME player identities at t and t-1s.
    #
    # This prevents centroid changes caused only
    # by player death from being interpreted as motion.
    # --------------------------------------------------

    current_t = (
        current
        .filter(
            (pl.col("side") == "t")
            & (pl.col("health") > 0)
        )
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
            f"No comparable T players at "
            f"round={round_num}, "
            f"horizon={horizon_sec}"
        )


    # --------------------------------------------------
    # Individual player displacement / speed
    # --------------------------------------------------

    movement = (
        movement
        .with_columns([

            (
                (
                    pl.col("current_X")
                    - pl.col("previous_X")
                )
                / MOTION_WINDOW_SEC
            ).alias("vx_1s"),

            (
                (
                    pl.col("current_Y")
                    - pl.col("previous_Y")
                )
                / MOTION_WINDOW_SEC
            ).alias("vy_1s"),
        ])
        .with_columns(

            (
                pl.col("vx_1s") ** 2
                + pl.col("vy_1s") ** 2
            )
            .sqrt()
            .alias("speed_1s")
        )
    )


    t_mean_speed = (
        movement["speed_1s"].mean()
    )


    # --------------------------------------------------
    # Team centroid motion
    #
    # Same current-alive players are used
    # for both time points.
    # --------------------------------------------------

    current_centroid_x = (
        movement["current_X"].mean()
    )

    current_centroid_y = (
        movement["current_Y"].mean()
    )

    previous_centroid_x = (
        movement["previous_X"].mean()
    )

    previous_centroid_y = (
        movement["previous_Y"].mean()
    )


    centroid_vx = (
        current_centroid_x
        - previous_centroid_x
    ) / MOTION_WINDOW_SEC


    centroid_vy = (
        current_centroid_y
        - previous_centroid_y
    ) / MOTION_WINDOW_SEC


    # --------------------------------------------------
    # Bomb motion
    # --------------------------------------------------

    bomb_x_now, bomb_y_now, _, bomb_state_now = (
        get_bomb_position(
            round_num,
            target_tick,
        )
    )


    bomb_x_prev, bomb_y_prev, _, bomb_state_prev = (
        get_bomb_position(
            round_num,
            previous_tick,
        )
    )


    bomb_dx = (
        bomb_x_now - bomb_x_prev
    )

    bomb_dy = (
        bomb_y_now - bomb_y_prev
    )


    bomb_speed = math.sqrt(
        bomb_dx * bomb_dx
        + bomb_dy * bomb_dy
    ) / MOTION_WINDOW_SEC


    # --------------------------------------------------
    # Save A2 features
    # --------------------------------------------------

    motion_rows.append({

        "round_num": round_num,
        "horizon_sec": horizon_sec,
        "target_tick": target_tick,

        "t_mean_speed_1s":
            t_mean_speed,

        "t_centroid_velocity_x_1s":
            centroid_vx,

        "t_centroid_velocity_y_1s":
            centroid_vy,

        "bomb_speed_1s":
            bomb_speed,

        # QA only
        "motion_players_used":
            movement.height,

        "bomb_state_now":
            bomb_state_now,

        "bomb_state_prev":
            bomb_state_prev,
    })


motion = pl.DataFrame(
    motion_rows
)


# --------------------------------------------------
# 4. Join A2 onto A1
# --------------------------------------------------

a2 = (
    a1
    .join(
        motion,
        on=[
            "round_num",
            "horizon_sec",
            "target_tick",
        ],
        how="left",
    )
)


# --------------------------------------------------
# 5. QA
# --------------------------------------------------

print("\n" + "=" * 100)
print("A2 MOTION DATASET")
print("=" * 100)

print(
    "Shape:",
    a2.shape,
)


print("\n" + "=" * 100)
print("NULL COUNTS")
print("=" * 100)

print(
    a2.null_count()
)


print("\n" + "=" * 100)
print("ROUND 1 MOTION EVOLUTION")
print("=" * 100)

print(
    a2
    .filter(
        pl.col("round_num") == 1
    )
    .select([
        "round_num",
        "horizon_sec",
        "label",

        "t_centroid_x",
        "t_centroid_y",

        "t_mean_speed_1s",

        "t_centroid_velocity_x_1s",
        "t_centroid_velocity_y_1s",

        "bomb_speed_1s",

        "motion_players_used",

        "bomb_state_prev",
        "bomb_state_now",
    ])
    .sort("horizon_sec")
)


print("\n" + "=" * 100)
print("MOTION FEATURE RANGES")
print("=" * 100)

for feature in [
    "t_mean_speed_1s",
    "t_centroid_velocity_x_1s",
    "t_centroid_velocity_y_1s",
    "bomb_speed_1s",
]:

    print(
        f"{feature:35s}"
        f" min={a2[feature].min():10.2f}"
        f" mean={a2[feature].mean():10.2f}"
        f" max={a2[feature].max():10.2f}"
    )


# --------------------------------------------------
# 6. Safety checks
# --------------------------------------------------

assert a2.height == 49

assert (
    a2
    .select([
        "t_mean_speed_1s",
        "t_centroid_velocity_x_1s",
        "t_centroid_velocity_y_1s",
        "bomb_speed_1s",
    ])
    .null_count()
    .row(0)
    == (0, 0, 0, 0)
)


assert (
    a2
    .filter(
        pl.col("t_mean_speed_1s") < 0
    )
    .height
    == 0
)


assert (
    a2
    .filter(
        pl.col("bomb_speed_1s") < 0
    )
    .height
    == 0
)


# --------------------------------------------------
# 7. Save
# --------------------------------------------------

a2.write_parquet(
    OUTPUT_PATH
)


print("\nSaved:")
print(OUTPUT_PATH)
