import polars as pl
from awpy import Demo


DEMO_PATH = "data/raw/havu-vs-mellren-m1-mirage.dem"
OUTPUT_PATH = "data/interim/v0_snapshot_players.parquet"

HORIZONS = [10, 20, 30, 40]

V0_PROPS = [
    "armor_value",
    "active_weapon_name",
    "current_equip_value",
    "velocity_X",
    "velocity_Y",
    "velocity_Z",
    "inventory",
    "has_helmet",
    "has_defuser",
    "pitch",
    "yaw",
]


# --------------------------------------------------
# 1. Parse demo
# --------------------------------------------------

demo = Demo(DEMO_PATH, verbose=True)
demo.parse(player_props=V0_PROPS)

tickrate = demo.tickrate


# --------------------------------------------------
# 2. Build ground-truth labels
# --------------------------------------------------

plants = (
    demo.bomb
    .filter(pl.col("event") == "plant")
    .select([
        "round_num",
        "tick",
        "bombsite",
    ])
    .rename({
        "tick": "plant_tick",
    })
)


rounds = (
    demo.rounds
    .select([
        "round_num",
        "freeze_end",
        "end",
    ])
    .join(
        plants,
        on="round_num",
        how="left",
    )
    .with_columns(
        pl.when(pl.col("bombsite") == "BombsiteA")
        .then(pl.lit("A_PLANT"))
        .when(pl.col("bombsite") == "BombsiteB")
        .then(pl.lit("B_PLANT"))
        .otherwise(pl.lit("NO_PLANT"))
        .alias("label")
    )
)


# --------------------------------------------------
# 3. Build valid observation points
# --------------------------------------------------

rows = []

for round_row in rounds.iter_rows(named=True):

    for horizon_sec in HORIZONS:

        target_tick = (
            round_row["freeze_end"]
            + horizon_sec * tickrate
        )

        before_round_end = (
            target_tick < round_row["end"]
        )

        before_plant = (
            round_row["plant_tick"] is None
            or target_tick < round_row["plant_tick"]
        )

        if before_round_end and before_plant:
            rows.append({
                "round_num": round_row["round_num"],
                "horizon_sec": horizon_sec,
                "target_tick": target_tick,
                "label": round_row["label"],
            })


observations = (
    pl.DataFrame(rows)
    .with_columns([
        pl.col("round_num").cast(pl.UInt32),
        pl.col("target_tick").cast(pl.Int32),
    ])
)


# --------------------------------------------------
# 4. Join observation points to player tick states
# --------------------------------------------------

snapshot_players = (
    observations
    .join(
        demo.ticks,
        left_on=[
            "round_num",
            "target_tick",
        ],
        right_on=[
            "round_num",
            "tick",
        ],
        how="inner",
    )
    .with_columns(
        (pl.col("health") > 0).alias("alive")
    )
)


# --------------------------------------------------
# 5. Validate: every observation should have 10 players
# --------------------------------------------------

counts = (
    snapshot_players
    .group_by([
        "round_num",
        "horizon_sec",
    ])
    .len()
    .sort([
        "round_num",
        "horizon_sec",
    ])
)


print("\n" + "=" * 90)
print("DATASET SUMMARY")
print("=" * 90)

print("Valid observations:", observations.height)
print("Player rows:", snapshot_players.height)

print("\nPlayers per observation:")
print(
    counts
    .rename({
        "len": "n_players",
    })
    .group_by("n_players")
    .agg(
        pl.len().alias("n_observations")
    )
    .sort("n_players")
)


assert counts["len"].min() == 10
assert counts["len"].max() == 10


# --------------------------------------------------
# 6. Show one real observation
# --------------------------------------------------

print("\n" + "=" * 90)
print("ROUND 1 @ 10 SEC")
print("=" * 90)

sample = (
    snapshot_players
    .filter(
        (pl.col("round_num") == 1)
        & (pl.col("horizon_sec") == 10)
    )
    .select([
        "round_num",
        "horizon_sec",
        "label",
        "name",
        "side",
        "alive",
        "place",
        "X",
        "Y",
        "Z",
        "velocity_X",
        "velocity_Y",
        "velocity_Z",
        "health",
        "armor",
        "active_weapon_name",
        "current_equip_value",
        "inventory",
    ])
    .sort([
        "side",
        "name",
    ])
)

print(sample)


# --------------------------------------------------
# 7. Save dataset
# --------------------------------------------------

snapshot_players.write_parquet(OUTPUT_PATH)

print("\nSaved:")
print(OUTPUT_PATH)
