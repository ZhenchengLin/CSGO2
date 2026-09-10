from awpy import Demo
import polars as pl

DEMO_PATH = "data/raw/havu-vs-mellren-m1-mirage.dem"

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


demo = Demo(DEMO_PATH, verbose=True)
demo.parse(player_props=V0_PROPS)


print("\n" + "=" * 70)
print("V0 TICK FIELD AUDIT")
print("=" * 70)

print("Shape:")
print(demo.ticks.shape)

print("\nColumns:")
for column in demo.ticks.columns:
    print(column)

print("\nFirst 5 rows:")
print(demo.ticks.head(5))


print("\n" + "=" * 70)
print("NULL COUNT")
print("=" * 70)

print(demo.ticks.null_count())


print("\n" + "=" * 70)
print("ONE PLAYER ACROSS TICKS")
print("=" * 70)

first_player = demo.ticks["steamid"][0]

player_sample = (
    demo.ticks
    .filter(demo.ticks["steamid"] == first_player)
    .select([
        "tick",
        "round_num",
        "name",
        "side",
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
    ])
    .head(20)
)

print(player_sample)


print("\n" + "=" * 70)
print("PLAYERS PER TICK")
print("=" * 70)

players_per_tick = (
    demo.ticks
    .group_by("tick")
    .agg(
        pl.col("steamid").n_unique().alias("n_players")
    )
)

print(
    players_per_tick
    .group_by("n_players")
    .len()
    .sort("n_players")
)