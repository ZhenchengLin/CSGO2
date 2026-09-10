import polars as pl
from awpy import Demo


DEMO_PATH = "data/raw/havu-vs-mellren-m1-mirage.dem"

demo = Demo(DEMO_PATH, verbose=True)
demo.parse(player_props=["inventory"])


print("\n" + "=" * 90)
print("ALL BOMB PLANT EVENTS")
print("=" * 90)

plants = (
    demo.bomb
    .filter(pl.col("event") == "plant")
    .select([
        "round_num",
        "tick",
        "name",
        "X",
        "Y",
        "Z",
        "bombsite",
    ])
)

print(plants)


print("\n" + "=" * 90)
print("ROUND TABLE vs BOMB EVENT")
print("=" * 90)

comparison = (
    demo.rounds
    .select([
        "round_num",
        "bomb_plant",
        "bomb_site",
    ])
    .join(
        plants.rename({
            "tick": "event_plant_tick",
            "bombsite": "event_bombsite",
            "name": "planter",
            "X": "plant_X",
            "Y": "plant_Y",
            "Z": "plant_Z",
        }),
        on="round_num",
        how="left",
    )
)

print(comparison)


print("\n" + "=" * 90)
print("PLANTER LOCATION AROUND EACH PLANT")
print("=" * 90)

for row in plants.iter_rows(named=True):

    round_num = row["round_num"]
    plant_tick = row["tick"]
    planter = row["name"]

    player_state = (
        demo.ticks
        .filter(
            (pl.col("round_num") == round_num)
            & (pl.col("name") == planter)
            & (pl.col("tick") >= plant_tick - 3)
            & (pl.col("tick") <= plant_tick + 3)
        )
        .select([
            "round_num",
            "tick",
            "name",
            "place",
            "X",
            "Y",
            "Z",
            "health",
            "inventory",
        ])
    )

    print(
        f"\nRound {round_num}"
        f" | plant tick {plant_tick}"
        f" | planter {planter}"
        f" | Awpy bombsite {row['bombsite']}"
    )

    print(player_state)
