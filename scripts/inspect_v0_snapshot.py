import polars as pl
from awpy import Demo


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

ROUND_NUM = 1
HORIZONS = [10, 20, 30, 40]


# 让 Terminal 尽量不要隐藏 columns
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_tbl_rows(20)


demo = Demo(DEMO_PATH, verbose=True)
demo.parse(player_props=V0_PROPS)


round_info = (
    demo.rounds
    .filter(pl.col("round_num") == ROUND_NUM)
    .row(0, named=True)
)

freeze_end = round_info["freeze_end"]
round_end = round_info["end"]
plant_tick = round_info["bomb_plant"]
bomb_site = round_info["bomb_site"]
tickrate = demo.tickrate


print("\n" + "=" * 80)
print("ROUND INFO")
print("=" * 80)

print("Round:", ROUND_NUM)
print("Tickrate:", tickrate)
print("Freeze end:", freeze_end)
print("Round end:", round_end)
print("Bomb plant:", plant_tick)
print("Bomb site candidate:", bomb_site)

if plant_tick is not None:
    plant_seconds = (plant_tick - freeze_end) / tickrate
    print(f"Plant time after freeze end: {plant_seconds:.2f} sec")


print("\n" + "=" * 80)
print("OBSERVATION HORIZONS")
print("=" * 80)


for seconds in HORIZONS:

    target_tick = freeze_end + seconds * tickrate

    before_round_end = target_tick < round_end

    before_plant = (
        plant_tick is None
        or target_tick < plant_tick
    )

    eligible = before_round_end and before_plant

    print(
        f"{seconds:>2}s -> tick {target_tick} "
        f"| eligible = {eligible}"
    )

    if not eligible:
        continue

    snapshot = (
        demo.ticks
        .filter(
            (pl.col("round_num") == ROUND_NUM)
            & (pl.col("tick") == target_tick)
        )
        .select([
            "tick",
            "name",
            "side",
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
        .sort(["side", "name"])
    )

    print(f"\n--- {seconds} SECOND SNAPSHOT ---")
    print(snapshot)
