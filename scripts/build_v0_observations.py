import polars as pl
from awpy import Demo


DEMO_PATH = "data/raw/havu-vs-mellren-m1-mirage.dem"
HORIZONS = [10, 20, 30, 40]


demo = Demo(DEMO_PATH, verbose=True)
demo.parse()

tickrate = demo.tickrate


# --------------------------------------------------
# Build labels from real bomb plant events
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
# Build observation candidates
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

        eligible = (
            before_round_end
            and before_plant
        )

        if not before_round_end:
            reason = "ROUND_ALREADY_ENDED"
        elif not before_plant:
            reason = "PLANT_ALREADY_HAPPENED"
        else:
            reason = "VALID"

        rows.append({
            "round_num": round_row["round_num"],
            "horizon_sec": horizon_sec,
            "target_tick": target_tick,
            "plant_tick": round_row["plant_tick"],
            "round_end": round_row["end"],
            "label": round_row["label"],
            "eligible": eligible,
            "reason": reason,
        })


observations = pl.DataFrame(rows)


print("\n" + "=" * 90)
print("V0 OBSERVATION TABLE")
print("=" * 90)

print(observations)


print("\n" + "=" * 90)
print("ELIGIBLE COUNTS BY HORIZON")
print("=" * 90)

print(
    observations
    .group_by([
        "horizon_sec",
        "eligible",
    ])
    .len()
    .sort([
        "horizon_sec",
        "eligible",
    ])
)


print("\n" + "=" * 90)
print("VALID OBSERVATIONS ONLY")
print("=" * 90)

valid = observations.filter(
    pl.col("eligible")
)

print(valid)


print("\nTOTAL VALID OBSERVATIONS:", valid.height)
