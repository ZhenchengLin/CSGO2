import polars as pl
from awpy import Demo


DEMO_PATH = "data/raw/havu-vs-mellren-m1-mirage.dem"

demo = Demo(DEMO_PATH, verbose=True)
demo.parse()


plants = (
    demo.bomb
    .filter(pl.col("event") == "plant")
    .select([
        "round_num",
        "tick",
        "bombsite",
    ])
)


plant_counts = (
    plants
    .group_by("round_num")
    .len()
)

assert plant_counts["len"].max() <= 1


labels = (
    demo.rounds
    .select([
        "round_num",
        "freeze_end",
        "end",
    ])
    .join(
        plants.rename({
            "tick": "plant_tick",
        }),
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


print(labels)

print("\nLABEL COUNTS")
print(
    labels
    .group_by("label")
    .len()
    .sort("label")
)
