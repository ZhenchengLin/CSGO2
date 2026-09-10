import polars as pl


FEATURE_PATH = "data/processed/v0_features_a1.parquet"


pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(240)
pl.Config.set_tbl_rows(50)


features = pl.read_parquet(FEATURE_PATH)


# --------------------------------------------------
# 1. Round 1 evolution
# --------------------------------------------------

print("\n" + "=" * 100)
print("ROUND 1 — A1 FEATURE EVOLUTION")
print("=" * 100)

round_1 = (
    features
    .filter(
        pl.col("round_num") == 1
    )
    .select([
        "round_num",
        "horizon_sec",
        "label",

        "t_centroid_x",
        "t_centroid_y",

        "t_stretch_xy",

        "t_range_x",
        "t_range_y",

        "t_mean_pairwise_distance",
        "t_convex_hull_area",

        "bomb_x",
        "bomb_y",

        "bomb_to_t_centroid_distance",
    ])
    .sort("horizon_sec")
)

print(round_1)


# --------------------------------------------------
# 2. Basic numerical ranges
# --------------------------------------------------

print("\n" + "=" * 100)
print("FEATURE RANGES")
print("=" * 100)

numeric_features = [
    "t_centroid_x",
    "t_centroid_y",
    "t_stretch_xy",
    "t_range_x",
    "t_range_y",
    "t_mean_pairwise_distance",
    "t_convex_hull_area",
    "bomb_x",
    "bomb_y",
    "bomb_to_t_centroid_distance",
]

for feature in numeric_features:

    print(
        f"{feature:32s}"
        f" min={features[feature].min():10.2f}"
        f" mean={features[feature].mean():10.2f}"
        f" max={features[feature].max():10.2f}"
    )


# --------------------------------------------------
# 3. Check suspicious geometry
# --------------------------------------------------

print("\n" + "=" * 100)
print("GEOMETRY QA")
print("=" * 100)

checks = {
    "negative_stretch":
        features.filter(
            pl.col("t_stretch_xy") < 0
        ).height,

    "negative_range_x":
        features.filter(
            pl.col("t_range_x") < 0
        ).height,

    "negative_range_y":
        features.filter(
            pl.col("t_range_y") < 0
        ).height,

    "negative_pairwise":
        features.filter(
            pl.col("t_mean_pairwise_distance") < 0
        ).height,

    "negative_hull_area":
        features.filter(
            pl.col("t_convex_hull_area") < 0
        ).height,

    "negative_bomb_distance":
        features.filter(
            pl.col("bomb_to_t_centroid_distance") < 0
        ).height,
}

for name, count in checks.items():
    print(f"{name:30s}: {count}")


# --------------------------------------------------
# 4. Biggest / smallest formations
# --------------------------------------------------

print("\n" + "=" * 100)
print("MOST SPREAD T FORMATIONS")
print("=" * 100)

print(
    features
    .select([
        "round_num",
        "horizon_sec",
        "label",
        "t_stretch_xy",
        "t_mean_pairwise_distance",
        "t_convex_hull_area",
    ])
    .sort(
        "t_stretch_xy",
        descending=True,
    )
    .head(5)
)


print("\n" + "=" * 100)
print("MOST COMPACT T FORMATIONS")
print("=" * 100)

print(
    features
    .select([
        "round_num",
        "horizon_sec",
        "label",
        "t_stretch_xy",
        "t_mean_pairwise_distance",
        "t_convex_hull_area",
    ])
    .sort("t_stretch_xy")
    .head(5)
)
