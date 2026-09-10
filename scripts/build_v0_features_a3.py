import polars as pl


PLAYER_PATH = "data/interim/v0_snapshot_players.parquet"
A2_PATH = "data/processed/v0_features_a2.parquet"
OUTPUT_PATH = "data/processed/v0_features_a3.parquet"


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

players = pl.read_parquet(PLAYER_PATH)
a2 = pl.read_parquet(A2_PATH)


# --------------------------------------------------
# 2. Build combat features
# --------------------------------------------------

combat_rows = []


groups = players.partition_by(
    [
        "round_num",
        "horizon_sec",
        "target_tick",
    ],
    maintain_order=True,
)


for group in groups:

    round_num = group["round_num"][0]
    horizon_sec = group["horizon_sec"][0]
    target_tick = group["target_tick"][0]


    # Alive players only
    t_alive_players = group.filter(
        (pl.col("side") == "t")
        & (pl.col("alive"))
    )

    ct_alive_players = group.filter(
        (pl.col("side") == "ct")
        & (pl.col("alive"))
    )


    # --------------------------------------------------
    # Alive counts
    # --------------------------------------------------

    t_alive = t_alive_players.height
    ct_alive = ct_alive_players.height


    # --------------------------------------------------
    # Health
    # --------------------------------------------------

    t_health_sum = t_alive_players["health"].sum()
    ct_health_sum = ct_alive_players["health"].sum()


    # --------------------------------------------------
    # Armor
    #
    # Only surviving players matter for current
    # tactical combat capability.
    # --------------------------------------------------

    t_armor_sum = t_alive_players["armor"].sum()
    ct_armor_sum = ct_alive_players["armor"].sum()


    combat_rows.append({

        "round_num": round_num,
        "horizon_sec": horizon_sec,
        "target_tick": target_tick,

        "t_alive": t_alive,
        "ct_alive": ct_alive,
        "alive_difference":
            t_alive - ct_alive,

        "t_health_sum":
            t_health_sum,

        "ct_health_sum":
            ct_health_sum,

        "health_difference":
            t_health_sum - ct_health_sum,

        "t_armor_sum":
            t_armor_sum,

        "ct_armor_sum":
            ct_armor_sum,

        "armor_difference":
            t_armor_sum - ct_armor_sum,
    })


combat = pl.DataFrame(combat_rows)


# --------------------------------------------------
# 3. Join A3 onto A2
# --------------------------------------------------

a3 = (
    a2
    .join(
        combat,
        on=[
            "round_num",
            "horizon_sec",
            "target_tick",
        ],
        how="left",
    )
)


# --------------------------------------------------
# 4. QA
# --------------------------------------------------

print("\n" + "=" * 100)
print("A3 COMBAT DATASET")
print("=" * 100)

print("Shape:", a3.shape)


print("\n" + "=" * 100)
print("NULL COUNTS — NEW COMBAT FEATURES")
print("=" * 100)

combat_features = [
    "t_alive",
    "ct_alive",
    "alive_difference",
    "t_health_sum",
    "ct_health_sum",
    "health_difference",
    "t_armor_sum",
    "ct_armor_sum",
    "armor_difference",
]

print(
    a3
    .select(combat_features)
    .null_count()
)


# --------------------------------------------------
# 5. Round 1 combat evolution
# --------------------------------------------------

print("\n" + "=" * 100)
print("ROUND 1 COMBAT EVOLUTION")
print("=" * 100)

print(
    a3
    .filter(
        pl.col("round_num") == 1
    )
    .select([
        "round_num",
        "horizon_sec",
        "label",
        *combat_features,
    ])
    .sort("horizon_sec")
)


# --------------------------------------------------
# 6. Explain zero-geometry observations
# --------------------------------------------------

print("\n" + "=" * 100)
print("ZERO STRETCH — COMBAT CONTEXT")
print("=" * 100)

print(
    a3
    .filter(
        pl.col("t_stretch_xy") == 0
    )
    .select([
        "round_num",
        "horizon_sec",
        "label",
        "t_alive",
        "ct_alive",
        "t_health_sum",
        "t_stretch_xy",
        "t_mean_pairwise_distance",
        "t_convex_hull_area",
    ])
    .sort([
        "round_num",
        "horizon_sec",
    ])
)


# --------------------------------------------------
# 7. Safety checks
# --------------------------------------------------

assert a3.height == 49

assert (
    a3
    .select(combat_features)
    .null_count()
    .row(0)
    == tuple(
        0 for _ in combat_features
    )
)

assert a3["t_alive"].min() >= 1
assert a3["t_alive"].max() <= 5

assert a3["ct_alive"].min() >= 0
assert a3["ct_alive"].max() <= 5

assert a3["t_health_sum"].min() >= 1
assert a3["t_health_sum"].max() <= 500

assert a3["ct_health_sum"].min() >= 0
assert a3["ct_health_sum"].max() <= 500


# --------------------------------------------------
# 8. Save
# --------------------------------------------------

a3.write_parquet(OUTPUT_PATH)

print("\nSaved:")
print(OUTPUT_PATH)
