import polars as pl


PLAYER_PATH = "data/interim/v0_snapshot_players.parquet"
A3_PATH = "data/processed/v0_features_a3.parquet"
OUTPUT_PATH = "data/processed/v0_features_a4.parquet"


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

players = pl.read_parquet(PLAYER_PATH)
a3 = pl.read_parquet(A3_PATH)


# --------------------------------------------------
# 2. Build economy features
# --------------------------------------------------

economy_rows = []

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

    # Only alive players contribute to current
    # tactical equipment capability.
    t_alive = group.filter(
        (pl.col("side") == "t")
        & (pl.col("alive"))
    )

    ct_alive = group.filter(
        (pl.col("side") == "ct")
        & (pl.col("alive"))
    )

    t_equip_value_sum = (
        t_alive["current_equip_value"].sum()
    )

    ct_equip_value_sum = (
        ct_alive["current_equip_value"].sum()
    )

    economy_rows.append({
        "round_num": round_num,
        "horizon_sec": horizon_sec,
        "target_tick": target_tick,

        "t_equip_value_sum":
            t_equip_value_sum,

        "ct_equip_value_sum":
            ct_equip_value_sum,

        "equip_value_difference":
            t_equip_value_sum
            - ct_equip_value_sum,
    })


economy = pl.DataFrame(economy_rows)


# --------------------------------------------------
# 3. Join A4 onto A3
# --------------------------------------------------

a4 = (
    a3
    .join(
        economy,
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

economy_features = [
    "t_equip_value_sum",
    "ct_equip_value_sum",
    "equip_value_difference",
]


print("\n" + "=" * 100)
print("A4 ECONOMY DATASET")
print("=" * 100)

print("Shape:", a4.shape)


print("\n" + "=" * 100)
print("NULL COUNTS — ECONOMY")
print("=" * 100)

print(
    a4
    .select(economy_features)
    .null_count()
)


print("\n" + "=" * 100)
print("ROUND 1 ECONOMY EVOLUTION")
print("=" * 100)

print(
    a4
    .filter(
        pl.col("round_num") == 1
    )
    .select([
        "round_num",
        "horizon_sec",
        "label",

        "t_alive",
        "ct_alive",

        "t_equip_value_sum",
        "ct_equip_value_sum",
        "equip_value_difference",
    ])
    .sort("horizon_sec")
)


print("\n" + "=" * 100)
print("ECONOMY FEATURE RANGES")
print("=" * 100)

for feature in economy_features:

    print(
        f"{feature:30s}"
        f" min={a4[feature].min():8}"
        f" mean={a4[feature].mean():10.2f}"
        f" max={a4[feature].max():8}"
    )


# --------------------------------------------------
# 5. Safety checks
# --------------------------------------------------

assert a4.height == 49

assert (
    a4
    .select(economy_features)
    .null_count()
    .row(0)
    == (0, 0, 0)
)

assert a4["t_equip_value_sum"].min() >= 0
assert a4["ct_equip_value_sum"].min() >= 0


# --------------------------------------------------
# 6. Save
# --------------------------------------------------

a4.write_parquet(OUTPUT_PATH)

print("\nSaved:")
print(OUTPUT_PATH)
