import math
from itertools import combinations

import numpy as np
import polars as pl
from scipy.spatial import ConvexHull, QhullError


PLAYER_PATH = "data/interim/v0_snapshot_players.parquet"
BOMB_PATH = "data/interim/v0_bomb_state_audit.parquet"
OUTPUT_PATH = "data/processed/v0_features_a1.parquet"


# --------------------------------------------------
# 1. Load observation-level source data
# --------------------------------------------------

players = pl.read_parquet(PLAYER_PATH)

bomb_states = (
    pl.read_parquet(BOMB_PATH)
    .select([
        "round_num",
        "horizon_sec",
        "target_tick",
        "bomb_state",
        "bomb_x",
        "bomb_y",
        "bomb_z",
    ])
)


# --------------------------------------------------
# 2. Geometry helpers
# --------------------------------------------------

def centroid(team):
    return (
        team["X"].mean(),
        team["Y"].mean(),
        team["Z"].mean(),
    )


def stretch_xy(team, cx, cy):
    """
    Mean distance from each alive player
    to the team XY centroid.
    """

    distances = []

    for row in team.iter_rows(named=True):
        dx = row["X"] - cx
        dy = row["Y"] - cy

        distances.append(
            math.sqrt(dx * dx + dy * dy)
        )

    return sum(distances) / len(distances)


def mean_pairwise_distance(team):
    """
    Mean XY distance across all unique
    alive-player pairs.
    """

    points = [
        (row["X"], row["Y"])
        for row in team.iter_rows(named=True)
    ]

    if len(points) < 2:
        return 0.0

    distances = []

    for p1, p2 in combinations(points, 2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]

        distances.append(
            math.sqrt(dx * dx + dy * dy)
        )

    return sum(distances) / len(distances)


def convex_hull_area_xy(team):
    """
    Area occupied by alive T players in XY.

    Fewer than 3 players cannot form
    a 2D polygon, so area = 0.
    """

    if team.height < 3:
        return 0.0

    points = np.array([
        [row["X"], row["Y"]]
        for row in team.iter_rows(named=True)
    ])

    try:
        hull = ConvexHull(points)

        # In scipy ConvexHull for 2D:
        # .volume = enclosed polygon area
        return float(hull.volume)

    except QhullError:
        # Collinear / degenerate configuration
        return 0.0


# --------------------------------------------------
# 3. Build one feature row per observation
# --------------------------------------------------

feature_rows = []


groups = players.partition_by(
    [
        "round_num",
        "horizon_sec",
        "target_tick",
        "label",
    ],
    maintain_order=True,
)


for group in groups:

    round_num = group["round_num"][0]
    horizon_sec = group["horizon_sec"][0]
    target_tick = group["target_tick"][0]
    label = group["label"][0]


    # --------------------------------------------------
    # Offensive formation:
    # only alive T players
    # --------------------------------------------------

    t_alive = (
        group
        .filter(
            (pl.col("side") == "t")
            & (pl.col("alive"))
        )
    )


    if t_alive.height == 0:
        raise ValueError(
            f"No alive T players at "
            f"round={round_num}, horizon={horizon_sec}"
        )


    # --------------------------------------------------
    # T centroid
    # --------------------------------------------------

    t_cx, t_cy, t_cz = centroid(t_alive)


    # --------------------------------------------------
    # T stretch
    # --------------------------------------------------

    t_stretch = stretch_xy(
        t_alive,
        t_cx,
        t_cy,
    )


    # --------------------------------------------------
    # T coordinate ranges
    # --------------------------------------------------

    t_range_x = (
        t_alive["X"].max()
        - t_alive["X"].min()
    )

    t_range_y = (
        t_alive["Y"].max()
        - t_alive["Y"].min()
    )


    # --------------------------------------------------
    # Pairwise distance
    # --------------------------------------------------

    t_pairwise = mean_pairwise_distance(
        t_alive
    )


    # --------------------------------------------------
    # Convex hull area
    # --------------------------------------------------

    t_hull_area = convex_hull_area_xy(
        t_alive
    )


    # --------------------------------------------------
    # Bomb state for this observation
    # --------------------------------------------------

    bomb = (
        bomb_states
        .filter(
            (pl.col("round_num") == round_num)
            & (pl.col("horizon_sec") == horizon_sec)
            & (pl.col("target_tick") == target_tick)
        )
    )


    if bomb.height != 1:
        raise ValueError(
            f"Expected exactly one bomb state for "
            f"round={round_num}, horizon={horizon_sec}"
        )


    bomb_x = bomb["bomb_x"][0]
    bomb_y = bomb["bomb_y"][0]
    bomb_z = bomb["bomb_z"][0]


    # --------------------------------------------------
    # Bomb ↔ T centroid distance
    # --------------------------------------------------

    bomb_to_t_centroid = math.sqrt(
        (bomb_x - t_cx) ** 2
        + (bomb_y - t_cy) ** 2
    )


    # --------------------------------------------------
    # Final A1 feature row
    # --------------------------------------------------

    feature_rows.append({

        # Metadata
        "round_num": round_num,
        "horizon_sec": horizon_sec,
        "target_tick": target_tick,

        # F1: Offensive spatial structure
        "t_centroid_x": t_cx,
        "t_centroid_y": t_cy,
        "t_centroid_z": t_cz,

        "t_stretch_xy": t_stretch,

        "t_range_x": t_range_x,
        "t_range_y": t_range_y,

        "t_mean_pairwise_distance": t_pairwise,

        "t_convex_hull_area": t_hull_area,

        # F7: Objective geometry
        "bomb_x": bomb_x,
        "bomb_y": bomb_y,
        "bomb_z": bomb_z,

        "bomb_to_t_centroid_distance":
            bomb_to_t_centroid,

        # Target
        "label": label,
    })


features = pl.DataFrame(feature_rows)


# --------------------------------------------------
# 4. QA
# --------------------------------------------------

print("\n" + "=" * 90)
print("A1 OFFENSIVE GEOMETRY DATASET")
print("=" * 90)

print("Shape:", features.shape)

print("\nColumns:")
for column in features.columns:
    print(column)


print("\n" + "=" * 90)
print("NULL COUNTS")
print("=" * 90)

print(features.null_count())


print("\n" + "=" * 90)
print("ROUND 1 FEATURE EVOLUTION")
print("=" * 90)

print(
    features
    .filter(
        pl.col("round_num") == 1
    )
    .sort("horizon_sec")
)


# --------------------------------------------------
# 5. Safety checks
# --------------------------------------------------

assert features.height == 49

assert (
    features
    .null_count()
    .row(0)
    == tuple(
        0 for _ in features.columns
    )
)


# --------------------------------------------------
# 6. Save
# --------------------------------------------------

features.write_parquet(OUTPUT_PATH)


print("\nSaved:")
print(OUTPUT_PATH)
