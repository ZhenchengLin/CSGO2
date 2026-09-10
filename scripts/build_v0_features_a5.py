import math
from itertools import combinations

import numpy as np
import polars as pl
from scipy.spatial import ConvexHull, QhullError


PLAYER_PATH = "data/interim/v0_snapshot_players.parquet"
A4_PATH = "data/processed/v0_features_a4.parquet"
OUTPUT_PATH = "data/processed/v0_features_a5.parquet"


players = pl.read_parquet(PLAYER_PATH)
a4 = pl.read_parquet(A4_PATH)


# --------------------------------------------------
# Geometry helpers
# --------------------------------------------------

def centroid(team):

    return (
        team["X"].mean(),
        team["Y"].mean(),
        team["Z"].mean(),
    )


def stretch_xy(team, cx, cy):

    distances = []

    for row in team.iter_rows(named=True):

        dx = row["X"] - cx
        dy = row["Y"] - cy

        distances.append(
            math.sqrt(dx * dx + dy * dy)
        )

    return sum(distances) / len(distances)


def mean_pairwise_distance(team):

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

    if team.height < 3:
        return 0.0

    points = np.array([
        [row["X"], row["Y"]]
        for row in team.iter_rows(named=True)
    ])

    try:

        hull = ConvexHull(points)

        return float(hull.volume)

    except QhullError:

        return 0.0


def distance_xy(x1, y1, x2, y2):

    return math.sqrt(
        (x1 - x2) ** 2
        + (y1 - y2) ** 2
    )


# --------------------------------------------------
# Build defensive + interaction features
# --------------------------------------------------

rows = []


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


    t_alive = group.filter(
        (pl.col("side") == "t")
        & (pl.col("alive"))
    )

    ct_alive = group.filter(
        (pl.col("side") == "ct")
        & (pl.col("alive"))
    )


    # --------------------------------------------------
    # CT formation
    # --------------------------------------------------

    if ct_alive.height > 0:

        ct_cx, ct_cy, ct_cz = centroid(
            ct_alive
        )

        ct_stretch = stretch_xy(
            ct_alive,
            ct_cx,
            ct_cy,
        )

        ct_range_x = (
            ct_alive["X"].max()
            - ct_alive["X"].min()
        )

        ct_range_y = (
            ct_alive["Y"].max()
            - ct_alive["Y"].min()
        )

        ct_pairwise = mean_pairwise_distance(
            ct_alive
        )

        ct_hull = convex_hull_area_xy(
            ct_alive
        )

    else:

        # No surviving defenders.
        # Geometry no longer exists.
        ct_cx = 0.0
        ct_cy = 0.0
        ct_cz = 0.0

        ct_stretch = 0.0
        ct_range_x = 0.0
        ct_range_y = 0.0
        ct_pairwise = 0.0
        ct_hull = 0.0


    # --------------------------------------------------
    # T centroid
    # --------------------------------------------------

    t_cx, t_cy, _ = centroid(
        t_alive
    )


    # --------------------------------------------------
    # T ↔ CT centroid distance
    # --------------------------------------------------

    if ct_alive.height > 0:

        centroid_distance = distance_xy(
            t_cx,
            t_cy,
            ct_cx,
            ct_cy,
        )

    else:

        centroid_distance = 0.0


    # --------------------------------------------------
    # Pairwise attacker-defender distances
    # --------------------------------------------------

    cross_distances = []

    nearest_distances = []


    if ct_alive.height > 0:

        for t_player in t_alive.iter_rows(
            named=True
        ):

            distances_to_ct = []

            for ct_player in ct_alive.iter_rows(
                named=True
            ):

                d = distance_xy(
                    t_player["X"],
                    t_player["Y"],
                    ct_player["X"],
                    ct_player["Y"],
                )

                cross_distances.append(d)
                distances_to_ct.append(d)

            nearest_distances.append(
                min(distances_to_ct)
            )


        minimum_t_ct_distance = min(
            cross_distances
        )

        mean_nearest_opponent_distance = (
            sum(nearest_distances)
            / len(nearest_distances)
        )

    else:

        minimum_t_ct_distance = 0.0
        mean_nearest_opponent_distance = 0.0


    rows.append({

        "round_num": round_num,
        "horizon_sec": horizon_sec,
        "target_tick": target_tick,

        # Defensive geometry
        "ct_centroid_x": ct_cx,
        "ct_centroid_y": ct_cy,
        "ct_centroid_z": ct_cz,

        "ct_stretch_xy": ct_stretch,

        "ct_range_x": ct_range_x,
        "ct_range_y": ct_range_y,

        "ct_mean_pairwise_distance":
            ct_pairwise,

        "ct_convex_hull_area":
            ct_hull,

        # Inter-team geometry
        "t_ct_centroid_distance":
            centroid_distance,

        "minimum_t_ct_distance":
            minimum_t_ct_distance,

        "mean_nearest_opponent_distance":
            mean_nearest_opponent_distance,
    })


defense = pl.DataFrame(rows)


# --------------------------------------------------
# Join onto A4
# --------------------------------------------------

a5 = (
    a4
    .join(
        defense,
        on=[
            "round_num",
            "horizon_sec",
            "target_tick",
        ],
        how="left",
    )
)


new_features = [
    "ct_centroid_x",
    "ct_centroid_y",
    "ct_centroid_z",
    "ct_stretch_xy",
    "ct_range_x",
    "ct_range_y",
    "ct_mean_pairwise_distance",
    "ct_convex_hull_area",
    "t_ct_centroid_distance",
    "minimum_t_ct_distance",
    "mean_nearest_opponent_distance",
]


# --------------------------------------------------
# QA
# --------------------------------------------------

print("\n" + "=" * 100)
print("A5 DEFENSE + INTERACTION DATASET")
print("=" * 100)

print("Shape:", a5.shape)


print("\n" + "=" * 100)
print("NULL COUNTS — A5")
print("=" * 100)

print(
    a5
    .select(new_features)
    .null_count()
)


print("\n" + "=" * 100)
print("ROUND 1 DEFENSIVE EVOLUTION")
print("=" * 100)

print(
    a5
    .filter(
        pl.col("round_num") == 1
    )
    .select([
        "round_num",
        "horizon_sec",
        "label",

        "t_alive",
        "ct_alive",

        "ct_centroid_x",
        "ct_centroid_y",

        "ct_stretch_xy",

        "t_ct_centroid_distance",
        "minimum_t_ct_distance",
        "mean_nearest_opponent_distance",
    ])
    .sort("horizon_sec")
)


print("\n" + "=" * 100)
print("INTERACTION FEATURE RANGES")
print("=" * 100)

for feature in [
    "t_ct_centroid_distance",
    "minimum_t_ct_distance",
    "mean_nearest_opponent_distance",
]:

    print(
        f"{feature:35s}"
        f" min={a5[feature].min():10.2f}"
        f" mean={a5[feature].mean():10.2f}"
        f" max={a5[feature].max():10.2f}"
    )


# --------------------------------------------------
# Safety checks
# --------------------------------------------------

assert a5.height == 49

assert (
    a5
    .select(new_features)
    .null_count()
    .row(0)
    == tuple(
        0 for _ in new_features
    )
)

for feature in [
    "ct_stretch_xy",
    "ct_range_x",
    "ct_range_y",
    "ct_mean_pairwise_distance",
    "ct_convex_hull_area",
    "t_ct_centroid_distance",
    "minimum_t_ct_distance",
    "mean_nearest_opponent_distance",
]:

    assert a5[feature].min() >= 0


# --------------------------------------------------
# Save
# --------------------------------------------------

a5.write_parquet(
    OUTPUT_PATH
)

print("\nSaved:")
print(OUTPUT_PATH)
