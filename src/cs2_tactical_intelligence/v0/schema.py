"""
Frozen feature contract for the V0 XGB-A5 model.

V0 evaluation is frozen. Runtime inference must use exactly
these features, in exactly this order.

Economy features are intentionally excluded because the V0
ablation did not support their inclusion.
"""


V0_LABELS = [
    "A_PLANT",
    "B_PLANT",
    "NO_PLANT",
]


V0_FEATURE_GROUPS = {

    "horizon": [
        "horizon_sec",
    ],

    "offensive_geometry": [
        "t_centroid_x",
        "t_centroid_y",
        "t_centroid_z",
        "t_stretch_xy",
        "t_range_x",
        "t_range_y",
        "t_mean_pairwise_distance",
        "t_convex_hull_area",
        "bomb_x",
        "bomb_y",
        "bomb_z",
        "bomb_to_t_centroid_distance",
    ],

    "motion": [
        "t_mean_speed_1s",
        "t_centroid_velocity_x_1s",
        "t_centroid_velocity_y_1s",
        "bomb_speed_1s",
    ],

    "combat": [
        "t_alive",
        "ct_alive",
        "alive_difference",
        "t_health_sum",
        "ct_health_sum",
        "health_difference",
        "t_armor_sum",
        "ct_armor_sum",
        "armor_difference",
    ],

    "defense": [
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
    ],
}


V0_MODEL_FEATURES = [
    feature
    for group in V0_FEATURE_GROUPS.values()
    for feature in group
]


V0_N_FEATURES = len(
    V0_MODEL_FEATURES
)


assert V0_N_FEATURES == 37


# ==================================================
# Canonical V0 dataframe dtypes
# ==================================================

V0_INTEGER_COLUMNS = [
    "round_num",
    "horizon_sec",
    "target_tick",
    "t_alive",
    "ct_alive",
    "alive_difference",
    "qa_motion_players_used",
]


V0_FLOAT_COLUMNS = [
    # A1
    "t_centroid_x",
    "t_centroid_y",
    "t_centroid_z",
    "t_stretch_xy",
    "t_range_x",
    "t_range_y",
    "t_mean_pairwise_distance",
    "t_convex_hull_area",
    "bomb_x",
    "bomb_y",
    "bomb_z",
    "bomb_to_t_centroid_distance",

    # A2
    "t_mean_speed_1s",
    "t_centroid_velocity_x_1s",
    "t_centroid_velocity_y_1s",
    "bomb_speed_1s",

    # A3
    "t_health_sum",
    "ct_health_sum",
    "health_difference",
    "t_armor_sum",
    "ct_armor_sum",
    "armor_difference",

    # A4
    "t_equip_value_sum",
    "ct_equip_value_sum",
    "equip_value_difference",

    # A5
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


V0_STRING_COLUMNS = [
    "demo_filename",
    "qa_bomb_state_prev",
    "qa_bomb_state_now",
    "label",
]


def normalize_v0_frame(frame):
    """
    Normalize a V0 dataframe to the canonical dtype contract.

    Columns not present in the dataframe are ignored.

    This allows the same function to normalize:

    - offline training rows
    - replay rows
    - future live inference rows
    """

    import polars as pl

    expressions = []

    for column in V0_INTEGER_COLUMNS:
        if column in frame.columns:
            expressions.append(
                pl.col(column).cast(pl.Int64)
            )

    for column in V0_FLOAT_COLUMNS:
        if column in frame.columns:
            expressions.append(
                pl.col(column).cast(pl.Float64)
            )

    for column in V0_STRING_COLUMNS:
        if column in frame.columns:
            expressions.append(
                pl.col(column).cast(pl.String)
            )

    return frame.with_columns(
        expressions
    )
