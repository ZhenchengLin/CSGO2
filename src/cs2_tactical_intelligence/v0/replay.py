"""
Historical replay adapter for V0.

Purpose
-------
Replay a historical CS2 demo through a live-like feature path.

The adapter may use historical demo files as its source, but feature
generation at observation time uses only state available at or before
that observation.

It does NOT use the final A/B/NO_PLANT label to build model features.
"""

from pathlib import Path

import polars as pl
from awpy import Demo

from cs2_tactical_intelligence.v0.schema import (
    normalize_v0_frame,
)

from cs2_tactical_intelligence.v0.features import (
    HORIZONS_SEC,
    MOTION_WINDOW_SEC,
    V0_PLAYER_PROPS,
    build_feature_row,
    build_snapshot_index,
    get_bomb_position,
    require_snapshot,
)


class V0ReplayAdapter:

    def __init__(
        self,
        demo_path,
    ):
        self.demo_path = Path(
            demo_path
        )

        self.demo = Demo(
            str(self.demo_path),
            verbose=False,
        )

        self.demo.parse(
            player_props=V0_PLAYER_PROPS
        )

        map_name = self.demo.header.get(
            "map_name"
        )

        if map_name != "de_mirage":
            raise ValueError(
                f"V0 expects de_mirage, "
                f"got {map_name}"
            )

    # ==================================================
    # Replay observation schedule
    # ==================================================

    def build_observations(self):
        """
        Construct V0 replay observation times.

        Important:
        No final plant-site label is created here.

        A prediction is produced only if, at that moment:

        - the round is still active
        - the bomb has not already been planted

        The historical round end is used here only to know whether
        the requested replay timestamp exists inside the round.

        A future live adapter will instead receive round lifecycle
        events incrementally.
        """

        observations = []
        invalid_rounds = []

        for round_row in (
            self.demo.rounds
            .sort("round_num")
            .iter_rows(named=True)
        ):
            round_num = round_row[
                "round_num"
            ]

            freeze_end = round_row[
                "freeze_end"
            ]

            round_end = round_row[
                "end"
            ]

            # ------------------------------------------
            # Timing validation
            # ------------------------------------------

            if freeze_end is None:
                invalid_rounds.append({
                    "round_num":
                        round_num,

                    "reason":
                        "MISSING_FREEZE_END",
                })

                continue

            if round_end is None:
                invalid_rounds.append({
                    "round_num":
                        round_num,

                    "reason":
                        "MISSING_ROUND_END",
                })

                continue

            if freeze_end >= round_end:
                invalid_rounds.append({
                    "round_num":
                        round_num,

                    "reason":
                        "INVALID_TIMING_ORDER",
                })

                continue

            # ------------------------------------------
            # Observation horizons
            # ------------------------------------------

            for horizon_sec in HORIZONS_SEC:

                target_tick = (
                    freeze_end
                    + int(
                        round(
                            horizon_sec
                            * self.demo.tickrate
                        )
                    )
                )

                # If the round has already ended,
                # there is no live observation here.
                if target_tick >= round_end:
                    continue

                # Determine whether a plant has already
                # happened BY the observation time.
                #
                # This is past/current information,
                # not future label information.
                plants_so_far = (
                    self.demo.bomb
                    .filter(
                        (pl.col("round_num") == round_num)
                        & (pl.col("event") == "plant")
                        & (pl.col("tick") <= target_tick)
                    )
                )

                if plants_so_far.height > 0:
                    continue

                observations.append({
                    "round_num":
                        round_num,

                    "horizon_sec":
                        horizon_sec,

                    "target_tick":
                        target_tick,
                })

        return (
            observations,
            invalid_rounds,
        )

    # ==================================================
    # Replay feature generation
    # ==================================================

    def build_features(self):
        """
        Generate V0 feature rows through the replay path.

        No training label is produced.
        """

        (
            observations,
            invalid_rounds,
        ) = self.build_observations()

        snapshot_index = (
            build_snapshot_index(
                self.demo,
                observations,
            )
        )

        lag_ticks = int(
            round(
                MOTION_WINDOW_SEC
                * self.demo.tickrate
            )
        )

        rows = []

        for obs in observations:

            round_num = obs[
                "round_num"
            ]

            horizon_sec = obs[
                "horizon_sec"
            ]

            target_tick = obs[
                "target_tick"
            ]

            previous_tick = (
                target_tick
                - lag_ticks
            )

            current = require_snapshot(
                snapshot_index,
                round_num,
                target_tick,
            )

            previous = require_snapshot(
                snapshot_index,
                round_num,
                previous_tick,
            )

            # ------------------------------------------
            # Demo-specific bomb state adapter
            # ------------------------------------------

            (
                bomb_x,
                bomb_y,
                bomb_z,
                bomb_state_now,
            ) = get_bomb_position(
                self.demo,
                current,
                round_num,
                target_tick,
            )

            (
                bomb_x_previous,
                bomb_y_previous,
                bomb_z_previous,
                bomb_state_previous,
            ) = get_bomb_position(
                self.demo,
                previous,
                round_num,
                previous_tick,
            )

            # ------------------------------------------
            # Shared runtime-safe feature builder
            # ------------------------------------------

            feature_row = build_feature_row(
                horizon_sec=horizon_sec,

                current=current,

                previous=previous,

                bomb_now=(
                    bomb_x,
                    bomb_y,
                    bomb_z,
                ),

                bomb_previous=(
                    bomb_x_previous,
                    bomb_y_previous,
                    bomb_z_previous,
                ),
            )

            rows.append({

                "demo_filename":
                    self.demo_path.name,

                "round_num":
                    round_num,

                "horizon_sec":
                    horizon_sec,

                "target_tick":
                    target_tick,

                **feature_row,

                "qa_bomb_state_prev":
                    bomb_state_previous,

                "qa_bomb_state_now":
                    bomb_state_now,
            })

        features = pl.DataFrame(
            rows
        )

        features = normalize_v0_frame(
            features
        )

        return (
            features,
            invalid_rounds,
        )
