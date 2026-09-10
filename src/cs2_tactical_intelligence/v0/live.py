"""
Live V0 inference engine.

Responsibilities
----------------
1. Maintain recent canonical V0Frames.
2. Detect freeze_end from freezetime -> live.
3. Trigger only frozen V0 horizons:
   10 / 20 / 30 / 40 seconds.
4. Retrieve approximately one-second historical state.
5. Build runtime-safe features.
6. Run V0Predictor.

This module does not know anything about GSI JSON.
It consumes canonical V0Frame objects only.
"""

from collections import deque
from dataclasses import dataclass

from cs2_tactical_intelligence.v0.features import (
    HORIZONS_SEC,
    build_feature_row,
)

from cs2_tactical_intelligence.v0.predictor import (
    V0Predictor,
)

from cs2_tactical_intelligence.v0.state import (
    V0Frame,
    frame_to_snapshot,
)


@dataclass(frozen=True)
class V0LivePrediction:
    round_num: int
    horizon_sec: int

    target_timestamp_sec: float
    observation_timestamp_sec: float

    horizon_lateness_sec: float
    motion_window_sec: float

    prediction: str
    confidence: float

    p_a_plant: float
    p_b_plant: float
    p_no_plant: float


class V0FrameHistory:
    """
    Small causal history buffer.

    Frames must arrive in nondecreasing timestamp order.
    """

    def __init__(
        self,
        max_age_sec=45.0,
    ):
        self.max_age_sec = float(
            max_age_sec
        )

        self.frames = deque()

    def clear(self):
        self.frames.clear()

    def add(
        self,
        frame: V0Frame,
    ):
        if (
            self.frames
            and frame.timestamp_sec
            < self.frames[-1].timestamp_sec
        ):
            raise ValueError(
                "V0Frame timestamps must be monotonic"
            )

        self.frames.append(
            frame
        )

        cutoff = (
            frame.timestamp_sec
            - self.max_age_sec
        )

        while (
            self.frames
            and self.frames[0].timestamp_sec
            < cutoff
        ):
            self.frames.popleft()

    def latest_at_or_before(
        self,
        *,
        timestamp_sec,
        round_num,
    ):
        for frame in reversed(
            self.frames
        ):
            if (
                frame.round_num
                != round_num
            ):
                continue

            if (
                frame.timestamp_sec
                <= timestamp_sec
            ):
                return frame

        return None


class V0LiveEngine:
    """
    Strict development version of the live engine.

    If timing quality is insufficient, fail loudly rather than
    silently producing a semantically different V0 feature.
    """

    def __init__(
        self,
        predictor=None,
        *,
        horizons_sec=HORIZONS_SEC,
        max_horizon_lateness_sec=0.25,
        max_motion_window_error_sec=0.25,
    ):
        self.predictor = (
            predictor
            if predictor is not None
            else V0Predictor()
        )

        self.horizons_sec = tuple(
            horizons_sec
        )

        self.max_horizon_lateness_sec = float(
            max_horizon_lateness_sec
        )

        self.max_motion_window_error_sec = float(
            max_motion_window_error_sec
        )

        self.history = (
            V0FrameHistory()
        )

        self.current_round_num = None
        self.previous_phase = None

        self.saw_freezetime = False
        self.freeze_end_timestamp_sec = None

        self.emitted_horizons = set()

    # ==================================================
    # Round lifecycle
    # ==================================================

    def _reset_round(
        self,
        round_num,
    ):
        self.current_round_num = (
            round_num
        )

        self.previous_phase = None

        self.saw_freezetime = False
        self.freeze_end_timestamp_sec = None

        self.emitted_horizons = set()

        self.history.clear()

    # ==================================================
    # One incoming canonical frame
    # ==================================================

    def process_frame(
        self,
        frame: V0Frame,
    ):
        """
        Process one causal live frame.

        Returns zero or more V0LivePrediction objects.
        """

        if (
            self.current_round_num
            != frame.round_num
        ):
            self._reset_round(
                frame.round_num
            )

        old_phase = (
            self.previous_phase
        )

        self.history.add(
            frame
        )

        # ------------------------------------------
        # Observe freeze time
        # ------------------------------------------

        if (
            frame.phase
            == "freezetime"
        ):
            self.saw_freezetime = True

        # ------------------------------------------
        # Detect freeze_end
        #
        # We require seeing the transition.
        # If the program starts halfway through a
        # round, we refuse to guess freeze_end.
        # ------------------------------------------

        if (
            frame.phase == "live"
            and old_phase == "freezetime"
            and self.saw_freezetime
            and self.freeze_end_timestamp_sec
            is None
        ):
            self.freeze_end_timestamp_sec = (
                frame.timestamp_sec
            )

        self.previous_phase = (
            frame.phase
        )

        if (
            self.freeze_end_timestamp_sec
            is None
        ):
            return []

        # V0 predictions are pre-plant/live-round only.
        if frame.phase != "live":
            return []

        if (
            frame.bomb.state.lower()
            in {
                "planted",
                "defused",
                "exploded",
            }
        ):
            return []

        results = []

        # ==================================================
        # Frozen V0 horizons
        # ==================================================

        for horizon_sec in (
            self.horizons_sec
        ):

            if (
                horizon_sec
                in self.emitted_horizons
            ):
                continue

            target_timestamp = (
                self.freeze_end_timestamp_sec
                + horizon_sec
            )

            if (
                frame.timestamp_sec
                < target_timestamp
            ):
                continue

            # First causal frame received at/after
            # the nominal horizon.
            horizon_lateness = (
                frame.timestamp_sec
                - target_timestamp
            )

            if (
                horizon_lateness
                > self.max_horizon_lateness_sec
            ):
                raise RuntimeError(
                    f"Live frame too late for "
                    f"{horizon_sec}s horizon: "
                    f"{horizon_lateness:.3f}s"
                )

            # ------------------------------------------
            # Find state ~1 second earlier
            # ------------------------------------------

            desired_previous_time = (
                frame.timestamp_sec
                - 1.0
            )

            previous = (
                self.history
                .latest_at_or_before(
                    timestamp_sec=(
                        desired_previous_time
                    ),
                    round_num=(
                        frame.round_num
                    ),
                )
            )

            if previous is None:
                raise RuntimeError(
                    "No historical frame available "
                    "for V0 motion calculation"
                )

            motion_window_sec = (
                frame.timestamp_sec
                - previous.timestamp_sec
            )

            motion_window_error = abs(
                motion_window_sec
                - 1.0
            )

            if (
                motion_window_error
                > self.max_motion_window_error_sec
            ):
                raise RuntimeError(
                    "Live motion history is too sparse: "
                    f"Δt={motion_window_sec:.3f}s"
                )

            # ------------------------------------------
            # Canonical state -> shared FeatureBuilder
            # ------------------------------------------

            current_snapshot = (
                frame_to_snapshot(
                    frame
                )
            )

            previous_snapshot = (
                frame_to_snapshot(
                    previous
                )
            )

            feature_row = (
                build_feature_row(
                    horizon_sec=horizon_sec,

                    current=(
                        current_snapshot
                    ),

                    previous=(
                        previous_snapshot
                    ),

                    bomb_now=(
                        frame.bomb.x,
                        frame.bomb.y,
                        frame.bomb.z,
                    ),

                    bomb_previous=(
                        previous.bomb.x,
                        previous.bomb.y,
                        previous.bomb.z,
                    ),

                    motion_window_sec=(
                        motion_window_sec
                    ),
                )
            )

            prediction = (
                self.predictor.predict(
                    feature_row
                )
            )

            result = V0LivePrediction(
                round_num=(
                    frame.round_num
                ),

                horizon_sec=(
                    horizon_sec
                ),

                target_timestamp_sec=(
                    target_timestamp
                ),

                observation_timestamp_sec=(
                    frame.timestamp_sec
                ),

                horizon_lateness_sec=(
                    horizon_lateness
                ),

                motion_window_sec=(
                    motion_window_sec
                ),

                prediction=(
                    prediction[
                        "prediction"
                    ]
                ),

                confidence=(
                    prediction[
                        "confidence"
                    ]
                ),

                p_a_plant=(
                    prediction[
                        "p_a_plant"
                    ]
                ),

                p_b_plant=(
                    prediction[
                        "p_b_plant"
                    ]
                ),

                p_no_plant=(
                    prediction[
                        "p_no_plant"
                    ]
                ),
            )

            results.append(
                result
            )

            self.emitted_horizons.add(
                horizon_sec
            )

        return results
