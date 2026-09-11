from pathlib import Path

from cs2_tactical_intelligence.v0.predictor import (
    V0Predictor,
)

from cs2_tactical_intelligence.v0.live import (
    V0LiveEngine,
)

from cs2_tactical_intelligence.v0.state import (
    V0BombState,
    V0Frame,
    V0PlayerState,
)


def make_frame(
    timestamp_sec,
    phase,
):
    """
    Deterministic fake live state.

    Players move gradually so motion features
    are non-zero.
    """

    t = timestamp_sec

    players = (
        V0PlayerState(
            steamid="111",
            side="t",
            x=100.0 + 20.0 * t,
            y=200.0 + 5.0 * t,
            z=10.0,
            health=100,
            armor=100,
            current_equip_value=4200,
        ),

        V0PlayerState(
            steamid="222",
            side="t",
            x=150.0 + 15.0 * t,
            y=250.0 + 4.0 * t,
            z=10.0,
            health=100,
            armor=50,
            current_equip_value=3500,
        ),

        V0PlayerState(
            steamid="333",
            side="ct",
            x=-500.0 + 3.0 * t,
            y=-300.0,
            z=20.0,
            health=100,
            armor=100,
            current_equip_value=4800,
        ),

        V0PlayerState(
            steamid="444",
            side="ct",
            x=-450.0,
            y=-350.0 + 2.0 * t,
            z=20.0,
            health=100,
            armor=100,
            current_equip_value=4000,
        ),
    )

    bomb = V0BombState(
        x=100.0 + 20.0 * t,
        y=200.0 + 5.0 * t,
        z=10.0,
        state="carried",
        carrier_steamid="111",
    )

    return V0Frame(
        timestamp_sec=float(
            timestamp_sec
        ),

        round_num=1,

        phase=phase,

        players=players,

        bomb=bomb,
    )


predictor = V0Predictor(
    model_path=Path(
        "artifacts/v0_tc_xgb_a5_runtime.json"
    ),
    metadata_path=Path(
        "artifacts/v0_tc_xgb_a5_runtime_metadata.json"
    ),
)

engine = V0LiveEngine(
    predictor=predictor,
)


# ==================================================
# Observe freeze time
# ==================================================

assert (
    engine.process_frame(
        make_frame(
            -0.1,
            "freezetime",
        )
    )
    == []
)


# ==================================================
# Transition to live at t = 0
# ==================================================

assert (
    engine.process_frame(
        make_frame(
            0.0,
            "live",
        )
    )
    == []
)


assert (
    engine.freeze_end_timestamp_sec
    == 0.0
)


print(
    "✅ freeze_end detected at t=0.0"
)


# ==================================================
# Simulate 0.1 second GSI updates
# ==================================================

predictions = []


for step in range(
    1,
    402,
):
    timestamp = (
        step / 10.0
    )

    results = (
        engine.process_frame(
            make_frame(
                timestamp,
                "live",
            )
        )
    )

    predictions.extend(
        results
    )


print(
    f"\nPredictions emitted: "
    f"{len(predictions)}"
)


assert (
    len(predictions)
    == 4
)


assert [
    result.horizon_sec
    for result in predictions
] == [
    10,
    20,
    30,
    40,
]


# ==================================================
# Timing + probability validation
# ==================================================

for result in predictions:

    probability_sum = (
        result.p_a_plant
        + result.p_b_plant
        + result.p_no_plant
    )

    assert (
        abs(
            probability_sum
            - 1.0
        )
        < 1e-12
    )

    assert (
        abs(
            result.motion_window_sec
            - 1.0
        )
        < 1e-9
    )

    print(
        f"\n{result.horizon_sec}s"
    )

    print(
        f"  actual time: "
        f"{result.observation_timestamp_sec:.3f}"
    )

    print(
        f"  lateness: "
        f"{result.horizon_lateness_sec:.3f}s"
    )

    print(
        f"  motion Δt: "
        f"{result.motion_window_sec:.3f}s"
    )

    print(
        f"  A:  "
        f"{100 * result.p_a_plant:.2f}%"
    )

    print(
        f"  B:  "
        f"{100 * result.p_b_plant:.2f}%"
    )

    print(
        f"  NO: "
        f"{100 * result.p_no_plant:.2f}%"
    )

    print(
        f"  prediction: "
        f"{result.prediction}"
    )


print(
    "\n✅ EXACTLY ONE prediction emitted "
    "for every frozen V0 horizon"
)

print(
    "✅ Live probabilities sum to 1"
)

print(
    "✅ Live 1-second history retrieval works"
)

print(
    "\n✅ V0-TC LIVE ENGINE SYNTHETIC TEST PASSED"
)
