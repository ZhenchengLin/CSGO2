"""
End-to-end V0 GSI HTTP integration test.

Tests:

synthetic GSI JSON
    ↓
HTTP
    ↓
V0GSIAdapter
    ↓
V0Frame
    ↓
V0LiveEngine
    ↓
FeatureBuilder
    ↓
V0Predictor
    ↓
10 / 20 / 30 / 40 second predictions
"""

import json
import os
import urllib.request


PORT = int(
    os.environ.get(
        "V0_GSI_PORT",
        "3001",
    )
)

URL = (
    f"http://127.0.0.1:{PORT}/gsi"
)

HEALTH_URL = (
    f"http://127.0.0.1:{PORT}/health"
)

EXPECTED_VERSION = (
    "V0-TIMING-CORRECTED"
)

EXPECTED_DATASET_SHA256 = (
    "092268fa06b0ed93e21aae444eeed81d"
    "3794887dc7bc8e7cf0ba9f4aa806e8ff"
)


def make_payload(
    timestamp,
    phase,
):
    t = timestamp

    return {
        "map": {
            "name":
                "de_mirage",

            "mode":
                "competitive",

            "round":
                0,
        },

        "round": {
            "phase":
                phase,
        },

        "phase_countdowns": {
            "phase":
                phase,

            "phase_ends_in":
                "10.0",
        },

        "allplayers": {

            "111": {
                "team":
                    "T",

                "position":
                    (
                        f"{100 + 20*t}, "
                        f"{200 + 5*t}, "
                        "10"
                    ),

                "state": {
                    "health":
                        100,

                    "armor":
                        100,

                    "equip_value":
                        4200,
                },
            },

            "222": {
                "team":
                    "T",

                "position":
                    (
                        f"{150 + 15*t}, "
                        f"{250 + 4*t}, "
                        "10"
                    ),

                "state": {
                    "health":
                        100,

                    "armor":
                        50,

                    "equip_value":
                        3500,
                },
            },

            "333": {
                "team":
                    "CT",

                "position":
                    (
                        f"{-500 + 3*t}, "
                        "-300, "
                        "20"
                    ),

                "state": {
                    "health":
                        100,

                    "armor":
                        100,

                    "equip_value":
                        4800,
                },
            },

            "444": {
                "team":
                    "CT",

                "position":
                    (
                        "-450, "
                        f"{-350 + 2*t}, "
                        "20"
                    ),

                "state": {
                    "health":
                        100,

                    "armor":
                        100,

                    "equip_value":
                        4000,
                },
            },
        },

        "bomb": {
            "state":
                "carried",

            "player":
                "111",

            "position":
                (
                    f"{100 + 20*t}, "
                    f"{200 + 5*t}, "
                    "10"
                ),
        },
    }


def send(
    timestamp,
    phase,
):
    payload = make_payload(
        timestamp,
        phase,
    )

    body = json.dumps(
        payload
    ).encode(
        "utf-8"
    )

    request = urllib.request.Request(
        URL,

        data=body,

        method="POST",

        headers={
            "Content-Type":
                "application/json",

            "X-V0-Test-Timestamp":
                str(timestamp),
        },
    )

    with urllib.request.urlopen(
        request
    ) as response:

        return json.loads(
            response.read().decode(
                "utf-8"
            )
        )


print(
    "\n"
    + "=" * 80
)

print(
    "V0 — HTTP GSI END-TO-END TEST"
)

print(
    "=" * 80
)


# ==================================================
# Runtime identity
# ==================================================

with urllib.request.urlopen(
    HEALTH_URL
) as response:

    health = json.loads(
        response.read().decode(
            "utf-8"
        )
    )


assert (
    health["model_version"]
    == EXPECTED_VERSION
)

assert (
    health["dataset_sha256"]
    == EXPECTED_DATASET_SHA256
)

assert (
    health["n_observations"]
    == 1686
)

assert (
    health["n_features"]
    == 37
)


print(
    "\n✅ corrected runtime identity verified"
)

print(
    f"   version: "
    f"{health['model_version']}"
)

print(
    f"   dataset: "
    f"{health['dataset_sha256']}"
)


# ==================================================
# Freeze time
# ==================================================

response = send(
    -0.1,
    "freezetime",
)

assert (
    response[
        "predictions_emitted"
    ]
    == 0
)

print(
    "\n✅ freezetime frame accepted"
)


# ==================================================
# freeze_end transition
# ==================================================

response = send(
    0.0,
    "live",
)

assert (
    response[
        "predictions_emitted"
    ]
    == 0
)

print(
    "✅ freezetime -> live transition accepted"
)


# ==================================================
# Synthetic 10 Hz stream
# ==================================================

predictions = []


for step in range(
    1,
    402,
):

    timestamp = (
        step / 10.0
    )

    response = send(
        timestamp,
        "live",
    )

    predictions.extend(
        response[
            "predictions"
        ]
    )


print(
    f"\nPredictions emitted: "
    f"{len(predictions)}"
)


assert (
    len(predictions)
    == 4
)


horizons = [
    prediction[
        "horizon_sec"
    ]
    for prediction
    in predictions
]


assert horizons == [
    10,
    20,
    30,
    40,
]


print(
    "✅ Horizons: "
    "10 / 20 / 30 / 40"
)


# ==================================================
# Validate every prediction
# ==================================================

for prediction in predictions:

    probability_sum = (
        prediction[
            "p_a_plant"
        ]
        + prediction[
            "p_b_plant"
        ]
        + prediction[
            "p_no_plant"
        ]
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
            prediction[
                "motion_window_sec"
            ]
            - 1.0
        )
        < 1e-9
    )

    print(
        "\n"
        f"{prediction['horizon_sec']}s"
    )

    print(
        f"  A:  "
        f"{100 * prediction['p_a_plant']:.2f}%"
    )

    print(
        f"  B:  "
        f"{100 * prediction['p_b_plant']:.2f}%"
    )

    print(
        f"  NO: "
        f"{100 * prediction['p_no_plant']:.2f}%"
    )

    print(
        f"  prediction: "
        f"{prediction['prediction']}"
    )

    print(
        f"  motion Δt: "
        f"{prediction['motion_window_sec']:.3f}s"
    )


print(
    "\n✅ HTTP transport preserved "
    "canonical timing"
)

print(
    "✅ HTTP transport preserved "
    "1-second motion history"
)

print(
    "✅ All probabilities valid"
)

print(
    "\n✅ V0-TC GSI HTTP END-TO-END TEST PASSED"
)
