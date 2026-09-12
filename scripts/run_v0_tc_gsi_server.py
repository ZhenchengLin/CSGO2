"""
Minimal HTTP receiver for CS2 Game State Integration.

Pipeline:

HTTP POST /gsi
    ↓
JSON payload
    ↓
V0GSIAdapter
    ↓
V0Frame
    ↓
V0LiveEngine
    ↓
V0Predictor
    ↓
probabilities

Uses only Python's standard library for HTTP.
"""

import json
import os
import time

from http.server import (
    BaseHTTPRequestHandler,
    HTTPServer,
)

from pathlib import Path

from cs2_tactical_intelligence.v0.gsi import (
    V0GSIAdapter,
)

from cs2_tactical_intelligence.v0.live import (
    V0LiveEngine,
)

from cs2_tactical_intelligence.v0.predictor import (
    V0Predictor,
)


HOST = os.environ.get(
    "V0_GSI_HOST",
    "127.0.0.1",
)

PORT = int(
    os.environ.get(
        "V0_GSI_PORT",
        "3000",
    )
)

ALLOW_TEST_TIME = (
    os.environ.get(
        "V0_GSI_ALLOW_TEST_TIME",
        "0",
    )
    == "1"
)

LOG_FRAMES = (
    os.environ.get(
        "V0_GSI_LOG_FRAMES",
        "0",
    )
    == "1"
)

CAPTURE_GSI = (
    os.environ.get(
        "V0_GSI_CAPTURE",
        "0",
    )
    == "1"
)

CAPTURE_PATH = Path(
    os.environ.get(
        "V0_GSI_CAPTURE_PATH",
        "data/interim/v0_gsi_capture.jsonl",
    )
)

RAW_DEBUG_PATH = Path(
    "data/interim/v0_last_gsi_payload.json"
)

CAPTURE_GSI = (
    os.environ.get(
        "V0_GSI_CAPTURE",
        "0",
    )
    == "1"
)

CAPTURE_PATH = Path(
    os.environ.get(
        "V0_GSI_CAPTURE_PATH",
        "data/interim/v0_tc_gsi_capture.jsonl",
    )
)


adapter = V0GSIAdapter()

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


class Handler(
    BaseHTTPRequestHandler
):

    def log_message(
        self,
        format,
        *args,
    ):
        # Disable default noisy HTTP logs.
        return

    def _send_json(
        self,
        status_code,
        payload,
    ):
        body = json.dumps(
            payload
        ).encode(
            "utf-8"
        )

        self.send_response(
            status_code
        )

        self.send_header(
            "Content-Type",
            "application/json",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.end_headers()

        self.wfile.write(
            body
        )

    # ==================================================
    # Health check
    # ==================================================

    def do_GET(
        self,
    ):
        if self.path != "/health":

            self._send_json(
                404,
                {
                    "error":
                        "not found"
                },
            )

            return

        self._send_json(
            200,
            {
                "status":
                    "ok",

                "service":
                    "V0-TC GSI receiver",

                "model_version":
                    predictor.metadata[
                        "version"
                    ],

                "dataset_sha256":
                    predictor.metadata[
                        "dataset_sha256"
                    ],

                "n_observations":
                    predictor.metadata[
                        "n_observations"
                    ],

                "n_features":
                    predictor.metadata[
                        "n_features"
                    ],
            },
        )

    # ==================================================
    # GSI POST
    # ==================================================

    def do_POST(
        self,
    ):
        if self.path != "/gsi":

            self._send_json(
                404,
                {
                    "error":
                        "not found"
                },
            )

            return

        received_monotonic_sec = (
            time.monotonic()
        )

        content_length = int(
            self.headers.get(
                "Content-Length",
                "0",
            )
        )

        raw_body = self.rfile.read(
            content_length
        )

        try:

            payload = json.loads(
                raw_body.decode(
                    "utf-8"
                )
            )

        except Exception as exc:

            print(
                f"❌ Invalid JSON: {exc}"
            )

            self._send_json(
                400,
                {
                    "error":
                        "invalid json"
                },
            )

            return

        # ------------------------------------------
        # Save most recent real payload for debugging.
        # data/interim is gitignored.
        # ------------------------------------------

        RAW_DEBUG_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        RAW_DEBUG_PATH.write_text(
            json.dumps(
                payload,
                indent=2,
            )
        )

        if CAPTURE_GSI:
            CAPTURE_PATH.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with CAPTURE_PATH.open(
                "a",
                encoding="utf-8",
            ) as capture_file:

                capture_file.write(
                    json.dumps({
                        "received_monotonic_sec":
                            time.monotonic(),

                        "payload":
                            payload,
                    })
                    + "\n"
                )

        test_timestamp = (
            self.headers.get(
                "X-V0-Test-Timestamp"
            )
        )

        if (
            ALLOW_TEST_TIME
            and test_timestamp
            is not None
        ):
            timestamp_sec = float(
                test_timestamp
            )

        else:
            timestamp_sec = (
                received_monotonic_sec
            )

        # ------------------------------------------
        # Optional raw-source capture.
        #
        # IMPORTANT:
        # Capture happens BEFORE V0GSIAdapter.
        # This preserves malformed/incomplete payloads
        # for later source auditing.
        # ------------------------------------------

        if CAPTURE_GSI:

            CAPTURE_PATH.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            record = {
                "received_monotonic_sec":
                    received_monotonic_sec,

                "frame_timestamp_sec":
                    timestamp_sec,

                "payload":
                    payload,
            }

            with CAPTURE_PATH.open(
                "a",
                encoding="utf-8",
            ) as capture_file:

                capture_file.write(
                    json.dumps(
                        record,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        try:

            frame = adapter.to_frame(
                payload,
                timestamp_sec=(
                    timestamp_sec
                ),
            )

        except ValueError as exc:

            # GSI may send incomplete states while
            # loading, in menus, or outside observer
            # gameplay. Do not kill the server.

            print(
                f"⏭️  skipped payload: {exc}"
            )

            self._send_json(
                200,
                {
                    "status":
                        "skipped",

                    "reason":
                        str(exc),
                },
            )

            return

        if LOG_FRAMES:
            print(
                f"📡 round={frame.round_num}"
                f" phase={frame.phase}"
                f" players={len(frame.players)}"
                f" bomb={frame.bomb.state}"
            )

        try:

            predictions = (
                engine.process_frame(
                    frame
                )
            )

        except Exception as exc:

            print(
                f"❌ live engine error: "
                f"{exc}"
            )

            self._send_json(
                500,
                {
                    "error":
                        str(exc)
                },
            )

            return

        for result in predictions:

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"🎯 V0-TC LIVE PREDICTION "
                f"| ROUND {result.round_num} "
                f"| {result.horizon_sec}s"
            )

            print(
                "=" * 70
            )

            print(
                f"A_PLANT  "
                f"{100 * result.p_a_plant:6.2f}%"
            )

            print(
                f"B_PLANT  "
                f"{100 * result.p_b_plant:6.2f}%"
            )

            print(
                f"NO_PLANT "
                f"{100 * result.p_no_plant:6.2f}%"
            )

            print(
                f"\nPrediction: "
                f"{result.prediction}"
            )

            print(
                f"Confidence: "
                f"{100 * result.confidence:.2f}%"
            )

            print(
                f"Horizon lateness: "
                f"{result.horizon_lateness_sec:.3f}s"
            )

            print(
                f"Motion Δt: "
                f"{result.motion_window_sec:.3f}s"
            )

        prediction_payloads = [
            {
                "round_num":
                    result.round_num,

                "horizon_sec":
                    result.horizon_sec,

                "prediction":
                    result.prediction,

                "confidence":
                    result.confidence,

                "p_a_plant":
                    result.p_a_plant,

                "p_b_plant":
                    result.p_b_plant,

                "p_no_plant":
                    result.p_no_plant,

                "horizon_lateness_sec":
                    result.horizon_lateness_sec,

                "motion_window_sec":
                    result.motion_window_sec,
            }
            for result in predictions
        ]

        self._send_json(
            200,
            {
                "status":
                    "ok",

                "round_num":
                    frame.round_num,

                "phase":
                    frame.phase,

                "predictions_emitted":
                    len(predictions),

                "predictions":
                    prediction_payloads,
            },
        )


print("\n" + "=" * 80)

print(
    "CS2 TACTICAL INTELLIGENCE — V0-TC GSI RECEIVER"
)

print("=" * 80)

print(
    f"\nListening on:"
)

print(
    f"http://{HOST}:{PORT}/gsi"
)

print(
    f"\nHealth check:"
)

print(
    f"http://{HOST}:{PORT}/health"
)

print(
    "\nCtrl+C to stop."
)

print()


server = HTTPServer(
    (
        HOST,
        PORT,
    ),
    Handler,
)


try:

    server.serve_forever()

except KeyboardInterrupt:

    print(
        "\nStopping V0 GSI receiver."
    )

finally:

    server.server_close()
