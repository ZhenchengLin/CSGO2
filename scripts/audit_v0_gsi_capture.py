"""
Audit a captured CS2 GSI JSONL stream.

This script does NOT train or evaluate the model.

It answers source-contract questions:

- What payload cadence did we receive?
- Are map / phase fields present?
- How many observer players are present?
- Are position / health / armor / equip_value available?
- What bomb states occur?
- Can V0GSIAdapter consume the raw payloads?
"""

import argparse
import json
import statistics

from collections import Counter
from pathlib import Path

from cs2_tactical_intelligence.v0.gsi import (
    V0GSIAdapter,
)


# ============================================================
# Helpers
# ============================================================

def percentile(
    values,
    q,
):
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (
        (len(ordered) - 1)
        * q
    )

    lower = int(position)
    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    fraction = (
        position - lower
    )

    return (
        ordered[lower]
        * (1.0 - fraction)
        +
        ordered[upper]
        * fraction
    )


def summarize_cadence(
    name,
    timestamps,
):
    print(
        f"\n{name}"
    )
    print("-" * len(name))

    if len(timestamps) < 2:
        print(
            "Not enough timestamps."
        )
        return

    deltas = [
        b - a
        for a, b
        in zip(
            timestamps[:-1],
            timestamps[1:],
        )
    ]

    positive = [
        delta
        for delta in deltas
        if delta > 0
    ]

    nonpositive = (
        len(deltas)
        - len(positive)
    )

    if not positive:
        print(
            "No positive intervals."
        )
        return

    mean_dt = statistics.fmean(
        positive
    )

    median_dt = statistics.median(
        positive
    )

    p95_dt = percentile(
        positive,
        0.95,
    )

    max_dt = max(
        positive
    )

    print(
        f"Intervals:           "
        f"{len(deltas)}"
    )

    print(
        f"Non-positive Δt:     "
        f"{nonpositive}"
    )

    print(
        f"Mean Δt:             "
        f"{mean_dt:.6f}s"
    )

    print(
        f"Median Δt:           "
        f"{median_dt:.6f}s"
    )

    print(
        f"P95 Δt:              "
        f"{p95_dt:.6f}s"
    )

    print(
        f"Maximum gap:         "
        f"{max_dt:.6f}s"
    )

    print(
        f"Median effective Hz: "
        f"{1.0 / median_dt:.3f}"
    )


def print_counter(
    title,
    counter,
):
    print(
        f"\n{title}"
    )
    print("-" * len(title))

    if not counter:
        print("None")
        return

    for key, value in (
        counter.most_common()
    ):
        print(
            f"{str(key):<30}"
            f"{value}"
        )


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "capture_path",
    nargs="?",
    default=(
        "data/interim/"
        "v0_gsi_capture.jsonl"
    ),
)

args = parser.parse_args()

capture_path = Path(
    args.capture_path
)

if not capture_path.exists():
    raise SystemExit(
        f"Capture does not exist: "
        f"{capture_path}"
    )


# ============================================================
# Counters
# ============================================================

records = 0
malformed_lines = 0

received_times = []
frame_times = []

map_names = Counter()
map_rounds = Counter()

phases = Counter()
phase_transitions = Counter()

player_counts = Counter()
team_count_pairs = Counter()

missing_player_position = 0
missing_player_state = 0
missing_health = 0
missing_armor = 0
missing_equip_value = 0

bomb_states = Counter()
bomb_position_present = 0
bomb_carrier_present = 0

adapter_accepted = 0
adapter_rejected = 0
adapter_errors = Counter()

previous_phase = None

adapter = V0GSIAdapter()


# ============================================================
# Read raw capture
# ============================================================

with capture_path.open(
    encoding="utf-8"
) as file:

    for line_number, line in (
        enumerate(
            file,
            start=1,
        )
    ):

        line = line.strip()

        if not line:
            continue

        try:
            record = json.loads(
                line
            )

        except json.JSONDecodeError:
            malformed_lines += 1
            continue

        records += 1

        received_time = (
            record.get(
                "received_monotonic_sec"
            )
        )

        frame_time = (
            record.get(
                "frame_timestamp_sec"
            )
        )

        if isinstance(
            received_time,
            (int, float),
        ):
            received_times.append(
                float(received_time)
            )

        if isinstance(
            frame_time,
            (int, float),
        ):
            frame_times.append(
                float(frame_time)
            )

        payload = record.get(
            "payload"
        )

        if not isinstance(
            payload,
            dict,
        ):
            adapter_rejected += 1
            adapter_errors[
                "payload is not a dict"
            ] += 1
            continue


        # ====================================================
        # Map
        # ====================================================

        map_data = payload.get(
            "map",
            {},
        )

        if isinstance(
            map_data,
            dict,
        ):

            map_names[
                map_data.get(
                    "name",
                    "<missing>",
                )
            ] += 1

            map_rounds[
                str(
                    map_data.get(
                        "round",
                        "<missing>",
                    )
                )
            ] += 1

        else:
            map_names[
                "<missing>"
            ] += 1


        # ====================================================
        # Phase
        # ====================================================

        phase_countdowns = (
            payload.get(
                "phase_countdowns",
                {},
            )
        )

        round_data = payload.get(
            "round",
            {},
        )

        phase = None

        if isinstance(
            phase_countdowns,
            dict,
        ):
            phase = (
                phase_countdowns.get(
                    "phase"
                )
            )

        if (
            phase is None
            and isinstance(
                round_data,
                dict,
            )
        ):
            phase = (
                round_data.get(
                    "phase"
                )
            )

        phase_key = (
            phase
            if phase is not None
            else "<missing>"
        )

        phases[
            phase_key
        ] += 1

        if (
            previous_phase is not None
            and phase_key
            != previous_phase
        ):
            phase_transitions[
                (
                    previous_phase,
                    phase_key,
                )
            ] += 1

        previous_phase = (
            phase_key
        )


        # ====================================================
        # Players
        # ====================================================

        allplayers = payload.get(
            "allplayers"
        )

        if isinstance(
            allplayers,
            dict,
        ):

            player_counts[
                len(allplayers)
            ] += 1

            t_count = 0
            ct_count = 0

            for player in (
                allplayers.values()
            ):

                if not isinstance(
                    player,
                    dict,
                ):
                    continue

                team = player.get(
                    "team"
                )

                if team == "T":
                    t_count += 1

                elif team == "CT":
                    ct_count += 1

                if (
                    "position"
                    not in player
                ):
                    missing_player_position += 1

                state = player.get(
                    "state"
                )

                if not isinstance(
                    state,
                    dict,
                ):
                    missing_player_state += 1
                    continue

                if (
                    "health"
                    not in state
                ):
                    missing_health += 1

                if (
                    "armor"
                    not in state
                ):
                    missing_armor += 1

                if (
                    "equip_value"
                    not in state
                ):
                    missing_equip_value += 1

            team_count_pairs[
                (
                    t_count,
                    ct_count,
                )
            ] += 1

        else:

            player_counts[
                "<missing>"
            ] += 1


        # ====================================================
        # Bomb
        # ====================================================

        bomb = payload.get(
            "bomb"
        )

        if isinstance(
            bomb,
            dict,
        ):

            bomb_states[
                bomb.get(
                    "state",
                    "<missing>",
                )
            ] += 1

            if (
                "position"
                in bomb
            ):
                bomb_position_present += 1

            if (
                "player"
                in bomb
            ):
                bomb_carrier_present += 1

        else:

            bomb_states[
                "<missing>"
            ] += 1


        # ====================================================
        # Adapter contract
        # ====================================================

        try:

            adapter.to_frame(
                payload,
                timestamp_sec=(
                    float(
                        frame_time
                        if frame_time
                        is not None
                        else 0.0
                    )
                ),
            )

            adapter_accepted += 1

        except Exception as exc:

            adapter_rejected += 1

            adapter_errors[
                str(exc)
            ] += 1


# ============================================================
# Report
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "V0 — GSI SOURCE CAPTURE AUDIT"
)

print(
    "=" * 80
)

print(
    f"\nCapture: {capture_path}"
)

print(
    f"Records:             "
    f"{records}"
)

print(
    f"Malformed JSON lines:"
    f" {malformed_lines}"
)


summarize_cadence(
    "FRAME TIMESTAMP CADENCE",
    frame_times,
)

summarize_cadence(
    "HTTP RECEIVE CADENCE",
    received_times,
)


print_counter(
    "MAP NAMES",
    map_names,
)

print_counter(
    "PHASES",
    phases,
)

print_counter(
    "PHASE TRANSITIONS",
    phase_transitions,
)

print_counter(
    "PLAYER COUNTS PER PAYLOAD",
    player_counts,
)

print_counter(
    "T / CT PLAYER COUNT PAIRS",
    team_count_pairs,
)


print(
    "\nPLAYER FIELD COVERAGE"
)
print(
    "-" * 21
)

print(
    f"Missing position:    "
    f"{missing_player_position}"
)

print(
    f"Missing state:       "
    f"{missing_player_state}"
)

print(
    f"Missing health:      "
    f"{missing_health}"
)

print(
    f"Missing armor:       "
    f"{missing_armor}"
)

print(
    f"Missing equip_value: "
    f"{missing_equip_value}"
)


print_counter(
    "BOMB STATES",
    bomb_states,
)

print(
    "\nBOMB FIELD COVERAGE"
)
print(
    "-" * 19
)

print(
    f"Position present:    "
    f"{bomb_position_present}"
    f" / {records}"
)

print(
    f"Carrier present:     "
    f"{bomb_carrier_present}"
    f" / {records}"
)


print(
    "\nADAPTER CONTRACT"
)
print(
    "-" * 16
)

print(
    f"Accepted:            "
    f"{adapter_accepted}"
)

print(
    f"Rejected:            "
    f"{adapter_rejected}"
)

if adapter_errors:

    print(
        "\nAdapter rejection reasons:"
    )

    for reason, count in (
        adapter_errors.most_common()
    ):

        print(
            f"  {count:>6} × "
            f"{reason}"
        )


print(
    "\n"
    + "=" * 80
)

print(
    "AUDIT COMPLETE"
)

print(
    "=" * 80
)
