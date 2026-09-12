"""
V1 Stage 0A — Horizon Feasibility Audit

Purpose
-------
Before changing the V0 prediction schedule, measure how much usable
data remains if observations are extended from 5 to 60 seconds.

This script does NOT:
- modify V0
- build model features
- train a model
- evaluate model performance

It only audits observation eligibility and class coverage.

Eligibility matches the frozen V0 logic:

    target_tick < round_end
    and
    target_tick < plant_tick

when a plant exists.

Labels are still derived from the eventual real bomb plant event:

    BombsiteA -> A_PLANT
    BombsiteB -> B_PLANT
    no plant  -> NO_PLANT
"""

from collections import Counter, defaultdict
from pathlib import Path

import polars as pl
from awpy import Demo


# ============================================================
# Configuration
# ============================================================

RAW_DIR = Path("data/raw")

MANIFEST_PATH = (
    RAW_DIR
    / "demo_manifest.csv"
)

HORIZONS_SEC = tuple(
    range(
        5,
        61,
        5,
    )
)

SUMMARY_PATH = Path(
    "data/interim/"
    "v1_horizon_feasibility.csv"
)

PAIRED_PATH = Path(
    "data/interim/"
    "v1_horizon_paired_retention.csv"
)

DETAIL_PATH = Path(
    "data/interim/"
    "v1_horizon_observations.csv"
)


# ============================================================
# Label helper
# ============================================================

def get_round_plant_info(
    demo,
):
    """
    Return one eventual plant event per round.

    If an unusual demo contains multiple plant events in one round,
    the first plant event defines the V1 audit label, matching the
    meaning of "eventual first plant outcome".
    """

    plants = (
        demo.bomb
        .filter(
            pl.col("event")
            == "plant"
        )
        .sort([
            "round_num",
            "tick",
        ])
    )

    plant_info = {}

    for row in (
        plants
        .iter_rows(named=True)
    ):

        round_num = row[
            "round_num"
        ]

        if round_num in plant_info:
            continue

        bombsite = row.get(
            "bombsite"
        )

        if bombsite == "BombsiteA":
            label = "A_PLANT"

        elif bombsite == "BombsiteB":
            label = "B_PLANT"

        else:
            label = "UNKNOWN_PLANT_SITE"

        plant_info[
            round_num
        ] = {
            "plant_tick":
                row["tick"],

            "label":
                label,
        }

    return plant_info


# ============================================================
# Load manifest
# ============================================================

manifest = pl.read_csv(
    MANIFEST_PATH,
    null_values=[""],
    infer_schema_length=1000,
)

print(
    "\n"
    + "=" * 100
)

print(
    "V1 STAGE 0A — HORIZON FEASIBILITY AUDIT"
)

print(
    "=" * 100
)

print(
    f"\nDemos in manifest: "
    f"{manifest.height}"
)

print(
    "Candidate horizons: "
    + ", ".join(
        f"{h}s"
        for h in HORIZONS_SEC
    )
)


# ============================================================
# Build observation audit
# ============================================================

rows = []

invalid_rounds = []

failed_demos = []

parsed_demos = 0


for demo_index, manifest_row in enumerate(
    manifest.iter_rows(
        named=True
    ),
    start=1,
):

    filename = manifest_row[
        "demo filename"
    ]

    demo_path = (
        RAW_DIR
        / filename
    )

    print(
        f"\n"
        f"[{demo_index:02d}/"
        f"{manifest.height:02d}] "
        f"{filename}"
    )

    try:

        demo = Demo(
            str(demo_path),
            verbose=False,
        )

        demo.parse()

        map_name = (
            demo.header.get(
                "map_name"
            )
        )

        if map_name != "de_mirage":
            raise ValueError(
                "Expected de_mirage, "
                f"got {map_name!r}"
            )

        parsed_demos += 1

        plant_info = (
            get_round_plant_info(
                demo
            )
        )


        # ----------------------------------------------------
        # One row per round x candidate horizon
        # ----------------------------------------------------

        for round_row in (
            demo.rounds
            .sort("round_num")
            .iter_rows(named=True)
        ):

            round_num = (
                round_row[
                    "round_num"
                ]
            )

            freeze_end = (
                round_row[
                    "freeze_end"
                ]
            )

            round_end = (
                round_row[
                    "end"
                ]
            )


            # ------------------------------------------------
            # Validate timing
            # ------------------------------------------------

            if freeze_end is None:

                invalid_rounds.append({
                    "demo_filename":
                        filename,

                    "round_num":
                        round_num,

                    "reason":
                        "MISSING_FREEZE_END",
                })

                continue


            if round_end is None:

                invalid_rounds.append({
                    "demo_filename":
                        filename,

                    "round_num":
                        round_num,

                    "reason":
                        "MISSING_ROUND_END",
                })

                continue


            if freeze_end >= round_end:

                invalid_rounds.append({
                    "demo_filename":
                        filename,

                    "round_num":
                        round_num,

                    "reason":
                        "INVALID_TIMING_ORDER",
                })

                continue


            # ------------------------------------------------
            # Eventual label
            # ------------------------------------------------

            plant = (
                plant_info.get(
                    round_num
                )
            )

            if plant is None:

                plant_tick = None
                label = "NO_PLANT"

            else:

                plant_tick = (
                    plant[
                        "plant_tick"
                    ]
                )

                label = (
                    plant[
                        "label"
                    ]
                )


            # ------------------------------------------------
            # Candidate observations
            # ------------------------------------------------

            for horizon_sec in (
                HORIZONS_SEC
            ):

                target_tick = (
                    freeze_end
                    + int(
                        round(
                            horizon_sec
                            * demo.tickrate
                        )
                    )
                )


                before_round_end = (
                    target_tick
                    < round_end
                )


                before_plant = (
                    plant_tick is None
                    or target_tick
                    < plant_tick
                )


                eligible = (
                    before_round_end
                    and before_plant
                )


                if not before_round_end:

                    reason = (
                        "ROUND_ALREADY_ENDED"
                    )

                elif not before_plant:

                    reason = (
                        "PLANT_ALREADY_HAPPENED"
                    )

                else:

                    reason = "VALID"


                rows.append({

                    "demo_filename":
                        filename,

                    "round_num":
                        round_num,

                    "horizon_sec":
                        horizon_sec,

                    "target_tick":
                        target_tick,

                    "freeze_end":
                        freeze_end,

                    "plant_tick":
                        plant_tick,

                    "round_end":
                        round_end,

                    "label":
                        label,

                    "eligible":
                        eligible,

                    "reason":
                        reason,
                })


        print(
            "  ✅ parsed"
        )


    except Exception as exc:

        print(
            f"  ❌ FAILED: {exc}"
        )

        failed_demos.append({
            "demo_filename":
                filename,

            "error":
                str(exc),
        })


# ============================================================
# Create detail table
# ============================================================

if not rows:
    raise RuntimeError(
        "No observation candidates "
        "were created."
    )


detail = pl.DataFrame(
    rows
)


valid = detail.filter(
    pl.col("eligible")
)


# ============================================================
# Summary by horizon
# ============================================================

summary_rows = []

baseline_n = None

previous_n = None


for horizon_sec in (
    HORIZONS_SEC
):

    current = valid.filter(
        pl.col("horizon_sec")
        == horizon_sec
    )

    n_total = (
        current.height
    )

    counts = Counter(
        current[
            "label"
        ].to_list()
    )


    n_matches = (
        current[
            "demo_filename"
        ].n_unique()
        if n_total
        else 0
    )


    n_rounds = (
        current.select([
            "demo_filename",
            "round_num",
        ])
        .unique()
        .height
        if n_total
        else 0
    )


    if baseline_n is None:
        baseline_n = n_total


    baseline_retention = (
        n_total
        / baseline_n
        if baseline_n
        else 0.0
    )


    if previous_n is None:
        adjacent_count_retention = (
            1.0
        )

    else:
        adjacent_count_retention = (
            n_total
            / previous_n
            if previous_n
            else 0.0
        )


    summary_rows.append({

        "horizon_sec":
            horizon_sec,

        "n_observations":
            n_total,

        "n_rounds":
            n_rounds,

        "n_matches":
            n_matches,

        "A_PLANT":
            counts.get(
                "A_PLANT",
                0,
            ),

        "B_PLANT":
            counts.get(
                "B_PLANT",
                0,
            ),

        "NO_PLANT":
            counts.get(
                "NO_PLANT",
                0,
            ),

        "UNKNOWN_PLANT_SITE":
            counts.get(
                "UNKNOWN_PLANT_SITE",
                0,
            ),

        "retention_from_5s":
            baseline_retention,

        "count_retention_from_previous":
            adjacent_count_retention,
    })


    previous_n = (
        n_total
    )


summary = pl.DataFrame(
    summary_rows
)


# ============================================================
# Paired adjacent-horizon retention
# ============================================================

eligible_keys = defaultdict(
    set
)


for row in (
    valid
    .select([
        "demo_filename",
        "round_num",
        "horizon_sec",
    ])
    .iter_rows(named=True)
):

    eligible_keys[
        row["horizon_sec"]
    ].add(
        (
            row[
                "demo_filename"
            ],
            row[
                "round_num"
            ],
        )
    )


paired_rows = []


for left, right in zip(
    HORIZONS_SEC[:-1],
    HORIZONS_SEC[1:],
):

    left_keys = (
        eligible_keys[left]
    )

    right_keys = (
        eligible_keys[right]
    )

    paired = (
        left_keys
        & right_keys
    )


    paired_rows.append({

        "from_horizon_sec":
            left,

        "to_horizon_sec":
            right,

        "n_from":
            len(
                left_keys
            ),

        "n_to":
            len(
                right_keys
            ),

        "n_paired_rounds":
            len(
                paired
            ),

        "paired_retention":
            (
                len(paired)
                / len(left_keys)
                if left_keys
                else 0.0
            ),
    })


paired = pl.DataFrame(
    paired_rows
)


# ============================================================
# Save exploratory audit
# ============================================================

SUMMARY_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


detail.write_csv(
    DETAIL_PATH
)

summary.write_csv(
    SUMMARY_PATH
)

paired.write_csv(
    PAIRED_PATH
)


# ============================================================
# Print results
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "HORIZON COVERAGE"
)

print(
    "=" * 100
)


print(
    summary.select([

        "horizon_sec",

        "n_observations",

        "n_matches",

        "A_PLANT",

        "B_PLANT",

        "NO_PLANT",

        "retention_from_5s",

    ])
)


print(
    "\n"
    + "=" * 100
)

print(
    "ADJACENT PAIRED RETENTION"
)

print(
    "=" * 100
)


print(
    paired
)


# ============================================================
# Reason counts
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "INELIGIBILITY REASONS BY HORIZON"
)

print(
    "=" * 100
)


print(
    detail
    .filter(
        ~pl.col(
            "eligible"
        )
    )
    .group_by([
        "horizon_sec",
        "reason",
    ])
    .len()
    .sort([
        "horizon_sec",
        "reason",
    ])
)


# ============================================================
# QA
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "AUDIT QA"
)

print(
    "=" * 100
)


print(
    f"Demos parsed:       "
    f"{parsed_demos}"
)

print(
    f"Demos failed:       "
    f"{len(failed_demos)}"
)

print(
    f"Invalid rounds:     "
    f"{len(invalid_rounds)}"
)

print(
    f"Candidate rows:     "
    f"{detail.height}"
)

print(
    f"Eligible rows:      "
    f"{valid.height}"
)


unknown_sites = (
    valid
    .filter(
        pl.col("label")
        == "UNKNOWN_PLANT_SITE"
    )
    .height
)


print(
    f"Unknown plant site: "
    f"{unknown_sites}"
)


if failed_demos:

    print(
        "\nFAILED DEMOS"
    )

    print(
        pl.DataFrame(
            failed_demos
        )
    )


if invalid_rounds:

    print(
        "\nINVALID ROUND REASONS"
    )

    print(
        pl.DataFrame(
            invalid_rounds
        )
        .group_by(
            "reason"
        )
        .len()
    )


print(
    "\nSaved:"
)

print(
    f"  {SUMMARY_PATH}"
)

print(
    f"  {PAIRED_PATH}"
)

print(
    f"  {DETAIL_PATH}"
)


print(
    "\n✅ V1 HORIZON FEASIBILITY AUDIT COMPLETE"
)
