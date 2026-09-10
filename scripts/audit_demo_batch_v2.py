from pathlib import Path

import polars as pl
from awpy import Demo


RAW_DIR = Path("data/raw")
MANIFEST_PATH = RAW_DIR / "demo_manifest.csv"
AUDIT_OUTPUT_PATH = Path("data/interim/v0_demo_batch_audit_v2.csv")

HORIZONS_SEC = [10, 20, 30, 40]


def build_plant_lookup(bomb: pl.DataFrame):

    plants = (
        bomb
        .filter(pl.col("event") == "plant")
        .select([
            "round_num",
            "tick",
            "bombsite",
        ])
        .sort([
            "round_num",
            "tick",
        ])
    )

    lookup = {}

    for row in plants.iter_rows(named=True):

        round_num = row["round_num"]

        if round_num in lookup:
            raise ValueError(
                f"Multiple plant events in round {round_num}"
            )

        if row["bombsite"] == "BombsiteA":
            label = "A_PLANT"

        elif row["bombsite"] == "BombsiteB":
            label = "B_PLANT"

        else:
            raise ValueError(
                f"Unknown bombsite: {row['bombsite']}"
            )

        lookup[round_num] = {
            "plant_tick": row["tick"],
            "label": label,
        }

    return lookup


def audit_demo(demo_path: Path):

    demo = Demo(
        str(demo_path),
        verbose=False,
    )

    demo.parse()

    map_name = demo.header.get("map_name")

    if map_name != "de_mirage":
        raise ValueError(
            f"Expected de_mirage, got {map_name}"
        )

    tickrate = demo.tickrate
    rounds = demo.rounds.sort("round_num")

    plant_lookup = build_plant_lookup(
        demo.bomb
    )

    n_valid_rounds = 0
    invalid_rounds = []

    a_plant = 0
    b_plant = 0
    no_plant = 0

    valid_observations = 0

    horizon_counts = {
        10: 0,
        20: 0,
        30: 0,
        40: 0,
    }


    for round_row in rounds.iter_rows(named=True):

        round_num = round_row["round_num"]
        freeze_end = round_row["freeze_end"]
        round_end = round_row["end"]


        # --------------------------------------------------
        # Round-level timing validation
        # --------------------------------------------------

        if freeze_end is None:

            invalid_rounds.append({
                "round_num": round_num,
                "reason": "MISSING_FREEZE_END",
            })

            continue


        if round_end is None:

            invalid_rounds.append({
                "round_num": round_num,
                "reason": "MISSING_ROUND_END",
            })

            continue


        if freeze_end >= round_end:

            invalid_rounds.append({
                "round_num": round_num,
                "reason": "INVALID_TIMING_ORDER",
            })

            continue


        n_valid_rounds += 1


        # --------------------------------------------------
        # Label
        # --------------------------------------------------

        plant_info = plant_lookup.get(
            round_num
        )


        if plant_info is None:

            plant_tick = None
            label = "NO_PLANT"
            no_plant += 1

        else:

            plant_tick = plant_info["plant_tick"]
            label = plant_info["label"]

            if label == "A_PLANT":
                a_plant += 1

            elif label == "B_PLANT":
                b_plant += 1


        # --------------------------------------------------
        # Observation eligibility
        # --------------------------------------------------

        for horizon_sec in HORIZONS_SEC:

            target_tick = (
                freeze_end
                + int(
                    round(
                        horizon_sec
                        * tickrate
                    )
                )
            )


            before_round_end = (
                target_tick < round_end
            )


            before_plant = (
                plant_tick is None
                or target_tick < plant_tick
            )


            if (
                before_round_end
                and before_plant
            ):

                valid_observations += 1

                horizon_counts[
                    horizon_sec
                ] += 1


    invalid_round_nums = ";".join(
        str(x["round_num"])
        for x in invalid_rounds
    )

    invalid_round_reasons = ";".join(
        f"{x['round_num']}:{x['reason']}"
        for x in invalid_rounds
    )


    return {
        "map_name": map_name,
        "tickrate": tickrate,

        "n_rounds": rounds.height,
        "n_valid_rounds": n_valid_rounds,

        "n_invalid_timing_rounds":
            len(invalid_rounds),

        "invalid_round_nums":
            invalid_round_nums,

        "invalid_round_reasons":
            invalid_round_reasons,

        "n_a_plant": a_plant,
        "n_b_plant": b_plant,
        "n_no_plant": no_plant,

        "n_observations":
            valid_observations,

        "n_obs_10":
            horizon_counts[10],

        "n_obs_20":
            horizon_counts[20],

        "n_obs_30":
            horizon_counts[30],

        "n_obs_40":
            horizon_counts[40],
    }


# ==================================================
# Main
# ==================================================

manifest = pl.read_csv(
    MANIFEST_PATH,
    null_values=[""],
    infer_schema_length=1000,
)


updated_manifest_rows = []
audit_rows = []


print("\n" + "=" * 100)
print("V0 — 20 DEMO BATCH AUDIT V2")
print("=" * 100)


for index, manifest_row in enumerate(
    manifest.iter_rows(named=True),
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
        f"\n[{index:02d}/{manifest.height:02d}] "
        f"{filename}"
    )


    updated_row = dict(
        manifest_row
    )


    if not demo_path.exists():

        print("  ❌ FILE NOT FOUND")

        updated_row["parse_status"] = "file_missing"
        updated_row["n_rounds"] = None
        updated_row["n_observations"] = None

        updated_manifest_rows.append(
            updated_row
        )

        audit_rows.append({
            "demo_filename": filename,
            "status": "file_missing",
            "error": "File not found",
        })

        continue


    try:

        result = audit_demo(
            demo_path
        )


        invalid_note = ""

        if result[
            "n_invalid_timing_rounds"
        ] > 0:

            invalid_note = (
                f" | invalid timing="
                f"{result['n_invalid_timing_rounds']}"
                f" ({result['invalid_round_reasons']})"
            )


        print(
            "  ✅ parsed"
            f" | rounds={result['n_rounds']}"
            f" | valid={result['n_valid_rounds']}"
            f" | obs={result['n_observations']}"
            f" | labels="
            f"A:{result['n_a_plant']} "
            f"B:{result['n_b_plant']} "
            f"N:{result['n_no_plant']}"
            f"{invalid_note}"
        )


        updated_row[
            "parse_status"
        ] = "parsed_ok"

        updated_row[
            "n_rounds"
        ] = result["n_rounds"]

        updated_row[
            "n_observations"
        ] = result["n_observations"]


        updated_manifest_rows.append(
            updated_row
        )


        audit_rows.append({
            "demo_filename": filename,
            "status": "parsed_ok",
            **result,
            "error": None,
        })


    except Exception as exc:

        print(
            f"  ❌ FAILED: {exc}"
        )

        updated_row[
            "parse_status"
        ] = "parse_failed"

        updated_row[
            "n_rounds"
        ] = None

        updated_row[
            "n_observations"
        ] = None


        updated_manifest_rows.append(
            updated_row
        )


        audit_rows.append({
            "demo_filename": filename,
            "status": "parse_failed",
            "error": str(exc),
        })


# ==================================================
# Save
# ==================================================

updated_manifest = pl.DataFrame(
    updated_manifest_rows
)

audit = pl.DataFrame(
    audit_rows
)


updated_manifest.write_csv(
    MANIFEST_PATH
)

AUDIT_OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

audit.write_csv(
    AUDIT_OUTPUT_PATH
)


# ==================================================
# Summary
# ==================================================

successful = audit.filter(
    pl.col("status") == "parsed_ok"
)

failed = audit.filter(
    pl.col("status") != "parsed_ok"
)


print("\n" + "=" * 100)
print("BATCH SUMMARY")
print("=" * 100)


print(
    f"Total demos:              {audit.height}"
)

print(
    f"Parsed OK:                {successful.height}"
)

print(
    f"Failed demos:             {failed.height}"
)


if successful.height > 0:

    print(
        f"Total parsed rounds:      "
        f"{successful['n_rounds'].sum()}"
    )

    print(
        f"Valid timing rounds:      "
        f"{successful['n_valid_rounds'].sum()}"
    )

    print(
        f"Invalid timing rounds:    "
        f"{successful['n_invalid_timing_rounds'].sum()}"
    )

    print(
        f"Total observations:       "
        f"{successful['n_observations'].sum()}"
    )


    print("\nLabel counts — valid rounds only:")

    print(
        f"A_PLANT:  "
        f"{successful['n_a_plant'].sum()}"
    )

    print(
        f"B_PLANT:  "
        f"{successful['n_b_plant'].sum()}"
    )

    print(
        f"NO_PLANT: "
        f"{successful['n_no_plant'].sum()}"
    )


    print("\nObservations by horizon:")

    for horizon in HORIZONS_SEC:

        print(
            f"{horizon:>2}s: "
            f"{successful[f'n_obs_{horizon}'].sum()}"
        )


print("\n" + "=" * 100)
print("ROUND-LEVEL DATA QUALITY ISSUES")
print("=" * 100)


issues = successful.filter(
    pl.col("n_invalid_timing_rounds") > 0
)


if issues.height == 0:

    print("None")

else:

    print(
        issues.select([
            "demo_filename",
            "n_invalid_timing_rounds",
            "invalid_round_reasons",
        ])
    )


if failed.height == 0:

    print(
        "\n✅ ALL 20 DEMOS PARSED SUCCESSFULLY"
    )

else:

    print(
        "\n❌ DEMO-LEVEL FAILURES REMAIN"
    )


print(
    "\nUpdated manifest:"
)

print(
    MANIFEST_PATH
)

print(
    "\nDetailed audit:"
)

print(
    AUDIT_OUTPUT_PATH
)
