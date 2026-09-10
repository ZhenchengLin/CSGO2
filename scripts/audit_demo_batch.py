from pathlib import Path

import polars as pl
from awpy import Demo


RAW_DIR = Path("data/raw")
MANIFEST_PATH = RAW_DIR / "demo_manifest.csv"
AUDIT_OUTPUT_PATH = Path("data/interim/v0_demo_batch_audit.csv")

HORIZONS_SEC = [10, 20, 30, 40]


def build_plant_lookup(bomb: pl.DataFrame):
    """
    Build:
        round_num -> {
            plant_tick,
            label
        }

    IMPORTANT:
    We use bomb plant events, NOT demo.rounds.bomb_site,
    because we already discovered that rounds.bomb_site
    is unreliable in Awpy 2.0.2.
    """

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
                f"Multiple plant events detected in round {round_num}"
            )

        bombsite = row["bombsite"]

        if bombsite == "BombsiteA":
            label = "A_PLANT"

        elif bombsite == "BombsiteB":
            label = "B_PLANT"

        else:
            raise ValueError(
                f"Unknown bombsite value: {bombsite}"
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

    # --------------------------------------------------
    # Map verification
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Label counts
    # --------------------------------------------------

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

            eligible = (
                before_round_end
                and before_plant
            )

            if eligible:

                valid_observations += 1

                horizon_counts[
                    horizon_sec
                ] += 1


    return {
        "map_name": map_name,
        "tickrate": tickrate,

        "n_rounds": rounds.height,
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
print("V0 — 20 DEMO BATCH AUDIT")
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

        updated_row[
            "parse_status"
        ] = "file_missing"

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
            "demo_filename":
                filename,

            "status":
                "file_missing",

            "error":
                "File not found",
        })

        continue


    try:

        result = audit_demo(
            demo_path
        )

        print(
            "  ✅ parsed"
            f" | rounds={result['n_rounds']}"
            f" | obs={result['n_observations']}"
            f" | labels="
            f"A:{result['n_a_plant']} "
            f"B:{result['n_b_plant']} "
            f"N:{result['n_no_plant']}"
        )

        updated_row[
            "parse_status"
        ] = "parsed_ok"

        updated_row[
            "n_rounds"
        ] = result[
            "n_rounds"
        ]

        updated_row[
            "n_observations"
        ] = result[
            "n_observations"
        ]

        updated_manifest_rows.append(
            updated_row
        )


        audit_rows.append({

            "demo_filename":
                filename,

            "status":
                "parsed_ok",

            **result,

            "error":
                None,
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

            "demo_filename":
                filename,

            "status":
                "parse_failed",

            "error":
                str(exc),
        })


# ==================================================
# Save results
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
# Dataset summary
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
    f"Total demos:      {audit.height}"
)

print(
    f"Parsed OK:        {successful.height}"
)

print(
    f"Failed:           {failed.height}"
)


if successful.height > 0:

    print(
        f"Total rounds:     "
        f"{successful['n_rounds'].sum()}"
    )

    print(
        f"Total observations: "
        f"{successful['n_observations'].sum()}"
    )

    print(
        "\nLabel counts:"
    )

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

    print(
        "\nObservations by horizon:"
    )

    for horizon in HORIZONS_SEC:

        column = f"n_obs_{horizon}"

        print(
            f"{horizon:>2}s: "
            f"{successful[column].sum()}"
        )


if failed.height == 0:

    print(
        "\n✅ ALL DEMOS PASSED BATCH AUDIT"
    )

else:

    print(
        "\n❌ SOME DEMOS REQUIRE ATTENTION"
    )

    print(
        failed.select([
            "demo_filename",
            "status",
            "error",
        ])
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
