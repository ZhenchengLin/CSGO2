"""
V2 new-demo intake audit.

Purpose
-------
Audit newly acquired demos BEFORE any V0/H2 evaluation.

This script does NOT:
- train a model
- evaluate V0
- evaluate H2
- modify the legacy demo manifest
- assign scientific conclusions

It DOES:
- hash every incoming demo
- detect duplicates against the legacy corpus
- detect duplicates inside the incoming batch
- audit raw demo ticks/sec from game_time
- enforce Mirage scope
- parse using the frozen 64-tick historical timing contract
- count rounds, labels, and eligible 10/20/30/40s observations
- attach acquisition metadata
- produce a V2 intake audit artifact
"""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import median

import polars as pl
from awpy import Demo

from cs2_tactical_intelligence.v0.timing import (
    V0_DEMO_TICKS_PER_SECOND,
    open_v0_demo,
    seconds_to_demo_ticks,
)


LEGACY_RAW_DIR = Path("data/raw")
INCOMING_DIR = Path("data/raw/v2_incoming")

METADATA_PATH = (
    INCOMING_DIR
    / "v2_intake_metadata.csv"
)

OUTPUT_PATH = Path(
    "data/interim/v2_intake_audit.csv"
)

HORIZONS_SEC = [10, 20, 30, 40]

EXPECTED_MAP = "de_mirage"
EXPECTED_RAW_TICKS_PER_SEC = 64.0
TICKRATE_TOLERANCE = 0.25


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def legacy_demo_paths() -> list[Path]:
    paths = []

    for path in LEGACY_RAW_DIR.rglob("*.dem"):
        try:
            path.relative_to(INCOMING_DIR)
            continue
        except ValueError:
            pass

        paths.append(path)

    return sorted(paths)


def build_legacy_hash_index() -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}

    print("\nHashing legacy demos...")

    paths = legacy_demo_paths()

    for i, path in enumerate(paths, start=1):
        digest = sha256_file(path)

        index.setdefault(
            digest,
            [],
        ).append(
            str(path)
        )

        print(
            f"  [{i:02d}/{len(paths):02d}] "
            f"{path.name}"
        )

    return index


def measure_raw_ticks_per_second(
    path: Path,
) -> tuple[float, int]:
    """
    Independently estimate raw ticks / game-second
    using parsed game_time.

    We explicitly configure 64 so Awpy's default
    128-tick assumption cannot leak into downstream
    timing semantics.
    """

    demo = Demo(
        path,
        tickrate=V0_DEMO_TICKS_PER_SECOND,
        verbose=False,
    )

    clock = (
        demo.parse_ticks(
            other_props=[
                "game_time",
            ]
        )
        .select([
            "tick",
            "game_time",
        ])
        .drop_nulls()
        .unique()
        .sort("tick")
    )

    ratios = []
    previous = None

    for current in clock.iter_rows(
        named=True
    ):
        if previous is not None:
            dtick = (
                current["tick"]
                - previous["tick"]
            )

            dtime = (
                current["game_time"]
                - previous["game_time"]
            )

            if (
                dtick > 0
                and dtime > 0
                and dtime < 2.0
            ):
                ratios.append(
                    dtick / dtime
                )

        previous = current

    if not ratios:
        raise RuntimeError(
            "No usable game_time intervals "
            "for raw tick-clock audit."
        )

    return (
        float(median(ratios)),
        len(ratios),
    )


def build_plant_lookup(
    bomb: pl.DataFrame,
) -> dict[int, dict]:
    plants = (
        bomb
        .filter(
            pl.col("event")
            == "plant"
        )
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

    for row in plants.iter_rows(
        named=True
    ):
        round_num = row[
            "round_num"
        ]

        if round_num in lookup:
            raise ValueError(
                "Multiple plant events "
                f"in round {round_num}"
            )

        bombsite = row[
            "bombsite"
        ]

        if bombsite == "BombsiteA":
            label = "A_PLANT"

        elif bombsite == "BombsiteB":
            label = "B_PLANT"

        else:
            raise ValueError(
                "Unknown bombsite: "
                f"{bombsite}"
            )

        lookup[round_num] = {
            "plant_tick":
                row["tick"],
            "label":
                label,
        }

    return lookup


def audit_scientific_content(
    path: Path,
) -> dict:
    """
    Full parse under the frozen corrected timing contract.
    """

    demo = open_v0_demo(
        path,
        verbose=False,
    )

    demo.parse()

    if (
        demo.tickrate
        != V0_DEMO_TICKS_PER_SECOND
    ):
        raise RuntimeError(
            "Configured timing contract changed."
        )

    map_name = demo.header.get(
        "map_name"
    )

    rounds = demo.rounds.sort(
        "round_num"
    )

    plant_lookup = (
        build_plant_lookup(
            demo.bomb
        )
    )

    invalid_rounds = []

    labels = {
        "A_PLANT": 0,
        "B_PLANT": 0,
        "NO_PLANT": 0,
    }

    horizon_counts = {
        horizon: 0
        for horizon
        in HORIZONS_SEC
    }

    n_valid_rounds = 0

    for round_row in rounds.iter_rows(
        named=True
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

        if freeze_end is None:
            invalid_rounds.append(
                f"{round_num}:"
                "MISSING_FREEZE_END"
            )
            continue

        if round_end is None:
            invalid_rounds.append(
                f"{round_num}:"
                "MISSING_ROUND_END"
            )
            continue

        if freeze_end >= round_end:
            invalid_rounds.append(
                f"{round_num}:"
                "INVALID_TIMING_ORDER"
            )
            continue

        n_valid_rounds += 1

        plant_info = (
            plant_lookup.get(
                round_num
            )
        )

        if plant_info is None:
            plant_tick = None
            label = "NO_PLANT"

        else:
            plant_tick = (
                plant_info[
                    "plant_tick"
                ]
            )

            label = (
                plant_info[
                    "label"
                ]
            )

        labels[label] += 1

        for horizon_sec in (
            HORIZONS_SEC
        ):
            target_tick = (
                freeze_end
                + seconds_to_demo_ticks(
                    horizon_sec
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

            if (
                before_round_end
                and before_plant
            ):
                horizon_counts[
                    horizon_sec
                ] += 1

    n_observations = sum(
        horizon_counts.values()
    )

    return {
        "map_name":
            map_name,

        "configured_tickrate":
            demo.tickrate,

        "n_rounds":
            rounds.height,

        "n_valid_rounds":
            n_valid_rounds,

        "n_invalid_rounds":
            len(invalid_rounds),

        "invalid_round_reasons":
            ";".join(
                invalid_rounds
            ),

        "n_a_plant":
            labels["A_PLANT"],

        "n_b_plant":
            labels["B_PLANT"],

        "n_no_plant":
            labels["NO_PLANT"],

        "n_observations":
            n_observations,

        "n_obs_10":
            horizon_counts[10],

        "n_obs_20":
            horizon_counts[20],

        "n_obs_30":
            horizon_counts[30],

        "n_obs_40":
            horizon_counts[40],
    }


def load_metadata() -> dict[str, dict]:
    if not METADATA_PATH.exists():
        return {}

    metadata = pl.read_csv(
        METADATA_PATH,
        null_values=[""],
        infer_schema_length=1000,
    )

    if metadata.height == 0:
        return {}

    if (
        "demo_filename"
        not in metadata.columns
    ):
        raise ValueError(
            "Metadata must contain "
            "demo_filename."
        )

    duplicate_names = (
        metadata
        .group_by(
            "demo_filename"
        )
        .len()
        .filter(
            pl.col("len") > 1
        )
    )

    if duplicate_names.height > 0:
        raise ValueError(
            "Duplicate filenames in "
            "V2 metadata:\n"
            f"{duplicate_names}"
        )

    return {
        row["demo_filename"]: row
        for row
        in metadata.iter_rows(
            named=True
        )
    }


def metadata_complete(
    metadata_row: dict | None,
) -> bool:
    if metadata_row is None:
        return False

    required = [
        "match_date",
        "event",
        "team1",
        "team2",
        "source",
        "source_url",
        "scientific_role",
    ]

    return all(
        metadata_row.get(field)
        not in (
            None,
            "",
        )
        for field
        in required
    )


def main():
    print(
        "\n"
        + "=" * 100
    )

    print(
        "V2 — NEW DEMO INTAKE AUDIT"
    )

    print(
        "=" * 100
    )

    print(
        "\nFrozen timing contract:"
        f" {V0_DEMO_TICKS_PER_SECOND}"
        " raw ticks/sec"
    )

    print(
        f"Expected map: {EXPECTED_MAP}"
    )

    awpy_version = (
        package_version(
            "awpy"
        )
    )

    demoparser_version = (
        package_version(
            "demoparser2"
        )
    )

    print(
        f"awpy: {awpy_version}"
    )

    print(
        "demoparser2: "
        f"{demoparser_version}"
    )

    incoming_paths = sorted(
        INCOMING_DIR.glob(
            "*.dem"
        )
    )

    print(
        "\nIncoming demos: "
        f"{len(incoming_paths)}"
    )

    if not incoming_paths:
        print(
            "\nNo .dem files found in:"
        )

        print(
            INCOMING_DIR
        )

        print(
            "\nAdd new Mirage demos "
            "there, fill the metadata CSV, "
            "then rerun this audit."
        )

        return

    metadata_lookup = (
        load_metadata()
    )

    legacy_hashes = (
        build_legacy_hash_index()
    )

    incoming_hash_owner = {}

    rows = []

    for index, path in enumerate(
        incoming_paths,
        start=1,
    ):
        print(
            "\n"
            f"[{index:02d}/"
            f"{len(incoming_paths):02d}] "
            f"{path.name}"
        )

        row = {
            "demo_filename":
                path.name,

            "file_size_bytes":
                path.stat().st_size,

            "sha256":
                None,

            "duplicate_legacy":
                False,

            "legacy_duplicate_paths":
                None,

            "duplicate_incoming":
                False,

            "incoming_duplicate_of":
                None,

            "awpy_version":
                awpy_version,

            "demoparser2_version":
                demoparser_version,

            "measured_raw_ticks_per_sec":
                None,

            "n_clock_intervals":
                None,

            "tick_clock_ok":
                False,

            "map_name":
                None,

            "map_ok":
                False,

            "configured_tickrate":
                None,

            "n_rounds":
                None,

            "n_valid_rounds":
                None,

            "n_invalid_rounds":
                None,

            "invalid_round_reasons":
                None,

            "n_a_plant":
                None,

            "n_b_plant":
                None,

            "n_no_plant":
                None,

            "n_observations":
                None,

            "n_obs_10":
                None,

            "n_obs_20":
                None,

            "n_obs_30":
                None,

            "n_obs_40":
                None,

            "metadata_complete":
                False,

            "scientific_role":
                None,

            "status":
                None,

            "error":
                None,
        }

        metadata_row = (
            metadata_lookup.get(
                path.name
            )
        )

        row[
            "metadata_complete"
        ] = metadata_complete(
            metadata_row
        )

        if metadata_row:
            for field in [
                "match_date",
                "event",
                "team1",
                "team2",
                "source",
                "source_url",
                "scientific_role",
                "notes",
            ]:
                row[field] = (
                    metadata_row.get(
                        field
                    )
                )

        try:
            digest = sha256_file(
                path
            )

            row["sha256"] = digest

            print(
                "  sha256: "
                f"{digest[:16]}..."
            )

            if digest in legacy_hashes:
                row[
                    "duplicate_legacy"
                ] = True

                row[
                    "legacy_duplicate_paths"
                ] = ";".join(
                    legacy_hashes[
                        digest
                    ]
                )

                print(
                    "  ❌ exact duplicate "
                    "of legacy demo"
                )

            if digest in (
                incoming_hash_owner
            ):
                row[
                    "duplicate_incoming"
                ] = True

                row[
                    "incoming_duplicate_of"
                ] = (
                    incoming_hash_owner[
                        digest
                    ]
                )

                print(
                    "  ❌ duplicate inside "
                    "incoming batch"
                )

            else:
                incoming_hash_owner[
                    digest
                ] = path.name

            measured, n_intervals = (
                measure_raw_ticks_per_second(
                    path
                )
            )

            row[
                "measured_raw_ticks_per_sec"
            ] = measured

            row[
                "n_clock_intervals"
            ] = n_intervals

            tick_clock_ok = (
                abs(
                    measured
                    - EXPECTED_RAW_TICKS_PER_SEC
                )
                <= TICKRATE_TOLERANCE
            )

            row[
                "tick_clock_ok"
            ] = tick_clock_ok

            print(
                "  raw clock: "
                f"{measured:.6f} "
                "ticks/sec"
            )

            if not tick_clock_ok:
                print(
                    "  ❌ timing contract "
                    "not confirmed"
                )

            scientific = (
                audit_scientific_content(
                    path
                )
            )

            row.update(
                scientific
            )

            map_ok = (
                row["map_name"]
                == EXPECTED_MAP
            )

            row["map_ok"] = map_ok

            print(
                "  map: "
                f"{row['map_name']}"
            )

            print(
                "  rounds: "
                f"{row['n_rounds']}"
                " | observations: "
                f"{row['n_observations']}"
            )

            print(
                "  labels: "
                f"A={row['n_a_plant']} "
                f"B={row['n_b_plant']} "
                f"NO={row['n_no_plant']}"
            )

            if (
                row["duplicate_legacy"]
            ):
                status = (
                    "exclude_duplicate_legacy"
                )

            elif (
                row[
                    "duplicate_incoming"
                ]
            ):
                status = (
                    "exclude_duplicate_incoming"
                )

            elif not tick_clock_ok:
                status = (
                    "exclude_timing_contract"
                )

            elif not map_ok:
                status = (
                    "exclude_wrong_map"
                )

            elif (
                row["n_invalid_rounds"]
                > 0
            ):
                status = (
                    "review_round_timing"
                )

            elif not row[
                "metadata_complete"
            ]:
                status = (
                    "review_metadata"
                )

            else:
                status = "eligible_intake"

            row["status"] = status

            print(
                f"  status: {status}"
            )

        except Exception as exc:
            row[
                "status"
            ] = "audit_failed"

            row[
                "error"
            ] = str(exc)

            print(
                f"  ❌ AUDIT FAILED: "
                f"{exc}"
            )

        rows.append(
            row
        )

    audit = pl.DataFrame(
        rows,
        infer_schema_length=1000,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.write_csv(
        OUTPUT_PATH
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "V2 INTAKE SUMMARY"
    )

    print(
        "=" * 100
    )

    summary = (
        audit
        .group_by(
            "status"
        )
        .len()
        .sort(
            "status"
        )
    )

    print(
        summary
    )

    eligible = (
        audit.filter(
            pl.col("status")
            == "eligible_intake"
        )
    )

    print(
        "\nEligible demos: "
        f"{eligible.height}"
    )

    if eligible.height > 0:
        print(
            "\nEligible observation "
            "total: "
            f"{eligible['n_observations'].sum()}"
        )

        print(
            "\nEligible horizons:"
        )

        for horizon in (
            HORIZONS_SEC
        ):
            print(
                f"  {horizon:>2}s: "
                f"{eligible[f'n_obs_{horizon}'].sum()}"
            )

    print(
        "\nSaved audit:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\n✅ V2 INTAKE AUDIT COMPLETE"
    )


if __name__ == "__main__":
    main()
