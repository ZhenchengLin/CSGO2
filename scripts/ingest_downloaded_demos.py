"""
V2 local demo-ingestion pipeline.

What it does
------------
Scans ~/Downloads for demos / archives corresponding to matches listed in:

    docs/v2_acquisition_queue.csv

For each matching match:

1. Finds .dem files directly or inside .zip/.rar/.7z archives
2. Parses each demo under the frozen V0 timing contract
3. Keeps only de_mirage
4. Computes SHA256
5. Rejects D_legacy duplicates
6. Rejects duplicate incoming demos
7. Renames canonically
8. Copies into data/raw/v2_incoming/
9. Updates v2_intake_metadata.csv
10. Runs V2 intake audit + pool summary

IMPORTANT
---------
This script does NOT scrape HLTV.
It only processes files already downloaded locally.

It does NOT train or evaluate V0/H2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from cs2_tactical_intelligence.v0.timing import (
    open_v0_demo,
)


# ============================================================
# Paths
# ============================================================

REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DEFAULT_SCAN_DIR = (
    Path.home()
    / "Downloads"
)

QUEUE_PATH = (
    REPO_ROOT
    / "docs"
    / "v2_acquisition_queue.csv"
)

LEGACY_RAW_DIR = (
    REPO_ROOT
    / "data"
    / "raw"
)

INCOMING_DIR = (
    LEGACY_RAW_DIR
    / "v2_incoming"
)

METADATA_PATH = (
    INCOMING_DIR
    / "v2_intake_metadata.csv"
)


EXPECTED_MAP = "de_mirage"

SUPPORTED_ARCHIVES = {
    ".zip",
    ".rar",
    ".7z",
}


QUEUE_REQUIRED_FIELDS = [
    "match_date",
    "event",
    "team1",
    "team2",
    "source",
    "source_url",
    "scientific_role",
]


METADATA_FIELDS = [
    "demo_filename",
    "match_date",
    "event",
    "team1",
    "team2",
    "source",
    "source_url",
    "scientific_role",
    "notes",
]


# ============================================================
# Helpers
# ============================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def compact(
    value,
) -> str:

    if value is None:
        return ""

    return "".join(
        character.lower()
        for character
        in str(value)
        if character.isalnum()
    )


def slugify(
    value: str,
) -> str:

    output = []
    last_was_dash = False

    for character in (
        value
        .lower()
        .strip()
    ):

        if character.isalnum():

            output.append(
                character
            )

            last_was_dash = False

        else:

            if not last_was_dash:

                output.append(
                    "-"
                )

                last_was_dash = True

    return (
        "".join(output)
        .strip("-")
    )


def canonical_filename(
    row: dict,
) -> str:

    return (
        f"{row['match_date']}_"
        f"{slugify(row['team1'])}"
        "_vs_"
        f"{slugify(row['team2'])}"
        "_mirage.dem"
    )


# ============================================================
# CSV helpers
# ============================================================

def read_csv_rows(
    path: Path,
):

    if not path.exists():
        return [], []

    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f
        )

        fields = list(
            reader.fieldnames
            or []
        )

        rows = [
            dict(row)
            for row
            in reader
        ]

    return (
        fields,
        rows,
    )


def write_csv_rows(
    path: Path,
    fields,
    rows,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def load_queue(
    path: Path,
):

    fields, rows = (
        read_csv_rows(
            path
        )
    )

    if not rows:

        raise RuntimeError(
            "Acquisition queue missing or empty:\n"
            f"{path}"
        )

    missing_columns = [
        field
        for field
        in QUEUE_REQUIRED_FIELDS
        if field
        not in fields
    ]

    if missing_columns:

        raise RuntimeError(
            "Queue missing columns: "
            f"{missing_columns}"
        )

    for index, row in enumerate(
        rows,
        start=2,
    ):

        missing_values = [
            field
            for field
            in QUEUE_REQUIRED_FIELDS
            if not (
                row.get(field)
                or ""
            ).strip()
        ]

        if missing_values:

            raise RuntimeError(
                f"Queue row {index} "
                "missing values: "
                f"{missing_values}"
            )

    return rows


# ============================================================
# Queue matching
# ============================================================

def path_matches_match(
    path: Path,
    row: dict,
) -> bool:

    haystack = compact(
        str(path)
    )

    team1 = compact(
        row["team1"]
    )

    team2 = compact(
        row["team2"]
    )

    return (
        team1
        and team2
        and team1 in haystack
        and team2 in haystack
    )


def match_queue_row(
    path: Path,
    queue,
):

    matches = [
        row
        for row
        in queue
        if path_matches_match(
            path,
            row,
        )
    ]

    if not matches:
        return None

    if len(matches) == 1:
        return matches[0]

    # Try event name as disambiguator.

    haystack = compact(
        str(path)
    )

    event_matches = [
        row
        for row
        in matches
        if compact(
            row.get(
                "event"
            )
        )
        in haystack
    ]

    if len(event_matches) == 1:
        return event_matches[0]

    descriptions = [
        (
            f"{row['match_date']} | "
            f"{row['team1']} vs "
            f"{row['team2']} | "
            f"{row['event']}"
        )
        for row
        in matches
    ]

    raise RuntimeError(
        "Ambiguous queue match:\n"
        f"{path}\n  "
        + "\n  ".join(
            descriptions
        )
    )


# ============================================================
# Demo identity
# ============================================================

def demo_map(
    path: Path,
):

    demo = open_v0_demo(
        path,
        verbose=False,
    )

    demo.parse()

    return demo.header.get(
        "map_name"
    )


def legacy_demo_paths():

    output = []

    for path in (
        LEGACY_RAW_DIR
        .rglob("*.dem")
    ):

        try:

            path.relative_to(
                INCOMING_DIR
            )

            continue

        except ValueError:
            pass

        output.append(
            path
        )

    return sorted(
        output
    )


def build_hash_index(
    paths,
):

    index = {}

    for path in paths:

        digest = sha256_file(
            path
        )

        index.setdefault(
            digest,
            [],
        ).append(
            path
        )

    return index


# ============================================================
# Archive extraction
# ============================================================

def extract_archive(
    archive: Path,
    destination: Path,
):

    suffix = (
        archive
        .suffix
        .lower()
    )

    if suffix == ".zip":

        with zipfile.ZipFile(
            archive
        ) as zf:

            zf.extractall(
                destination
            )

        return


    if suffix == ".rar":

        unar = shutil.which(
            "unar"
        )

        if unar is None:

            raise RuntimeError(
                "RAR archive detected but "
                "'unar' is not installed.\n"
                "Run once:\n"
                "brew install unar"
            )

        subprocess.run(
            [
                unar,
                "-q",
                "-o",
                str(destination),
                str(archive),
            ],
            check=True,
        )

        return


    if suffix == ".7z":

        seven_zip = (
            shutil.which(
                "7zz"
            )
            or
            shutil.which(
                "7z"
            )
        )

        if seven_zip is None:

            raise RuntimeError(
                "7z archive detected but "
                "7zz/7z is not installed."
            )

        subprocess.run(
            [
                seven_zip,
                "x",
                "-y",
                f"-o{destination}",
                str(archive),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

        return


    raise RuntimeError(
        "Unsupported archive: "
        f"{archive}"
    )


# ============================================================
# Metadata
# ============================================================

def build_metadata_note(
    source_path: Path,
    original_demo: Path,
):

    return (
        "Auto-ingested from local download pipeline; "
        f"source_path={source_path}; "
        f"original_demo_filename={original_demo.name}"
    )


def upsert_metadata(
    queue_row,
    target_filename,
    source_path,
    original_demo,
    *,
    dry_run,
):

    fields, rows = (
        read_csv_rows(
            METADATA_PATH
        )
    )

    if not fields:

        fields = list(
            METADATA_FIELDS
        )

    for field in (
        METADATA_FIELDS
    ):

        if field not in fields:

            fields.append(
                field
            )


    desired = {

        "demo_filename":
            target_filename,

        "match_date":
            queue_row[
                "match_date"
            ],

        "event":
            queue_row[
                "event"
            ],

        "team1":
            queue_row[
                "team1"
            ],

        "team2":
            queue_row[
                "team2"
            ],

        "source":
            queue_row[
                "source"
            ],

        "source_url":
            queue_row[
                "source_url"
            ],

        "scientific_role":
            queue_row[
                "scientific_role"
            ],

        "notes":
            build_metadata_note(
                source_path,
                original_demo,
            ),
    }


    existing = next(
        (
            row
            for row
            in rows
            if row.get(
                "demo_filename"
            )
            == target_filename
        ),
        None,
    )


    if existing is not None:

        compare_fields = [
            "match_date",
            "event",
            "team1",
            "team2",
            "source",
            "source_url",
            "scientific_role",
        ]

        for field in (
            compare_fields
        ):

            old = (
                existing.get(
                    field
                )
                or ""
            ).strip()

            new = (
                desired.get(
                    field
                )
                or ""
            ).strip()

            if (
                old
                and old != new
            ):

                raise RuntimeError(
                    "Metadata conflict:\n"
                    f"{target_filename}\n"
                    f"{field}: "
                    f"{old!r} != {new!r}"
                )


        old_notes = (
            existing.get(
                "notes"
            )
            or ""
        ).strip()

        new_note = (
            desired[
                "notes"
            ]
        )

        if (
            new_note
            not in old_notes
        ):

            existing[
                "notes"
            ] = (
                f"{old_notes}; "
                f"{new_note}"
                if old_notes
                else new_note
            )


    else:

        rows.append(
            desired
        )


    if not dry_run:

        write_csv_rows(
            METADATA_PATH,
            fields,
            rows,
        )


# ============================================================
# Process one demo
# ============================================================

def process_demo(
    demo_path,
    queue_row,
    source_path,
    legacy_hashes,
    incoming_hashes,
    seen_this_run,
    *,
    dry_run,
):

    try:

        map_name = demo_map(
            demo_path
        )

    except Exception as exc:

        print(
            "    ⚠️ parse failed: "
            f"{demo_path.name}"
        )

        print(
            f"       {exc}"
        )

        return "parse_failed"


    if map_name != EXPECTED_MAP:

        print(
            f"    skip "
            f"{demo_path.name}: "
            f"{map_name}"
        )

        return "wrong_map"


    digest = sha256_file(
        demo_path
    )


    if digest in seen_this_run:

        print(
            "    ↪ duplicate in current scan: "
            f"{demo_path.name}"
        )

        return "duplicate_in_scan"


    seen_this_run.add(
        digest
    )


    target_filename = (
        canonical_filename(
            queue_row
        )
    )

    target_path = (
        INCOMING_DIR
        / target_filename
    )


    if digest in legacy_hashes:

        print(
            "    ⛔ legacy duplicate"
        )

        for owner in (
            legacy_hashes[
                digest
            ]
        ):

            print(
                f"       {owner}"
            )

        return "legacy_duplicate"


    if digest in incoming_hashes:

        existing_path = (
            incoming_hashes[
                digest
            ][0]
        )

        print(
            "    ✅ already ingested: "
            f"{existing_path.name}"
        )

        upsert_metadata(
            queue_row,
            existing_path.name,
            source_path,
            demo_path,
            dry_run=dry_run,
        )

        return "already_ingested"


    if target_path.exists():

        existing_digest = (
            sha256_file(
                target_path
            )
        )

        if existing_digest != digest:

            raise RuntimeError(
                "Canonical filename conflict:\n"
                f"{target_path}\n"
                "Existing SHA256:\n"
                f"{existing_digest}\n"
                "New SHA256:\n"
                f"{digest}"
            )

        print(
            "    ✅ target already exists"
        )

        upsert_metadata(
            queue_row,
            target_filename,
            source_path,
            demo_path,
            dry_run=dry_run,
        )

        return "already_ingested"


    print(
        f"    Mirage: "
        f"{demo_path.name}"
    )

    print(
        f"    → "
        f"{target_filename}"
    )

    print(
        f"    SHA256: "
        f"{digest}"
    )


    if not dry_run:

        INCOMING_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            demo_path,
            target_path,
        )

        incoming_hashes.setdefault(
            digest,
            [],
        ).append(
            target_path
        )


    upsert_metadata(
        queue_row,
        target_filename,
        source_path,
        demo_path,
        dry_run=dry_run,
    )


    return "ingested"


# ============================================================
# Scan
# ============================================================

def direct_demos(
    scan_dir,
):

    return sorted(
        path
        for path
        in scan_dir.rglob(
            "*.dem"
        )
        if path.is_file()
    )


def archive_files(
    scan_dir,
):

    return sorted(
        path
        for path
        in scan_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_ARCHIVES
        )
    )


# ============================================================
# Existing V2 QA pipeline
# ============================================================

def run_v2_qa():

    commands = [

        [
            sys.executable,
            "scripts/audit_v2_intake.py",
        ],

        [
            sys.executable,
            "scripts/summarize_v2_confirm_pool.py",
        ],
    ]


    for command in commands:

        print(
            "\n$ "
            + " ".join(
                command
            )
        )

        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=DEFAULT_SCAN_DIR,
        help=(
            "Directory scanned recursively. "
            "Default: ~/Downloads"
        ),
    )


    parser.add_argument(
        "--queue",
        type=Path,
        default=QUEUE_PATH,
        help=(
            "Acquisition queue CSV."
        ),
    )


    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Parse/report only. "
            "Do not copy files or "
            "change metadata."
        ),
    )


    parser.add_argument(
        "--no-audit",
        action="store_true",
        help=(
            "Do not automatically run "
            "audit + pool summary."
        ),
    )


    args = parser.parse_args()


    scan_dir = (
        args.scan_dir
        .expanduser()
        .resolve()
    )

    queue_path = (
        args.queue
        .expanduser()
        .resolve()
    )


    if not scan_dir.exists():

        raise FileNotFoundError(
            scan_dir
        )


    queue = load_queue(
        queue_path
    )


    print(
        "\n"
        + "=" * 100
    )

    print(
        "V2 — LOCAL DEMO INGESTION"
    )

    print(
        "=" * 100
    )


    print(
        f"\nScan directory: "
        f"{scan_dir}"
    )

    print(
        f"Queue:          "
        f"{queue_path}"
    )

    print(
        f"Queue matches:  "
        f"{len(queue)}"
    )

    print(
        f"Dry run:        "
        f"{args.dry_run}"
    )


    print(
        "\nHashing legacy demos..."
    )

    legacy_hashes = (
        build_hash_index(
            legacy_demo_paths()
        )
    )


    incoming_paths = (

        sorted(
            INCOMING_DIR
            .glob("*.dem")
        )

        if INCOMING_DIR.exists()

        else []
    )


    incoming_hashes = (
        build_hash_index(
            incoming_paths
        )
    )


    seen_this_run = set()


    counts = {

        "ingested":
            0,

        "already_ingested":
            0,

        "duplicate_in_scan":
            0,

        "legacy_duplicate":
            0,

        "wrong_map":
            0,

        "parse_failed":
            0,
    }


    # ========================================================
    # 1. Already-extracted demos
    # ========================================================

    for demo_path in (
        direct_demos(
            scan_dir
        )
    ):

        queue_row = (
            match_queue_row(
                demo_path,
                queue,
            )
        )

        if queue_row is None:
            continue


        print(
            "\nDirect candidate:"
        )

        print(
            f"  {queue_row['match_date']} | "
            f"{queue_row['team1']} vs "
            f"{queue_row['team2']}"
        )

        print(
            f"  {demo_path}"
        )


        status = process_demo(
            demo_path,
            queue_row,
            demo_path,
            legacy_hashes,
            incoming_hashes,
            seen_this_run,
            dry_run=args.dry_run,
        )


        counts[
            status
        ] += 1


    # ========================================================
    # 2. Archives
    # ========================================================

    for archive in (
        archive_files(
            scan_dir
        )
    ):

        queue_row = (
            match_queue_row(
                archive,
                queue,
            )
        )

        if queue_row is None:
            continue


        print(
            "\nArchive candidate:"
        )

        print(
            f"  {queue_row['match_date']} | "
            f"{queue_row['team1']} vs "
            f"{queue_row['team2']}"
        )

        print(
            f"  {archive}"
        )


        with tempfile.TemporaryDirectory(
            prefix=(
                "csgo2_v2_ingest_"
            )
        ) as temporary:

            temp_dir = Path(
                temporary
            )


            try:

                extract_archive(
                    archive,
                    temp_dir,
                )

            except Exception as exc:

                print(
                    "    ⚠️ extraction failed"
                )

                print(
                    f"       {exc}"
                )

                continue


            demos = sorted(
                temp_dir.rglob(
                    "*.dem"
                )
            )


            if not demos:

                print(
                    "    ⚠️ no .dem files "
                    "inside archive"
                )

                continue


            for demo_path in demos:

                status = process_demo(
                    demo_path,
                    queue_row,
                    archive,
                    legacy_hashes,
                    incoming_hashes,
                    seen_this_run,
                    dry_run=args.dry_run,
                )


                counts[
                    status
                ] += 1


    # ========================================================
    # Summary
    # ========================================================

    print(
        "\n"
        + "=" * 100
    )

    print(
        "INGEST SUMMARY"
    )

    print(
        "=" * 100
    )


    for name, count in (
        counts.items()
    ):

        print(
            f"{name:20s}: "
            f"{count}"
        )


    if args.dry_run:

        print(
            "\nDry run complete."
        )

        print(
            "No files or metadata changed."
        )

        return


    if not args.no_audit:

        run_v2_qa()


if __name__ == "__main__":
    main()
