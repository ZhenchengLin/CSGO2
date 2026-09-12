"""
Freeze the first V2 independent-confirmation manifest.

This script performs selection only.

It does NOT:
- train V0
- train H2
- calculate model probabilities
- calculate model metrics

The selection rule is defined in:
    docs/v2_confirm_selection.md
"""

from pathlib import Path

import polars as pl


AUDIT_PATH = Path(
    "data/interim/v2_intake_audit.csv"
)

OUTPUT_PATH = Path(
    "docs/v2_confirm_manifest.csv"
)

TARGET_N = 30

LEGACY_CUTOFF = (
    "2026-09-08"
)

EXPECTED_MAP = (
    "de_mirage"
)

EXPECTED_ROLE = (
    "confirm_candidate"
)


def main():
    print(
        "\n"
        + "=" * 100
    )

    print(
        "V2 — FREEZE D_CONFIRM MANIFEST"
    )

    print(
        "=" * 100
    )

    if OUTPUT_PATH.exists():
        raise RuntimeError(
            "Frozen confirmation manifest already exists:\n"
            f"{OUTPUT_PATH}\n"
            "Refusing to overwrite it."
        )

    if not AUDIT_PATH.exists():
        raise FileNotFoundError(
            "V2 intake audit does not exist:\n"
            f"{AUDIT_PATH}"
        )

    audit = pl.read_csv(
        AUDIT_PATH,
        null_values=[""],
        infer_schema_length=1000,
    )

    required_columns = [
        "demo_filename",
        "sha256",
        "match_date",
        "event",
        "team1",
        "team2",
        "source",
        "source_url",
        "scientific_role",
        "status",
        "map_name",
        "tick_clock_ok",
        "metadata_complete",
        "measured_raw_ticks_per_sec",
        "n_rounds",
        "n_observations",
    ]

    missing = [
        column
        for column
        in required_columns
        if column not in audit.columns
    ]

    if missing:
        raise RuntimeError(
            "Audit is missing required columns: "
            + ", ".join(missing)
        )

    candidates = (
        audit
        .with_columns(
            pl.col("match_date")
            .str.to_date(
                strict=False
            )
            .alias(
                "_match_date"
            )
        )
        .filter(
            pl.col("status")
            == "eligible_intake"
        )
        .filter(
            pl.col("scientific_role")
            == EXPECTED_ROLE
        )
        .filter(
            pl.col("map_name")
            == EXPECTED_MAP
        )
        .filter(
            pl.col("tick_clock_ok")
            == True
        )
        .filter(
            pl.col("metadata_complete")
            == True
        )
        .filter(
            pl.col("_match_date")
            > pl.lit(
                LEGACY_CUTOFF
            ).str.to_date()
        )
    )

    if candidates.height == 0:
        print(
            "\nEligible confirmation candidates: 0"
        )

        print(
            "\nNo manifest created."
        )

        return

    duplicate_hashes = (
        candidates
        .group_by(
            "sha256"
        )
        .len()
        .filter(
            pl.col("len") > 1
        )
    )

    if duplicate_hashes.height > 0:
        raise RuntimeError(
            "Duplicate SHA256 values remain "
            "among eligible candidates."
        )

    candidates = (
        candidates
        .sort([
            "_match_date",
            "demo_filename",
        ])
    )

    print(
        "\nEligible confirmation candidates: "
        f"{candidates.height}"
    )

    print(
        "Target confirmation size: "
        f"{TARGET_N}"
    )

    if (
        candidates.height
        < TARGET_N
    ):
        remaining = (
            TARGET_N
            - candidates.height
        )

        print(
            "\nNot enough eligible demos yet."
        )

        print(
            f"Need {remaining} more."
        )

        print(
            "\nNo manifest created."
        )

        return

    selected = (
        candidates
        .head(
            TARGET_N
        )
        .with_row_index(
            "selection_rank",
            offset=1,
        )
        .select([
            "selection_rank",
            "demo_filename",
            "sha256",
            "match_date",
            "event",
            "team1",
            "team2",
            "source",
            "source_url",
            "scientific_role",
            "map_name",
            "measured_raw_ticks_per_sec",
            "n_rounds",
            "n_observations",
        ])
    )

    assert (
        selected.height
        == TARGET_N
    )

    assert (
        selected[
            "sha256"
        ].n_unique()
        == TARGET_N
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.write_csv(
        OUTPUT_PATH
    )

    print(
        "\nSelected confirmation demos:"
    )

    print(
        selected.select([
            "selection_rank",
            "match_date",
            "team1",
            "team2",
            "demo_filename",
        ])
    )

    print(
        "\nFrozen manifest:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Commit this manifest before "
        "running V0 or H2 confirmation."
    )

    print(
        "\n✅ D_CONFIRM SELECTION FROZEN"
    )


if __name__ == "__main__":
    main()
