"""
Summarize the V2 confirmation candidate pool.

This script is outcome-blind.

It does NOT:
- run V0
- run H2
- inspect predictions
- calculate model metrics
- change confirmation membership

It only summarizes acquisition and data diversity.
"""

from pathlib import Path

import polars as pl


AUDIT_PATH = Path(
    "data/interim/v2_intake_audit.csv"
)

LEGACY_MANIFEST_PATH = Path(
    "data/raw/demo_manifest.csv"
)

TARGET_MATCHES = 30

ELIGIBLE_STATUSES = [
    "eligible_intake",
    "eligible_intake_with_round_exclusions",
]


def normalized_team(value):
    if value is None:
        return None

    return (
        str(value)
        .strip()
        .lower()
    )


def main():
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing intake audit: {AUDIT_PATH}"
        )

    audit = pl.read_csv(
        AUDIT_PATH,
        null_values=[""],
        infer_schema_length=1000,
    )

    eligible = (
        audit
        .filter(
            pl.col("status")
            .is_in(
                ELIGIBLE_STATUSES
            )
        )
        .sort([
            "match_date",
            "demo_filename",
        ])
    )

    n_matches = eligible.height

    remaining = max(
        TARGET_MATCHES
        - n_matches,
        0,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "V2 — D_CONFIRM ACQUISITION PROGRESS"
    )

    print(
        "=" * 100
    )

    print(
        f"\nTarget matches:        {TARGET_MATCHES}"
    )

    print(
        f"Eligible matches:      {n_matches}"
    )

    print(
        f"Still needed:          {remaining}"
    )

    if n_matches > 0:
        print(
            f"Total observations:    "
            f"{eligible['n_observations'].sum()}"
        )

        print(
            f"Total parsed rounds:   "
            f"{eligible['n_rounds'].sum()}"
        )

        print(
            f"Valid rounds:          "
            f"{eligible['n_valid_rounds'].sum()}"
        )

        print(
            f"Excluded rounds:       "
            f"{eligible['n_invalid_rounds'].sum()}"
        )

        print(
            "\nObservations by horizon:"
        )

        for horizon in [
            10,
            20,
            30,
            40,
        ]:
            print(
                f"  {horizon:>2}s: "
                f"{eligible[f'n_obs_{horizon}'].sum()}"
            )

        print(
            "\nRound labels:"
        )

        print(
            "  A_PLANT:  "
            f"{eligible['n_a_plant'].sum()}"
        )

        print(
            "  B_PLANT:  "
            f"{eligible['n_b_plant'].sum()}"
        )

        print(
            "  NO_PLANT: "
            f"{eligible['n_no_plant'].sum()}"
        )

        dates = (
            eligible
            .select(
                pl.col("match_date")
                .str.to_date(
                    strict=False
                )
                .alias("date")
            )
            ["date"]
            .drop_nulls()
        )

        if len(dates) > 0:
            print(
                "\nDate range:"
            )

            print(
                f"  {dates.min()} "
                f"→ {dates.max()}"
            )

    # --------------------------------------------------
    # Candidate team diversity
    # --------------------------------------------------

    candidate_teams = set()

    for row in eligible.select([
        "team1",
        "team2",
    ]).iter_rows(named=True):

        for field in [
            "team1",
            "team2",
        ]:
            team = normalized_team(
                row[field]
            )

            if team:
                candidate_teams.add(
                    team
                )

    print(
        "\nCandidate diversity:"
    )

    print(
        f"  Unique teams:        "
        f"{len(candidate_teams)}"
    )

    if (
        "event"
        in eligible.columns
    ):
        events = (
            eligible[
                "event"
            ]
            .drop_nulls()
            .unique()
        )

        print(
            f"  Unique events:       "
            f"{len(events)}"
        )

    # --------------------------------------------------
    # Legacy team overlap
    # --------------------------------------------------

    if LEGACY_MANIFEST_PATH.exists():

        legacy = pl.read_csv(
            LEGACY_MANIFEST_PATH,
            null_values=[""],
            infer_schema_length=1000,
        )

        legacy_teams = set()

        for row in legacy.select([
            "team1",
            "team2",
        ]).iter_rows(named=True):

            for field in [
                "team1",
                "team2",
            ]:
                team = normalized_team(
                    row[field]
                )

                if team:
                    legacy_teams.add(
                        team
                    )

        overlap = sorted(
            candidate_teams
            & legacy_teams
        )

        new_teams = sorted(
            candidate_teams
            - legacy_teams
        )

        print(
            "\nLegacy-team overlap:"
        )

        print(
            f"  Legacy unique teams: "
            f"{len(legacy_teams)}"
        )

        print(
            f"  Candidate teams seen "
            f"in legacy: {len(overlap)}"
        )

        print(
            f"  Candidate teams new: "
            f"{len(new_teams)}"
        )

        if overlap:
            print(
                "\n  Overlapping teams:"
            )

            for team in overlap:
                print(
                    f"    - {team}"
                )

    # --------------------------------------------------
    # Match table
    # --------------------------------------------------

    print(
        "\n"
        + "-" * 100
    )

    print(
        "ELIGIBLE MATCHES"
    )

    print(
        "-" * 100
    )

    if n_matches > 0:
        for i, row in enumerate(
            eligible.iter_rows(
                named=True
            ),
            start=1,
        ):
            print(
                f"{i:02d}. "
                f"{row['match_date']} | "
                f"{row['team1']} vs "
                f"{row['team2']} | "
                f"obs={row['n_observations']} | "
                f"{row['status']}"
            )

    # --------------------------------------------------
    # Freeze readiness
    # --------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    if n_matches >= TARGET_MATCHES:

        print(
            "✅ READY TO FREEZE D_CONFIRM"
        )

        print(
            "\nNext command:"
        )

        print(
            "python "
            "scripts/freeze_v2_confirm_manifest.py"
        )

    else:

        print(
            "⏳ D_CONFIRM NOT READY"
        )

        print(
            f"Acquire at least "
            f"{remaining} more eligible "
            "Mirage demos."
        )

        print(
            "\nDO NOT run V0 or H2 "
            "confirmation yet."
        )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
