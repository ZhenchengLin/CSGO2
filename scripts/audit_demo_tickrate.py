from pathlib import Path
from statistics import median

import polars as pl
from awpy import Demo


RAW_DIR = Path("data/raw")
MANIFEST = RAW_DIR / "demo_manifest.csv"

manifest = pl.read_csv(
    MANIFEST,
    null_values=[""],
    infer_schema_length=1000,
)

results = []

print("\n" + "=" * 90)
print("CS2 DEMO RAW TICK CLOCK AUDIT")
print("=" * 90)

for i, row in enumerate(
    manifest.iter_rows(named=True),
    start=1,
):
    filename = row["demo filename"]
    path = RAW_DIR / filename

    print(
        f"\n[{i:02d}/{manifest.height:02d}] "
        f"{filename}"
    )

    demo = Demo(
        path,
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
                and dtime < 2
            ):
                ratios.append(
                    dtick / dtime
                )

        previous = current

    if not ratios:
        raise RuntimeError(
            f"No usable clock samples for {filename}"
        )

    measured = median(
        ratios
    )

    print(
        f"  configured Awpy tickrate: "
        f"{demo.tickrate}"
    )

    print(
        f"  measured raw ticks/sec:   "
        f"{measured:.6f}"
    )

    results.append({
        "demo_filename":
            filename,

        "awpy_configured_tickrate":
            demo.tickrate,

        "measured_raw_ticks_per_sec":
            measured,

        "n_clock_intervals":
            len(ratios),
    })


result = pl.DataFrame(
    results
)

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

print(
    result.select([
        "demo_filename",
        "awpy_configured_tickrate",
        "measured_raw_ticks_per_sec",
    ])
)

print("\nMeasured values:")

print(
    result
    .group_by(
        "measured_raw_ticks_per_sec"
    )
    .len()
    .sort(
        "measured_raw_ticks_per_sec"
    )
)

output = Path(
    "data/interim/demo_tickrate_audit.csv"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result.write_csv(
    output
)

print(
    f"\nSaved: {output}"
)

print(
    "\n✅ DEMO TICK CLOCK AUDIT COMPLETE"
)
