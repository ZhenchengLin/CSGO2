"""
V0 historical-demo timing contract.

The 20-demo development corpus was independently audited using
the parsed `game_time` clock.

Measured raw demo clock:
    64 raw ticks / game second

Important:
Awpy currently defaults Demo.tickrate to 128 when no explicit
tickrate is supplied. That default does NOT match the raw tick clock
of the V0 CS2 demo corpus.

All historical V0 demo parsing must therefore explicitly use 64.
"""

from pathlib import Path

from awpy import Demo


V0_DEMO_TICKS_PER_SECOND = 64


def open_v0_demo(
    path,
    *,
    verbose=False,
):
    """
    Open a historical V0 CS2 demo with the audited raw tick clock.
    """

    demo = Demo(
        Path(path),
        tickrate=V0_DEMO_TICKS_PER_SECOND,
        verbose=verbose,
    )

    if (
        demo.tickrate
        != V0_DEMO_TICKS_PER_SECOND
    ):
        raise RuntimeError(
            "V0 demo timing contract violated: "
            f"expected "
            f"{V0_DEMO_TICKS_PER_SECOND}, "
            f"got {demo.tickrate}"
        )

    return demo


def seconds_to_demo_ticks(
    seconds,
):
    """
    Convert real game seconds into raw demo ticks.
    """

    return int(
        round(
            float(seconds)
            * V0_DEMO_TICKS_PER_SECOND
        )
    )
