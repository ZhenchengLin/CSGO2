"""
Canonical live/replay state representation for V0.

Adapters convert source-specific data into these structures.

Examples:
    ReplayAdapter
    GSIAdapter

The FeatureBuilder should depend on canonical state,
not directly on GSI JSON or Awpy objects.
"""

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class V0PlayerState:
    """
    One player's observable state at one moment.
    """

    steamid: str
    side: str

    x: float
    y: float
    z: float

    health: float
    armor: float

    current_equip_value: float


@dataclass(frozen=True)
class V0BombState:
    """
    Current observable bomb state.
    """

    x: float
    y: float
    z: float

    state: str

    carrier_steamid: str | None = None


@dataclass(frozen=True)
class V0Frame:
    """
    Canonical observable game frame.

    timestamp_sec should increase monotonically
    within the source stream.
    """

    timestamp_sec: float

    round_num: int

    phase: str

    players: tuple[V0PlayerState, ...]

    bomb: V0BombState


def frame_to_snapshot(
    frame: V0Frame,
) -> pl.DataFrame:
    """
    Convert a canonical V0Frame into the player
    snapshot format expected by build_feature_row().
    """

    rows = []

    for player in frame.players:

        side = player.side.lower()

        if side not in {
            "t",
            "ct",
        }:
            raise ValueError(
                f"Unknown side: {player.side}"
            )

        rows.append({
            "steamid":
                player.steamid,

            "side":
                side,

            "X":
                float(player.x),

            "Y":
                float(player.y),

            "Z":
                float(player.z),

            "health":
                float(player.health),

            "armor":
                float(player.armor),

            "current_equip_value":
                float(
                    player.current_equip_value
                ),
        })

    if not rows:
        raise ValueError(
            "V0Frame contains no players"
        )

    return pl.DataFrame(
        rows,
        schema={
            "steamid":
                pl.String,

            "side":
                pl.String,

            "X":
                pl.Float64,

            "Y":
                pl.Float64,

            "Z":
                pl.Float64,

            "health":
                pl.Float64,

            "armor":
                pl.Float64,

            "current_equip_value":
                pl.Float64,
        },
    )
