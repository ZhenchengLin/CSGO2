"""
CS2 Game State Integration adapter for V0.

The adapter converts one observer/spectator GSI payload into
the canonical V0Frame representation.

It performs source-specific parsing only.

It does NOT:
- compute model features
- run the model
- use future round outcomes
- infer final plant site
"""

from collections.abc import Mapping

from cs2_tactical_intelligence.v0.state import (
    V0BombState,
    V0Frame,
    V0PlayerState,
)


# ==================================================
# Basic parsing helpers
# ==================================================

def parse_vector3(value):
    """
    Parse CS2 GSI vector strings such as:

        "-717.51, -1059.59, -202.17"

    into:

        (-717.51, -1059.59, -202.17)
    """

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            f"Expected GSI vector string, got: "
            f"{value!r}"
        )

    parts = [
        part.strip()
        for part in value.split(",")
    ]

    if len(parts) != 3:
        raise ValueError(
            f"Expected three coordinates, got: "
            f"{value!r}"
        )

    try:
        return tuple(
            float(part)
            for part in parts
        )

    except ValueError as exc:
        raise ValueError(
            f"Invalid GSI vector: "
            f"{value!r}"
        ) from exc


def normalize_side(value):
    """
    Convert GSI team names to the internal V0 side contract.
    """

    if value == "T":
        return "t"

    if value == "CT":
        return "ct"

    raise ValueError(
        f"Unsupported GSI team: {value!r}"
    )


# ==================================================
# GSI Adapter
# ==================================================

class V0GSIAdapter:

    def __init__(
        self,
        *,
        expected_map="de_mirage",
    ):
        self.expected_map = (
            expected_map
        )

    # ==================================================
    # Player parsing
    # ==================================================

    def _parse_players(
        self,
        payload: Mapping,
    ):
        allplayers = payload.get(
            "allplayers"
        )

        if not isinstance(
            allplayers,
            Mapping,
        ):
            raise ValueError(
                "GSI payload does not contain "
                "observer allplayers data."
            )

        players = []

        for steamid, player in (
            allplayers.items()
        ):

            if not isinstance(
                player,
                Mapping,
            ):
                raise ValueError(
                    f"Invalid player payload "
                    f"for {steamid}"
                )

            team = player.get(
                "team"
            )

            side = normalize_side(
                team
            )

            if "position" not in player:
                raise ValueError(
                    f"Missing position for "
                    f"player {steamid}"
                )

            (
                x,
                y,
                z,
            ) = parse_vector3(
                player["position"]
            )

            state = player.get(
                "state"
            )

            if not isinstance(
                state,
                Mapping,
            ):
                raise ValueError(
                    f"Missing state for "
                    f"player {steamid}"
                )

            required_state = [
                "health",
                "armor",
                "equip_value",
            ]

            missing = [
                field
                for field
                in required_state
                if field not in state
            ]

            if missing:
                raise ValueError(
                    f"Missing GSI player state "
                    f"for {steamid}: {missing}"
                )

            players.append(
                V0PlayerState(
                    steamid=str(
                        steamid
                    ),

                    side=side,

                    x=float(x),
                    y=float(y),
                    z=float(z),

                    health=float(
                        state["health"]
                    ),

                    armor=float(
                        state["armor"]
                    ),

                    current_equip_value=float(
                        state[
                            "equip_value"
                        ]
                    ),
                )
            )

        if not players:
            raise ValueError(
                "GSI allplayers is empty."
            )

        return tuple(
            players
        )

    # ==================================================
    # Bomb parsing
    # ==================================================

    def _parse_bomb(
        self,
        payload: Mapping,
        players,
    ):
        bomb = payload.get(
            "bomb"
        )

        if not isinstance(
            bomb,
            Mapping,
        ):
            raise ValueError(
                "GSI payload does not contain "
                "bomb state."
            )

        bomb_state = bomb.get(
            "state"
        )

        if bomb_state is None:
            raise ValueError(
                "GSI bomb state is missing."
            )

        carrier = bomb.get(
            "player"
        )

        if carrier is not None:
            carrier = str(
                carrier
            )

        # ------------------------------------------
        # Preferred source:
        # explicit GSI bomb position
        # ------------------------------------------

        position = bomb.get(
            "position"
        )

        if position is not None:

            (
                x,
                y,
                z,
            ) = parse_vector3(
                position
            )

        # ------------------------------------------
        # Fallback:
        # carried bomb may be represented by
        # carrier identity, so use carrier position.
        # ------------------------------------------

        elif carrier is not None:

            carrier_state = next(
                (
                    player
                    for player
                    in players
                    if player.steamid
                    == carrier
                ),
                None,
            )

            if carrier_state is None:
                raise ValueError(
                    "Bomb carrier not found "
                    "in allplayers."
                )

            x = carrier_state.x
            y = carrier_state.y
            z = carrier_state.z

        else:
            raise ValueError(
                "Cannot reconstruct GSI bomb "
                "position."
            )

        return V0BombState(
            x=float(x),
            y=float(y),
            z=float(z),

            state=str(
                bomb_state
            ).lower(),

            carrier_steamid=carrier,
        )

    # ==================================================
    # Public conversion
    # ==================================================

    def to_frame(
        self,
        payload: Mapping,
        *,
        timestamp_sec: float,
    ):
        """
        Convert one full observer GSI payload
        into a canonical V0Frame.

        timestamp_sec is deliberately supplied
        externally.

        Later the HTTP receiver will use a
        monotonic local receive clock so motion
        history is not dependent on wall-clock
        timestamps contained in the GSI payload.
        """

        if not isinstance(
            payload,
            Mapping,
        ):
            raise ValueError(
                "GSI payload must be a mapping."
            )

        map_data = payload.get(
            "map"
        )

        if not isinstance(
            map_data,
            Mapping,
        ):
            raise ValueError(
                "Missing GSI map block."
            )

        map_name = map_data.get(
            "name"
        )

        if (
            self.expected_map
            is not None
            and map_name
            != self.expected_map
        ):
            raise ValueError(
                f"V0 expects "
                f"{self.expected_map}, "
                f"got {map_name!r}"
            )

        if "round" not in map_data:
            raise ValueError(
                "Missing map.round."
            )

        # GSI map.round means number of rounds
        # already completed.
        #
        # Offline V0 round_num is 1-based.
        round_num = (
            int(
                map_data["round"]
            )
            + 1
        )

        round_data = payload.get(
            "round",
            {},
        )

        phase_data = payload.get(
            "phase_countdowns",
            {},
        )

        phase = (
            phase_data.get(
                "phase"
            )
            or round_data.get(
                "phase"
            )
        )

        if phase is None:
            raise ValueError(
                "Cannot determine current "
                "round phase."
            )

        players = (
            self._parse_players(
                payload
            )
        )

        bomb = (
            self._parse_bomb(
                payload,
                players,
            )
        )

        return V0Frame(
            timestamp_sec=float(
                timestamp_sec
            ),

            round_num=round_num,

            phase=str(
                phase
            ).lower(),

            players=players,

            bomb=bomb,
        )
