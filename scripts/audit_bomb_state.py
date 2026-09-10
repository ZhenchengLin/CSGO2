import polars as pl
from awpy import Demo


DEMO_PATH = "data/raw/havu-vs-mellren-m1-mirage.dem"
SNAPSHOT_PATH = "data/interim/v0_snapshot_players.parquet"
OUTPUT_PATH = "data/interim/v0_bomb_state_audit.parquet"


pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(220)


# --------------------------------------------------
# 1. Load our 49 valid observation snapshots
# --------------------------------------------------

players = pl.read_parquet(SNAPSHOT_PATH)

observations = (
    players
    .select([
        "round_num",
        "horizon_sec",
        "target_tick",
        "label",
    ])
    .unique()
    .sort([
        "round_num",
        "horizon_sec",
    ])
)


print("\n" + "=" * 90)
print("OBSERVATIONS")
print("=" * 90)

print("Observation count:", observations.height)


# --------------------------------------------------
# 2. Parse bomb events from the demo
# --------------------------------------------------

demo = Demo(DEMO_PATH, verbose=True)
demo.parse()

bomb_events = (
    demo.bomb
    .select([
        "round_num",
        "tick",
        "event",
        "X",
        "Y",
        "Z",
        "steamid",
        "name",
        "bombsite",
    ])
    .sort([
        "round_num",
        "tick",
    ])
)


# --------------------------------------------------
# Helper:
# Does a player currently have the C4?
# --------------------------------------------------

def inventory_has_c4(inventory):
    if inventory is None:
        return False

    return any(
        "C4" in item
        for item in inventory
    )


# --------------------------------------------------
# 3. Reconstruct bomb state for each observation
# --------------------------------------------------

results = []


for observation in observations.iter_rows(named=True):

    round_num = observation["round_num"]
    horizon_sec = observation["horizon_sec"]
    target_tick = observation["target_tick"]
    label = observation["label"]


    # ----------------------------------------------
    # Player states at this exact observation
    # ----------------------------------------------

    snapshot = (
        players
        .filter(
            (pl.col("round_num") == round_num)
            & (pl.col("horizon_sec") == horizon_sec)
        )
    )


    # ----------------------------------------------
    # Find T players whose inventory contains C4
    # ----------------------------------------------

    carriers = []

    for player in snapshot.iter_rows(named=True):

        if (
            player["side"] == "t"
            and inventory_has_c4(player["inventory"])
        ):
            carriers.append(player)


    latest_event = (
        bomb_events
        .filter(
            (pl.col("round_num") == round_num)
            & (pl.col("tick") <= target_tick)
        )
        .tail(1)
    )


    latest_event_name = None
    latest_event_tick = None


    if latest_event.height == 1:
        latest_event_name = latest_event["event"][0]
        latest_event_tick = latest_event["tick"][0]


    # ----------------------------------------------
    # Case 1:
    # Exactly one player carries C4
    # ----------------------------------------------

    if len(carriers) == 1:

        carrier = carriers[0]

        bomb_state = "CARRIED"

        bomb_x = carrier["X"]
        bomb_y = carrier["Y"]
        bomb_z = carrier["Z"]

        carrier_name = carrier["name"]
        carrier_steamid = carrier["steamid"]

        reconstruction_source = "PLAYER_INVENTORY"


    # ----------------------------------------------
    # Case 2:
    # More than one player appears to carry C4
    # This should never happen
    # ----------------------------------------------

    elif len(carriers) > 1:

        bomb_state = "ERROR_MULTIPLE_CARRIERS"

        bomb_x = None
        bomb_y = None
        bomb_z = None

        carrier_name = None
        carrier_steamid = None

        reconstruction_source = "INVALID"


    # ----------------------------------------------
    # Case 3:
    # Nobody carries C4
    #
    # Check latest bomb event.
    # If latest event was DROP,
    # bomb should be lying on the map.
    # ----------------------------------------------

    else:

        carrier_name = None
        carrier_steamid = None

        if latest_event.height == 0:

            bomb_state = "UNKNOWN_NO_EVENT"

            bomb_x = None
            bomb_y = None
            bomb_z = None

            reconstruction_source = "UNKNOWN"


        elif latest_event_name == "drop":

            bomb_state = "DROPPED"

            bomb_x = latest_event["X"][0]
            bomb_y = latest_event["Y"][0]
            bomb_z = latest_event["Z"][0]

            reconstruction_source = "BOMB_DROP_EVENT"


        elif latest_event_name == "pickup":

            bomb_state = "INCONSISTENT_PICKUP_NO_C4"

            bomb_x = latest_event["X"][0]
            bomb_y = latest_event["Y"][0]
            bomb_z = latest_event["Z"][0]

            reconstruction_source = "BOMB_PICKUP_EVENT"


        elif latest_event_name == "plant":

            bomb_state = "ERROR_PLANT_BEFORE_OBSERVATION"

            bomb_x = latest_event["X"][0]
            bomb_y = latest_event["Y"][0]
            bomb_z = latest_event["Z"][0]

            reconstruction_source = "BOMB_PLANT_EVENT"


        else:

            bomb_state = f"UNKNOWN_EVENT_{latest_event_name}"

            bomb_x = latest_event["X"][0]
            bomb_y = latest_event["Y"][0]
            bomb_z = latest_event["Z"][0]

            reconstruction_source = "BOMB_EVENT"


    results.append({
        "round_num": round_num,
        "horizon_sec": horizon_sec,
        "target_tick": target_tick,
        "label": label,

        "bomb_state": bomb_state,

        "bomb_x": bomb_x,
        "bomb_y": bomb_y,
        "bomb_z": bomb_z,

        "carrier_name": carrier_name,
        "carrier_steamid": carrier_steamid,

        "latest_bomb_event": latest_event_name,
        "latest_bomb_event_tick": latest_event_tick,

        "reconstruction_source": reconstruction_source,
    })


audit = pl.DataFrame(results)


# --------------------------------------------------
# 4. Summary
# --------------------------------------------------

print("\n" + "=" * 90)
print("BOMB STATE SUMMARY")
print("=" * 90)

print(
    audit
    .group_by([
        "bomb_state",
        "reconstruction_source",
    ])
    .len()
    .sort("bomb_state")
)


# --------------------------------------------------
# 5. Show all reconstructed observation bomb states
# --------------------------------------------------

print("\n" + "=" * 90)
print("ALL OBSERVATION BOMB STATES")
print("=" * 90)

print(
    audit.select([
        "round_num",
        "horizon_sec",
        "target_tick",
        "label",
        "bomb_state",
        "carrier_name",
        "bomb_x",
        "bomb_y",
        "bomb_z",
        "latest_bomb_event",
        "latest_bomb_event_tick",
        "reconstruction_source",
    ])
)


# --------------------------------------------------
# 6. Find suspicious observations
# --------------------------------------------------

valid_states = [
    "CARRIED",
    "DROPPED",
]

suspicious = (
    audit
    .filter(
        ~pl.col("bomb_state").is_in(valid_states)
    )
)


print("\n" + "=" * 90)
print("SUSPICIOUS OBSERVATIONS")
print("=" * 90)

if suspicious.height == 0:
    print("None")
else:
    print(suspicious)


# --------------------------------------------------
# 7. Basic safety checks
# --------------------------------------------------

assert (
    audit
    .filter(
        pl.col("bomb_state") == "ERROR_MULTIPLE_CARRIERS"
    )
    .height
    == 0
)

assert (
    audit
    .filter(
        pl.col("bomb_state") == "ERROR_PLANT_BEFORE_OBSERVATION"
    )
    .height
    == 0
)


# --------------------------------------------------
# 8. Save audit result
# --------------------------------------------------

audit.write_parquet(OUTPUT_PATH)

print("\nSaved:")
print(OUTPUT_PATH)
