from awpy import Demo


DEMO_PATH = "data/raw/havu-vs-mellren-m1-mirage.dem"


demo = Demo(DEMO_PATH, verbose=True)
demo.parse()


print("\n" + "=" * 70)
print("HEADER")
print("=" * 70)

for key, value in demo.header.items():
    print(f"{key}: {value}")


print("\n" + "=" * 70)
print("ROUNDS")
print("=" * 70)

print("Shape:", demo.rounds.shape)
print("Columns:", demo.rounds.columns)
print(demo.rounds)


print("\n" + "=" * 70)
print("DETECTED EVENTS")
print("=" * 70)

print("Number of detected event types:", len(demo.detected_events))

for event in sorted(demo.detected_events):
    print(event)


print("\n" + "=" * 70)
print("DATA TABLE INVENTORY")
print("=" * 70)


tables = [
    "ticks",
    "kills",
    "damages",
    "bomb",
    "grenades",
    "shots",
    "smokes",
    "infernos",
    "footsteps",
    "player_round_totals",
]


for table_name in tables:
    print(f"\n--- {table_name.upper()} ---")

    try:
        table = getattr(demo, table_name)

        print("Shape:", table.shape)
        print("Columns:", table.columns)

        if table.height > 0:
            print(table.head(3))
        else:
            print("EMPTY TABLE")

    except Exception as error:
        print("Could not inspect:", error)
