from awpy import Demo

demo = Demo(
    "data/raw/havu-vs-mellren-m1-mirage.dem",
    verbose=True,
)

demo.parse()

print("\n=== HEADER ===")
print(demo.header)

print("\n=== FIRST 5 ROUNDS ===")
print(demo.rounds.head(5))
