from pathlib import Path

import polars as pl


OLD_FOLD_PATH = Path(
    "data/processed/"
    "v0_a1_oof_predictions.parquet"
)

NEW_DATASET_PATH = Path(
    "data/processed/"
    "v0_dataset_v2_timing_corrected.parquet"
)

OUTPUT_PATH = Path(
    "data/interim/"
    "v0_timing_corrected_fold_map.csv"
)


old = pl.read_parquet(
    OLD_FOLD_PATH
)

new = pl.read_parquet(
    NEW_DATASET_PATH
)


print("\n" + "=" * 90)
print("V0 CORRECTED EVALUATION — FOLD FREEZE")
print("=" * 90)


# ==================================================
# One old fold per match
# ==================================================

fold_map = (
    old
    .select([
        "demo_filename",
        "cv_fold",
    ])
    .unique()
    .sort([
        "cv_fold",
        "demo_filename",
    ])
)


fold_count_per_match = (
    fold_map
    .group_by(
        "demo_filename"
    )
    .len()
)


assert (
    fold_count_per_match[
        "len"
    ].max()
    == 1
)


print(
    "\nOld match → fold mapping:"
)

print(
    fold_map
)


print(
    "\nMatches:",
    fold_map[
        "demo_filename"
    ].n_unique(),
)


print(
    "\nMatches per fold:"
)

print(
    fold_map
    .group_by(
        "cv_fold"
    )
    .len()
    .sort(
        "cv_fold"
    )
)


# ==================================================
# Confirm corrected dataset contains same matches
# ==================================================

old_matches = set(
    fold_map[
        "demo_filename"
    ].to_list()
)

new_matches = set(
    new[
        "demo_filename"
    ].unique().to_list()
)


print(
    "\nOld matches:",
    len(old_matches),
)

print(
    "New matches:",
    len(new_matches),
)


assert (
    old_matches
    == new_matches
), (
    f"Match sets differ.\n"
    f"Only old: {old_matches - new_matches}\n"
    f"Only new: {new_matches - old_matches}"
)


# ==================================================
# Attach frozen fold to all corrected observations
# ==================================================

new_with_fold = (
    new
    .join(
        fold_map,
        on="demo_filename",
        how="left",
    )
)


assert (
    new_with_fold[
        "cv_fold"
    ].null_count()
    == 0
)

assert (
    new_with_fold.height
    == 1686
)


print(
    "\nCorrected observations per frozen fold:"
)

print(
    new_with_fold
    .group_by(
        "cv_fold"
    )
    .agg([
        pl.len().alias(
            "n_observations"
        ),

        pl.col(
            "demo_filename"
        )
        .n_unique()
        .alias(
            "n_matches"
        ),
    ])
    .sort(
        "cv_fold"
    )
)


# ==================================================
# Leakage check
# ==================================================

for fold in sorted(
    new_with_fold[
        "cv_fold"
    ].unique().to_list()
):

    test_matches = set(
        new_with_fold
        .filter(
            pl.col(
                "cv_fold"
            )
            == fold
        )[
            "demo_filename"
        ]
        .unique()
        .to_list()
    )

    train_matches = (
        new_matches
        - test_matches
    )

    assert not (
        train_matches
        & test_matches
    )


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fold_map.write_csv(
    OUTPUT_PATH
)


print(
    "\nSaved frozen fold map:"
)

print(
    OUTPUT_PATH
)


print(
    "\n✅ OLD MATCH-LEVEL CV FOLDS "
    "SUCCESSFULLY FROZEN FOR CORRECTED V0"
)
