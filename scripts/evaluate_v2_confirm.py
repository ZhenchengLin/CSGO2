from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, f1_score, log_loss
from xgboost import XGBClassifier


DATASET = Path("data/processed/v2_confirm_dataset.parquet")
FREEZE = Path("docs/v2_confirm_model_freeze.json")
VERIFY = Path("docs/v2_confirm_model_verification.json")

RESULT = Path("artifacts/v2_confirm_evaluation.json")
METRICS = Path("artifacts/v2_confirm_metrics.csv")
PREDICTIONS = Path("data/processed/v2_confirm_predictions.parquet")

EXPECTED_SHA = (
    "e781dbebe0956322476cd8841675ad9a22e9d680c6dec7a47f5d23c35829c367"
)

EXPECTED_ROWS = 2269
EXPECTED_MATCHES = 30

BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 42


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def tracked_clean() -> bool:
    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip() == ""


def logit(p):
    p = np.clip(
        p,
        1e-6,
        1.0 - 1e-6,
    )

    return np.log(
        p / (1.0 - p)
    )


def sigmoid(x):
    return 1.0 / (
        1.0 + np.exp(-x)
    )


def brier(y, p):
    one_hot = np.eye(3)[y]

    return float(
        np.mean(
            np.sum(
                (p - one_hot) ** 2,
                axis=1,
            )
        )
    )


def score(y, p):
    pred = np.argmax(
        p,
        axis=1,
    )

    return {
        "n": int(len(y)),
        "log_loss": float(
            log_loss(
                y,
                p,
                labels=[0, 1, 2],
            )
        ),
        "brier_score": brier(
            y,
            p,
        ),
        "accuracy": float(
            accuracy_score(
                y,
                pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
    }


def main():
    print("=" * 100)
    print("V2 — ONE FORMAL D_CONFIRM EVALUATION")
    print("=" * 100)

    # ---------------------------------------------------------
    # One-shot protection
    # ---------------------------------------------------------

    for path in [
        RESULT,
        METRICS,
        PREDICTIONS,
    ]:
        require(
            not path.exists(),
            f"Formal output already exists: {path}",
        )

    require(
        tracked_clean(),
        "Tracked worktree is not clean. "
        "Commit evaluator before scoring.",
    )

    # ---------------------------------------------------------
    # Frozen identities
    # ---------------------------------------------------------

    require(
        sha256(DATASET) == EXPECTED_SHA,
        "D_CONFIRM SHA mismatch.",
    )

    freeze = json.loads(
        FREEZE.read_text()
    )

    verification = json.loads(
        VERIFY.read_text()
    )

    require(
        verification["verification_status"]
        == "PASS",
        "Model freeze verification is not PASS.",
    )

    require(
        verification[
            "scientific_metrics_calculated"
        ]
        is False,
        "Verification record already contains metrics.",
    )

    df = pl.read_parquet(
        DATASET
    )

    require(
        df.height == EXPECTED_ROWS,
        "D_CONFIRM row count mismatch.",
    )

    require(
        df["demo_filename"].n_unique()
        == EXPECTED_MATCHES,
        "D_CONFIRM match count mismatch.",
    )

    labels = freeze["labels"]

    require(
        labels
        == [
            "A_PLANT",
            "B_PLANT",
            "NO_PLANT",
        ],
        "Unexpected label order.",
    )

    label_to_id = {
        label: i
        for i, label
        in enumerate(labels)
    }

    y = np.array([
        label_to_id[label]
        for label
        in df["label"].to_list()
    ])

    # ---------------------------------------------------------
    # Frozen feature contracts
    # ---------------------------------------------------------

    v0_features = freeze[
        "v0"
    ][
        "features"
    ]

    plant_features = freeze[
        "h2"
    ][
        "plant_features"
    ]

    site_features = freeze[
        "h2"
    ][
        "site_features"
    ]

    X_v0 = (
        df
        .select(v0_features)
        .to_pandas()
    )

    X_plant = (
        df
        .select(plant_features)
        .to_pandas()
    )

    X_site = (
        df
        .select(site_features)
        .to_pandas()
    )

    # ---------------------------------------------------------
    # Reload frozen models
    # ---------------------------------------------------------

    artifacts = freeze["artifacts"]

    for name, item in artifacts.items():
        path = Path(
            item["path"]
        )

        require(
            sha256(path)
            == item["sha256"],
            f"{name} artifact SHA mismatch.",
        )

    v0 = XGBClassifier()
    v0.load_model(
        artifacts["v0"]["path"]
    )

    plant = XGBClassifier()
    plant.load_model(
        artifacts["h2_plant"]["path"]
    )

    site = XGBClassifier()
    site.load_model(
        artifacts["h2_site"]["path"]
    )

    platt = json.loads(
        Path(
            artifacts["h2_platt"]["path"]
        ).read_text()
    )

    # ---------------------------------------------------------
    # ONE inference pass
    # ---------------------------------------------------------

    p_v0 = (
        v0.predict_proba(
            X_v0
        )
        .astype(np.float64)
    )

    q = (
        plant.predict_proba(
            X_plant
        )[:, 1]
        .astype(np.float64)
    )

    raw_r = (
        site.predict_proba(
            X_site
        )[:, 1]
        .astype(np.float64)
    )

    calibrated_r = sigmoid(
        platt["slope"]
        * logit(raw_r)
        + platt["intercept"]
    )

    p_h2 = np.column_stack([
        q * calibrated_r,
        q * (
            1.0
            - calibrated_r
        ),
        1.0 - q,
    ])

    for name, p in [
        ("V0", p_v0),
        ("H2", p_h2),
    ]:
        require(
            np.isfinite(p).all(),
            f"{name} probabilities non-finite.",
        )

        require(
            np.allclose(
                p.sum(axis=1),
                1.0,
                atol=1e-8,
            ),
            f"{name} probabilities do not sum to 1.",
        )

    # ---------------------------------------------------------
    # Formal metrics
    # ---------------------------------------------------------

    v0_metrics = score(
        y,
        p_v0,
    )

    h2_metrics = score(
        y,
        p_h2,
    )

    overall_delta = {
        key:
            h2_metrics[key]
            - v0_metrics[key]

        for key in [
            "log_loss",
            "brier_score",
            "accuracy",
            "macro_f1",
        ]
    }

    # ---------------------------------------------------------
    # Horizon metrics
    # ---------------------------------------------------------

    horizon_rows = []

    horizons = (
        df["horizon_sec"]
        .to_numpy()
    )

    for horizon in [
        10,
        20,
        30,
        40,
    ]:
        mask = (
            horizons == horizon
        )

        v = score(
            y[mask],
            p_v0[mask],
        )

        h = score(
            y[mask],
            p_h2[mask],
        )

        horizon_rows.append({
            "horizon_sec": horizon,
            "v0": v,
            "h2": h,
            "delta_h2_minus_v0_log_loss":
                h["log_loss"]
                - v["log_loss"],
        })

    # ---------------------------------------------------------
    # Equal-match paired log-loss
    # ---------------------------------------------------------

    demos = (
        df["demo_filename"]
        .to_numpy()
    )

    per_match = []

    for demo in sorted(
        np.unique(demos)
    ):
        mask = (
            demos == demo
        )

        v = float(
            log_loss(
                y[mask],
                p_v0[mask],
                labels=[0, 1, 2],
            )
        )

        h = float(
            log_loss(
                y[mask],
                p_h2[mask],
                labels=[0, 1, 2],
            )
        )

        per_match.append({
            "demo_filename": str(demo),
            "v0_log_loss": v,
            "h2_log_loss": h,
            "delta_h2_minus_v0": h - v,
        })

    deltas = np.array([
        row["delta_h2_minus_v0"]
        for row in per_match
    ])

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    bootstrap = np.empty(
        BOOTSTRAP_REPS
    )

    for i in range(
        BOOTSTRAP_REPS
    ):
        bootstrap[i] = rng.choice(
            deltas,
            size=len(deltas),
            replace=True,
        ).mean()

    ci = np.quantile(
        bootstrap,
        [
            0.025,
            0.975,
        ],
    )

    paired = {
        "n_matches":
            int(len(deltas)),

        "h2_better_matches":
            int(
                np.sum(
                    deltas < 0
                )
            ),

        "v0_better_matches":
            int(
                np.sum(
                    deltas > 0
                )
            ),

        "mean_delta_h2_minus_v0":
            float(
                deltas.mean()
            ),

        "bootstrap_95_ci": [
            float(ci[0]),
            float(ci[1]),
        ],
    }

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    pred = df.with_columns([
        pl.Series(
            "v0_p_a",
            p_v0[:, 0],
        ),
        pl.Series(
            "v0_p_b",
            p_v0[:, 1],
        ),
        pl.Series(
            "v0_p_no",
            p_v0[:, 2],
        ),
        pl.Series(
            "h2_p_a",
            p_h2[:, 0],
        ),
        pl.Series(
            "h2_p_b",
            p_h2[:, 1],
        ),
        pl.Series(
            "h2_p_no",
            p_h2[:, 2],
        ),
    ])

    pred.write_parquet(
        PREDICTIONS
    )

    metric_rows = [
        {
            "model": "V0",
            "horizon_sec": None,
            **v0_metrics,
        },
        {
            "model": "H2",
            "horizon_sec": None,
            **h2_metrics,
        },
    ]

    for row in horizon_rows:
        metric_rows.append({
            "model": "V0",
            "horizon_sec":
                row["horizon_sec"],
            **row["v0"],
        })

        metric_rows.append({
            "model": "H2",
            "horizon_sec":
                row["horizon_sec"],
            **row["h2"],
        })

    pl.DataFrame(
        metric_rows
    ).write_csv(
        METRICS
    )

    result = {
        "evaluation_name":
            "V2 independent confirmation",

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "evaluation_git_commit":
            git_head(),

        "d_confirm": {
            "sha256":
                EXPECTED_SHA,

            "rows":
                EXPECTED_ROWS,

            "matches":
                EXPECTED_MATCHES,
        },

        "overall": {
            "v0":
                v0_metrics,

            "h2":
                h2_metrics,

            "delta_h2_minus_v0":
                overall_delta,
        },

        "by_horizon":
            horizon_rows,

        "paired_equal_match_log_loss":
            paired,

        "per_match":
            per_match,
    }

    RESULT.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n"
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print()
    print("FORMAL OVERALL RESULTS")
    print("-" * 100)

    for name, m in [
        ("V0", v0_metrics),
        ("H2", h2_metrics),
    ]:
        print(
            f"{name} | "
            f"LL={m['log_loss']:.6f} | "
            f"Brier={m['brier_score']:.6f} | "
            f"Acc={m['accuracy']:.6f} | "
            f"F1={m['macro_f1']:.6f}"
        )

    print()
    print(
        "H2 - V0 LL: "
        f"{overall_delta['log_loss']:+.6f}"
    )

    print()
    print("BY HORIZON")

    for row in horizon_rows:
        print(
            f"{row['horizon_sec']:>2d}s | "
            f"V0={row['v0']['log_loss']:.6f} | "
            f"H2={row['h2']['log_loss']:.6f} | "
            f"Δ={row['delta_h2_minus_v0_log_loss']:+.6f}"
        )

    print()
    print("PAIRED EQUAL-MATCH")
    print(
        f"H2 better: "
        f"{paired['h2_better_matches']}/30"
    )
    print(
        f"Mean Δ: "
        f"{paired['mean_delta_h2_minus_v0']:+.6f}"
    )
    print(
        "95% bootstrap CI: "
        f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
    )

    print()
    print("✅ V2 FORMAL CONFIRMATION EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
