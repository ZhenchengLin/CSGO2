"""
Runtime inference interface for V0.

Input:
    one feature dictionary following the frozen
    V0 feature contract

Output:
    calibrated-style model probabilities for

    A_PLANT
    B_PLANT
    NO_PLANT

The predictor never uses future round information.
"""

from pathlib import Path
import json
from collections.abc import Mapping

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from cs2_tactical_intelligence.v0.schema import (
    V0_LABELS,
    V0_MODEL_FEATURES,
)


DEFAULT_MODEL_PATH = Path(
    "artifacts/v0_xgb_a5_runtime.json"
)

DEFAULT_METADATA_PATH = Path(
    "artifacts/v0_xgb_a5_runtime_metadata.json"
)


class V0Predictor:

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        metadata_path=DEFAULT_METADATA_PATH,
    ):

        self.model_path = Path(
            model_path
        )

        self.metadata_path = Path(
            metadata_path
        )


        if not self.model_path.exists():
            raise FileNotFoundError(
                f"V0 model not found: "
                f"{self.model_path}"
            )


        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"V0 metadata not found: "
                f"{self.metadata_path}"
            )


        self.metadata = json.loads(
            self.metadata_path.read_text()
        )


        # ==========================================
        # Validate metadata against code contract
        # ==========================================

        if (
            self.metadata["features"]
            != V0_MODEL_FEATURES
        ):
            raise ValueError(
                "Runtime model feature contract "
                "does not match V0 schema."
            )


        if (
            self.metadata["labels"]
            != V0_LABELS
        ):
            raise ValueError(
                "Runtime model label contract "
                "does not match V0 schema."
            )


        # ==========================================
        # Load model
        # ==========================================

        self.model = XGBClassifier()

        self.model.load_model(
            self.model_path
        )


        booster_features = (
            self.model
            .get_booster()
            .feature_names
        )


        if (
            booster_features
            != V0_MODEL_FEATURES
        ):
            raise ValueError(
                "Loaded model feature ordering "
                "does not match V0 schema."
            )


    def _prepare_input(
        self,
        feature_row: Mapping,
    ):
        """
        Select and order exactly the 37 frozen
        model features.

        Extra fields are allowed because the
        FeatureBuilder may also produce QA or
        historical ablation fields.
        """

        missing = [
            feature
            for feature
            in V0_MODEL_FEATURES
            if feature
            not in feature_row
        ]


        if missing:
            raise ValueError(
                f"Missing V0 model features: "
                f"{missing}"
            )


        values = {}


        for feature in V0_MODEL_FEATURES:

            value = float(
                feature_row[
                    feature
                ]
            )


            if not np.isfinite(
                value
            ):
                raise ValueError(
                    f"Non-finite feature "
                    f"{feature}={value}"
                )


            values[
                feature
            ] = value


        return pd.DataFrame(
            [values],
            columns=V0_MODEL_FEATURES,
        )


    def predict(
        self,
        feature_row: Mapping,
    ):
        """
        Predict one V0 observation.
        """

        X = self._prepare_input(
            feature_row
        )


        probabilities = (
            self.model
            .predict_proba(X)[0]
            .astype(np.float64)
        )


        probabilities = (
            probabilities
            / probabilities.sum()
        )


        if len(probabilities) != 3:
            raise ValueError(
                "Expected three V0 "
                "class probabilities."
            )


        predicted_id = int(
            np.argmax(
                probabilities
            )
        )


        predicted_label = (
            V0_LABELS[
                predicted_id
            ]
        )


        return {

            "prediction":
                predicted_label,

            "confidence":
                float(
                    probabilities[
                        predicted_id
                    ]
                ),

            "p_a_plant":
                float(
                    probabilities[0]
                ),

            "p_b_plant":
                float(
                    probabilities[1]
                ),

            "p_no_plant":
                float(
                    probabilities[2]
                ),
        }
