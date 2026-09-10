"""
Frozen runtime model definition for V0.

These parameters reproduce the selected XGB-A5
development model configuration.

IMPORTANT:
Changing these parameters means creating a new
model experiment/version. Do not silently tune V0.
"""

from xgboost import XGBClassifier


V0_XGB_CONFIG = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,

    "min_child_weight": 5,

    "subsample": 0.8,
    "colsample_bytree": 0.8,

    "reg_alpha": 0.5,
    "reg_lambda": 5.0,

    "objective": "multi:softprob",
    "num_class": 3,

    "eval_metric": "mlogloss",

    "tree_method": "hist",

    "random_state": 42,
    "n_jobs": -1,
}


def make_v0_model():
    """
    Construct the frozen V0 XGB-A5 classifier.
    """

    return XGBClassifier(
        **V0_XGB_CONFIG
    )
