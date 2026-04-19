"""
model_utils.py
--------------
Training, evaluation and prediction helpers for the Loan Prediction app.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from .preprocessing import TARGET_COL

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return X (features) and y (target) from the processed dataframe."""
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. "
            f"Run preprocessing first."
        )
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return X, y


def make_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
):
    X, y = split_features_target(df)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 1.0,
    max_iter: int = 1000,
) -> LogisticRegression:
    model = LogisticRegression(
        C=C, max_iter=max_iter, solver="liblinear", random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


def get_coefficient_table(
    model: LogisticRegression, feature_names: List[str]
) -> pd.DataFrame:
    """Tidy coefficient table with odds-ratio and absolute importance."""
    coefs = model.coef_[0]
    odds = np.exp(coefs)
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "odds_ratio": odds,
            "abs_coefficient": np.abs(coefs),
        }
    ).sort_values("abs_coefficient", ascending=False, ignore_index=True)
    return table


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    model: LogisticRegression, X: pd.DataFrame, y: pd.Series
) -> Dict[str, Any]:
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y, preds),
        "classification_report": classification_report(
            y, preds, zero_division=0, output_dict=False
        ),
        "y_pred": preds,
        "y_prob": probs,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_with_probability(
    model: LogisticRegression, X: pd.DataFrame
) -> pd.DataFrame:
    """Return a dataframe with Prediction, Prob_Approval, Prob_Rejection."""
    probs = model.predict_proba(X)
    preds = model.predict(X)
    # class index of '1' = approved
    approved_idx = list(model.classes_).index(1) if 1 in model.classes_ else 1
    rejected_idx = 1 - approved_idx
    out = pd.DataFrame(
        {
            "Prediction": np.where(preds == 1, "Approved", "Rejected"),
            "Probability_Approval": probs[:, approved_idx],
            "Probability_Rejection": probs[:, rejected_idx],
        }
    )
    return out
