"""
preprocessing.py
----------------
Reusable preprocessing pipeline for the Banking Loan Prediction project.

Design goals
============
* Keep all data transformations in ONE place so the same logic is applied
  during training, evaluation, bulk prediction and manual prediction.
* Return a transformation LOG (list[str]) so the UI can display what
  happened in plain English for teaching / audit purposes.
* Never mutate inputs in place: always return a fresh DataFrame.

Pipeline steps (in order)
=========================
1. Standardize target column name  ("loan_status" -> "outcome")
2. Concatenate train + test into a single unified dataframe
3. Fix "dependence" column  ("3+" -> 3, cast to numeric)
4. Impute missing values
     * categorical -> mode
     * numerical   -> median
5. One-hot encode categorical features  (drop_first=True, int dtype)
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COL = "outcome"
TRAIN_TARGET_ORIGINAL = "loan_status"
DEPENDENCE_COL = "dependence"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Case-insensitive lookup — returns the actual column name or None."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


# ---------------------------------------------------------------------------
# Step 1 + 2 : standardize target and combine train / test
# ---------------------------------------------------------------------------
def standardize_and_combine(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """Rename train's target to `outcome` and concatenate both frames."""
    log: List[str] = []
    train = train_df.copy()
    test = test_df.copy()

    # rename train target
    train_target = _find_column(train, [TRAIN_TARGET_ORIGINAL, TARGET_COL])
    if train_target and train_target != TARGET_COL:
        train = train.rename(columns={train_target: TARGET_COL})
        log.append(f"Renamed train column '{train_target}' -> '{TARGET_COL}'.")
    elif train_target == TARGET_COL:
        log.append("Train already uses 'outcome' as target — no rename needed.")
    else:
        log.append(
            "⚠️ Could not find 'loan_status' / 'outcome' in train dataset."
        )

    # make sure test target is named 'outcome'
    test_target = _find_column(test, [TARGET_COL, TRAIN_TARGET_ORIGINAL])
    if test_target and test_target != TARGET_COL:
        test = test.rename(columns={test_target: TARGET_COL})
        log.append(f"Renamed test column '{test_target}' -> '{TARGET_COL}'.")

    combined = pd.concat([train, test], axis=0, ignore_index=True)
    log.append(
        f"Combined datasets — train: {train.shape}, test: {test.shape}, "
        f"combined: {combined.shape}."
    )
    return combined, log


# ---------------------------------------------------------------------------
# Step 3 : fix the dependence / dependents column
# ---------------------------------------------------------------------------
def fix_dependence(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Replace '3+' with 3 and cast to numeric."""
    log: List[str] = []
    out = df.copy()

    col = _find_column(out, [DEPENDENCE_COL, "dependents"])
    if col is None:
        log.append("No 'dependence' / 'dependents' column found — skipping.")
        return out, log

    out[col] = out[col].replace({"3+": 3, "3 +": 3})
    out[col] = pd.to_numeric(out[col], errors="coerce")
    log.append(f"Column '{col}': replaced '3+' with 3 and cast to numeric.")
    return out, log


# ---------------------------------------------------------------------------
# Step 4 : impute missing values
# ---------------------------------------------------------------------------
def impute_missing(
    df: pd.DataFrame, impute_stats: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Fill missing values.

    Parameters
    ----------
    df
        Input dataframe.
    impute_stats
        If provided, reuse the stored mode/median values (used at prediction
        time so new data is imputed with the SAME values learnt at training
        time).  When None, stats are computed from `df`.

    Returns
    -------
    filled_df, log_lines, stats_dict
    """
    log: List[str] = []
    out = df.copy()
    stats: Dict[str, Any] = {} if impute_stats is None else dict(impute_stats)

    # decide column dtypes (exclude target from imputation stats computation
    # but still fill it if asked)
    for col in out.columns:
        if out[col].isna().sum() == 0 and impute_stats is None:
            continue  # nothing to learn / fill

        is_numeric = pd.api.types.is_numeric_dtype(out[col])

        if impute_stats is None:
            if is_numeric:
                value = out[col].median()
            else:
                mode_vals = out[col].mode(dropna=True)
                value = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
            stats[col] = value
        else:
            value = stats.get(col)
            if value is None:
                # fall back if new column encountered
                value = out[col].median() if is_numeric else "Unknown"
                stats[col] = value

        n_missing = int(out[col].isna().sum())
        if n_missing > 0:
            out[col] = out[col].fillna(value)
            kind = "median" if is_numeric else "mode"
            log.append(
                f"Filled {n_missing} missing values in '{col}' with {kind} = {value!r}."
            )

    if not log:
        log.append("No missing values found — nothing to impute.")
    return out, log, stats


# ---------------------------------------------------------------------------
# Step 5 : one-hot encoding
# ---------------------------------------------------------------------------
def detect_categorical_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    """Return object/category/bool columns, optionally excluding some names."""
    exclude = set(exclude or [])
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return [c for c in cat_cols if c not in exclude]


def one_hot_encode(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    reference_columns: Optional[List[str]] = None,
    encode_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply pd.get_dummies with drop_first=True and cast new cols to int.

    Parameters
    ----------
    encode_columns
        Explicit list of columns to one-hot encode.  When supplied, only these
        columns are encoded AND any OTHER non-numeric columns in the frame are
        dropped (because LogisticRegression can't consume strings).
        When None, every object / category / bool column is encoded.
    reference_columns
        If supplied (i.e. prediction time) the output is re-indexed so it has
        EXACTLY the same columns as the training feature matrix — missing
        columns get filled with 0, extra columns are dropped.
    """
    log: List[str] = []
    out = df.copy()

    # Separate target so we don't encode it
    target_series = None
    if target_col in out.columns:
        target_series = out[target_col]
        out = out.drop(columns=[target_col])

    if encode_columns is None:
        cat_cols = out.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    else:
        # Only encode the user-selected columns that actually exist
        cat_cols = [c for c in encode_columns if c in out.columns]
        # Any OTHER non-numeric column has to be dropped — LR can't read strings
        all_non_numeric = out.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        to_drop = [c for c in all_non_numeric if c not in cat_cols]
        if to_drop:
            out = out.drop(columns=to_drop)
            log.append(
                f"Dropped {len(to_drop)} non-numeric columns not selected for encoding: {to_drop}."
            )

    encoded = pd.get_dummies(out, columns=cat_cols, drop_first=True)

    # force every dummy (and any boolean) to int for numerical stability
    for c in encoded.columns:
        if encoded[c].dtype == bool:
            encoded[c] = encoded[c].astype(int)
    new_cols = [c for c in encoded.columns if c not in out.columns]
    for c in new_cols:
        encoded[c] = encoded[c].astype(int)

    log.append(
        f"One-hot encoded {len(cat_cols)} categorical columns "
        f"({cat_cols}) → produced {len(new_cols)} dummy columns."
    )

    # align with training columns at prediction time
    if reference_columns is not None:
        missing = [c for c in reference_columns if c not in encoded.columns]
        extra = [c for c in encoded.columns if c not in reference_columns]
        for c in missing:
            encoded[c] = 0
        encoded = encoded[reference_columns]
        if missing:
            log.append(f"Added {len(missing)} missing training columns filled with 0.")
        if extra:
            log.append(f"Dropped {len(extra)} extra columns not seen during training.")

    if target_series is not None:
        encoded[target_col] = target_series.values

    return encoded, log


# ---------------------------------------------------------------------------
# Full training-time pipeline
# ---------------------------------------------------------------------------
def run_full_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    encode_columns: Optional[List[str]] = None,
    drop_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    End-to-end preprocessing used on Page 2.

    Parameters
    ----------
    encode_columns
        Explicit categorical columns to one-hot encode.  Any non-numeric
        column NOT in this list is dropped from the feature matrix.
        When None, every detected categorical column is encoded.
    drop_columns
        Raw columns to drop entirely before any other step (e.g. ID columns
        or columns the user doesn't want the model to see at all).

    Returns
    -------
    processed_df
        Fully cleaned, encoded dataframe (still contains `outcome`).
    artifacts
        Dict persisted into session_state and re-used at prediction time:
          - impute_stats   : dict of fill values
          - feature_columns: list of feature columns after encoding
          - target_mapping : dict mapping raw target -> 0/1
          - encode_columns : categorical columns selected for OHE
          - drop_columns   : raw columns dropped before preprocessing
    logs
        Human-readable list describing every step.
    """
    logs: List[str] = []
    df, l = standardize_and_combine(train_df, test_df); logs.extend(l)

    # User-requested drops happen FIRST so those columns never influence
    # imputation statistics or encoding.
    if drop_columns:
        present = [c for c in drop_columns if c in df.columns]
        if present:
            df = df.drop(columns=present)
            logs.append(f"Dropped user-selected columns before preprocessing: {present}.")

    df, l = fix_dependence(df);                         logs.extend(l)
    df, l, stats = impute_missing(df);                  logs.extend(l)
    df, l = one_hot_encode(
        df, target_col=TARGET_COL, encode_columns=encode_columns
    )
    logs.extend(l)

    # encode the target to 0/1 if it is textual (e.g. "Y"/"N" or "Approved"/"Rejected")
    target_mapping: Dict[Any, int] = {}
    if TARGET_COL in df.columns and df[TARGET_COL].dtype == object:
        uniques = sorted(df[TARGET_COL].dropna().unique().tolist())
        # prefer Y=1, Approved=1 semantics
        positive_tokens = {"y", "yes", "approved", "1", "true", "t"}
        ordered = sorted(
            uniques, key=lambda x: 0 if str(x).strip().lower() in positive_tokens else 1
        )
        # ensure exactly two classes
        target_mapping = {v: (1 if str(v).strip().lower() in positive_tokens else 0) for v in uniques}
        # if we accidentally made all-0 or all-1, fall back to alphabetic binary
        if len(set(target_mapping.values())) < 2 and len(uniques) == 2:
            target_mapping = {ordered[0]: 1, ordered[1]: 0}
        df[TARGET_COL] = df[TARGET_COL].map(target_mapping).astype(int)
        logs.append(f"Encoded target '{TARGET_COL}' using mapping {target_mapping}.")

    feature_columns = [c for c in df.columns if c != TARGET_COL]

    artifacts = {
        "impute_stats": stats,
        "feature_columns": feature_columns,
        "target_mapping": target_mapping,
        "encode_columns": list(encode_columns) if encode_columns is not None else None,
        "drop_columns": list(drop_columns) if drop_columns else [],
    }
    return df, artifacts, logs


# ---------------------------------------------------------------------------
# Inference-time preprocessing (bulk / manual prediction)
# ---------------------------------------------------------------------------
def preprocess_for_inference(
    raw_df: pd.DataFrame, artifacts: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply the EXACT same preprocessing learnt at training time."""
    logs: List[str] = []
    df = raw_df.copy()

    # drop any leftover target column
    for tgt in (TARGET_COL, TRAIN_TARGET_ORIGINAL):
        if tgt in df.columns:
            df = df.drop(columns=[tgt])
            logs.append(f"Dropped target column '{tgt}' from inference input.")

    # mirror the user-selected drops from training
    drop_cols = artifacts.get("drop_columns") or []
    present_drops = [c for c in drop_cols if c in df.columns]
    if present_drops:
        df = df.drop(columns=present_drops)
        logs.append(f"Dropped user-selected columns: {present_drops}.")

    df, l = fix_dependence(df);                                         logs.extend(l)
    df, l, _ = impute_missing(df, impute_stats=artifacts["impute_stats"]); logs.extend(l)
    df, l = one_hot_encode(
        df,
        target_col=TARGET_COL,
        reference_columns=artifacts["feature_columns"],
        encode_columns=artifacts.get("encode_columns"),
    )
    logs.extend(l)
    return df, logs
