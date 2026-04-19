"""
app.py
------
Banking Loan Prediction System — multi-page Streamlit dashboard.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from utils.preprocessing import (
    TARGET_COL,
    standardize_and_combine,
    fix_dependence,
    impute_missing,
    one_hot_encode,
    run_full_pipeline,
    preprocess_for_inference,
    detect_categorical_columns,
)
from utils.model_utils import (
    split_features_target,
    make_train_test_split,
    train_logistic_regression,
    get_coefficient_table,
    evaluate_model,
    predict_with_probability,
)
from utils.ui import (
    inject_css,
    app_header,
    page_header,
    metric_card,
    PRIMARY,
    ACCENT,
)


# ---------------------------------------------------------------------------
# Page config & global styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Banking Loan Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    "train_df": None,           # raw training CSV
    "test_df": None,            # raw test CSV
    "combined_raw": None,       # train+test (before any cleaning)
    "processed_df": None,       # fully preprocessed feature matrix + target
    "artifacts": None,          # impute_stats + feature_columns + target map
    "logs": [],                 # human-readable preprocessing logs
    "model": None,
    "split": None,              # tuple (X_train, X_test, y_train, y_test)
    "eval_results": None,
    "raw_feature_schema": None, # mapping of raw col -> {"dtype","options"/"range"}
    "encode_columns_sel": None, # user-selected columns to one-hot encode
    "drop_columns_sel": [],     # user-selected columns to drop entirely
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_csv(uploader) -> pd.DataFrame | None:
    if uploader is None:
        return None
    try:
        return pd.read_csv(uploader)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read CSV: {exc}")
        return None


def _build_raw_feature_schema(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Introspect the RAW (pre-encoding) dataframe to power the manual form."""
    schema: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        if col == TARGET_COL:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            schema[col] = {
                "dtype": "numeric",
                "min": float(np.nanmin(series)) if series.notna().any() else 0.0,
                "max": float(np.nanmax(series)) if series.notna().any() else 1.0,
                "median": float(np.nanmedian(series)) if series.notna().any() else 0.0,
            }
        else:
            options = (
                series.dropna().astype(str).unique().tolist()
                if series.notna().any()
                else []
            )
            schema[col] = {"dtype": "categorical", "options": options}
    return schema


def _model_ready() -> bool:
    return st.session_state.model is not None and st.session_state.artifacts is not None


# ---------------------------------------------------------------------------
# PAGE 1 — Data Upload & Overview
# ---------------------------------------------------------------------------
def page_data_upload() -> None:
    page_header(
        "📂 Page 1 · Data Upload & Overview",
        "Upload the training and test CSVs to begin the ML workflow.",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'><b>🟢 Train dataset</b><br>"
                    "<small>Expected target column: <code>loan_status</code></small></div>",
                    unsafe_allow_html=True)
        train_file = st.file_uploader("Upload Train CSV", type=["csv"], key="train_up")
    with col2:
        st.markdown("<div class='card'><b>🩷 Test dataset</b><br>"
                    "<small>Expected target column: <code>outcome</code></small></div>",
                    unsafe_allow_html=True)
        test_file = st.file_uploader("Upload Test CSV", type=["csv"], key="test_up")

    if train_file is not None:
        st.session_state.train_df = _read_csv(train_file)
    if test_file is not None:
        st.session_state.test_df = _read_csv(test_file)

    if st.session_state.train_df is None or st.session_state.test_df is None:
        st.info("⬆️ Upload BOTH train and test CSVs to continue.")
        return

    # Standardize + combine immediately so every page can use the union
    combined, logs = standardize_and_combine(
        st.session_state.train_df, st.session_state.test_df
    )
    st.session_state.combined_raw = combined
    st.session_state.raw_feature_schema = _build_raw_feature_schema(combined)

    # summary metrics
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Train rows", f"{st.session_state.train_df.shape[0]:,}")
    metric_card(c2, "Test rows", f"{st.session_state.test_df.shape[0]:,}")
    metric_card(c3, "Combined rows", f"{combined.shape[0]:,}")
    metric_card(c4, "Columns", f"{combined.shape[1]:,}")
    st.write("")

    with st.expander("🧾 Combine log", expanded=False):
        for line in logs:
            st.markdown(f"- {line}")

    tab_preview, tab_schema, tab_missing, tab_stats = st.tabs(
        ["👀 Preview", "🧬 Schema", "❓ Missing values", "📈 Describe"]
    )
    with tab_preview:
        st.dataframe(combined.head(20), use_container_width=True)
    with tab_schema:
        schema = pd.DataFrame(
            {"column": combined.columns, "dtype": combined.dtypes.astype(str)}
        ).reset_index(drop=True)
        st.dataframe(schema, use_container_width=True)
    with tab_missing:
        miss = combined.isna().sum()
        miss_pct = (miss / len(combined) * 100).round(2)
        miss_df = pd.DataFrame(
            {"missing": miss, "missing_%": miss_pct}
        ).sort_values("missing", ascending=False)
        st.dataframe(miss_df, use_container_width=True)
    with tab_stats:
        st.dataframe(combined.describe(include="all").T, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 2 — Preprocessing
# ---------------------------------------------------------------------------
def page_preprocessing() -> None:
    page_header(
        "🧹 Page 2 · Data Preprocessing",
        "Clean, transform and one-hot encode the data in a single reproducible pipeline.",
    )

    if st.session_state.combined_raw is None:
        st.warning("Upload both datasets on Page 1 first.")
        return

    raw = st.session_state.combined_raw
    st.markdown("<div class='card'><b>BEFORE preprocessing</b></div>", unsafe_allow_html=True)
    st.dataframe(raw.head(10), use_container_width=True)
    st.write(f"Shape: **{raw.shape[0]} rows × {raw.shape[1]} columns**")

    # ---- Column selection (runs on the post-fix_dependence frame so that
    # columns like `dependence` are no longer listed as categorical) -------
    preview_df, _ = fix_dependence(raw)
    cat_candidates = detect_categorical_columns(preview_df, exclude=[TARGET_COL])
    all_columns = [c for c in raw.columns if c != TARGET_COL]

    st.markdown(
        "<div class='card'><b>🎛️ Column selection</b><br>"
        "<small>Pick which categorical columns should be one-hot encoded. "
        "Any categorical column you leave <i>unchecked</i> will be dropped "
        "from the feature matrix (Logistic Regression can't consume strings). "
        "Use the second list to drop any column entirely — e.g. ID columns.</small></div>",
        unsafe_allow_html=True,
    )

    # Default the selection to ALL detected cat columns the first time round
    if st.session_state.encode_columns_sel is None:
        st.session_state.encode_columns_sel = cat_candidates

    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        encode_selected = st.multiselect(
            "✅ Categorical columns to ONE-HOT ENCODE",
            options=cat_candidates,
            default=[c for c in st.session_state.encode_columns_sel if c in cat_candidates],
            help="Only columns you select here will become dummy variables. "
                 "Unselected categorical columns will be dropped.",
            key="ohe_multiselect",
        )
    with sel_col2:
        drop_selected = st.multiselect(
            "🗑️ Columns to DROP entirely (e.g. ID columns)",
            options=all_columns,
            default=[c for c in st.session_state.drop_columns_sel if c in all_columns],
            help="These columns are removed before any preprocessing step — "
                 "the model will never see them.",
            key="drop_multiselect",
        )

    # persist user's picks so they survive reruns
    st.session_state.encode_columns_sel = encode_selected
    st.session_state.drop_columns_sel = drop_selected

    # Show what will happen
    will_drop_cat = [c for c in cat_candidates if c not in encode_selected]
    if will_drop_cat or drop_selected:
        info_lines = []
        if drop_selected:
            info_lines.append(f"🗑️ Will DROP entirely: `{drop_selected}`")
        if will_drop_cat:
            info_lines.append(
                f"⚠️ Categorical columns NOT encoded (will also be dropped): `{will_drop_cat}`"
            )
        st.info("  \n".join(info_lines))

    st.write("")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        impute_btn = st.button("① Handle Missing Values", use_container_width=True)
    with c2:
        encode_btn = st.button("② Apply One-Hot Encoding", use_container_width=True)
    with c3:
        full_btn = st.button("🚀 Run Full Pipeline", use_container_width=True)

    logs: List[str] = []
    result: pd.DataFrame | None = None

    if impute_btn:
        df = raw.copy()
        if drop_selected:
            df = df.drop(columns=[c for c in drop_selected if c in df.columns])
            logs.append(f"Dropped user-selected columns: {drop_selected}.")
        df, l1 = fix_dependence(df); logs.extend(l1)
        df, l2, _ = impute_missing(df); logs.extend(l2)
        result = df
    elif encode_btn:
        df = raw.copy()
        if drop_selected:
            df = df.drop(columns=[c for c in drop_selected if c in df.columns])
            logs.append(f"Dropped user-selected columns: {drop_selected}.")
        df, l1 = fix_dependence(df); logs.extend(l1)
        df, l2, _ = impute_missing(df); logs.extend(l2)
        df, l3 = one_hot_encode(
            df, target_col=TARGET_COL, encode_columns=encode_selected
        ); logs.extend(l3)
        result = df
    elif full_btn:
        df, artifacts, logs = run_full_pipeline(
            st.session_state.train_df,
            st.session_state.test_df,
            encode_columns=encode_selected,
            drop_columns=drop_selected,
        )
        st.session_state.processed_df = df
        st.session_state.artifacts = artifacts
        st.session_state.logs = logs
        # reset downstream state
        st.session_state.model = None
        st.session_state.split = None
        st.session_state.eval_results = None
        result = df
        st.success("✅ Full preprocessing pipeline executed successfully.")

    if result is not None:
        st.markdown("<div class='card'><b>AFTER preprocessing</b></div>", unsafe_allow_html=True)
        st.dataframe(result.head(10), use_container_width=True)
        st.write(f"Shape: **{result.shape[0]} rows × {result.shape[1]} columns**")
        with st.expander("📝 Transformation log", expanded=True):
            for line in logs:
                st.markdown(f"- {line}")

    if st.session_state.processed_df is not None and not (impute_btn or encode_btn or full_btn):
        st.info("A full pipeline result already exists in session. Preview below.")
        st.dataframe(st.session_state.processed_df.head(10), use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 3 — Train/Test Split & Model Training
# ---------------------------------------------------------------------------
def page_train_model() -> None:
    page_header(
        "🧠 Page 3 · Train-Test Split & Model Training",
        "Fit a Logistic Regression model and inspect its coefficients.",
    )

    if st.session_state.processed_df is None:
        st.warning("Run the full preprocessing pipeline on Page 2 first.")
        return

    df = st.session_state.processed_df

    c1, c2, c3 = st.columns(3)
    with c1:
        test_size = st.slider("Test size (%)", 10, 40, 20, step=5) / 100
    with c2:
        C = st.number_input("Regularization C", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
    with c3:
        max_iter = st.number_input("Max iterations", 100, 5000, 1000, step=100)

    if st.button("🚀 Train Logistic Regression", use_container_width=True):
        try:
            X_train, X_test, y_train, y_test = make_train_test_split(df, test_size=test_size)
            model = train_logistic_regression(
                X_train, y_train, C=float(C), max_iter=int(max_iter)
            )
            st.session_state.split = (X_train, X_test, y_train, y_test)
            st.session_state.model = model
            st.session_state.eval_results = None  # force re-eval on page 4
            st.success("✅ Model trained.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Training failed: {exc}")

    if st.session_state.model is None:
        return

    X_train, X_test, y_train, y_test = st.session_state.split
    model = st.session_state.model

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Train rows", f"{len(X_train):,}")
    metric_card(c2, "Test rows", f"{len(X_test):,}")
    metric_card(c3, "Features", f"{X_train.shape[1]:,}")
    metric_card(c4, "Intercept", f"{model.intercept_[0]:.4f}")
    st.write("")

    coef_tbl = get_coefficient_table(model, X_train.columns.tolist())

    tab_coef, tab_chart, tab_explain = st.tabs(
        ["📋 Coefficients", "📊 Feature importance", "📘 How to read this"]
    )
    with tab_coef:
        st.dataframe(coef_tbl, use_container_width=True)
    with tab_chart:
        top = coef_tbl.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top))))
        colors = [PRIMARY if c > 0 else ACCENT for c in top["coefficient"]]
        ax.barh(top["feature"], top["coefficient"], color=colors)
        ax.axvline(0, color="#888", linewidth=0.8)
        ax.set_xlabel("Coefficient")
        ax.set_title("Top 15 features by |coefficient|")
        fig.tight_layout()
        st.pyplot(fig)
    with tab_explain:
        st.markdown(
            f"""
            **Intercept** `{model.intercept_[0]:.4f}` — the log-odds of approval when every feature equals 0.

            **Coefficients** are in **log-odds space**:
            - A **positive** coefficient means the feature *increases* the probability of loan approval.
            - A **negative** coefficient means the feature *decreases* the probability.
            - `odds_ratio = exp(coefficient)` tells you by what factor the odds of approval
              are multiplied for a one-unit increase in the feature, holding everything else constant.
              For example `odds_ratio = 1.8` ⇒ **+80% odds** per unit increase.
            - Since features were one-hot encoded with `drop_first=True`, each dummy is read
              *relative to its dropped baseline category*.
            """
        )


# ---------------------------------------------------------------------------
# PAGE 4 — Model Evaluation
# ---------------------------------------------------------------------------
def page_evaluate() -> None:
    page_header(
        "🧪 Page 4 · Model Evaluation",
        "Score the trained model on the held-out test partition.",
    )

    if not _model_ready() or st.session_state.split is None:
        st.warning("Train a model on Page 3 first.")
        return

    _, X_test, _, y_test = st.session_state.split
    model = st.session_state.model

    results = evaluate_model(model, X_test, y_test)
    st.session_state.eval_results = results

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Accuracy",  f"{results['accuracy']*100:.2f}%")
    metric_card(c2, "Precision", f"{results['precision']*100:.2f}%")
    metric_card(c3, "Recall",    f"{results['recall']*100:.2f}%")
    metric_card(c4, "F1 Score",  f"{results['f1']*100:.2f}%")
    st.write("")

    col_cm, col_report = st.columns([1, 1])
    with col_cm:
        st.markdown("**Confusion Matrix**")
        cm = results["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="mako",
            xticklabels=["Rejected", "Approved"],
            yticklabels=["Rejected", "Approved"],
            cbar=False, ax=ax,
        )
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    with col_report:
        st.markdown("**Classification Report**")
        st.code(results["classification_report"])

    tn, fp, fn, tp = results["confusion_matrix"].ravel()
    st.markdown(
        f"""
        #### 🧠 Interpretation
        - **True Positives ({tp})** — applicants correctly *approved*.
        - **True Negatives ({tn})** — applicants correctly *rejected*.
        - **False Positives ({fp})** — rejected applicants incorrectly approved  _(credit risk)_.
        - **False Negatives ({fn})** — eligible applicants incorrectly rejected _(lost business)_.

        In banking, the cost of **False Positives** (defaults) is usually higher than **False
        Negatives**, so monitor **Precision** closely. If you want to capture more eligible
        customers, optimise for **Recall** and lower the decision threshold.
        """
    )


# ---------------------------------------------------------------------------
# PAGE 5 — Bulk Prediction
# ---------------------------------------------------------------------------
def page_bulk_prediction() -> None:
    page_header(
        "📦 Page 5 · Bulk Prediction",
        "Upload a CSV of new customers — the same preprocessing pipeline is applied end-to-end.",
    )

    if not _model_ready():
        st.warning("Train a model on Page 3 first.")
        return

    uploaded = st.file_uploader("Upload new customer CSV", type=["csv"], key="bulk_up")
    if uploaded is None:
        st.info("Upload a CSV with the same RAW columns as the training data "
                "(target column optional — it will be ignored).")
        return

    raw_new = _read_csv(uploaded)
    if raw_new is None:
        return

    st.markdown("**Raw input preview**")
    st.dataframe(raw_new.head(10), use_container_width=True)

    try:
        X_new, logs = preprocess_for_inference(raw_new, st.session_state.artifacts)
        # strip target if the helper re-added it
        if TARGET_COL in X_new.columns:
            X_new = X_new.drop(columns=[TARGET_COL])
        preds = predict_with_probability(st.session_state.model, X_new)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        return

    result = pd.concat([raw_new.reset_index(drop=True), preds], axis=1)

    with st.expander("📝 Preprocessing log", expanded=False):
        for line in logs:
            st.markdown(f"- {line}")

    c1, c2, c3 = st.columns(3)
    metric_card(c1, "Total records", f"{len(result):,}")
    metric_card(c2, "Predicted approved",
                f"{(preds['Prediction'] == 'Approved').sum():,}")
    metric_card(c3, "Predicted rejected",
                f"{(preds['Prediction'] == 'Rejected').sum():,}")
    st.write("")

    st.markdown("**Predictions**")
    st.dataframe(result, use_container_width=True)

    csv_bytes = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download predictions CSV",
        data=csv_bytes,
        file_name="loan_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# PAGE 6 — Manual Prediction
# ---------------------------------------------------------------------------
def page_manual_prediction() -> None:
    page_header(
        "🧍 Page 6 · Manual Prediction",
        "Fill out an applicant profile and get an instant approval probability.",
    )

    if not _model_ready() or st.session_state.raw_feature_schema is None:
        st.warning("Train a model on Page 3 first.")
        return

    schema = st.session_state.raw_feature_schema
    artifacts = st.session_state.artifacts or {}
    dropped = set(artifacts.get("drop_columns") or [])
    encode_cols = artifacts.get("encode_columns")  # may be None = encode all

    # Only show fields for columns that actually influence the model:
    #  - skip columns the user dropped entirely
    #  - skip categorical columns that were NOT selected for encoding
    visible_schema = {}
    for col, meta in schema.items():
        if col in dropped:
            continue
        if meta["dtype"] == "categorical" and encode_cols is not None and col not in encode_cols:
            continue
        visible_schema[col] = meta

    st.markdown("<div class='card'><b>Applicant details</b></div>", unsafe_allow_html=True)

    with st.form("manual_pred_form", clear_on_submit=False):
        inputs: Dict[str, Any] = {}
        cols = st.columns(2)
        for i, (col_name, meta) in enumerate(visible_schema.items()):
            target = cols[i % 2]
            with target:
                if meta["dtype"] == "numeric":
                    inputs[col_name] = st.number_input(
                        col_name,
                        min_value=float(meta["min"]),
                        max_value=float(meta["max"]) if meta["max"] > meta["min"] else float(meta["min"]) + 1.0,
                        value=float(meta["median"]),
                    )
                else:
                    options = meta["options"] or ["Unknown"]
                    inputs[col_name] = st.selectbox(col_name, options)
        submit = st.form_submit_button("🔮 Predict", use_container_width=True)

    if not submit:
        return

    try:
        raw_row = pd.DataFrame([inputs])
        X_row, _ = preprocess_for_inference(raw_row, st.session_state.artifacts)
        if TARGET_COL in X_row.columns:
            X_row = X_row.drop(columns=[TARGET_COL])
        proba = st.session_state.model.predict_proba(X_row)[0]
        classes = list(st.session_state.model.classes_)
        approved_idx = classes.index(1) if 1 in classes else 1
        rejected_idx = 1 - approved_idx
        p_approve = proba[approved_idx]
        p_reject = proba[rejected_idx]
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        return

    approved = p_approve >= 0.5
    c1, c2 = st.columns(2)
    with c1:
        metric_card(c1, "Probability of Approval", f"{p_approve*100:.2f}%")
    with c2:
        metric_card(c2, "Probability of Rejection", f"{p_reject*100:.2f}%")

    if approved:
        st.markdown(
            f"<div class='pred-box pred-approved'>✅ LOAN APPROVED · "
            f"{p_approve*100:.2f}% confidence</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='pred-box pred-rejected'>❌ LOAN REJECTED · "
            f"{p_reject*100:.2f}% confidence</div>",
            unsafe_allow_html=True,
        )

    fig, ax = plt.subplots(figsize=(6, 1.3))
    ax.barh([0], [p_approve], color=PRIMARY, label="Approval")
    ax.barh([0], [p_reject], left=[p_approve], color=ACCENT, label="Rejection")
    ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xlabel("Probability")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False)
    fig.tight_layout()
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
def sidebar() -> str:
    with st.sidebar:
        st.markdown(
            "<div style='font-size:20px;font-weight:800;"
            f"background:linear-gradient(90deg,{PRIMARY},{ACCENT});"
            "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
            "margin-bottom:4px'>🏦 Loan Predictor</div>"
            "<div style='color:#9BA3AF;font-size:12px;margin-bottom:14px;'>"
            "Logistic Regression Dashboard</div>",
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigate",
            [
                "📂  Data Upload & Overview",
                "🧹  Data Preprocessing",
                "🧠  Train-Test Split & Training",
                "🧪  Model Evaluation",
                "📦  Bulk Prediction",
                "🧍  Manual Prediction",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("**Pipeline status**")
        status = {
            "Data loaded":     st.session_state.combined_raw is not None,
            "Preprocessed":    st.session_state.processed_df is not None,
            "Model trained":   st.session_state.model is not None,
            "Evaluated":       st.session_state.eval_results is not None,
        }
        for name, ok in status.items():
            st.markdown(f"- {'🟢' if ok else '⚪'} {name}")
        st.markdown("---")
        st.caption("© Loan Prediction · Built with Streamlit + scikit-learn")
    return page


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    app_header()
    page = sidebar()

    if page.startswith("📂"):
        page_data_upload()
    elif page.startswith("🧹"):
        page_preprocessing()
    elif page.startswith("🧠"):
        page_train_model()
    elif page.startswith("🧪"):
        page_evaluate()
    elif page.startswith("📦"):
        page_bulk_prediction()
    elif page.startswith("🧍"):
        page_manual_prediction()


if __name__ == "__main__":
    main()
