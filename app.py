
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10

# Increment whenever the structure of session-state ML results changes.
# This prevents Streamlit hot-reload from displaying results created by an
# older version of the application.
ML_RESULTS_SCHEMA_VERSION = 2

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "kiwifruit_research_outputs"

st.set_page_config(
    page_title="Kiwifruit Smart Irrigation XAI",
    page_icon="🥝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clear stale ML results after code/schema changes. Streamlit preserves
# session_state during reruns, so an older results DataFrame can otherwise
# trigger KeyError when new metric columns are requested.
if st.session_state.get("_ml_results_schema_version") != ML_RESULTS_SCHEMA_VERSION:
    for _key in [
        "trained_models",
        "ml_results",
        "oof_predictions",
        "ml_class_names",
        "ml_label_encoder",
        "ml_features",
        "ml_target",
        "ml_X",
        "ml_y",
    ]:
        st.session_state.pop(_key, None)
    st.session_state["_ml_results_schema_version"] = ML_RESULTS_SCHEMA_VERSION


# =============================================================================
# HELPERS
# =============================================================================

def out_path(name: str) -> Path:
    return OUT_DIR / name


@st.cache_data(show_spinner=False)
def load_csv(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not read {path.name}: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_excel(path: Path, sheet=0):
    try:
        return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    except Exception as e:
        st.error(f"Could not read {path.name}: {e}")
        return None


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().replace("\n", " ").replace("\t", " ")
        for c in df.columns
    ]
    return df


def normalize_name(value):
    x = str(value).strip().lower()
    x = re.sub(r"[%()/\-]+", "_", x)
    x = re.sub(r"\s+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def score_sheet(df: pd.DataFrame) -> int:
    score = 0
    if len(df) >= 10:
        score += 3
    if df.shape[1] >= 8:
        score += 3

    keywords = [
        "irrigation", "stage", "phenological", "replication", "replicate",
        "vine", "plant", "canopy", "temperature", "spad", "rwc",
        "relative_water", "ndvi", "yellowing", "par", "proline", "mda",
        "fruit", "tss", "yield", "stress", "class", "chlorophyll", "leaf",
        "water", "moisture", "growth",
    ]

    for col in df.columns:
        name = normalize_name(col)
        for key in keywords:
            if key in name:
                score += 2

    return score


def select_best_sheet(path: Path):
    xls = pd.ExcelFile(path, engine="openpyxl")
    rows = []

    for sheet in xls.sheet_names:
        try:
            tmp = clean_columns(
                pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
            )
            rows.append({
                "Sheet": sheet,
                "Rows": tmp.shape[0],
                "Columns": tmp.shape[1],
                "Score": score_sheet(tmp),
            })
        except Exception:
            continue

    return (
        pd.DataFrame(rows)
        .sort_values("Score", ascending=False)
        .reset_index(drop=True)
    )


def encode_target(y_raw):
    """
    Always encode target labels consistently so both string and numeric
    targets are handled safely.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw.astype(str))
    return y_encoded, le


def make_model(name):
    """
    Model factory.

    The models intentionally use conservative regularisation/depth so that
    the application compares genuinely different model families instead of
    relying only on highly flexible classifiers.

    NOTE:
    A different model does NOT guarantee lower accuracy. If the biological
    predictors strongly separate the experimental treatment classes, several
    models may legitimately obtain perfect classification.
    """

    if name == "Logistic Regression":
        return LogisticRegression(
            C=0.1,
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )

    if name == "LDA":
        return LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
        )

    if name == "QDA":
        return QuadraticDiscriminantAnalysis(
            reg_param=0.5,
        )

    if name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            metric="euclidean",
        )

    if name == "Linear SVM":
        return SVC(
            kernel="linear",
            C=0.1,
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    if name == "SVM (RBF)":
        return SVC(
            kernel="rbf",
            C=0.5,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    if name == "Naive Bayes":
        return GaussianNB(
            var_smoothing=1e-8,
        )

    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=2,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if name == "Extra Trees":
        return ExtraTreesClassifier(
            n_estimators=200,
            max_depth=2,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.03,
            max_depth=1,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )

    raise ValueError(f"Unknown model: {name}")


def make_pipeline_model(name):
    """
    Leakage-safe preprocessing.

    Imputation and scaling are fitted separately inside each training fold.
    Tree/boosting models do not require scaling; distance/linear/probabilistic
    models are scaled.
    """
    model = make_model(name)

    tree_models = [
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
    ]

    if name in tree_models:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ])

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def metric_dict(y_true, y_pred):
    """
    Core multiclass evaluation metrics.

    Precision, recall and F1 are macro-averaged so that each stress class
    (Low, Moderate, High) contributes equally, regardless of class size.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision Macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "Recall Macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "F1 Macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate_models(
    X,
    y,
    chosen_models,
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
):
    """
    Main evaluation:
    - repeated stratified CV for mean +/- SD
    - one independent stratified 5-fold OOF prediction for reporting
    - final model fitted on all observations ONLY for future/new prediction
    """
    repeated_cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )

    oof_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []
    trained = {}
    oof_predictions = {}

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    }

    for name in chosen_models:
        pipeline_model = make_pipeline_model(name)

        # Repeated CV: mean and SD across 50 validation folds
        cv_scores = cross_val_score(
            pipeline_model,
            X,
            y,
            cv=repeated_cv,
            scoring="accuracy",
            n_jobs=-1,
        )

        precision_scores = cross_val_score(
            pipeline_model,
            X,
            y,
            cv=repeated_cv,
            scoring="precision_macro",
            n_jobs=-1,
        )

        recall_scores = cross_val_score(
            pipeline_model,
            X,
            y,
            cv=repeated_cv,
            scoring="recall_macro",
            n_jobs=-1,
        )

        bal_scores = cross_val_score(
            pipeline_model,
            X,
            y,
            cv=repeated_cv,
            scoring="balanced_accuracy",
            n_jobs=-1,
        )

        f1_scores = cross_val_score(
            pipeline_model,
            X,
            y,
            cv=repeated_cv,
            scoring="f1_macro",
            n_jobs=-1,
        )

        # Single 5-fold OOF prediction for confusion matrix/report
        oof = cross_val_predict(
            pipeline_model,
            X,
            y,
            cv=oof_cv,
            method="predict",
        )

        oof_metrics = metric_dict(y, oof)

        results.append({
            "Model": name,
            "Accuracy": oof_metrics["Accuracy"],
            "Accuracy Mean": cv_scores.mean(),
            "Accuracy SD": cv_scores.std(),

            "Precision Macro": oof_metrics["Precision Macro"],
            "Precision Macro Mean": precision_scores.mean(),
            "Precision Macro SD": precision_scores.std(),

            "Recall Macro": oof_metrics["Recall Macro"],
            "Recall Macro Mean": recall_scores.mean(),
            "Recall Macro SD": recall_scores.std(),

            "F1 Macro": oof_metrics["F1 Macro"],
            "F1 Macro Mean": f1_scores.mean(),
            "F1 Macro SD": f1_scores.std(),

            "Balanced Accuracy": oof_metrics["Balanced Accuracy"],
            "Balanced Accuracy Mean": bal_scores.mean(),
            "Balanced Accuracy SD": bal_scores.std(),

            "MCC": oof_metrics["MCC"],
        })

        oof_predictions[name] = oof

        # Final model is ONLY for prediction of genuinely new observations.
        final_model = clone(pipeline_model)
        final_model.fit(X, y)

        trained[name] = {
            "model": final_model,
            "features": list(X.columns),
        }

    return (
        pd.DataFrame(results),
        trained,
        oof_predictions,
    )


def group_holdout_evaluation(X, y, groups, chosen_model, group_name):
    """
    Leave-one-group-out evaluation.
    Used for phenological stage and replication robustness.
    """
    groups = pd.Series(groups).reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = np.asarray(y)

    unique_groups = list(pd.unique(groups))
    rows = []
    all_true = []
    all_pred = []

    for held_out in unique_groups:
        train_idx = groups != held_out
        test_idx = groups == held_out

        model = make_pipeline_model(chosen_model)
        model.fit(X.loc[train_idx], y[train_idx])

        pred = model.predict(X.loc[test_idx])
        true = y[test_idx]

        m = metric_dict(true, pred)

        rows.append({
            "Held-out group": str(held_out),
            "N test": int(test_idx.sum()),
            "Accuracy": m["Accuracy"],
            "Precision Macro": m["Precision Macro"],
            "Recall Macro": m["Recall Macro"],
            "F1 Macro": m["F1 Macro"],
            "Balanced Accuracy": m["Balanced Accuracy"],
            "MCC": m["MCC"],
        })

        all_true.extend(true.tolist())
        all_pred.extend(pred.tolist())

    overall = metric_dict(
        np.asarray(all_true),
        np.asarray(all_pred),
    )

    summary = pd.DataFrame(rows)

    return summary, overall


def permutation_test(
    X,
    y,
    model_name,
    n_permutations=200,
    random_state=RANDOM_STATE,
):
    """
    Permutation-label test using the same stratified 5-fold CV.
    This tests whether the observed CV accuracy is greater than expected
    after breaking the association between predictors and labels.
    """
    rng = np.random.default_rng(random_state)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model = make_pipeline_model(model_name)

    observed_pred = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict",
    )
    observed_accuracy = accuracy_score(y, observed_pred)

    perm_scores = []

    progress = st.progress(0.0)

    for i in range(n_permutations):
        y_perm = rng.permutation(y)

        score = cross_val_score(
            model,
            X,
            y_perm,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        ).mean()

        perm_scores.append(score)
        progress.progress((i + 1) / n_permutations)

    progress.empty()

    perm_scores = np.asarray(perm_scores)

    # +1 correction prevents zero empirical p-values.
    p_value = (
        np.sum(perm_scores >= observed_accuracy) + 1
    ) / (len(perm_scores) + 1)

    return observed_accuracy, perm_scores, p_value


def get_tree_model_and_X_for_shap(pipeline_model, X):
    """
    Extract the fitted tree estimator and transformed X for SHAP.
    """
    if not hasattr(pipeline_model, "named_steps"):
        return None, None, None

    final_estimator = pipeline_model.named_steps.get("model")

    if not hasattr(final_estimator, "feature_importances_"):
        return None, None, None

    imputer = pipeline_model.named_steps.get("imputer")
    X_imp = imputer.transform(X) if imputer is not None else X.values

    return final_estimator, X_imp, list(X.columns)


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🥝 Kiwifruit XAI")
st.sidebar.markdown(
    "Stage-aware explainable machine-learning dashboard for "
    "*Actinidia deliciosa* cv. Allison."
)

uploaded_files = st.sidebar.file_uploader(
    "Upload raw Excel workbooks (optional)",
    type=["xlsx", "xls"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Datasets",
        "Data Separability Audit",
        "Feature Ablation",
        "Train & Evaluate ML",
        "Robustness Validation",
        "OOF Predictions",
        "SHAP / XAI",
        "Irrigation Decision Support",
    ],
)


# =============================================================================
# LOAD CORE ARTIFACTS
# =============================================================================

artifact_names = {
    "True biological features": "09C_true_biological_feature_dataset.csv",
    "ML dataset": "10_full_biological_ml_dataset.csv",
    "OOF predictions": "11_cell13_all_physiological_oof_predictions.csv",
    "SHAP global": "13_cell15_global_shap_importance.csv",
    "Multimodal fused": "03_multimodal_fused_dataset.csv",
}

artifacts = {
    key: load_csv(out_path(filename))
    for key, filename in artifact_names.items()
}


# =============================================================================
# PAGE: OVERVIEW
# =============================================================================

if page == "Overview":

    st.title("Smart Irrigation Scheduling in Kiwifruit cv. Allison")

    st.markdown(
        """
        ### Research workflow

        **Kiwifruit cv. Allison → Soil Moisture Monitoring → Phenological Stage
        → Physiological Monitoring → Explainable Machine Learning
        → Experimental Water-Stress Condition → SHAP Explanation
        → Stage-Aware Irrigation Decision → Smart Irrigation Scheduling**

        This version includes leakage-safe preprocessing, multiple classifier
        families, repeated stratified cross-validation, true out-of-fold
        evaluation, feature ablation, leave-one-stage-out, leave-one-replication-
        out and permutation testing.
        """
    )

    st.info(
        "Scientific interpretation: the current target represents an "
        "experimentally imposed irrigation/water-stress condition. It should "
        "not be described as an independently validated physiological stress "
        "diagnosis."
    )

    col1, col2, col3, col4 = st.columns(4)

    ml_df = artifacts.get("ML dataset")

    if ml_df is not None:
        col1.metric("Records", len(ml_df))

        target_features = [
            c for c in ml_df.columns
            if c not in [
                "irrigation_block",
                "phenological_stage",
                "replication",
                "stress_condition",
                "stress_severity",
            ]
        ]

        col2.metric("Biological features", len(target_features))

        if "stress_condition" in ml_df.columns:
            col3.metric(
                "Stress classes",
                ml_df["stress_condition"].nunique(),
            )

        if "phenological_stage" in ml_df.columns:
            col4.metric(
                "Phenological stages",
                ml_df["phenological_stage"].nunique(),
            )

        st.subheader("Stress distribution")

        if "stress_condition" in ml_df.columns:
            st.bar_chart(
                ml_df["stress_condition"].value_counts()
            )

        st.subheader("Stage × experimental stress condition")

        if (
            "phenological_stage" in ml_df.columns
            and "stress_condition" in ml_df.columns
        ):
            stage_table = pd.crosstab(
                ml_df["phenological_stage"],
                ml_df["stress_condition"],
            )
            st.dataframe(stage_table)

    else:
        st.warning(
            "ML dataset not found. Expected: "
            "10_full_biological_ml_dataset.csv"
        )


# =============================================================================
# PAGE: DATASETS
# =============================================================================

elif page == "Datasets":

    st.title("Datasets")

    if uploaded_files:
        st.subheader("Uploaded Excel workbooks")

        for uf in uploaded_files:
            st.markdown(f"**{uf.name}**")

            sheet_rank = select_best_sheet(uf)
            st.dataframe(sheet_rank.head(10))

            if not sheet_rank.empty:
                best = sheet_rank.iloc[0]["Sheet"]
                st.write(f"Best sheet: `{best}`")

                best_df = clean_columns(
                    pd.read_excel(
                        uf,
                        sheet_name=best,
                        engine="openpyxl",
                    )
                )

                st.dataframe(best_df.head(20))

    choice = st.selectbox(
        "Select output dataset",
        list(artifact_names.keys()),
    )

    df = artifacts.get(choice)

    if df is None:
        st.warning(
            f"{choice} is not available in {OUT_DIR}"
        )
    else:
        st.write(f"Shape: `{df.shape}`")
        st.dataframe(df.head(20))

        with st.expander("Descriptive statistics"):
            st.dataframe(df.describe(include="all").T)



# =============================================================================
# PAGE: DATA SEPARABILITY AUDIT
# =============================================================================

elif page == "Data Separability Audit":

    st.title("Data Separability & Leakage Audit")

    ml_df = artifacts.get("ML dataset")

    if ml_df is None:
        st.warning("ML dataset not found.")
        st.stop()

    st.markdown(
        """
        This page is designed to answer **why the classification accuracy may
        be 1.000**. It does not modify the data or artificially reduce
        accuracy. Instead, it quantifies how strongly each biological feature
        separates the experimentally defined stress classes.
        """
    )

    st.warning(
        "Important: the current target is derived from the experimental "
        "irrigation/stress treatment. Strong feature separation therefore "
        "means the model can reconstruct the experimental treatment condition; "
        "it does not by itself establish an independently validated "
        "physiological stress diagnosis."
    )

    target_options = [
        c for c in ["stress_condition", "stress_severity"]
        if c in ml_df.columns
    ]

    target = st.selectbox(
        "Target for audit",
        target_options,
    )

    exclude_cols = [
        "irrigation_block",
        "phenological_stage",
        "replication",
        "stress_condition",
        "stress_severity",
    ]

    feature_cols = [
        c for c in ml_df.columns
        if c not in exclude_cols
        and pd.api.types.is_numeric_dtype(ml_df[c])
    ]

    st.subheader("1. Dataset overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Observations", len(ml_df))
    c2.metric("Numeric biological features", len(feature_cols))
    c3.metric("Target classes", ml_df[target].nunique())

    st.subheader("Target distribution")

    target_counts = (
        ml_df[target]
        .astype(str)
        .value_counts()
        .rename_axis("Class")
        .reset_index(name="N")
    )

    st.dataframe(target_counts)

    st.subheader("2. Feature-wise class separation")

    rows = []

    y_codes, target_encoder = encode_target(
        ml_df[target]
    )

    for feature in feature_cols:

        x = pd.to_numeric(
            ml_df[feature],
            errors="coerce",
        )

        if x.notna().sum() < 3:
            continue

        # Spearman correlation with ordered encoded target is an exploratory
        # association only. It is not interpreted as causal.
        rho = x.corr(
            pd.Series(
                y_codes,
                index=ml_df.index,
            ),
            method="spearman",
        )

        # Mutual information is also exploratory and is calculated on
        # median-imputed values.
        x_mi = x.fillna(x.median()).to_numpy().reshape(-1, 1)

        try:
            mi = mutual_info_classif(
                x_mi,
                y_codes,
                random_state=RANDOM_STATE,
                discrete_features=False,
            )[0]
        except Exception:
            mi = np.nan

        rows.append({
            "Feature": feature,
            "Spearman rho": rho,
            "Absolute rho": abs(rho) if pd.notna(rho) else np.nan,
            "Mutual Information": mi,
            "Minimum": x.min(),
            "Maximum": x.max(),
            "Mean": x.mean(),
            "SD": x.std(),
        })

    audit_df = (
        pd.DataFrame(rows)
        .sort_values(
            ["Absolute rho", "Mutual Information"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    st.dataframe(
        audit_df.style.format(
            {
                "Spearman rho": "{:.3f}",
                "Absolute rho": "{:.3f}",
                "Mutual Information": "{:.3f}",
                "Minimum": "{:.3f}",
                "Maximum": "{:.3f}",
                "Mean": "{:.3f}",
                "SD": "{:.3f}",
            }
        )
    )

    st.subheader("3. Class-specific ranges")

    selected_feature = st.selectbox(
        "Select feature",
        feature_cols,
        index=(
            feature_cols.index("Proline")
            if "Proline" in feature_cols
            else 0
        ),
    )

    range_rows = []

    for cls, group in ml_df.groupby(target):

        vals = pd.to_numeric(
            group[selected_feature],
            errors="coerce",
        ).dropna()

        range_rows.append({
            "Class": str(cls),
            "N": len(vals),
            "Minimum": vals.min(),
            "Maximum": vals.max(),
            "Mean": vals.mean(),
            "SD": vals.std(),
        })

    range_df = pd.DataFrame(range_rows)

    st.dataframe(
        range_df.style.format(
            {
                "Minimum": "{:.3f}",
                "Maximum": "{:.3f}",
                "Mean": "{:.3f}",
                "SD": "{:.3f}",
            }
        )
    )

    # Plot distribution without relying on seaborn.
    fig, ax = plt.subplots(figsize=(8, 5))

    classes = list(
        ml_df[target]
        .astype(str)
        .drop_duplicates()
    )

    values = []

    for cls in classes:
        vals = pd.to_numeric(
            ml_df.loc[
                ml_df[target].astype(str) == cls,
                selected_feature,
            ],
            errors="coerce",
        ).dropna()

        values.append(vals.to_numpy())

    ax.boxplot(
        values,
        tick_labels=classes,
    )

    ax.set_xlabel("Experimental class")
    ax.set_ylabel(selected_feature)
    ax.set_title(
        f"{selected_feature} by experimental class"
    )

    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("4. Univariate 5-fold OOF classification")

    st.markdown(
        """
        This is a critical diagnostic. If **Proline alone** achieves near
        perfect OOF classification, the 1.000 result of the multivariable
        models is primarily explained by the underlying biological
        separability rather than model complexity.
        """
    )

    univariate_features = [
        f for f in ["Proline", "Leaf_Potassium"]
        if f in ml_df.columns
    ]

    uni_model = st.selectbox(
        "Univariate classifier",
        [
            "Logistic Regression",
            "LDA",
            "QDA",
            "KNN",
            "Linear SVM",
            "SVM (RBF)",
            "Naive Bayes",
            "Random Forest",
            "Extra Trees",
            "Gradient Boosting",
        ],
        key="uni_model",
    )

    if st.button(
        "Run Proline / Leaf-K univariate tests",
        type="primary",
    ):

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        uni_rows = []

        for feature in univariate_features:

            X_uni = ml_df[[feature]].copy()

            model = make_pipeline_model(
                uni_model
            )

            pred = cross_val_predict(
                model,
                X_uni,
                y_codes,
                cv=cv,
                method="predict",
            )

            m = metric_dict(
                y_codes,
                pred,
            )

            uni_rows.append({
                "Feature": feature,
                "Model": uni_model,
                "OOF Accuracy": m["Accuracy"],
                "OOF Balanced Accuracy": m["Balanced Accuracy"],
                "OOF Macro-F1": m["F1 Macro"],
                "OOF MCC": m["MCC"],
            })

        uni_results = pd.DataFrame(
            uni_rows
        )

        st.dataframe(
            uni_results.style.format(
                {
                    "OOF Accuracy": "{:.3f}",
                    "OOF Balanced Accuracy": "{:.3f}",
                    "OOF Macro-F1": "{:.3f}",
                    "OOF MCC": "{:.3f}",
                }
            )
        )

        if not uni_results.empty:

            best_row = uni_results.iloc[
                uni_results["OOF Accuracy"].argmax()
            ]

            st.info(
                f"Best univariate result: "
                f"{best_row['Feature']} with "
                f"OOF accuracy = "
                f"{best_row['OOF Accuracy']:.3f}."
            )

    st.subheader("5. Biological interpretation")

    if (
        "Proline" in ml_df.columns
        and "Leaf_Potassium" in ml_df.columns
    ):

        proline_summary = (
            ml_df.groupby(target)["Proline"]
            .agg(["min", "max", "mean", "std"])
            .reset_index()
        )

        k_summary = (
            ml_df.groupby(target)["Leaf_Potassium"]
            .agg(["min", "max", "mean", "std"])
            .reset_index()
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Proline**")
            st.dataframe(
                proline_summary.style.format(
                    {
                        "min": "{:.3f}",
                        "max": "{:.3f}",
                        "mean": "{:.3f}",
                        "std": "{:.3f}",
                    }
                )
            )

        with c2:
            st.markdown("**Leaf potassium**")
            st.dataframe(
                k_summary.style.format(
                    {
                        "min": "{:.3f}",
                        "max": "{:.3f}",
                        "mean": "{:.3f}",
                        "std": "{:.3f}",
                    }
                )
            )

        st.success(
            "Use these distributions to explain the 1.000 ML result. "
            "If class ranges do not overlap, perfect classification is "
            "mathematically plausible and should be reported rather than "
            "artificially suppressed."
        )


# =============================================================================
# PAGE: FEATURE ABLATION
# =============================================================================

elif page == "Feature Ablation":

    st.title("Feature Ablation & Model Sensitivity")

    ml_df = artifacts.get("ML dataset")

    if ml_df is None:
        st.warning("ML dataset not found.")
        st.stop()

    target = st.selectbox(
        "Target variable",
        [
            c for c in ["stress_condition", "stress_severity"]
            if c in ml_df.columns
        ],
        key="ablation_target",
    )

    feature_sets = {
        "Proline only": ["Proline"],
        "Leaf Potassium only": ["Leaf_Potassium"],
        "Proline + Leaf Potassium": [
            "Proline",
            "Leaf_Potassium",
        ],
        "Physiological (7)": [
            "Proline",
            "Leaf_Potassium",
            "TCSA",
            "Trunk_Girth",
            "Initial_Bloom",
            "Final_Bloom",
            "Bloom_Intensity",
        ],
        "All biological": [
            c for c in ml_df.columns
            if c not in [
                "irrigation_block",
                "phenological_stage",
                "replication",
                "stress_condition",
                "stress_severity",
            ]
        ],
    }

    model_name = st.selectbox(
        "Classifier",
        [
            "Logistic Regression",
            "LDA",
            "QDA",
            "KNN",
            "Linear SVM",
            "SVM (RBF)",
            "Naive Bayes",
            "Random Forest",
            "Extra Trees",
            "Gradient Boosting",
        ],
        key="ablation_model",
    )

    if st.button(
        "Run feature ablation",
        type="primary",
        key="run_ablation",
    ):

        y, le = encode_target(ml_df[target])

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        rows = []

        for set_name, features in feature_sets.items():

            features = [
                f for f in features
                if f in ml_df.columns
            ]

            X = ml_df[features].copy()
            model = make_pipeline_model(model_name)

            pred = cross_val_predict(
                model,
                X,
                y,
                cv=cv,
                method="predict",
            )

            m = metric_dict(y, pred)

            rows.append({
                "Feature set": set_name,
                "N features": len(features),
                "OOF Accuracy": m["Accuracy"],
                "OOF Precision": m["Precision Macro"],
                "OOF Recall": m["Recall Macro"],
                "OOF Macro-F1": m["F1 Macro"],
                "OOF Balanced Accuracy": m["Balanced Accuracy"],
                "OOF MCC": m["MCC"],
            })

        result = pd.DataFrame(rows)

        st.dataframe(
            result.style.format({
                "OOF Accuracy": "{:.3f}",
                "OOF Balanced Accuracy": "{:.3f}",
                "OOF Macro-F1": "{:.3f}",
                "OOF MCC": "{:.3f}",
            })
        )

        st.subheader("Interpretation")

        best = result.iloc[result["OOF Accuracy"].argmax()]

        st.write(
            f"The highest OOF accuracy was {best['OOF Accuracy']:.3f} "
            f"using **{best['Feature set']}**."
        )

        if (
            result.loc[
                result["Feature set"] == "Proline only",
                "OOF Accuracy",
            ].iloc[0] >= 0.99
        ):
            st.warning(
                "Proline alone provides near-perfect separation in the "
                "current dataset. Changing to a more sophisticated model "
                "will therefore not solve the 1.000-accuracy issue; the "
                "underlying experimental classes are already strongly "
                "separated by this predictor."
            )

        st.info(
            "This page is diagnostic. It should be used to explain why "
            "different classifiers obtain similar performance, not to "
            "select a model solely because it produces a lower accuracy."
        )


# =============================================================================
# PAGE: TRAIN & EVALUATE ML
# =============================================================================

elif page == "Train & Evaluate ML":

    st.title("Train & Evaluate ML Models")

    ml_df = artifacts.get("ML dataset")

    if ml_df is None:
        st.warning("ML dataset not found.")
        st.stop()

    meta_cols = [
        "irrigation_block",
        "phenological_stage",
        "replication",
        "stress_severity",
    ]

    target = st.selectbox(
        "Target variable",
        [
            c for c in ["stress_condition", "stress_severity"]
            if c in ml_df.columns
        ],
    )

    feature_sets = {
        "Physiological (7)": [
            "Proline",
            "Leaf_Potassium",
            "TCSA",
            "Trunk_Girth",
            "Initial_Bloom",
            "Final_Bloom",
            "Bloom_Intensity",
        ],
        "Physiological only (2)": [
            "Proline",
            "Leaf_Potassium",
        ],
        "All biological (15)": [
            c for c in ml_df.columns
            if c not in meta_cols + ["stress_condition"]
        ],
    }

    feature_sets = {
        name: [c for c in cols if c in ml_df.columns]
        for name, cols in feature_sets.items()
    }

    fs_choice = st.selectbox(
        "Feature set",
        list(feature_sets.keys()),
    )

    features = feature_sets[fs_choice]

    st.write(
        f"Using **{len(features)} features**: "
        + ", ".join(features)
    )

    available_models = [
        "Logistic Regression",
        "LDA",
        "QDA",
        "KNN",
        "Linear SVM",
        "SVM (RBF)",
        "Naive Bayes",
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
    ]

    chosen_models = st.multiselect(
        "Models",
        available_models,
        default=available_models,
    )

    st.caption(
        "Evaluation uses leakage-safe preprocessing. Scaling and imputation "
        "are fitted independently inside each cross-validation training fold."
    )

    if st.button("Train models", type="primary"):

        if not chosen_models:
            st.error("Select at least one model.")
            st.stop()

        X = ml_df[features].copy()

        y, le = encode_target(
            ml_df[target]
        )

        with st.spinner(
            "Running repeated stratified 5-fold cross-validation..."
        ):

            (
                results,
                trained,
                oof_predictions,
            ) = evaluate_models(
                X,
                y,
                chosen_models,
                n_splits=N_SPLITS,
                n_repeats=N_REPEATS,
            )

        st.session_state["trained_models"] = trained
        st.session_state["ml_results"] = results
        st.session_state["oof_predictions"] = oof_predictions
        st.session_state["ml_class_names"] = le.classes_
        st.session_state["ml_label_encoder"] = le
        st.session_state["ml_features"] = features
        st.session_state["ml_target"] = target
        st.session_state["ml_X"] = X
        st.session_state["_ml_results_schema_version"] = ML_RESULTS_SCHEMA_VERSION
        st.session_state["ml_y"] = y

        st.success(
            "Models trained and evaluated using leakage-safe cross-validation."
        )

        st.warning(
            "If several model families still achieve 1.000, do not interpret "
            "this as a model-selection problem. It indicates strong separation "
            "in the available experimental predictors. Use the Separability "
            "Audit, LOSO/LORO and independent validation to assess robustness."
        )

    if "ml_results" in st.session_state:

        st.subheader("Cross-validated metrics")

        display_cols = [
            "Model",

            "Accuracy",
            "Accuracy Mean",
            "Accuracy SD",

            "Precision Macro",
            "Precision Macro Mean",
            "Precision Macro SD",

            "Recall Macro",
            "Recall Macro Mean",
            "Recall Macro SD",

            "F1 Macro",
            "F1 Macro Mean",
            "F1 Macro SD",

            "Balanced Accuracy",
            "Balanced Accuracy Mean",
            "Balanced Accuracy SD",

            "MCC",
        ]

        results = st.session_state["ml_results"].copy()

        # Defensive compatibility check. If an old DataFrame survived a
        # Streamlit rerun, never crash with KeyError. Ask the user to rerun
        # training so all new metric columns are generated.
        missing_cols = [c for c in display_cols if c not in results.columns]

        if missing_cols:
            st.warning(
                "The displayed model results were created by an older "
                "version of the application. Please click 'Train models' "
                "again to regenerate Accuracy, Precision, Recall and F1 "
                "metrics."
            )
            st.caption(
                "Missing columns: " + ", ".join(missing_cols)
            )
            st.stop()

        st.dataframe(
            results[display_cols].style.format(
                {
                    c: "{:.3f}"
                    for c in display_cols
                    if c != "Model"
                }
            ),
            use_container_width=True,
        )

        st.download_button(
            "Download model metrics (CSV)",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="kiwifruit_model_metrics_precision_recall_f1_accuracy.csv",
            mime="text/csv",
        )

        if not results.empty:
            best_acc = results["Accuracy"].max()
            perfect_models = int(
                np.isclose(results["Accuracy"], 1.0).sum()
            )
            st.info(
                f"{perfect_models} of {len(results)} selected model(s) have "
                f"OOF accuracy = 1.000. If this persists across different "
                f"model families, inspect feature-wise separability rather "
                f"than deliberately weakening the classifier."
            )

        st.markdown(
            """
            **How to read the table**

            - **Accuracy, macro-Precision, macro-Recall, macro-F1, Balanced
              Accuracy and MCC** are calculated from the independent five-fold
              out-of-fold predictions.
            - **Mean ± SD** columns come from repeated stratified 5-fold CV
              (5 folds × 10 repetitions).
            - The final model fitted on all observations is used only for
              prediction of new observations.
            """
        )

        selected_model = st.selectbox(
            "Select model for detailed OOF evaluation",
            list(st.session_state["trained_models"].keys()),
        )

        tm = st.session_state["trained_models"][selected_model]

        oof = st.session_state["oof_predictions"][selected_model]
        y_true = st.session_state["ml_y"]
        class_names = st.session_state["ml_class_names"]

        st.subheader("True out-of-fold classification report")

        report = classification_report(
            y_true,
            oof,
            target_names=[str(c) for c in class_names],
            output_dict=True,
            zero_division=0,
        )

        class_report = pd.DataFrame(report).T.reset_index()
        class_report = class_report.rename(columns={"index": "Class"})

        st.dataframe(
            class_report.style.format({
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1-score": "{:.3f}",
                "support": "{:.0f}",
            }),
            use_container_width=True,
        )

        oof_metrics = metric_dict(y_true, oof)

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Accuracy",
            f"{oof_metrics['Accuracy']:.3f}",
        )
        c2.metric(
            "Macro Precision",
            f"{oof_metrics['Precision Macro']:.3f}",
        )
        c3.metric(
            "Macro Recall",
            f"{oof_metrics['Recall Macro']:.3f}",
        )
        c4.metric(
            "Macro F1",
            f"{oof_metrics['F1 Macro']:.3f}",
        )
        c5.metric(
            "MCC",
            f"{oof_metrics['MCC']:.3f}",
        )

        st.caption(
            "Precision, recall and F1 are reported per class and as macro "
            "averages. The confusion matrix below provides the class-by-class "
            "error pattern."
        )

        st.subheader("Out-of-fold confusion matrix")

        cm = confusion_matrix(
            y_true,
            oof,
        )

        fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(cm)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))

        ax.set_xticklabels(
            [str(c) for c in class_names],
            rotation=45,
            ha="right",
        )

        ax.set_yticklabels(
            [str(c) for c in class_names]
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(
            f"5-fold OOF Confusion Matrix — {selected_model}"
        )

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                )

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        st.pyplot(fig)

        st.subheader("Predict for a genuinely new observation")

        cols = st.columns(3)
        inputs = {}

        for i, f in enumerate(tm["features"]):

            series = pd.to_numeric(
                ml_df[f],
                errors="coerce",
            )

            default = float(series.median())
            minv = float(series.min())
            maxv = float(series.max())

            if minv == maxv:
                minv = minv - 1.0
                maxv = maxv + 1.0

            inputs[f] = cols[i % 3].number_input(
                f,
                min_value=minv,
                max_value=maxv,
                value=default,
            )

        x_new = pd.DataFrame([inputs])

        final_model = tm["model"]

        pred = final_model.predict(x_new)

        if hasattr(final_model, "predict_proba"):
            probs = final_model.predict_proba(x_new)[0]
        else:
            # Fallback for estimators without probability output.
            probs = np.zeros(len(class_names), dtype=float)
            probs[int(pred[0])] = 1.0

        pred_label = class_names[int(pred[0])]

        st.success(
            f"**Predicted {st.session_state['ml_target']}: "
            f"{pred_label}**"
        )

        prob_df = pd.DataFrame({
            "Class": class_names,
            "Probability": probs,
        })

        st.bar_chart(
            prob_df.set_index("Class")
        )

        st.warning(
            "This prediction is an application of the final model trained "
            "on the complete experimental dataset. It is not an independent "
            "validation result."
        )


# =============================================================================
# PAGE: ROBUSTNESS VALIDATION
# =============================================================================

elif page == "Robustness Validation":

    st.title("Robustness & Generalization Validation")

    ml_df = artifacts.get("ML dataset")

    if ml_df is None:
        st.warning("ML dataset not found.")
        st.stop()

    features = [
        "Proline",
        "Leaf_Potassium",
        "TCSA",
        "Trunk_Girth",
        "Initial_Bloom",
        "Final_Bloom",
        "Bloom_Intensity",
    ]

    features = [
        f for f in features
        if f in ml_df.columns
    ]

    target = st.selectbox(
        "Target",
        [
            c for c in ["stress_condition", "stress_severity"]
            if c in ml_df.columns
        ],
    )

    feature_choice = st.selectbox(
        "Feature set",
        [
            "Physiological only (2)",
            "Physiological (7)",
            "All biological (15)",
        ],
    )

    feature_map = {
        "Physiological only (2)": [
            "Proline",
            "Leaf_Potassium",
        ],
        "Physiological (7)": [
            "Proline",
            "Leaf_Potassium",
            "TCSA",
            "Trunk_Girth",
            "Initial_Bloom",
            "Final_Bloom",
            "Bloom_Intensity",
        ],
        "All biological (15)": [
            c for c in ml_df.columns
            if c not in [
                "irrigation_block",
                "phenological_stage",
                "replication",
                "stress_condition",
                "stress_severity",
            ]
        ],
    }

    features = [
        f for f in feature_map[feature_choice]
        if f in ml_df.columns
    ]

    X = ml_df[features].copy()
    y, le = encode_target(ml_df[target])

    model_name = st.selectbox(
        "Model",
        [
            "Logistic Regression",
            "LDA",
            "QDA",
            "KNN",
            "Linear SVM",
            "SVM (RBF)",
            "Naive Bayes",
            "Random Forest",
            "Extra Trees",
            "Gradient Boosting",
        ],
    )

    st.subheader("1. Leave-One-Stage-Out (LOSO)")

    if "phenological_stage" not in ml_df.columns:
        st.warning("Phenological stage column not available.")
    elif st.button("Run LOSO", key="run_loso"):

        summary, overall = group_holdout_evaluation(
            X,
            y,
            ml_df["phenological_stage"],
            model_name,
            "Phenological Stage",
        )

        st.dataframe(
            summary.style.format(
                {
                    "Accuracy": "{:.3f}",
                    "Precision Macro": "{:.3f}",
                    "Recall Macro": "{:.3f}",
                    "F1 Macro": "{:.3f}",
                    "Balanced Accuracy": "{:.3f}",
                    "MCC": "{:.3f}",
                }
            )
        )

        st.subheader("Overall LOSO performance")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Accuracy",
            f"{overall['Accuracy']:.3f}",
        )
        c2.metric(
            "Macro Precision",
            f"{overall['Precision Macro']:.3f}",
        )
        c3.metric(
            "Macro Recall",
            f"{overall['Recall Macro']:.3f}",
        )
        c4.metric(
            "Macro F1",
            f"{overall['F1 Macro']:.3f}",
        )
        c5.metric(
            "MCC",
            f"{overall['MCC']:.3f}",
        )

        st.info(
            "LOSO asks a stronger question than random CV: can the model "
            "classify observations from a phenological stage that was not "
            "used for training?"
        )

    st.divider()

    st.subheader("2. Leave-One-Replication-Out (LORO)")

    if "replication" not in ml_df.columns:
        st.warning("Replication column not available.")
    elif st.button("Run LORO", key="run_loro"):

        summary, overall = group_holdout_evaluation(
            X,
            y,
            ml_df["replication"],
            model_name,
            "Replication",
        )

        st.dataframe(
            summary.style.format(
                {
                    "Accuracy": "{:.3f}",
                    "Precision Macro": "{:.3f}",
                    "Recall Macro": "{:.3f}",
                    "F1 Macro": "{:.3f}",
                    "Balanced Accuracy": "{:.3f}",
                    "MCC": "{:.3f}",
                }
            )
        )

        st.subheader("Overall LORO performance")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Accuracy",
            f"{overall['Accuracy']:.3f}",
        )
        c2.metric(
            "Macro Precision",
            f"{overall['Precision Macro']:.3f}",
        )
        c3.metric(
            "Macro Recall",
            f"{overall['Recall Macro']:.3f}",
        )
        c4.metric(
            "Macro F1",
            f"{overall['F1 Macro']:.3f}",
        )
        c5.metric(
            "MCC",
            f"{overall['MCC']:.3f}",
        )

        st.info(
            "LORO evaluates whether the model generalizes across experimental "
            "replications rather than relying on replication-specific patterns."
        )

    st.divider()

    st.subheader("3. Permutation-label test")

    n_perm = st.slider(
        "Number of permutations",
        min_value=50,
        max_value=1000,
        value=200,
        step=50,
    )

    st.caption(
        "Use 1000 permutations for the final manuscript when runtime permits."
    )

    if st.button("Run permutation test", key="run_perm"):

        with st.spinner(
            f"Running {n_perm} label permutations..."
        ):

            (
                observed,
                perm_scores,
                p_value,
            ) = permutation_test(
                X,
                y,
                model_name,
                n_permutations=n_perm,
            )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Observed 5-fold CV accuracy",
            f"{observed:.3f}",
        )

        c2.metric(
            "Mean permuted accuracy",
            f"{perm_scores.mean():.3f}",
        )

        c3.metric(
            "Empirical p-value",
            f"{p_value:.4f}",
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(
            perm_scores,
            bins=20,
        )

        ax.axvline(
            observed,
            linestyle="--",
            linewidth=2,
            label=f"Observed = {observed:.3f}",
        )

        ax.set_xlabel("CV accuracy under permuted labels")
        ax.set_ylabel("Frequency")
        ax.set_title("Permutation Test")
        ax.legend()

        fig.tight_layout()
        st.pyplot(fig)

        st.info(
            "A small empirical p-value indicates that the observed predictive "
            "performance is unlikely to arise from random label assignment. "
            "This does not establish external generalization."
        )


# =============================================================================
# PAGE: OOF PREDICTIONS
# =============================================================================

elif page == "OOF Predictions":

    st.title("Out-of-Fold Predictions")

    if "oof_predictions" in st.session_state:

        y = st.session_state["ml_y"]
        class_names = st.session_state["ml_class_names"]

        selected_model = st.selectbox(
            "Model",
            list(st.session_state["oof_predictions"].keys()),
        )

        pred = st.session_state["oof_predictions"][selected_model]

        oof_table = pd.DataFrame({
            "True": [
                class_names[int(v)]
                for v in y
            ],
            "OOF_Predicted": [
                class_names[int(v)]
                for v in pred
            ],
        })

        oof_table["Correct"] = (
            oof_table["True"]
            == oof_table["OOF_Predicted"]
        )

        st.dataframe(oof_table)

        st.metric(
            "OOF accuracy",
            f"{oof_table['Correct'].mean():.3f}",
        )

    else:

        oof = artifacts.get("OOF predictions")

        if oof is None:
            st.warning(
                "No OOF predictions are available. Train models first."
            )
            st.stop()

        st.write(f"Shape: `{oof.shape}`")
        st.dataframe(oof.head(30))

        if "stress_condition" in oof.columns:

            model_cols = [
                c for c in oof.columns
                if c.startswith("OOF_")
            ]

            if model_cols:

                st.subheader(
                    "Pre-computed model agreement"
                )

                rows = []

                for col in model_cols:
                    rows.append({
                        "Model": col.replace("OOF_", ""),
                        "Accuracy": (
                            oof[col]
                            == oof["stress_condition"]
                        ).mean(),
                    })

                agreement = pd.DataFrame(rows)

                st.dataframe(
                    agreement.style.format(
                        {"Accuracy": "{:.3f}"}
                    )
                )


# =============================================================================
# PAGE: SHAP / XAI
# =============================================================================

elif page == "SHAP / XAI":

    st.title("SHAP / Explainable AI")

    shap_global = artifacts.get("SHAP global")

    if shap_global is not None:

        st.subheader(
            "Pre-computed global SHAP importance"
        )

        st.dataframe(shap_global)

        if {
            "Feature",
            "Mean_Absolute_SHAP",
        }.issubset(shap_global.columns):

            plot_df = shap_global.sort_values(
                "Mean_Absolute_SHAP"
            )

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.barh(
                plot_df["Feature"],
                plot_df["Mean_Absolute_SHAP"],
            )

            ax.set_xlabel(
                "Mean |SHAP value|"
            )

            ax.set_title(
                "Global SHAP Feature Importance"
            )

            fig.tight_layout()
            st.pyplot(fig)

    else:
        st.info(
            "No pre-computed SHAP file found."
        )

    if (
        SHAP_AVAILABLE
        and "trained_models" in st.session_state
    ):

        st.divider()

        st.subheader(
            "Compute SHAP for trained tree model"
        )

        model_names = [
            name
            for name in st.session_state["trained_models"].keys()
            if name in [
                "Random Forest",
                "Extra Trees",
            ]
        ]

        if not model_names:

            st.info(
                "Train Random Forest or Extra Trees first "
                "to compute live SHAP explanations."
            )

        else:

            model_name = st.selectbox(
                "Model for SHAP",
                model_names,
            )

            if st.button(
                "Compute SHAP summary"
            ):

                tm = st.session_state[
                    "trained_models"
                ][model_name]

                X = st.session_state[
                    "ml_X"
                ]

                fitted_model = tm["model"]

                estimator, X_transformed, feature_names = (
                    get_tree_model_and_X_for_shap(
                        fitted_model,
                        X,
                    )
                )

                if estimator is None:

                    st.error(
                        "Selected model does not expose tree "
                        "feature importances."
                    )

                else:

                    try:

                        explainer = shap.TreeExplainer(
                            estimator
                        )

                        shap_values = (
                            explainer.shap_values(
                                X_transformed
                            )
                        )

                        # SHAP versions can return either a list
                        # or an ndarray for multiclass models.
                        if isinstance(
                            shap_values,
                            list,
                        ):

                            shap_matrix = np.asarray(
                                shap_values
                            )

                            # Mean absolute SHAP over classes
                            mean_abs = np.mean(
                                np.abs(
                                    shap_matrix
                                ),
                                axis=(0, 1),
                            )

                            shap_importance = pd.DataFrame({
                                "Feature": feature_names,
                                "Mean_Absolute_SHAP": mean_abs,
                            }).sort_values(
                                "Mean_Absolute_SHAP",
                                ascending=False,
                            )

                            st.subheader(
                                "Global SHAP importance"
                            )

                            st.dataframe(
                                shap_importance
                            )

                            fig, ax = plt.subplots(
                                figsize=(8, 5)
                            )

                            p = (
                                shap_importance
                                .sort_values(
                                    "Mean_Absolute_SHAP"
                                )
                            )

                            ax.barh(
                                p["Feature"],
                                p["Mean_Absolute_SHAP"],
                            )

                            ax.set_xlabel(
                                "Mean |SHAP value|"
                            )

                            fig.tight_layout()
                            st.pyplot(fig)

                        else:

                            # New SHAP versions may return:
                            # samples x features x classes
                            arr = np.asarray(
                                shap_values
                            )

                            if arr.ndim == 3:

                                mean_abs = np.mean(
                                    np.abs(arr),
                                    axis=(0, 2),
                                )

                                shap_importance = pd.DataFrame({
                                    "Feature": feature_names,
                                    "Mean_Absolute_SHAP": mean_abs,
                                }).sort_values(
                                    "Mean_Absolute_SHAP",
                                    ascending=False,
                                )

                                st.dataframe(
                                    shap_importance
                                )

                    except Exception as e:

                        st.error(
                            f"SHAP computation failed: {e}"
                        )

    elif not SHAP_AVAILABLE:

        st.info(
            "`shap` is not installed. Install it to "
            "enable live SHAP analysis."
        )


# =============================================================================
# PAGE: IRRIGATION DECISION SUPPORT
# =============================================================================

elif page == "Irrigation Decision Support":

    st.title("Stage-Aware Irrigation Decision Support")

    ml_df = artifacts.get("ML dataset")

    if ml_df is None:
        st.warning("ML dataset not found.")
        st.stop()

    features = [
        "Proline",
        "Leaf_Potassium",
        "TCSA",
        "Trunk_Girth",
        "Initial_Bloom",
        "Final_Bloom",
        "Bloom_Intensity",
    ]

    features = [
        f for f in features
        if f in ml_df.columns
    ]

    st.info(
        "This module provides model-based experimental stress assessment. "
        "The irrigation recommendation is a decision-support prompt, not a "
        "validated field-capacity prescription."
    )

    st.subheader(
        "Input current observation"
    )

    if "irrigation_block" in ml_df.columns:

        block = st.selectbox(
            "Experimental irrigation block",
            sorted(
                ml_df["irrigation_block"]
                .dropna()
                .astype(str)
                .unique()
            ),
        )

    else:
        block = None

    if "phenological_stage" in ml_df.columns:

        stage = st.selectbox(
            "Phenological stage",
            sorted(
                ml_df["phenological_stage"]
                .dropna()
                .astype(str)
                .unique()
            ),
        )

    else:
        stage = None

    cols = st.columns(3)
    inputs = {}

    for i, f in enumerate(features):

        series = pd.to_numeric(
            ml_df[f],
            errors="coerce",
        )

        default = float(series.median())
        minv = float(series.min())
        maxv = float(series.max())

        if minv == maxv:
            minv -= 1
            maxv += 1

        inputs[f] = cols[i % 3].number_input(
            f,
            min_value=minv,
            max_value=maxv,
            value=default,
        )

    if st.button(
        "Predict & assess",
        type="primary",
    ):

        X = ml_df[features].copy()
        y, le = encode_target(
            ml_df["stress_condition"]
        )

        model = make_pipeline_model(
            "Random Forest"
        )

        model.fit(X, y)

        x_new = pd.DataFrame(
            [inputs]
        )

        pred = model.predict(
            x_new
        )[0]

        probs = model.predict_proba(
            x_new
        )[0]

        label = le.classes_[int(pred)]

        st.subheader(
            f"Predicted experimental condition: **{label}**"
        )

        prob_df = pd.DataFrame({
            "Condition": le.classes_,
            "Probability": probs,
        })

        st.bar_chart(
            prob_df.set_index(
                "Condition"
            )
        )

        # Conservative decision-support language.
        if label == "Low_Stress":

            rec = (
                f"Low experimental stress signal during {stage}. "
                "Continue monitoring soil moisture and plant response."
            )

        elif label == "Moderate_Stress":

            rec = (
                f"Moderate experimental stress signal during {stage}. "
                "Review soil-moisture status and irrigation timing; "
                "consider intervention according to the validated "
                "experimental irrigation protocol."
            )

        else:

            rec = (
                f"High experimental stress signal during {stage}. "
                "Prioritize assessment of soil moisture and plant status "
                "and consider irrigation intervention according to the "
                "validated experimental protocol."
            )

        st.success(rec)

        st.subheader(
            "Comparable experimental records"
        )

        if (
            block is not None
            and stage is not None
            and "irrigation_block" in ml_df.columns
            and "phenological_stage" in ml_df.columns
        ):

            mask = (
                ml_df["irrigation_block"].astype(str)
                == str(block)
            ) & (
                ml_df["phenological_stage"].astype(str)
                == str(stage)
            )

            subset = ml_df.loc[mask]

            st.dataframe(
                subset
            )

        st.warning(
            "Do not interpret the predicted class as a validated universal "
            "water-stress threshold. External field validation is required "
            "before operational irrigation scheduling."
        )
