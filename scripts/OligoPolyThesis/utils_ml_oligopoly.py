"""
Shared utilities for the oligometastatic vs. polymetastatic breast cancer thesis.

Contents:
  - full-availability modelling of the clinical, mutation and transcriptomic
    cohorts (nested cross-validation, out-of-fold validation, pooled metrics);
  - the N=110 common-cohort comparison and its reporting (``load_primary110``
    verifies the saved results);
  - interpretation-only fits for the common-cohort models;
  - transcriptomic representation support (PCA, GenomicSuperSignature,
    MultiPLIER, Tybalt and METABRIC-trained encoders, Integrated Gradients);
  - expression, overall-survival and TCGA-BRCA analyses of the eight
    model-prioritised genes;
  - plotting and presentation helpers used by the thesis notebooks.

Where a function name is modality-specific, the suffix (``_clinical``,
``_mutation``, ``_tpm``) marks the cohort it serves.
"""

from __future__ import annotations

import os
import re
# Logging only: retain WARNING/ERROR messages and all numerical defaults.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def _is_tensorflow_startup_info(line):
    """Recognise only the known C++ INFO startup records and absl banner."""
    if line.strip() == "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR":
        return True
    if line.startswith("To enable the following instructions:") and "rebuild TensorFlow" in line:
        return True
    return bool(re.match(r"^I(?:\d{4}|\d{4}-\d{2}-\d{2})", line)) and any(
        marker in line for marker in (
            "oneDNN custom operations are on", "Could not find cuda drivers",
            "This TensorFlow binary is optimized to use available CPU instructions",
        )
    )


from contextlib import contextmanager as _startup_contextmanager


@_startup_contextmanager
def _quiet_tensorflow_import():
    """Buffer C++ startup stderr briefly; replay every non-startup message.

    Some TensorFlow builds emit absl INFO before its logger is initialised and
    ignore TF_CPP_MIN_LOG_LEVEL there. This filter applies only during import.
    """
    import sys
    import tempfile
    sys.stderr.flush()
    original_stderr = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as captured:
        try:
            os.dup2(captured.fileno(), 2)
            yield
        finally:
            sys.stderr.flush()
            os.dup2(original_stderr, 2)
            os.close(original_stderr)
            captured.seek(0)
            for line in captured.readlines():
                if not _is_tensorflow_startup_info(line.decode("utf-8", errors="replace")):
                    os.write(2, line)


import json
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
with _quiet_tensorflow_import():
    import shap
import requests
import time
import mygene
import joblib
import random
with _quiet_tensorflow_import():
    import umap

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from flaml import AutoML
from itertools import combinations

import plotly.graph_objects as go
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    brier_score_loss,
    precision_recall_curve,
    average_precision_score,
    silhouette_score
    )
import re
from typing import Optional, Sequence, Tuple, List
from pathlib import Path
from IPython.display import display
with _quiet_tensorflow_import():
    import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import PCA


# ==============================================================================
# Clinical-Data Modelling Utilities (from utils_ml_clinical.py)
# ==============================================================================

def build_models_clinical(random_state=42, xgb_scale_pos_weight=1.0):
    """
    Build candidate model pipelines and hyperparameter grids.

    Each pipeline now includes a "sampler" step (SMOTE, random undersampling,
    or no resampling), selected via the grid search alongside the other
    hyperparameters, so the nested CV itself decides which imbalance
    strategy works best per fold rather than assuming one in advance.
    The sampler is applied inside the pipeline, so it only ever sees the
    training portion of each fold — never the held-out test data.

    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility.
    xgb_scale_pos_weight : float, default=1.0
        Class-imbalance weight for XGBoost. Usually calculated as
        number of negative samples / number of positive samples.

    Returns
    -------
    dict
        Dictionary of model pipelines and hyperparameter grids.
    """

    sampler_options = [
        "passthrough",
        SMOTE(random_state=random_state),
        RandomUnderSampler(random_state=random_state),
    ]

    models = {
        "logreg_l2": {
            "pipeline": ImbPipeline([
                ("scaler", StandardScaler()),
                ("sampler", "passthrough"),
                ("classifier", LogisticRegression(
                    solver="lbfgs",
                    penalty="l2",
                    max_iter=5000,
                    tol=1e-4,
                    random_state=random_state
                ))
            ]),
            "param_grid": {
                "classifier__C": [0.001, 0.01, 0.1, 1, 10],
                "classifier__class_weight": [None, "balanced"],
                "sampler": sampler_options,
            }
        },
        "logreg_l1": {
            "pipeline": ImbPipeline([
                ("scaler", StandardScaler()),
                ("sampler", "passthrough"),
                ("classifier", LogisticRegression(
                    solver="liblinear",
                    penalty="l1",
                    max_iter=5000,
                    tol=1e-4,
                    random_state=random_state
                ))
            ]),
            "param_grid": {
                "classifier__C": [0.001, 0.01, 0.1, 1, 10],
                "classifier__class_weight": [None, "balanced"],
                "sampler": sampler_options,
            }
        },
        "random_forest": {
            "pipeline": ImbPipeline([
                ("sampler", "passthrough"),
                ("clf", RandomForestClassifier(
                    random_state=random_state,
                    n_jobs=1
                ))
            ]),
            "param_grid": {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
                "clf__class_weight": [None, "balanced"],
                "sampler": sampler_options,
            }
        }
    }

    if XGBClassifier is not None:
        models["xgboost"] = {
            "pipeline": ImbPipeline([
                ("sampler", "passthrough"),
                ("clf", XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=1,
                    tree_method="hist"
                ))
            ]),
            "param_grid": {
                "clf__n_estimators": [50, 100],
                "clf__max_depth": [2, 3],
                "clf__learning_rate": [0.05, 0.1],
                "clf__subsample": [0.8],
                "clf__colsample_bytree": [0.8],
                "clf__reg_lambda": [1, 5],
                "clf__scale_pos_weight": [1.0, xgb_scale_pos_weight],
                "sampler": sampler_options,
            }
        }
    else:
        warnings.warn(
            "XGBoost is not installed. The 'xgboost' model will not be included. "
            "Install it with: pip install xgboost"
        )

    return models

def run_nested_cv_with_predictions_clinical(
    X,
    y,
    record_ids,
    sample_ids,
    models,
    outer_cv,
    inner_cv,
    scoring="f1",
    n_jobs=-1,
    coefficient_threshold=1e-8,
    top_n_tree=20
):
    """
    Run nested cross-validation for all candidate models and store metrics,
    predictions, inner-cross-validation scores, and fold-level feature rankings.

    For each candidate model, an inner cross-validation loop is used for
    hyperparameter tuning with GridSearchCV, while an outer cross-validation
    loop is used for performance estimation on held-out data.

    The function stores:
    - fold-level test metrics;
    - fold-level train metrics;
    - best inner-cross-validation score for each outer fold;
    - sample-level predictions for the held-out outer folds;
    - fold-level feature rankings extracted from the fitted best estimator
      selected by the inner cross-validation loop within each outer fold.

    For logistic regression models, feature rankings are based on coefficient
    magnitude. For tree-based models, feature rankings are based on feature
    importance.

    Parameters
    ----------
    X : pandas.DataFrame
        Predictor matrix.
    y : pandas.Series
        Binary target variable.
    record_ids : pandas.Series
        Patient record identifiers, used for saving fold-level predictions.
    sample_ids : pandas.Series
        Sample identifiers, used for saving fold-level predictions.
    models : dict
        Dictionary of candidate models and hyperparameter grids, usually
        returned by `build_models_clinical`.
    outer_cv : sklearn cross-validation splitter
        Outer cross-validation object used for out-of-training performance
        estimation.
    inner_cv : sklearn cross-validation splitter
        Inner cross-validation object used for hyperparameter selection.
    scoring : str, default="f1"
        Metric used by GridSearchCV for hyperparameter tuning.
    n_jobs : int, default=-1
        Number of parallel jobs used by GridSearchCV.
    coefficient_threshold : float, default=1e-8
        Threshold used to define whether a logistic regression coefficient is
        considered non-zero.
    top_n_tree : int, default=20
        Number of top-ranked features considered selected for tree-based models.

    Returns
    -------
    tuple of pandas.DataFrame
        ``results_test_df``:
            Fold-level test metrics for each model, including the
            corresponding best inner-cross-validation score.

        ``results_train_df``:
            Fold-level train metrics for each model, including the
            corresponding best inner-cross-validation score.

        ``predictions_df``:
            Sample-level predictions for each held-out outer test fold.

        ``feature_rankings_df``:
            Fold-level feature rankings for each model and outer fold.
    """

    all_test_results = []
    all_train_results = []
    all_predictions = []
    all_feature_rankings = []

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    record_ids = record_ids.reset_index(drop=True)
    sample_ids = sample_ids.reset_index(drop=True)

    outer_splits = clinical_outer_splits(outer_cv, X, y)

    for model_name, model_cfg in models.items():
        print(f"\n=== Running model: {model_name} ===")

        for fold_idx, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            print(f"Outer fold {fold_idx}/{len(outer_splits)}")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            search = GridSearchCV(
                estimator=model_cfg["pipeline"],
                param_grid=model_cfg["param_grid"],
                scoring=scoring,
                cv=inner_cv,
                n_jobs=n_jobs,
                refit=True
            )

            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            best_inner_score = search.best_score_
            best_params_json = json.dumps(search.best_params_, default=str)

            # ------------------------------------------------------------
            # Store feature rankings from the fitted best estimator selected
            # by inner cross-validation within this outer fold.
            # ------------------------------------------------------------
            if "logreg" in model_name:
                feature_ranking_df = extract_logreg_feature_rankings_clinical(
                    best_model,
                    X.columns
                )

                feature_ranking_df["selected_in_fold"] = (
                    feature_ranking_df["coefficient"].abs() > coefficient_threshold
                )

            elif model_name in ["random_forest", "xgboost"]:
                feature_ranking_df = extract_rf_feature_rankings_clinical(
                    best_model,
                    X.columns
                )

                feature_ranking_df["selected_in_fold"] = (
                    np.arange(1, len(feature_ranking_df) + 1) <= top_n_tree
                )

            else:
                raise ValueError(f"Unknown model type: {model_name}")

            feature_ranking_df = feature_ranking_df.reset_index(drop=True)
            feature_ranking_df["model"] = model_name
            feature_ranking_df["outer_fold"] = fold_idx
            feature_ranking_df["rank_in_fold"] = (
                np.arange(1, len(feature_ranking_df) + 1)
            )
            feature_ranking_df["best_params"] = best_params_json
            feature_ranking_df["best_inner_score"] = best_inner_score

            all_feature_rankings.append(feature_ranking_df)

            # ------------------------------------------------------------
            # Test performance on held-out outer fold
            # ------------------------------------------------------------
            y_test_proba = best_model.predict_proba(X_test)[:, 1]
            y_test_pred = best_model.predict(X_test)

            test_metrics = calculate_classification_metrics(
                y_true=y_test,
                y_pred=y_test_pred,
                y_prob=y_test_proba,
                y_true_constant=True
            )

            test_metrics["balanced_accuracy"] = (
                test_metrics.get("sensitivity", np.nan)
                + test_metrics.get("specificity", np.nan)
            ) / 2

            test_metrics.update(clinical_binary_metrics(y_test, y_test_pred, y_test_proba))

            test_row = {
                "model": model_name,
                "outer_fold": fold_idx,
                "dataset": "test",
                "best_params": best_params_json,
                "best_inner_score": best_inner_score,
                **test_metrics
            }

            all_test_results.append(test_row)

            # ------------------------------------------------------------
            # Train performance on outer training fold
            # ------------------------------------------------------------
            y_train_proba = best_model.predict_proba(X_train)[:, 1]
            y_train_pred = best_model.predict(X_train)

            train_metrics = calculate_classification_metrics(
                y_true=y_train,
                y_pred=y_train_pred,
                y_prob=y_train_proba,
                y_true_constant=True
            )

            train_metrics["balanced_accuracy"] = (
                train_metrics.get("sensitivity", np.nan)
                + train_metrics.get("specificity", np.nan)
            ) / 2

            train_metrics.update(clinical_binary_metrics(y_train, y_train_pred, y_train_proba))

            train_row = {
                "model": model_name,
                "outer_fold": fold_idx,
                "dataset": "train",
                "best_params": best_params_json,
                "best_inner_score": best_inner_score,
                **train_metrics
            }

            all_train_results.append(train_row)

            # ------------------------------------------------------------
            # Store sample-level predictions for the held-out test fold
            # ------------------------------------------------------------
            fold_predictions = pd.DataFrame({
                "model": model_name,
                "outer_fold": fold_idx,
                "record_id": record_ids.iloc[test_idx].values,
                "sample_id": sample_ids.iloc[test_idx].values,
                "dataset": "test",
                "row_position": test_idx,
                "y_true": y_test.values,
                "y_pred": y_test_pred,
                "y_proba": y_test_proba,
                "best_inner_score": best_inner_score,
                "best_params": best_params_json
            })

            all_predictions.append(fold_predictions)

    results_test_df = pd.DataFrame(all_test_results)
    results_train_df = pd.DataFrame(all_train_results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    feature_rankings_df = pd.concat(all_feature_rankings, ignore_index=True)

    return (
        results_test_df,
        results_train_df,
        predictions_df,
        feature_rankings_df
    )


def extract_logreg_feature_rankings_clinical(fitted_model, feature_names):
    """
    Extract feature rankings from a fitted logistic regression model.

    Parameters
    ----------
    fitted_model : sklearn Pipeline or estimator
        Fitted logistic regression pipeline or fitted logistic regression estimator.

    feature_names : list-like
        Names of the input features.

    Returns
    -------
    pandas.DataFrame
        DataFrame with feature names, coefficients and absolute coefficients,
        sorted by absolute coefficient magnitude.
    """

    if hasattr(fitted_model, "named_steps"):
        if "classifier" in fitted_model.named_steps:
            classifier = fitted_model.named_steps["classifier"]
        elif "clf" in fitted_model.named_steps:
            classifier = fitted_model.named_steps["clf"]
        else:
            raise KeyError(
                "Could not find classifier step in fitted_model. "
                "Expected either 'classifier' or 'clf'. "
                f"Available steps: {list(fitted_model.named_steps.keys())}"
            )
    else:
        classifier = fitted_model

    coefs = classifier.coef_[0]

    coef_df = pd.DataFrame({
        "feature": list(feature_names),
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    }).sort_values("abs_coefficient", ascending=False)

    return coef_df

def extract_rf_feature_rankings_clinical(fitted_model, feature_names):
    """
    Extract feature rankings from a fitted random forest model.

    Features are ranked according to their impurity-based feature importance
    values returned by the fitted random forest classifier.

    Parameters
    ----------
    fitted_model : sklearn.pipeline.Pipeline
        Fitted random forest pipeline.
    feature_names : list-like
        Names of the input features, in the same order used during model
        fitting.

    Returns
    -------
    pandas.DataFrame
        Table containing feature names and random forest importance values,
        sorted by decreasing importance.
    """

    importances = fitted_model.named_steps["clf"].feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)
    return importance_df


def plot_best_inner_score_histogram_and_folds(results_df, model_name=None, model_col="model"):
    """
    Two-panel plot: histogram of best_inner_score distribution across
    outer folds (left), and best_inner_score by fold in fold order (right),
    for one model. Helps spot whether the mean score is being driven by
    an unstable/outlier fold.
    """
    if model_name is not None:
        subset = results_df[results_df[model_col] == model_name].copy()
    else:
        subset = results_df.copy()

    subset = subset.reset_index(drop=True)
    subset["fold"] = subset.index + 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(subset["best_inner_score"], bins=min(10, len(subset)), color="steelblue", edgecolor="white")
    axes[0].set_xlabel("best_inner_score")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Distribution{f' — {model_name}' if model_name else ''}")

    axes[1].bar(subset["fold"], subset["best_inner_score"], color="steelblue")
    axes[1].axhline(subset["best_inner_score"].mean(), color="orange", linestyle="--", label="mean")
    axes[1].set_xlabel("Outer fold")
    axes[1].set_ylabel("best_inner_score")
    axes[1].set_title(f"By fold{f' — {model_name}' if model_name else ''}")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    
def plot_calibration_curve(predictions_df, model_name, n_bins=5):
    """
    Reliability diagram + Brier score, pooling predictions across all outer
    folds for one model. Checks whether predicted probabilities are
    trustworthy, not just whether the model discriminates well.
    """
    model_predictions_df = predictions_df[predictions_df["model"] == model_name]

    y_true = model_predictions_df["y_true"].values
    y_proba = model_predictions_df["y_proba"].values

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y_true, y_proba)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label=f"{model_name} (Brier = {brier:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return brier


# ==============================================================================
# Mutation-Data Modelling Utilities (from utils_ml_mutation.py)
# ==============================================================================

def build_models_mutation(random_state=42, xgb_scale_pos_weight=1.0):
    """
    Build candidate model pipelines and hyperparameter grids.

    Each pipeline includes a "selector" step (SelectKBest via F-test,
    SelectKBest via mutual information, or no selection) and a "sampler"
    step (SMOTE or no resampling), both chosen via the grid search alongside
    the other hyperparameters. Both steps are performed INSIDE the pipeline,
    so they only ever see the training portion of each fold — never the
    held-out test data — avoiding the leakage that would occur from
    selecting features or resampling on the full dataset beforehand.

    imblearn's Pipeline is used instead of sklearn's, since sklearn's
    Pipeline does not support samplers such as SMOTE.

    In the current scikit-learn API, logistic regression regularization is
    controlled through l1_ratio: l1_ratio=0.0 corresponds to L2 regularization,
    and l1_ratio=1.0 corresponds to L1 regularization. The C parameter controls
    the strength of regularization, with smaller C values corresponding to
    stronger regularization.

    Parameters
    ----------
    random_state : int, default=42
        Random seed used for reproducibility.
    xgb_scale_pos_weight : float, default=1.0
        Candidate class-imbalance weight for XGBoost, usually computed as the
        number of negative samples divided by the number of positive samples.

    Returns
    -------
    dict
        Dictionary where each key is a model name and each value contains
        ``pipeline`` and ``param_grid``.
    """

    selector_options = [
        "passthrough",
        SelectKBest(score_func=f_classif, k=200),
        SelectKBest(score_func=f_classif, k=500),
        SelectKBest(score_func=mutual_info_classif, k=200),
    ]

    sampler_options = [
        "passthrough",
        SMOTE(random_state=random_state),
    ]

    models = {
        "logreg_l1": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("selector", "passthrough"),
                ("sampler", "passthrough"),
                ("clf", LogisticRegression(
                    solver="saga",
                    penalty="l1",
                    max_iter=20000,
                    tol=1e-3,
                    random_state=random_state
                ))
            ]),
            "param_grid": {
                "clf__C": [0.0001, 0.001, 0.01, 0.05, 0.1],
                "clf__class_weight": [None, "balanced"],
                "selector": selector_options,
                "sampler": sampler_options,
            }
        },

        "random_forest": {
            "pipeline": Pipeline([
                ("selector", "passthrough"),
                ("sampler", "passthrough"),
                ("clf", RandomForestClassifier(
                    random_state=random_state,
                    n_jobs=1
                ))
            ]),
            "param_grid": {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
                "clf__class_weight": [None, "balanced"],
                "selector": selector_options,
                "sampler": sampler_options,
            }
        }
    }

    if XGBClassifier is not None:
        models["xgboost"] = {
            "pipeline": Pipeline([
                ("selector", "passthrough"),
                ("sampler", "passthrough"),
                ("clf", XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1,
                    tree_method="hist"
                ))
            ]),
            "param_grid": {
                "clf__n_estimators": [50, 100],
                "clf__max_depth": [2, 3],
                "clf__learning_rate": [0.05, 0.1],
                "clf__subsample": [0.8],
                "clf__colsample_bytree": [0.8],
                "clf__reg_lambda": [1, 5],
                "clf__scale_pos_weight": [1.0, xgb_scale_pos_weight],
                "selector": selector_options,
                "sampler": sampler_options,
            }
        }
    else:
        warnings.warn(
            "XGBoost is not installed. The 'xgboost' model will not be included. "
            "Install it with: pip install xgboost"
        )

    return models

def compute_metrics(y_true, y_pred, y_proba):
    """
    Compute classification performance metrics for binary prediction.

    The positive class is assumed to be the polymetastatic class, encoded as 1.
    The function returns threshold-dependent metrics, such as F1-score,
    macro-F1, weighted-F1, precision, recall, and balanced accuracy, as well as
    ROC-AUC based on the predicted probabilities.

    Parameters
    ----------
    y_true : array-like
        True binary class labels.
    y_pred : array-like
        Predicted binary class labels.
    y_proba : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Dictionary containing ROC-AUC, balanced accuracy, F1-score, macro-F1,
        weighted-F1, precision, and recall.
    """
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0)
    }

def run_nested_cv_with_predictions_mutation(
    X,
    y,
    sample_ids,
    models,
    outer_cv,
    inner_cv,
    scoring="f1_macro",
    n_jobs=-1,
    coefficient_threshold=1e-8,
    top_n_tree=20
):
    """
    Run nested cross-validation and store fold-level predictions, metrics,
    inner-cross-validation scores, and feature rankings.

    For each model, an inner cross-validation loop is used for hyperparameter
    tuning with GridSearchCV. The best model from the inner loop is then
    evaluated on the held-out fold from the outer cross-validation loop.

    The inner cross-validation score is stored for each outer fold. This allows
    candidate model families to be selected according to mean inner-CV score,
    while outer-fold test metrics are used only for out-of-training performance
    estimation.

    Both test and train metrics are stored for each outer fold, allowing model
    performance and possible overfitting to be inspected. The function also
    stores sample-level predictions for the held-out test folds.

    In addition, fold-level feature rankings are extracted from the fitted model
    selected in each outer fold. For logistic regression models, rankings are
    based on coefficient magnitude. For tree-based models, rankings are based on
    feature importance values. Each pipeline may also include a "selector" step
    (e.g. SelectKBest), which can reduce the feature set seen by the classifier
    to a subset of X.columns — the gene names reaching the classifier are
    resolved from the fitted selector itself, rather than assumed to be all of
    X.columns, so feature rankings stay correctly aligned regardless of which
    selector (if any) was chosen in a given outer fold.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix containing mutation features.
    y : pandas.Series
        Binary target variable, where 0 corresponds to oligometastatic and 1 to
        polymetastatic.
    sample_ids : pandas.Series
        Sample identifiers aligned with X and y.
    models : dict
        Dictionary of model configurations returned by build_models_mutation.
    outer_cv : cross-validation splitter
        Outer cross-validation object used for out-of-training performance
        estimation.
    inner_cv : cross-validation splitter
        Inner cross-validation object used for hyperparameter tuning.
    scoring : str, default="f1_macro"
        Scoring metric used by GridSearchCV in the inner loop. The default is
        macro-F1, which gives equal weight to both classes and is useful when the
        goal is to optimize balanced performance across oligometastatic and
        polymetastatic patients.
    n_jobs : int, default=-1
        Number of jobs used by GridSearchCV.
    coefficient_threshold : float, default=1e-8
        Threshold used to define whether a logistic regression coefficient is
        considered non-zero.
    top_n_tree : int, default=20
        Number of top-ranked features considered selected for tree-based models.

    Returns
    -------
    tuple of pandas.DataFrame
        results_test_df:
            Fold-level test metrics for each model, including the best
            inner-cross-validation score from the corresponding outer fold.

        results_train_df:
            Fold-level train metrics for each model, including the best
            inner-cross-validation score from the corresponding outer fold.

        predictions_df:
            Sample-level predictions for each held-out test fold.

        feature_rankings_df:
            Fold-level feature rankings for each model and outer fold.
    """

    all_test_results = []
    all_train_results = []
    all_predictions = []
    all_feature_rankings = []

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    sample_ids = sample_ids.reset_index(drop=True)

    outer_splits = mutation_outer_splits(outer_cv, X, y)

    for model_name, model_cfg in models.items():
        print(f"\n=== Running model: {model_name} ===")

        for fold_idx, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            print(f"Outer fold {fold_idx}/{len(outer_splits)}")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            search = GridSearchCV(
                estimator=model_cfg["pipeline"],
                param_grid=model_cfg["param_grid"],
                scoring=scoring,
                cv=inner_cv,
                n_jobs=n_jobs,
                refit=True
            )

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*Features.*are constant.*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=".*invalid value encountered in divide.*",
                    category=RuntimeWarning,
                )
                search.fit(X_train, y_train)

            best_estimator = search.best_estimator_
            best_inner_score = search.best_score_
            best_params_json = json.dumps(search.best_params_, default=str)

            # ------------------------------------------------------------
            # Resolve which genes actually reached the classifier — the
            # selector step may have kept only a subset of X.columns.
            # ------------------------------------------------------------
            selector_step = best_estimator.named_steps.get("selector", "passthrough")

            if selector_step == "passthrough" or not hasattr(selector_step, "get_support"):
                genes_reaching_classifier = X.columns
            else:
                selected_mask = selector_step.get_support()
                genes_reaching_classifier = X.columns[selected_mask]

            # ------------------------------------------------------------
            # Store feature rankings from the fitted model selected by inner
            # cross-validation within this outer fold.
            # ------------------------------------------------------------
            if "logreg" in model_name:
                coefs = best_estimator.named_steps["clf"].coef_[0]

                feature_ranking_df = pd.DataFrame({
                    "model": model_name,
                    "outer_fold": fold_idx,
                    "gene": genes_reaching_classifier,
                    "coefficient": coefs,
                    "abs_coefficient": np.abs(coefs),
                    "selected_in_fold": np.abs(coefs) > coefficient_threshold,
                    "best_params": best_params_json,
                    "best_inner_score": best_inner_score
                })

                feature_ranking_df = (
                    feature_ranking_df
                    .sort_values("abs_coefficient", ascending=False)
                    .reset_index(drop=True)
                )

                feature_ranking_df["rank_in_fold"] = (
                    np.arange(1, len(feature_ranking_df) + 1)
                )

            elif model_name in ["random_forest", "xgboost"]:
                importances = best_estimator.named_steps["clf"].feature_importances_

                feature_ranking_df = pd.DataFrame({
                    "model": model_name,
                    "outer_fold": fold_idx,
                    "gene": genes_reaching_classifier,
                    "importance": importances,
                    "selected_in_fold": False,
                    "best_params": best_params_json,
                    "best_inner_score": best_inner_score
                })

                feature_ranking_df = (
                    feature_ranking_df
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True)
                )

                feature_ranking_df["rank_in_fold"] = (
                    np.arange(1, len(feature_ranking_df) + 1)
                )

                feature_ranking_df["selected_in_fold"] = (
                    feature_ranking_df["rank_in_fold"] <= top_n_tree
                )

            else:
                raise ValueError(f"Unknown model type: {model_name}")

            all_feature_rankings.append(feature_ranking_df)

            # ------------------------------------------------------------
            # Test performance on held-out outer fold
            # ------------------------------------------------------------
            y_test_proba = best_estimator.predict_proba(X_test)[:, 1]
            y_test_pred = best_estimator.predict(X_test)

            test_metrics = compute_metrics(
                y_test,
                y_test_pred,
                y_test_proba
            )

            test_row = {
                "model": model_name,
                "outer_fold": fold_idx,
                "dataset": "test",
                "best_params": best_params_json,
                "best_inner_score": best_inner_score,
                **test_metrics
            }

            all_test_results.append(test_row)

            # ------------------------------------------------------------
            # Train performance on outer training fold
            # ------------------------------------------------------------
            y_train_proba = best_estimator.predict_proba(X_train)[:, 1]
            y_train_pred = best_estimator.predict(X_train)

            train_metrics = compute_metrics(
                y_train,
                y_train_pred,
                y_train_proba
            )

            train_row = {
                "model": model_name,
                "outer_fold": fold_idx,
                "dataset": "train",
                "best_params": best_params_json,
                "best_inner_score": best_inner_score,
                **train_metrics
            }

            all_train_results.append(train_row)

            # ------------------------------------------------------------
            # Store sample-level predictions for the held-out test fold
            # ------------------------------------------------------------
            fold_predictions = pd.DataFrame({
                "model": model_name,
                "outer_fold": fold_idx,
                "sample_id": sample_ids.iloc[test_idx].values,
                "dataset": "test",
                "row_position": test_idx,
                "y_true": y_test.values,
                "y_pred": y_test_pred,
                "y_proba": y_test_proba,
                "best_inner_score": best_inner_score,
                "best_params": best_params_json
            })

            all_predictions.append(fold_predictions)

    results_test_df = pd.DataFrame(all_test_results)
    results_train_df = pd.DataFrame(all_train_results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    feature_rankings_df = pd.concat(all_feature_rankings, ignore_index=True)

    return (
        results_test_df,
        results_train_df,
        predictions_df,
        feature_rankings_df
    )


def parse_best_params_mutation(x):
    """
    Parse stored best-parameter values into a dictionary.

    Best parameters may already be stored as dictionaries or as strings when
    loaded from CSV files. This function converts both formats into Python
    dictionaries.

    Parameters
    ----------
    x : dict or str
        Best-parameter object or string representation of a dictionary.

    Returns
    -------
    dict
        Parsed dictionary of best hyperparameters.

    Raises
    ------
    ValueError
        If ``x`` is neither a dictionary nor a string.
    """
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        return ast.literal_eval(x.replace("null", "None"))
    raise ValueError(f"Unsupported best_params format: {type(x)}")


def annotate_genes_with_symbols(
    genes_df,
    gene_col="gene",
    species="human",
    verbose=False,
    fill_unmapped=False
):
    """
    Annotate gene identifiers with gene symbols using MyGene.info.

    This function supports mixed gene identifier formats. It first queries
    Ensembl gene IDs and then attempts to map any remaining unmapped identifiers
    using Entrez Gene IDs. All input rows are preserved, even when no gene symbol
    is found.

    Parameters
    ----------
    genes_df : pandas.DataFrame
        Dataframe containing a column with gene identifiers.
    gene_col : str, default="gene"
        Name of the column containing gene identifiers.
    species : str, default="human"
        Species used by MyGene.info for the query.
    verbose : bool, default=False
        Whether to print unmapped gene identifiers.
    fill_unmapped : bool, default=False
        If True, missing gene symbols are filled with "unmapped".

    Returns
    -------
    pandas.DataFrame
        Copy of the input dataframe with an additional ``gene_symbol`` column.
    """

    annotated_df = genes_df.copy()

    gene_key = f"{gene_col}_key"
    annotated_df[gene_key] = annotated_df[gene_col].astype(str)

    gene_ids = annotated_df[gene_key].dropna().unique().tolist()

    mg = mygene.MyGeneInfo()

    def query_symbols(ids, scope):
        if len(ids) == 0:
            return pd.DataFrame(columns=["gene", "gene_symbol"])

        result = mg.querymany(
            ids,
            scopes=scope,
            fields="symbol",
            species=species,
            as_dataframe=True,
            verbose=False
        ).reset_index()

        result = result.rename(columns={
            "query": "gene",
            "symbol": "gene_symbol"
        })

        if "gene_symbol" not in result.columns:
            result["gene_symbol"] = np.nan

        result = result[["gene", "gene_symbol"]].copy()
        result["gene"] = result["gene"].astype(str)

        result = result.dropna(subset=["gene_symbol"])
        result = result.drop_duplicates(subset="gene", keep="first")

        return result

    ensembl_ids = [
        gene_id for gene_id in gene_ids
        if gene_id.startswith("ENSG")
    ]

    mapping_ensembl = query_symbols(
        ensembl_ids,
        scope="ensembl.gene"
    )

    mapped_genes = set(mapping_ensembl["gene"].astype(str))

    ids_to_try_as_entrez = [
        gene_id for gene_id in gene_ids
        if gene_id not in mapped_genes
    ]

    mapping_entrez = query_symbols(
        ids_to_try_as_entrez,
        scope="entrezgene"
    )

    mapping = pd.concat(
        [mapping_ensembl, mapping_entrez],
        ignore_index=True
    )

    mapping = mapping.drop_duplicates(subset="gene", keep="first")

    annotated_df = annotated_df.merge(
        mapping.rename(columns={
            "gene": gene_key
        }),
        on=gene_key,
        how="left"
    )

    mapped_after_all = set(mapping["gene"].astype(str))
    unmapped_ids = sorted(set(gene_ids) - mapped_after_all)

    if verbose and len(unmapped_ids) > 0:
        print("Warning: some gene identifiers could not be mapped to symbols:")
        print(unmapped_ids[:30])

    if fill_unmapped:
        annotated_df["gene_symbol"] = annotated_df["gene_symbol"].fillna("unmapped")

    annotated_df = annotated_df.drop(columns=[gene_key])

    return annotated_df


def annotate_gene_pairs_with_symbols(
    pairs_df,
    feature1_col="Feature1",
    feature2_col="Feature2",
    scopes="ensembl.gene,entrezgene",
    species="human",
    manual_symbol_map=None
):
    """
    Annotate pairs of gene identifiers with gene symbols using MyGene.info.

    This function supports mixed gene identifier formats, including Ensembl gene
    IDs and Entrez Gene IDs. It adds gene-symbol annotations for both genes in
    each pair, while preserving the original identifier columns.

    Parameters
    ----------
    pairs_df : pandas.DataFrame
        Dataframe containing gene pairs to annotate.
    feature1_col : str, default="Feature1"
        Name of the column containing the first gene identifier.
    feature2_col : str, default="Feature2"
        Name of the column containing the second gene identifier.
    scopes : str, default="ensembl.gene,entrezgene"
        Identifier namespaces used by MyGene.info for the query.
    species : str, default="human"
        Species used by MyGene.info for the query.
    manual_symbol_map : dict, optional
        Optional manual mapping from gene identifiers to symbols. This is useful
        for identifiers that are not automatically mapped by MyGene.info.

    Returns
    -------
    pandas.DataFrame
        Copy of the input dataframe with two additional columns:
        ``Feature1_symbol`` and ``Feature2_symbol``.
    """

    annotated_df = pairs_df.copy()

    # Temporary string columns for robust mapping and merging
    feature1_key = f"{feature1_col}_key"
    feature2_key = f"{feature2_col}_key"

    annotated_df[feature1_key] = annotated_df[feature1_col].astype(str)
    annotated_df[feature2_key] = annotated_df[feature2_col].astype(str)

    gene_ids = pd.unique(
        annotated_df[[feature1_key, feature2_key]].values.ravel()
    ).tolist()

    mg = mygene.MyGeneInfo()

    mapping = mg.querymany(
        gene_ids,
        scopes=scopes,
        fields="symbol",
        species=species,
        as_dataframe=True
    ).reset_index()

    mapping = mapping.rename(columns={
        "query": "gene",
        "symbol": "gene_symbol"
    })

    mapping = mapping[["gene", "gene_symbol"]].drop_duplicates(subset="gene")

    # Optional manual correction/fallback
    if manual_symbol_map is not None:
        manual_mapping = pd.DataFrame({
            "gene": [str(gene) for gene in manual_symbol_map.keys()],
            "gene_symbol": list(manual_symbol_map.values())
        })

        mapping = pd.concat([mapping, manual_mapping], ignore_index=True)
        mapping = mapping.drop_duplicates(subset="gene", keep="last")

    annotated_df = annotated_df.merge(
        mapping.rename(columns={
            "gene": feature1_key,
            "gene_symbol": f"{feature1_col}_symbol"
        }),
        on=feature1_key,
        how="left"
    )

    annotated_df = annotated_df.merge(
        mapping.rename(columns={
            "gene": feature2_key,
            "gene_symbol": f"{feature2_col}_symbol"
        }),
        on=feature2_key,
        how="left"
    )

    annotated_df = annotated_df.drop(columns=[feature1_key, feature2_key])

    return annotated_df

def plot_train_test_boxplots_mutation(results_train_df, results_test_df, metric="roc_auc", top_n=None):
    """
    Boxplot comparing train vs test values of a metric, per model, across outer folds.
    """
    models = results_test_df["model"].unique()

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = range(len(models))
    width = 0.35

    train_data = [results_train_df.loc[results_train_df["model"] == m, metric].values for m in models]
    test_data = [results_test_df.loc[results_test_df["model"] == m, metric].values for m in models]

    bp_train = ax.boxplot(train_data, positions=[p - width/2 for p in positions], widths=width,
                        patch_artist=True, boxprops=dict(facecolor="#7d99ad"), medianprops=dict(color="#222222", linewidth=1.5))
    bp_test = ax.boxplot(test_data, positions=[p + width/2 for p in positions], widths=width,
                        patch_artist=True, boxprops=dict(facecolor="#246a91"), medianprops=dict(color="#222222", linewidth=1.5))

    ax.set_xticks(list(positions))
    labels = {"logreg_l1": "L1 logistic", "logreg_l2": "L2 logistic", "random_forest": "Random Forest", "xgboost": "XGBoost", "flaml_automl": "FLAML"}
    ax.set_xticklabels([labels.get(m, m) for m in models], rotation=30, ha="right")
    metric_label = "ROC-AUC" if metric in {"roc_auc", "auc_score"} else metric
    ax.set_ylabel(metric_label)
    ax.set_title(f"Apparent outer-training and held-out outer-test {metric_label}")
    ax.legend([bp_train["boxes"][0], bp_test["boxes"][0]], ["Outer-train AUC (apparent)" if metric_label == "ROC-AUC" else "Outer-train (apparent)", "Outer-test AUC (held-out)" if metric_label == "ROC-AUC" else "Outer-test (held-out)"])
    plt.tight_layout()
    plt.show()


# ==============================================================================
# Transcriptomics Representation-Learning Utilities (from utils_ML_tpm.py)
# ==============================================================================

def split_transcriptomics_features_target_metadata(
    transcriptomics_df: pd.DataFrame,
    target_col: str = "Polymetastatic",
    feature_prefix: str = "transcriptomics__",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str], list[str]]:
    """
    Split a preprocessed transcriptomics dataframe into X, y and metadata.

    The preprocessing notebook creates a dataframe containing:

    - metadata columns, such as sample identifiers and transcriptomics source;
    - the target column, usually `Polymetastatic`;
    - transcriptomics expression features, usually prefixed with `transcriptomics__`.

    This function separates these components so that only transcriptomics
    expression features are used as model inputs.

    Parameters
    ----------
    transcriptomics_df : pandas.DataFrame
        Preprocessed transcriptomics dataframe containing metadata, target,
        and transcriptomics expression features.

    target_col : str, default="Polymetastatic"
        Name of the binary target column.

        Expected coding:
        - 0 = oligometastatic;
        - 1 = polymetastatic.

    feature_prefix : str, default="transcriptomics__"
        Prefix identifying transcriptomics feature columns.

    Returns
    -------
    X : pandas.DataFrame
        Transcriptomics feature matrix containing only columns whose names start
        with `feature_prefix`.

    y : pandas.Series
        Binary target series.

    metadata_df : pandas.DataFrame
        Non-feature columns retained for auditing, interpretation and linking
        predictions back to samples.

    feature_cols : list[str]
        Names of transcriptomics feature columns included in X.

    metadata_cols : list[str]
        Names of metadata/non-feature columns excluded from X.

    Raises
    ------
    ValueError
        If the target column is missing, no transcriptomics feature columns are
        found, X/y dimensions do not match, y is not binary, or X contains
        missing values.
    """

    if target_col not in transcriptomics_df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    feature_cols = [
        col for col in transcriptomics_df.columns
        if col.startswith(feature_prefix)
    ]

    if not feature_cols:
        raise ValueError(
            f"No feature columns found with prefix '{feature_prefix}'."
        )

    metadata_cols = [
        col for col in transcriptomics_df.columns
        if col not in feature_cols
    ]

    X = transcriptomics_df[feature_cols].copy()
    y = transcriptomics_df[target_col].astype(int).copy()
    metadata_df = transcriptomics_df[metadata_cols].copy()

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have different numbers of samples.")

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only binary values 0 and 1.")

    if X.isna().sum().sum() > 0:
        raise ValueError("X contains missing values.")

    if target_col in X.columns:
        raise ValueError("Target column was incorrectly included in X.")

    return X, y, metadata_df, feature_cols, metadata_cols

def get_positive_class_probability(estimator, X_test: pd.DataFrame) -> np.ndarray:
    """
    Return probability-like scores for the positive class.

    The positive class is assumed to be:

    - 1 = polymetastatic.

    The function tries, in this order:

    1. `predict_proba`, returning probability of class 1;
    2. `decision_function`, transformed to the interval [0, 1] using the logistic function;
    3. `predict`, cast to float.

    Parameters
    ----------
    estimator : sklearn estimator
        Fitted classifier or pipeline.

    X_test : pandas.DataFrame
        Test feature matrix.

    Returns
    -------
    numpy.ndarray
        Probability-like scores for the positive class.
    """

    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_test)[:, 1]

    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X_test)
        return 1.0 / (1.0 + np.exp(-scores))

    return estimator.predict(X_test).astype(float)

def calculate_binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
) -> dict:
    """
    Calculate binary classification metrics.

    Target coding:
    - 0 = oligometastatic;
    - 1 = polymetastatic.

    Notes
    -----
    `f1` is the F1-score for the positive class, where positive class = 1
    polymetastatic.

    `f1_macro` is the unweighted mean of F1 for both classes. This is the metric
    used for inner-CV model selection when scoring="f1_macro".
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "precision_poly": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "sensitivity_poly": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "recall_oligo": recall_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0,
        ),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = np.nan

    try:
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        metrics["specificity_oligo"] = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        )

    except ValueError:
        metrics["specificity_oligo"] = np.nan

    return metrics


def summarise_representation_results(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Summarise nested-CV results across transcriptomics representations.

    Selection principle
    -------------------
    The best representation/model should be selected using:

    - mean_inner_cv_score

    because this is based on the inner cross-validation model-selection process.

    Reporting principle
    --------------------
    Only unseen-data (test) metrics are reported for performance estimation,
    plus a single training metric (ROC-AUC) used purely as an overfitting
    diagnostic via the train-test gap. Additional metrics (binary/macro F1,
    precision, sensitivity, specificity, and their standard deviations) are
    not computed here, following standard reporting practice.
    """

    required_cols = [
        "representation",
        "best_inner_score",
        "test_roc_auc",
        "test_balanced_accuracy",
        "train_roc_auc",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in results_df.columns
    ]

    if missing_cols:
        raise ValueError(
            "The following required columns are missing from results_df: "
            f"{missing_cols}"
        )

    summary_df = (
        results_df
        .groupby("representation")
        .agg(
            mean_inner_cv_score=("best_inner_score", "mean"),
            std_inner_cv_score=("best_inner_score", "std"),
            mean_test_roc_auc=("test_roc_auc", "mean"),
            mean_test_balanced_accuracy=("test_balanced_accuracy", "mean"),
            mean_train_roc_auc=("train_roc_auc", "mean"),
        )
        .sort_values(by="mean_inner_cv_score", ascending=False)
    )

    summary_df["train_test_gap"] = (
        summary_df["mean_train_roc_auc"] - summary_df["mean_test_roc_auc"]
    )

    if output_path is not None:
        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        summary_df.to_csv(output_path)

    return summary_df

def fit_full_data_pca_exploratory(
    X: pd.DataFrame,
    metadata_df: pd.DataFrame,
    n_components: int = 20,
    random_state: int = 42,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fit PCA on the full dataset for exploratory visualisation only.

    This function is useful for visualising samples in PCA space and inspecting
    broad transcriptomics structure.

    Important
    ---------
    This PCA is fitted using all samples. Therefore, its output should not be
    used to estimate supervised predictive performance.

    For supervised modelling, PCA should be placed inside an sklearn Pipeline
    and fitted inside cross-validation folds.

    Parameters
    ----------
    X : pandas.DataFrame
        Transcriptomics feature matrix.

    metadata_df : pandas.DataFrame
        Sample metadata to append to the PCA scores.

    n_components : int, default=20
        Number of PCA components to compute.

    random_state : int, default=42
        Random seed used by PCA.

    output_path : str or None, default=None
        Optional path to save the PCA scores. If provided, the explained
        variance table is also saved with suffix `_explained_variance.csv`.

    Returns
    -------
    pca_scores_df : pandas.DataFrame
        DataFrame containing metadata columns plus PCA component scores.
    """

    pca_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=random_state,
        )),
    ])

    pca_scores = pca_pipeline.fit_transform(X)

    pca_feature_cols = [
        f"PC{i}"
        for i in range(1, n_components + 1)
    ]

    pca_scores_df = pd.DataFrame(
        pca_scores,
        columns=pca_feature_cols,
    )

    pca_scores_df = pd.concat(
        [
            metadata_df.reset_index(drop=True),
            pca_scores_df,
        ],
        axis=1,
    )

    explained_variance_df = pd.DataFrame({
        "component": pca_feature_cols,
        "explained_variance_ratio": (
            pca_pipeline.named_steps["pca"].explained_variance_ratio_
        ),
    })

    if output_path is not None:
        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        pca_scores_df.to_csv(output_path, index=False)

        explained_variance_path = output_path.replace(
            ".csv",
            "_explained_variance.csv",
        )

        explained_variance_df.to_csv(
            explained_variance_path,
            index=False,
        )

    return pca_scores_df


def set_global_seed(random_state: int = 42) -> None:
    """
    Set random seeds for numpy, Python random, and TensorFlow/Keras.
    """

    np.random.seed(random_state)
    random.seed(random_state)

    if tf is not None:
        tf.keras.utils.set_random_seed(random_state)


def scale_external_and_institutional_expression(
    external_selected_df: pd.DataFrame,
    institutional_selected_df: pd.DataFrame,
    validation_fraction: float = 0.2,
    random_state: int = 42,
    scaler_type: str = "standard",
) -> dict:
    """
    Split external expression data into train/validation sets, fit scaler on
    external train data, and apply it to external full and institutional data.
    """

    external_train_df, external_val_df = train_test_split(
        external_selected_df,
        test_size=validation_fraction,
        random_state=random_state,
    )

    if scaler_type != "standard":
        raise ValueError(
            "For now, use scaler_type='standard' for consistency with the "
            "existing transcriptomics modelling pipeline."
        )

    scaler = StandardScaler()

    X_external_train = scaler.fit_transform(external_train_df)
    X_external_val = scaler.transform(external_val_df)
    X_external_full = scaler.transform(external_selected_df)
    X_institutional = scaler.transform(institutional_selected_df)

    return {
        "external_train_df": external_train_df,
        "external_val_df": external_val_df,
        "X_external_train": X_external_train,
        "X_external_val": X_external_val,
        "X_external_full": X_external_full,
        "X_institutional": X_institutional,
        "scaler": scaler,
    }

class IntegrativeVAESampling(layers.Layer):
    """
    Reparameterisation trick for VAE latent sampling.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs

        batch_size = tf.shape(z_mean)[0]
        latent_dim = tf.shape(z_mean)[1]

        epsilon = tf.random.normal(shape=(batch_size, latent_dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class RNAOnlyKerasVAE(keras.Model):
    """
    RNA-only variational autoencoder for transcriptomic representation learning.

    This is inspired by IntegrativeVAEs/CancerAI-CL, but adapted to a single
    transcriptomic modality so that it can be applied to the institutional TPM
    cohort.
    """

    def __init__(self, encoder, decoder, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def build(self, input_shape):
        """Register the wrapper after its Functional components created their variables.

        Custom train/test steps call the components directly, bypassing Model.__call__.
        No stochastic forward pass or additional variable initialisation is needed.
        """
        if not self.encoder.built or not self.decoder.built:
            raise ValueError("Build the Functional encoder and decoder before the VAE wrapper.")
        if input_shape[-1] != self.encoder.input_shape[-1]:
            raise ValueError("VAE build shape does not match the encoder input.")
        super().build(input_shape)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def _compute_losses(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(data - reconstruction),
                axis=1,
            )
        )

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1,
            )
        )

        total_loss = reconstruction_loss + self.beta * kl_loss

        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self._compute_losses(data)

        gradients = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_weights)
        )

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        total_loss, reconstruction_loss, kl_loss = self._compute_losses(data)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def build_integrativevae_rna_only(
    input_dim: int,
    latent_dim: int = 100,
    hidden_dims: Sequence[int] = (1024, 256),
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
    beta: float = 1.0,
):
    """
    Build an RNA-only Keras VAE inspired by IntegrativeVAEs/CancerAI-CL.
    """

    if tf is None or keras is None or layers is None:
        raise ImportError(
            "TensorFlow/Keras is required to build the IntegrativeVAE-inspired model."
        )

    inputs = keras.Input(shape=(input_dim,), name="rna_expression_input")

    x = inputs

    for i, hidden_dim in enumerate(hidden_dims):
        x = layers.Dense(
            hidden_dim,
            activation="relu",
            name=f"rna_encoder_dense_{hidden_dim}_{i + 1}",
        )(x)
        x = layers.BatchNormalization(
            name=f"rna_encoder_batchnorm_{hidden_dim}_{i + 1}"
        )(x)
        x = layers.Dropout(
            dropout_rate,
            name=f"rna_encoder_dropout_{hidden_dim}_{i + 1}"
        )(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    z = IntegrativeVAESampling(name="z")([z_mean, z_log_var])

    encoder = keras.Model(
        inputs=inputs,
        outputs=[z_mean, z_log_var, z],
        name="integrativevae_rna_only_encoder",
    )

    z_mean_encoder = keras.Model(
        inputs=inputs,
        outputs=z_mean,
        name="integrativevae_rna_only_z_mean_encoder",
    )

    latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")

    x = latent_inputs

    for i, hidden_dim in enumerate(reversed(hidden_dims)):
        x = layers.Dense(
            hidden_dim,
            activation="relu",
            name=f"rna_decoder_dense_{hidden_dim}_{i + 1}",
        )(x)
        x = layers.BatchNormalization(
            name=f"rna_decoder_batchnorm_{hidden_dim}_{i + 1}"
        )(x)

    outputs = layers.Dense(
        input_dim,
        activation="linear",
        name="rna_reconstructed_expression",
    )(x)

    decoder = keras.Model(
        inputs=latent_inputs,
        outputs=outputs,
        name="integrativevae_rna_only_decoder",
    )

    vae = RNAOnlyKerasVAE(
        encoder=encoder,
        decoder=decoder,
        beta=beta,
        name="integrativevae_rna_only",
    )

    vae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )

    return vae, encoder, z_mean_encoder, decoder

def train_integrativevae_rna_only(
    X_train,
    X_val,
    input_dim: int,
    latent_dim: int = 100,
    hidden_dims: Sequence[int] = (1024, 256),
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
    beta: float = 1.0,
    epochs: int = 300,
    batch_size: int = 32,
    patience: int = 25,
    operational_callbacks=None,
):
    """
    Train the RNA-only IntegrativeVAE-inspired model.
    """

    vae, encoder, z_mean_encoder, decoder = build_integrativevae_rna_only(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        beta=beta,
    )

    # All variables already exist in the Functional encoder/decoder. Register the
    # outer wrapper without consuming a latent sample or changing any weights.
    vae.build((None, input_dim))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=max(5, patience // 3),
            factor=0.5,
            min_lr=1e-6,
        ),
    ]

    history = vae.fit(
        X_train,
        validation_data=(X_val,),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks + list(operational_callbacks or []),
        verbose=1,
    )

    history_df = pd.DataFrame(history.history)

    return vae, encoder, z_mean_encoder, decoder, history_df

def project_samples_with_z_mean_encoder(
    z_mean_encoder,
    X,
    sample_ids,
    prefix: str,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Project samples into VAE latent mean space.
    """

    latent_scores = z_mean_encoder.predict(
        X,
        batch_size=batch_size,
        verbose=1,
    )

    latent_scores_df = pd.DataFrame(
        latent_scores,
        index=sample_ids,
        columns=[
            f"{prefix}_latent_{i + 1:03d}"
            for i in range(latent_scores.shape[1])
        ],
    ).reset_index()

    latent_scores_df = latent_scores_df.rename(
        columns={latent_scores_df.columns[0]: "sample_id_normalised"}
    )

    return latent_scores_df

def save_integrativevae_outputs(
    output_dir,
    vae,
    encoder,
    z_mean_encoder,
    decoder,
    scaler,
    selected_genes_df: pd.DataFrame,
    overlap_summary_df: pd.DataFrame,
    history_df: pd.DataFrame,
    latent_scores_df: pd.DataFrame,
    metadata: dict,
    representation_name: str = "metabric_integrativevae_rna_only_latent100",
):
    """
    Save fitted encoder components, scaler, selected genes, summaries, training
    history, latent scores, and metadata.

    The full subclassed VAE object is not saved because custom Keras subclassed
    models require explicit serialization methods. For downstream use, the
    deterministic z_mean_encoder is the key model component needed to project
    new samples into the latent space.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save only directly reusable functional models.
    # z_mean_encoder is the most important object for downstream projection.
    z_mean_encoder.save(
        output_dir / f"{representation_name}_z_mean_encoder.keras"
    )

    decoder.save(
        output_dir / f"{representation_name}_decoder.keras"
    )

    # Optional: save the stochastic encoder weights/config if needed.
    # This may include the custom Sampling layer, so z_mean_encoder is preferred
    # for simple reloading and downstream use.
    try:
        encoder.save(
            output_dir / f"{representation_name}_encoder.keras"
        )
    except Exception as e:
        print("Warning: full stochastic encoder could not be saved.")
        print(e)

    joblib.dump(
        scaler,
        output_dir / f"{representation_name}_scaler.joblib"
    )

    selected_genes_df.to_csv(
        output_dir / f"{representation_name}_selected_genes.csv",
        index=False,
    )

    overlap_summary_df.to_csv(
        output_dir / f"{representation_name}_gene_overlap_summary.csv",
        index=False,
    )

    history_df.to_csv(
        output_dir / f"{representation_name}_training_history.csv",
        index=False,
    )

    latent_scores_path = (
        output_dir / f"transcriptomics_{representation_name}_scores.csv"
    )

    latent_scores_df.to_csv(latent_scores_path, index=False)

    with open(output_dir / f"{representation_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Saved z_mean encoder:")
    print(output_dir / f"{representation_name}_z_mean_encoder.keras")

    print("\nSaved latent scores:")
    print(latent_scores_path)

    return latent_scores_path

_TYBALT_ENCODER_CACHE = {}

class TybaltEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Apply a frozen pretrained Tybalt encoder with fold-specific MinMax scaling.

    MinMaxScaler is fitted only on the training samples supplied to fit().
    The pretrained Tybalt encoder weights are fixed and never updated.
    """

    def __init__(self, model_path, batch_size=32, clip=True):
        self.model_path = model_path
        self.batch_size = batch_size
        self.clip = clip

    def fit(self, X, y=None):
        self.scaler_ = MinMaxScaler(feature_range=(0, 1))
        self.scaler_.fit(X)

        model_path = str(self.model_path)

        if model_path not in _TYBALT_ENCODER_CACHE:
            _TYBALT_ENCODER_CACHE[model_path] = keras.models.load_model(
                model_path,
                compile=False,
            )

        self.encoder_ = _TYBALT_ENCODER_CACHE[model_path]

        if self.encoder_.input_shape[-1] != X.shape[1]:
            raise ValueError(
                f"Tybalt encoder expects {self.encoder_.input_shape[-1]} "
                f"features, but received {X.shape[1]}."
            )

        return self

    def transform(self, X):
        X_scaled = self.scaler_.transform(X)

        if self.clip:
            X_scaled = np.clip(X_scaled, 0.0, 1.0)

        X_scaled = np.asarray(X_scaled, dtype=np.float32)

        latent = self.encoder_.predict(
            X_scaled,
            batch_size=self.batch_size,
            verbose=0,
        )

        return np.asarray(latent, dtype=np.float32)

def build_representation_classifiers(
    random_state=42,
    n_features=None,
    use_feature_selection=False
):
    """
    Build candidate classifiers for transcriptomics representations.

    L2 logistic regression is excluded because transcriptomics is a
    high-dimensional, small-sample setting. L1 regularisation is retained
    because it can produce sparse solutions and is more appropriate when the
    number of candidate predictors is large relative to the number of patients.

    All pipelines begin with StandardScaler so that:
    - regularised logistic regression is fitted on comparable feature scales;
    - SMOTE, when selected, computes nearest neighbours in a standardised
    feature space.

    SMOTE is treated as a hyperparameter and is applied inside the
    cross-validation pipeline, preventing information leakage from held-out
    folds.

    Feature selection can optionally be included inside the pipeline. Candidate
    k values are restricted to values smaller than the number of input features,
    allowing the same function to be used for both high-dimensional gene-level
    expression and lower-dimensional transcriptomic representations.

    Parameters
    ----------
    random_state : int, default=42
        Random seed used for reproducibility.

    n_features : int or None, default=None
        Number of input features in the current representation. Required when
        use_feature_selection=True.

    use_feature_selection : bool, default=False
        Whether SelectKBest should be included as a searchable pipeline step.

    Returns
    -------
    dict
        Dictionary mapping classifier names to pipelines and parameter grids.
    """

    sampler_options = [
        "passthrough",
        SMOTE(random_state=random_state),
    ]

    # ------------------------------------------------------------
    # Optional feature-selection strategies
    # ------------------------------------------------------------
    if use_feature_selection:
        if n_features is None:
            raise ValueError(
                "n_features must be provided when "
                "use_feature_selection=True."
            )

        selector_options = ["passthrough"]

        if n_features > 200:
            selector_options.append(
                SelectKBest(
                    score_func=f_classif,
                    k=200
                )
            )

            selector_options.append(
                SelectKBest(
                    score_func=mutual_info_classif,
                    k=200
                )
            )

        if n_features > 500:
            selector_options.append(
                SelectKBest(
                    score_func=f_classif,
                    k=500
                )
            )

    else:
        selector_options = ["passthrough"]

    classifiers = {
        "logreg_l1": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("selector", "passthrough"),
                ("sampler", "passthrough"),
                ("classifier", LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=random_state,
                ))
            ]),
            "param_grid": {
                "classifier__C": [0.001, 0.01, 0.1, 1],
                "classifier__class_weight": [None, "balanced"],
                "selector": selector_options,
                "sampler": sampler_options,
            }
        },

        "random_forest": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("selector", "passthrough"),
                ("sampler", "passthrough"),
                ("classifier", RandomForestClassifier(
                    random_state=random_state,
                    n_jobs=1,
                ))
            ]),
            "param_grid": {
                "classifier__n_estimators": [100, 300],
                "classifier__max_depth": [2, 3],
                "classifier__min_samples_leaf": [5, 10],
                "classifier__max_features": [0.1, "sqrt"],
                "classifier__class_weight": [None, "balanced"],
                "selector": selector_options,
                "sampler": sampler_options,
            }
        }
    }

    if XGBClassifier is not None:
        classifiers["xgboost"] = {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("selector", "passthrough"),
                ("sampler", "passthrough"),
                ("classifier", XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=1,
                    tree_method="hist",
                ))
            ]),
            "param_grid": {
                "classifier__n_estimators": [30, 50],
                "classifier__max_depth": [1, 2],
                "classifier__learning_rate": [0.01, 0.05],
                "classifier__subsample": [0.6, 0.8],
                "classifier__colsample_bytree": [0.05, 0.1],
                "classifier__min_child_weight": [3, 5],
                "classifier__reg_lambda": [1, 5],
                "classifier__reg_alpha": [0, 1],
                "selector": selector_options,
                "sampler": sampler_options,
            }
        }

    return classifiers

def wrap_with_tybalt(classifiers, model_path):
    """
    Prepend fold-specific Tybalt projection to representation classifiers.

    Pipeline:
        fold-specific MinMaxScaler
        -> frozen pretrained Tybalt encoder
        -> StandardScaler
        -> SMOTE/passthrough
        -> classifier
    """
    wrapped = {}

    for name, info in classifiers.items():
        base_pipeline = clone(info["pipeline"])

        pipeline = Pipeline([
            (
                "tybalt",
                TybaltEncoderTransformer(
                    model_path=str(model_path),
                ),
            ),
            *base_pipeline.steps,
        ])

        wrapped[name] = {
            "pipeline": pipeline,
            "param_grid": dict(info["param_grid"]),
        }

    return wrapped

def wrap_with_pca(
    classifiers,
    n_components_grid=None,
    random_state=42,
):
    """
    Wrap candidate classifiers with StandardScaler + PCA.

    The resulting pipeline is:

        StandardScaler
        -> PCA
        -> sampler (SMOTE or passthrough)
        -> classifier

    PCA and SMOTE are both fitted exclusively within the training portion
    of each cross-validation fold, preventing information leakage.

    SMOTE is applied after PCA so that synthetic samples are generated in
    the lower-dimensional representation space rather than directly in the
    original high-dimensional transcriptomics feature space.

    Parameters
    ----------
    classifiers : dict
        Classifier specifications returned by
        ``build_representation_classifiers``.

    n_components_grid : list or None
        Candidate PCA dimensionalities.

    random_state : int, default=42
        Random seed.

    Returns
    -------
    dict
        PCA-wrapped classifier pipelines and parameter grids.
    """

    if n_components_grid is None:
        n_components_grid = [5, 10, 20, 50]

    pca_classifiers = {}

    for name, info in classifiers.items():

        classifier_estimator = clone(
            info["pipeline"].named_steps["classifier"]
        )

        pca_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(
                random_state=random_state
            )),
            ("sampler", "passthrough"),
            ("classifier", classifier_estimator),
        ])

        base_param_grid = {
            key: value
            for key, value in info["param_grid"].items()
            if key != "selector"
        }

        pca_param_grid = {
            "pca__n_components": n_components_grid,
            **base_param_grid,
        }

        pca_classifiers[name] = {
            "pipeline": pca_pipeline,
            "param_grid": pca_param_grid,
        }

    return pca_classifiers


def add_plot_label_columns(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a robust plotting label column to nested-CV results.
    """

    results_df = results_df.copy()

    if "representation" in results_df.columns:
        representation_col = "representation"
    elif "representation_name" in results_df.columns:
        representation_col = "representation_name"
    else:
        representation_col = None

    if representation_col is not None:
        results_df["plot_label"] = results_df[representation_col].astype(str)
    else:
        results_df["plot_label"] = "model"

    return results_df

def plot_metric_boxplots(
    all_results_df: pd.DataFrame,
    metric: str = "test_roc_auc",
    top_n: Optional[int] = 10,
    chance_level: Optional[float] = None,
):
    """
    Plot metric values across outer folds for each representation/model.
    """

    if metric not in all_results_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results dataframe.")

    plot_df = add_plot_label_columns(all_results_df)

    mean_scores = (
        plot_df
        .groupby("plot_label")[metric]
        .mean()
        .sort_values(ascending=False)
    )

    if top_n is not None:
        selected_labels = mean_scores.head(top_n).index
        plot_df = plot_df[plot_df["plot_label"].isin(selected_labels)]
    else:
        selected_labels = mean_scores.index

    data = [
        plot_df.loc[plot_df["plot_label"] == label, metric].dropna().values
        for label in selected_labels
    ]

    plt.figure(figsize=(12, max(5, 2 + 0.11 * max((len(str(label)) for label in selected_labels), default=0))))
    plt.boxplot(data, labels=selected_labels, patch_artist=True,
                boxprops=dict(facecolor="#246a91"), medianprops=dict(color="#222222", linewidth=1.5))
    plt.xticks(rotation=75, ha="right")
    plt.ylabel(metric)

    if chance_level is not None:
        plt.axhline(
            chance_level,
            linestyle="--",
            linewidth=1,
            label="reference level",
        )
        plt.legend()

    plt.title(f"{metric} across outer folds")
    plt.tight_layout()
    plt.show()

def plot_train_test_gap_boxplots(
    all_results_df: pd.DataFrame,
    train_col: str,
    test_col: str,
    top_n: Optional[int] = 10,
):
    """
    Plot train-test generalisation gaps across outer folds.
    """

    if train_col not in all_results_df.columns:
        raise ValueError(f"Column '{train_col}' not found.")

    if test_col not in all_results_df.columns:
        raise ValueError(f"Column '{test_col}' not found.")

    plot_df = add_plot_label_columns(all_results_df)
    gap_col = f"{train_col}_minus_{test_col}"
    plot_df[gap_col] = plot_df[train_col] - plot_df[test_col]

    mean_gaps = (
        plot_df
        .groupby("plot_label")[gap_col]
        .mean()
        .sort_values(ascending=False)
    )

    if top_n is not None:
        selected_labels = mean_gaps.head(top_n).index
        plot_df = plot_df[plot_df["plot_label"].isin(selected_labels)]
    else:
        selected_labels = mean_gaps.index

    data = [
        plot_df.loc[plot_df["plot_label"] == label, gap_col].dropna().values
        for label in selected_labels
    ]

    plt.figure(figsize=(12, max(5, 2 + 0.11 * max((len(str(label)) for label in selected_labels), default=0))))
    plt.boxplot(data, labels=selected_labels, medianprops=dict(color="#222222", linewidth=1.5))
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Train - test")
    plt.title(f"Train-test gap: {train_col} - {test_col}")
    plt.tight_layout()
    plt.show()

def plot_exploratory_pca_variance_and_scatter(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int = 20,
    random_state: int = 42,
    output_path: Optional[str] = None,
):
    """
    Fit full-data PCA for exploratory visualisation only and plot:

    - explained variance ratio;
    - PC1 versus PC2 scatter.

    This should not be used for predictive performance estimation.
    """

    pca_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(
            n_components=n_components,
            random_state=random_state,
        )),
    ])

    pca_scores = pca_pipeline.fit_transform(X)

    pca = pca_pipeline.named_steps["pca"]

    explained_variance_df = pd.DataFrame({
        "principal_component": [
            f"PC{i + 1}"
            for i in range(len(pca.explained_variance_ratio_))
        ],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(
            pca.explained_variance_ratio_
        ),
    })

    if output_path is not None:
        explained_variance_df.to_csv(output_path, index=False)

    display(explained_variance_df)

    plt.figure(figsize=(9, 5))
    plt.bar(
        explained_variance_df["principal_component"],
        explained_variance_df["explained_variance_ratio"],
    )
    plt.xticks(rotation=45)
    plt.ylabel("Explained variance ratio")
    plt.title("Exploratory full-data PCA explained variance")
    plt.tight_layout()
    plt.show()

    pca_plot_df = pd.DataFrame({
        "PC1": pca_scores[:, 0],
        "PC2": pca_scores[:, 1],
        "Polymetastatic": y.values,
    })

    plt.figure(figsize=(7, 6))

    for class_value, label in [
        (0, "Oligometastatic"),
        (1, "Polymetastatic"),
    ]:
        subset_df = pca_plot_df[
            pca_plot_df["Polymetastatic"] == class_value
        ]

        plt.scatter(
            subset_df["PC1"],
            subset_df["PC2"],
            label=label,
            alpha=0.8,
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Exploratory full-data PCA: PC1 vs PC2")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return explained_variance_df, pca_plot_df

def plot_vae_training_history(history_df, title_prefix="VAE"):
    """
    Plot loss, and optionally reconstruction/KL loss, from a Keras VAE
    training history dataframe. Prints final and best validation loss.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["loss"], label="train_loss")
    plt.plot(history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Training History")
    plt.legend()
    plt.show()

    if "reconstruction_loss" in history_df.columns:
        plt.figure(figsize=(7, 5))
        plt.plot(history_df["reconstruction_loss"], label="train_reconstruction_loss")
        if "val_reconstruction_loss" in history_df.columns:
            plt.plot(history_df["val_reconstruction_loss"], label="val_reconstruction_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction Loss")
        plt.title(f"{title_prefix} Reconstruction Loss")
        plt.legend()
        plt.show()

    if "kl_loss" in history_df.columns:
        plt.figure(figsize=(7, 5))
        plt.plot(history_df["kl_loss"], label="train_kl_loss")
        if "val_kl_loss" in history_df.columns:
            plt.plot(history_df["val_kl_loss"], label="val_kl_loss")
        plt.xlabel("Epoch")
        plt.ylabel("KL Loss")
        plt.title(f"{title_prefix} KL Loss")
        plt.legend()
        plt.show()

    print("Final train loss:", history_df["loss"].iloc[-1])
    print("Final validation loss:", history_df["val_loss"].iloc[-1])
    print("Best validation loss:", history_df["val_loss"].min())

def integrativevae_sampling(args):
    """
    Reparameterisation trick used by the original IntegrativeVAEs code.
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class IntegrativeVAERNALossLayer(layers.Layer):
    """
    Loss layer for the RNA-only adaptation of the original IntegrativeVAEs
    HVAE numerical branch.

    Adds reconstruction MSE + beta * KL regularisation to the model.
    """

    def __init__(self, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def call(self, inputs):
        original_input, reconstructed_input, z_mean, z_log_sigma = inputs

        reconstruction_loss = tf.reduce_mean(
            tf.square(original_input - reconstructed_input),
            axis=-1,
        )

        kl_loss = 1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        total_loss = tf.reduce_mean(
            reconstruction_loss + self.beta * kl_loss
        )

        self.add_loss(total_loss)

        return reconstructed_input

    def get_config(self):
        config = super().get_config()
        config.update({
            "beta": self.beta,
        })
        return config

def build_integrativevae_hvae_rna_branch(
    input_dim: int,
    ds: int = 256,
    activation: str = "elu",
    dropout_rate: float = 0.2,
    beta: float = 1.0,
    learning_rate: float = 1e-3,
):
    """
    Build an RNA-only adaptation of the original IntegrativeVAEs HVAE
    numerical branch, corresponding to HVAE(args, 'numVAE') / build_s2().

    Original structure:
    input -> Dense(ds) -> BatchNorm -> z_mean/z_log_sigma of size ds//2
          -> Sampling -> Dense(ds) -> BatchNorm -> Dropout -> linear output

    The latent dimension is ds // 2, matching the original HVAE build_s2().
    """

    inputs = keras.Input(shape=(input_dim,), name="rna_input")

    x = layers.Dense(
        ds,
        activation=activation,
        name="encoder_dense",
    )(inputs)

    x = layers.BatchNormalization(
        name="encoder_batchnorm",
    )(x)

    latent_dim = ds // 2

    z_mean = layers.Dense(
        latent_dim,
        name="z_mean",
    )(x)

    z_log_sigma = layers.Dense(
        latent_dim,
        kernel_initializer="zeros",
        name="z_log_sigma",
    )(x)

    z = layers.Lambda(
        integrativevae_sampling,
        output_shape=(latent_dim,),
        name="z",
    )([z_mean, z_log_sigma])

    encoder = keras.Model(
        inputs,
        [z_mean, z_log_sigma, z],
        name="integrativevae_hvae_rna_encoder",
    )

    z_mean_encoder = keras.Model(
        inputs,
        z_mean,
        name="integrativevae_hvae_rna_z_mean_encoder",
    )

    latent_inputs = keras.Input(
        shape=(latent_dim,),
        name="z_sampling",
    )

    x_decoded = layers.Dense(
        ds,
        activation=activation,
        name="decoder_dense",
    )(latent_inputs)

    x_decoded = layers.BatchNormalization(
        name="decoder_batchnorm",
    )(x_decoded)

    x_decoded = layers.Dropout(
        dropout_rate,
        name="decoder_dropout",
    )(x_decoded)

    decoded_outputs = layers.Dense(
        input_dim,
        name="rna_reconstruction",
    )(x_decoded)

    decoder = keras.Model(
        latent_inputs,
        decoded_outputs,
        name="integrativevae_hvae_rna_decoder",
    )

    reconstructed = decoder(z)

    outputs = IntegrativeVAERNALossLayer(
        beta=beta,
        name="rna_vae_loss",
    )([
        inputs,
        reconstructed,
        z_mean,
        z_log_sigma,
    ])

    vae = keras.Model(
        inputs,
        outputs,
        name="integrativevae_hvae_rna_branch",
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
    )

    # Functional construction already builds this model; explicitly enforce the
    # same pre-fit checkpoint contract without an extra stochastic forward pass.
    vae.build((None, input_dim))
    vae.compile(optimizer=optimizer)

    return vae, encoder, z_mean_encoder, decoder

def build_mobrcanet_expression_attention_model(
    input_dim: int,
    n_classes: int,
    latent_dim: int = 100,
    hidden_dim: int = 256,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
):
    inputs = keras.Input(
        shape=(input_dim,),
        name="gene_expression_input",
    )

    # Gene-level attention/gating branch.
    # Sigmoid gates avoid the pathological softmax collapse observed when
    # attention over hundreds of genes is forced to sum to one.
    attention_hidden = layers.Dense(
        hidden_dim,
        activation="relu",
        name="gene_attention_hidden",
    )(inputs)

    attention_hidden = layers.BatchNormalization(
        name="gene_attention_batchnorm",
    )(attention_hidden)

    attention_hidden = layers.Dropout(
        dropout_rate,
        name="gene_attention_dropout",
    )(attention_hidden)

    attention_weights = layers.Dense(
        input_dim,
        activation="sigmoid",
        name="gene_attention_weights",
    )(attention_hidden)

    attention_weighted_expression = layers.Multiply(
        name="attention_weighted_expression",
    )([inputs, attention_weights])

    x = layers.Dense(
        hidden_dim,
        activation="relu",
        name="hidden_dense",
    )(attention_weighted_expression)

    x = layers.BatchNormalization(
        name="hidden_batchnorm",
    )(x)

    x = layers.Dropout(
        dropout_rate,
        name="hidden_dropout",
    )(x)

    # Use a linear latent layer to avoid many projected institutional features
    # becoming exactly zero because of ReLU truncation.
    latent = layers.Dense(
        latent_dim,
        activation="linear",
        name="attention_latent_representation",
    )(x)

    outputs = layers.Dense(
        n_classes,
        activation="softmax",
        name="pam50_output",
    )(latent)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mobrcanet_inspired_expression_attention_classifier",
    )

    representation_model = keras.Model(
        inputs=inputs,
        outputs=latent,
        name="mobrcanet_inspired_expression_attention_encoder",
    )

    attention_model = keras.Model(
        inputs=inputs,
        outputs=attention_weights,
        name="mobrcanet_inspired_expression_attention_weights",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, representation_model, attention_model

def plot_classification_training_history(history_df, title_prefix="Model"):
    """
    Plot loss and accuracy from a Keras classification training history
    dataframe. Reports validation accuracy AT the best-val-loss epoch
    (the one EarlyStopping with monitor="val_loss" actually restores),
    not just the historical maximum across all epochs.
    """
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["loss"], label="train_loss")
    plt.plot(history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"{title_prefix} Training History")
    plt.legend()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(history_df["accuracy"], label="train_accuracy")
    plt.plot(history_df["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Accuracy")
    plt.legend()
    plt.show()

    best_epoch_idx = history_df["val_loss"].idxmin()

    print("Epochs trained:", history_df.shape[0])
    print("Best epoch (by val_loss):", best_epoch_idx)
    print("Val loss at best epoch:", history_df.loc[best_epoch_idx, "val_loss"])
    print("Val accuracy at best epoch:", history_df.loc[best_epoch_idx, "val_accuracy"])
    print("(Historical max val_accuracy across all epochs, for reference only:",
        history_df["val_accuracy"].max(), ")")

def build_rostami_autoencoder(
    input_dim: int,
    bottleneck_dim: int = 500,
    learning_rate: float = 1e-3,
):
    """
    Build a simple deterministic autoencoder inspired by Rostami et al. (2025).

    Architecture:
        input_dim -> bottleneck_dim -> input_dim

    The hidden layer uses ReLU; linear reconstruction permits signed standardised targets.
    """

    bottleneck_dim = min(bottleneck_dim, input_dim)

    inputs = keras.Input(
        shape=(input_dim,),
        name="gene_expression_input",
    )

    bottleneck = layers.Dense(
        bottleneck_dim,
        activation="relu",
        name="bottleneck",
    )(inputs)

    outputs = layers.Dense(
        input_dim,
        activation="linear",
        name="reconstruction",
    )(bottleneck)

    autoencoder = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="rostami_article_inspired_autoencoder",
    )

    encoder = keras.Model(
        inputs=inputs,
        outputs=bottleneck,
        name="rostami_article_inspired_encoder",
    )

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    return autoencoder, encoder

def compute_encoder_gene_scores(autoencoder, gene_names):
    """
    Compute article-inspired gene feature scores from the encoder weight matrix.

    The article interprets larger input-to-middle-layer weights as stronger
    feature importance. Here, each gene score is the L2 norm of its weights
    from the input layer to all bottleneck nodes.
    """

    encoder_weights = autoencoder.get_layer("bottleneck").get_weights()[0]

    gene_scores = np.linalg.norm(
        encoder_weights,
        ord=2,
        axis=1,
    )

    scores_df = pd.DataFrame({
        "gene": gene_names,
        "gene_score": gene_scores,
    })

    scores_df = scores_df.sort_values(
        "gene_score",
        ascending=False,
    )

    return scores_df


def resolve_gene_ids_via_primary_sources(
    gene_ids,
    max_retries=1,
    base_wait_seconds=2.0,
    request_delay_seconds=0.34,
):
    """
    ... (docstring igual)
    """

    if len(gene_ids) == 0:
        return pd.DataFrame(columns=["gene", "gene_symbol", "status"])

    results = []

    for gene_id in gene_ids:
        symbol = None
        status = None

        for attempt in range(max_retries + 1):
            try:
                if str(gene_id).startswith("ENSG"):
                    resp = requests.get(
                        f"https://rest.ensembl.org/lookup/id/{gene_id}",
                        params={"content-type": "application/json"},
                        headers={"Content-Type": "application/json"},
                        timeout=15,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        symbol = data.get("display_name")
                        status = "found" if symbol else "found_no_symbol"
                        break
                    elif resp.status_code == 404:
                        status = "withdrawn_or_not_found"
                        break
                    else:
                        status = f"http_error_{resp.status_code}"
                else:
                    resp = requests.get(
                        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                        params={"db": "gene", "id": gene_id, "retmode": "json"},
                        timeout=15,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        doc = data.get("result", {}).get(str(gene_id), {})
                        symbol = doc.get("name")
                        status = "found" if symbol else "found_no_symbol"
                        break
                    else:
                        status = f"http_error_{resp.status_code}"
            except Exception as e:
                status = f"error: {e}"

            if attempt < max_retries:
                time.sleep(base_wait_seconds * (attempt + 1))

        results.append({"gene": gene_id, "gene_symbol": symbol, "status": status})
        time.sleep(request_delay_seconds)

    return pd.DataFrame(results)


def run_nested_cv_with_flaml_clinical(
    X,
    y,
    record_ids,
    sample_ids,
    outer_cv,
    time_budget=None,
    eval_method="cv",
    n_splits=5,
    metric="macro_f1",
    random_state=42,
    n_jobs=1,
):
    """
    Run nested cross-validation using FLAML as the model search step.

    FLAML performs its own algorithm selection and hyperparameter tuning
    internally, via `eval_method` (its own "inner" loop — "cv" for k-fold
    internal resampling, matching the same nested-CV principle used
    elsewhere in this project). This function wraps that in the same outer
    cross-validation splitter used for every other candidate model, so the
    resulting test performance is directly comparable.

    Output columns match run_nested_cv_with_predictions_clinical as closely
    as possible, so results can be concatenated/compared directly.

    Parameters
    ----------
    time_budget : positive number, required explicitly
        Search time budget in seconds, PER OUTER FOLD. With 5 outer folds,
        total runtime is at least 5x this value.
    eval_method : str, default="cv"
        FLAML's own internal validation strategy (the "inner" loop).
        "cv" uses k-fold internal resampling instead of a single holdout
        split, better suited to the small sample size here.
    """

    if eval_method != "cv" or n_splits != 5 or metric != "macro_f1":
        raise ValueError("Clinical FLAML requires five-fold inner CV and macro-F1.")
    settings = clinical_flaml_settings(time_budget, "f1_macro", random_state, n_jobs)

    all_test_results = []
    all_train_results = []
    all_predictions = []

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    record_ids = record_ids.reset_index(drop=True)
    sample_ids = sample_ids.reset_index(drop=True)

    outer_splits = clinical_outer_splits(outer_cv, X, y)
    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        print(f"\n=== FLAML | Outer fold {fold_idx}/{len(outer_splits)} ===")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        automl = AutoML()
        automl.fit(X_train=X_train, y_train=y_train, **settings)

        y_test_proba = automl.predict_proba(X_test)[:, 1]
        y_test_pred = automl.predict(X_test)
        y_train_proba = automl.predict_proba(X_train)[:, 1]
        y_train_pred = automl.predict(X_train)

        test_metrics = calculate_classification_metrics(
            y_true=y_test, y_pred=y_test_pred, y_prob=y_test_proba, y_true_constant=True
        )
        train_metrics = calculate_classification_metrics(
            y_true=y_train, y_pred=y_train_pred, y_prob=y_train_proba, y_true_constant=True
        )

        test_metrics["balanced_accuracy"] = (
            test_metrics.get("sensitivity", np.nan) + test_metrics.get("specificity", np.nan)
        ) / 2
        train_metrics["balanced_accuracy"] = (
            train_metrics.get("sensitivity", np.nan) + train_metrics.get("specificity", np.nan)
        ) / 2

        test_metrics.update(clinical_binary_metrics(y_test, y_test_pred, y_test_proba))
        train_metrics.update(clinical_binary_metrics(y_train, y_train_pred, y_train_proba))

        best_pipeline_repr = json.dumps({
            "best_estimator": automl.best_estimator,
            "best_config": automl.best_config,
        }, default=str)

        best_inner_score = 1 - automl.best_loss  # FLAML macro_f1 loss is 1 - macro-F1

        test_row = {
            "model": "flaml_automl",
            "outer_fold": fold_idx,
            "best_params": best_pipeline_repr,
            "best_inner_score": best_inner_score,
            "selection_metric": metric,
            "inner_n_splits": n_splits,
            "time_budget_seconds": time_budget,
            **test_metrics,
        }
        train_row = {
            "model": "flaml_automl",
            "outer_fold": fold_idx,
            "best_params": best_pipeline_repr,
            "best_inner_score": best_inner_score,
            "selection_metric": metric,
            "inner_n_splits": n_splits,
            "time_budget_seconds": time_budget,
            **train_metrics,
        }

        all_test_results.append(test_row)
        all_train_results.append(train_row)

        fold_predictions = pd.DataFrame({
            "model": "flaml_automl",
            "outer_fold": fold_idx,
            "record_id": record_ids.iloc[test_idx].values,
            "sample_id": sample_ids.iloc[test_idx].values,
            "dataset": "test",
            "row_position": test_idx,
            "selection_metric": metric,
            "inner_n_splits": n_splits,
            "time_budget_seconds": time_budget,
            "y_true": y_test.values,
            "y_pred": y_test_pred,
            "y_proba": y_test_proba,
        })
        all_predictions.append(fold_predictions)

    results_test_df = pd.DataFrame(all_test_results)
    results_train_df = pd.DataFrame(all_train_results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    return results_test_df, results_train_df, predictions_df


def run_nested_cv_with_flaml_mutation(
    X,
    y,
    sample_ids,
    outer_cv,
    time_budget=None,
    eval_method="cv",
    n_splits=5,
    metric="macro_f1",
    random_state=42,
    n_jobs=1,
):
    """
    Run nested cross-validation using FLAML as the model search step, for
    the mutation dataset. Mirrors run_nested_cv_with_flaml_clinical, but
    uses compute_metrics (roc_auc, balanced_accuracy, f1, f1_macro,
    f1_weighted, precision, recall) for consistency with
    run_nested_cv_with_predictions_mutation's own metric naming, and has no
    record_ids column (mutation predictions only track sample_id).
    """
    from flaml import AutoML

    if eval_method != "cv" or n_splits != 5 or metric != "macro_f1":
        raise ValueError("Mutation FLAML requires five-fold inner CV and macro-F1.")
    settings = mutation_flaml_settings(time_budget, random_state, n_jobs)

    all_test_results = []
    all_train_results = []
    all_predictions = []

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    sample_ids = sample_ids.reset_index(drop=True)

    outer_splits = mutation_outer_splits(outer_cv, X, y)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        print(f"\n=== FLAML | Outer fold {fold_idx}/{len(outer_splits)} ===")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        automl = AutoML()
        automl.fit(X_train=X_train, y_train=y_train, **settings)

        y_test_proba = automl.predict_proba(X_test)[:, 1]
        y_test_pred = automl.predict(X_test)
        y_train_proba = automl.predict_proba(X_train)[:, 1]
        y_train_pred = automl.predict(X_train)

        test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)
        train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)

        best_pipeline_repr = json.dumps({
            "best_estimator": automl.best_estimator,
            "best_config": automl.best_config,
        }, default=str)

        best_inner_score = 1 - automl.best_loss

        test_row = {
            "model": "flaml_automl",
            "outer_fold": fold_idx,
            "dataset": "test",
            "best_params": best_pipeline_repr,
            "best_inner_score": best_inner_score,
            "selection_metric": metric,
            "inner_n_splits": n_splits,
            "time_budget_seconds": time_budget,
            **test_metrics,
        }
        train_row = {
            "model": "flaml_automl",
            "outer_fold": fold_idx,
            "dataset": "train",
            "best_params": best_pipeline_repr,
            "best_inner_score": best_inner_score,
            "selection_metric": metric,
            "inner_n_splits": n_splits,
            "time_budget_seconds": time_budget,
            **train_metrics,
        }

        all_test_results.append(test_row)
        all_train_results.append(train_row)

        fold_predictions = pd.DataFrame({
            "model": "flaml_automl",
            "outer_fold": fold_idx,
            "sample_id": sample_ids.iloc[test_idx].values,
                "dataset": "test",
                "row_position": test_idx,
            "y_true": y_test.values,
            "y_pred": y_test_pred,
            "y_proba": y_test_proba,
            "best_inner_score": best_inner_score,
            "selection_metric": metric,
            "inner_n_splits": n_splits,
            "time_budget_seconds": time_budget,
            "best_params": best_pipeline_repr,
        })
        all_predictions.append(fold_predictions)

    results_test_df = pd.DataFrame(all_test_results)
    results_train_df = pd.DataFrame(all_train_results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    return results_test_df, results_train_df, predictions_df


def compute_representation_silhouette(X, y, representation_name, scale=True):
    """Compute an exploratory silhouette score using oligo/poly labels."""
    X_array = np.asarray(X)
    y_array = np.asarray(y)

    if scale:
        X_array = StandardScaler().fit_transform(X_array)

    score = silhouette_score(X_array, y_array)

    return {
        "representation": representation_name,
        "silhouette_score": score,
        "n_samples": X_array.shape[0],
        "n_features": X_array.shape[1],
    }


def plot_umap_representation(
    X,
    y,
    representation_name,
    random_state=42,
    n_neighbors=15,
    min_dist=0.1,
    scale=True,
):
    """Create an exploratory 2D UMAP projection coloured by metastatic group."""
    X_array = np.asarray(X)
    y_array = np.asarray(y)

    if scale:
        X_array = StandardScaler().fit_transform(X_array)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )

    embedding = reducer.fit_transform(X_array)

    umap_df = pd.DataFrame({
        "UMAP1": embedding[:, 0],
        "UMAP2": embedding[:, 1],
        "Polymetastatic": y_array,
    })

    plt.figure(figsize=(7, 6))

    for class_value, label in [(0, "Oligometastatic"), (1, "Polymetastatic")]:
        subset = umap_df[umap_df["Polymetastatic"] == class_value]
        plt.scatter(subset["UMAP1"], subset["UMAP2"], label=label, alpha=0.8)

    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(f"UMAP — {representation_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return umap_df

def wrap_with_feature_selection(classifiers, k_grid=None):
    """
    Add supervised feature selection inside the CV pipeline.
    """
    if k_grid is None:
        k_grid = [5, 10, 20, 50]

    wrapped = {}

    for name, info in classifiers.items():
        classifier = clone(
            info["pipeline"].named_steps["classifier"]
        )

        pipeline = Pipeline([
            ("selector", SelectKBest(
                score_func=f_classif,
                k=5
            )),
            ("scaler", StandardScaler()),
            ("sampler", "passthrough"),
            ("classifier", classifier),
        ])

        base_param_grid = {
            key: value
            for key, value in info["param_grid"].items()
            if key != "selector"
        }

        param_grid = {
            "selector__k": k_grid,
            **base_param_grid,
        }

        wrapped[name] = {
            "pipeline": pipeline,
            "param_grid": param_grid,
        }

    return wrapped



# Thesis-owned plotting and metric implementations; shared utility is read-only.


def plot_roc_curves(
    curves_data,
    title="Receiver Operating Characteristic (ROC) Curve",
    save_plot_path=None,
    interactive=True,
    dpi: int = 144,
    width: int | None = None,
    height: int | None = None,
    margin: dict = dict(t=80, b=50, l=100, r=50),
    marker_size: int = 2,
    base_font_size: int | None = None,
    title_font_size: int | None = None,
    axis_title_font_size: int | None = None,
    x_tick_font_size: int | None = None,
    y_tick_font_size: int | None = None,
    legend_font_size: int | None = None,
):
    """
    Plot one or more ROC curves (TPR vs FPR) with AUC annotations.

    Each entry in `curves_data` describes one curve. One curve can be designated as
    the “main” curve (solid/thicker) while others are drawn as dotted comparison lines.
    A diagonal “random classifier” baseline is added for reference.

    Parameters
    ----------
    curves_data : list[dict]
        Each dict must contain:
        - y_true: array-like of binary labels
        - y_prob: array-like of predicted probabilities for the positive class
        - label: str, label displayed in legend
        - is_main_curve: bool, if True uses main styling
    title : str
        Plot title.
    interactive : bool, default=True
        If True, show interactive plot; else render PNG.
    dpi : int, default=144
        DPI used for PNG rendering and saving (96 -> scale=1).
    save_plot_path : str or None
        If provided, save the figure; format inferred from extension (requires kaleido).

    Dynamic sizing & fonts (optional)
    ---------------------------------
    width, height : int or None
        Override figure size (pixels). If None, fallbacks are used.
    margin : dict
        Margin dict for the figure layout.
    marker_size : int
        Size of markers on the ROC curves.
    base_font_size, title_font_size, axis_title_font_size : int or None
        Layout font sizes. If None, fallbacks are used.
    x_tick_font_size, y_tick_font_size : int or None
        Tick label font sizes. If None, fallbacks are used.
    legend_font_size : int or None
        Legend font size. If None, derived from base font size.

    Returns
    -------
    plotly.graph_objects.Figure or None
        ROC figure, or None if no valid curves are available.

    Notes
    -----
    - Curves whose y_true contains only one class are skipped (AUC undefined).
    - Requires sklearn: `roc_curve`, `auc`.
    """

    print("--- Plotting ROC Curve ---")

    if not isinstance(curves_data, list) or not curves_data:
        print("Cannot plot ROC curve: 'curves_data' must be a non-empty list of dictionaries.")
        return None

    fig_roc = go.Figure()
    has_valid_curve = False

    # Define a set of colors for comparison curves
    comparison_colors = ["red", "green", "purple", "orange", "brown", "pink", "teal"]
    color_idx = 0

    for i, curve_info in enumerate(curves_data):
        y_true = curve_info.get("y_true")
        y_prob = curve_info.get("y_prob")
        label = curve_info.get("label", f"Curve {i+1}")
        is_main = curve_info.get("is_main_curve", False)

        if y_true is None or y_prob is None or len(y_true) == 0:
            print(f"Skipping '{label}': y_true or y_prob data is missing or empty.")
            continue
        if len(np.unique(y_true)) < 2:
            print(f"Skipping '{label}': y_true contains only one class.")
            continue

        has_valid_curve = True
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        if is_main:
            line_config = dict(color="blue", width=marker_size)
        else:
            line_config = dict(
                color=comparison_colors[color_idx % len(comparison_colors)],
                dash="dot",
                width=marker_size,
            )
            color_idx += 1

        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{label} (AUC = {roc_auc:.2f})",
                line=line_config,
            )
        )

    if not has_valid_curve:
        print("No valid curve data found to plot ROC curve.")
        return None

    # Add the random classifier line (diagonal)
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color="grey", dash="dash", width=1),
        )
    )

    # Apply dynamic layout helper (no n_features for ROC)
    fig_roc = apply_dynamic_plotly_layout(
        fig_roc,
        title=title,
        n_features=None,
        width=width,
        height=height,
        base_font_size=base_font_size,
        title_font_size=title_font_size,
        axis_title_font_size=axis_title_font_size,
        x_tick_font_size=x_tick_font_size,
        y_tick_font_size=y_tick_font_size,
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        reverse_y=False,
        margin=margin,
    )

    # Axes ranges/grid (kept separate so apply_dynamic_plotly_layout stays generic)
    fig_roc.update_xaxes(range=[0, 1], showgrid=True)
    fig_roc.update_yaxes(range=[0, 1], showgrid=True)

    # Legend settings (with optional font control)
    final_base_font = int(fig_roc.layout.font.size) if fig_roc.layout.font and fig_roc.layout.font.size else 12
    legend_font_size = legend_font_size if legend_font_size is not None else _clamp(final_base_font, 9, 14)

    fig_roc.update_layout(
        showlegend=True,
        hovermode="closest",
        legend=dict(
            x=0.99,
            y=0.01,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="Black",
            borderwidth=1,
            font=dict(size=legend_font_size),
        ),
    )

    if interactive:
        fig_roc.show()
    else:
        fig_roc.show(
            renderer="png",
            width=fig_roc.layout.width,
            height=fig_roc.layout.height,
            scale=_plotly_scale_from_dpi(dpi),
        )

    if save_plot_path:
        try:
            output_dir = os.path.dirname(save_plot_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")

            file_extension = os.path.splitext(save_plot_path)[1].lower().replace(".", "")
            if file_extension not in ["png", "jpeg", "webp", "svg", "pdf"]:
                print(f"Warning: Unsupported file format '{file_extension}'. Saving as PNG.")
                save_plot_path = os.path.splitext(save_plot_path)[0] + ".png"
                file_extension = "png"

            fig_roc.write_image(
                save_plot_path,
                width=fig_roc.layout.width,
                height=fig_roc.layout.height,
                format=file_extension,
                scale=_plotly_scale_from_dpi(dpi),
            )
            print(f"ROC curve saved to: {save_plot_path}")
        except ImportError:
            print("Error: 'kaleido' package not found. Please install it (`pip install kaleido`) to save plots.")
        except Exception as e:
            print(f"An error occurred while saving the plot: {e}")

    return fig_roc

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_labels: Optional[list[str]] = None,
    title: str = "Confusion Matrix",
    save_plot_path: Optional[str] = None,
    interactive: bool = True,
    dpi: int = 144,
    width: Optional[int] = None,
    height: Optional[int] = None,
    margin: dict = dict(t=80, b=50, l=100, r=50),
    base_font_size: Optional[int] = None,
    title_font_size: Optional[int] = None,
    axis_title_font_size: Optional[int] = None,
    x_tick_font_size: Optional[int] = None,
    y_tick_font_size: Optional[int] = None,
    annotation_font_size: Optional[int] = None,
):
    """
    Plot a confusion matrix as a Plotly heatmap with readable cell annotations.

    The function computes the confusion matrix from `y_true` and `y_pred`,
    renders a heatmap, and overlays numeric cell annotations whose text color
    adapts to the cell intensity (white on dark, black on light) for legibility.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted class labels.
    class_labels : list[str] or None, optional
        Labels to display on axes. If None, labels are derived from the union of
        unique values in `y_true` and `y_pred` as "Class <label>".
    title : str, default="Confusion Matrix"
        Plot title.
    interactive : bool, default=True
        If True, show an interactive Plotly figure.
        If False, render as PNG via Plotly renderer (useful in headless contexts).
    save_plot_path : str or None, optional
        If provided, save the figure to this path. Format is inferred from the file
        extension (.png/.jpeg/.webp/.svg/.pdf). Requires `kaleido`.
    dpi : int, default=144
        DPI used for PNG rendering and saving (96 -> scale=1).
    margin : dict, optional
        Margin dictionary for the figure layout.

    Dynamic sizing & fonts (optional)
    ---------------------------------
    width, height : int or None
        If provided, override figure size (pixels). If None, fallbacks are used.
    base_font_size, title_font_size, axis_title_font_size : int or None
        Layout-level font sizes. If None, fallbacks are used.
    x_tick_font_size, y_tick_font_size : int or None
        Tick label font sizes. If None, fallbacks are used.
    annotation_font_size : int or None
        Font size for the values printed inside the matrix cells. If None, derived
        from `base_font_size`.

    Returns
    -------
    plotly.graph_objects.Figure or None
        The confusion matrix figure, or None if inputs are empty/invalid.

    Notes
    -----
    - This plot does not use n_features-based autoscaling (matrix size is typically small).
    - Saving requires `pip install kaleido`.
    """

    print("--- Plotting Confusion Matrix ---")

    if y_true is None or y_pred is None or len(y_true) == 0:
        print("Cannot plot Confusion Matrix: y_true or y_pred data is missing or empty.")
        return None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if class_labels is None:
        unique_labels = [0, 1]
        class_labels = [f"Class {label}" for label in unique_labels]

    heatmap = go.Heatmap(
        z=cm,
        x=class_labels,
        y=class_labels,
        colorscale="Blues",
        zmin=0,
        zmax=max(int(np.max(cm)), 1),
        showscale=False,
    )

    max_cm_value = int(np.max(cm)) if cm.size else 0

    # If not provided, make annotation font scale with base font a bit
    # (computed later if base_font_size is None, so set placeholder now)
    annotations = []

    # Create the figure first so we can compute final base_font_size via helper fallback logic
    fig_cm = go.Figure(data=[heatmap])

    # Apply dynamic layout (no n_features for confusion matrix)
    fig_cm = apply_dynamic_plotly_layout(
        fig_cm,
        title=title,
        n_features=None,
        width=width,
        height=height,
        base_font_size=base_font_size,
        title_font_size=title_font_size,
        axis_title_font_size=axis_title_font_size,
        x_tick_font_size=x_tick_font_size,
        y_tick_font_size=y_tick_font_size,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        reverse_y=True,
        margin=margin,
    )

    # Now that base font is finalized in the layout, set annotation size default if needed
    final_base_font = int(fig_cm.layout.font.size) if fig_cm.layout.font and fig_cm.layout.font.size else 12
    ann_size = annotation_font_size if annotation_font_size is not None else _clamp(final_base_font + 6, 14, 28)

    for i, row in enumerate(cm):  # True labels (rows)
        for j, value in enumerate(row):  # Pred labels (cols)
            font_color = _clinical_heatmap_text_colour(value, max_cm_value)
            annotations.append(
                dict(
                    x=class_labels[j],
                    y=class_labels[i],
                    text=str(int(value)),
                    font=dict(color=font_color, size=ann_size),
                    showarrow=False,
                    xref="x",
                    yref="y",
                )
            )

    fig_cm.update_layout(annotations=annotations)

    if interactive:
        fig_cm.show()
    else:
        fig_cm.show(
            renderer="png",
            width=fig_cm.layout.width,
            height=fig_cm.layout.height,
            scale=_plotly_scale_from_dpi(dpi),
        )

    if save_plot_path:
        try:
            output_dir = os.path.dirname(save_plot_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")

            file_extension = os.path.splitext(save_plot_path)[1].lower().replace(".", "")
            if file_extension not in ["png", "jpeg", "webp", "svg", "pdf"]:
                print(f"Warning: Unsupported file format '{file_extension}'. Saving as PNG.")
                save_plot_path = os.path.splitext(save_plot_path)[0] + ".png"
                file_extension = "png"

            fig_cm.write_image(
                save_plot_path,
                width=fig_cm.layout.width,
                height=fig_cm.layout.height,
                format=file_extension,
                scale=_plotly_scale_from_dpi(dpi),
            )
            print(f"Confusion matrix saved to: {save_plot_path}")
        except ImportError:
            print("Error: 'kaleido' package not found. Please install it (`pip install kaleido`) to save plots.")
        except Exception as e:
            print(f"An error occurred while saving the plot: {e}")

    return fig_cm

def plot_combined_ranking(
    df: pd.DataFrame,
    sort_by: str = "combined",
    metrics: list | None = None,
    pretty_names: dict | None = None,
    data_type_col: str = "data_type",
    title: str = "Combined Ranking",
    xaxis_title: str = "Score",
    yaxis_title: str = "Data Type",
    xaxis_range: list | None = None,
    save_plot_path: str | None = None,
    interactive: bool = True,
    dpi: int = 144,
    width: int | None = None,
    height: int | None = None,
    margin: dict = dict(t=80, b=60, l=120, r=40),
    base_font_size: int | None = None,
    title_font_size: int | None = None,
    axis_title_font_size: int | None = None,
    x_tick_font_size: int | None = None,
    y_tick_font_size: int | None = None,
    legend_font_size: int | None = None,
) -> go.Figure:
    """Horizontal grouped bar chart of multiple metrics per data type.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``data_type_col`` and all metric columns.
    sort_by : str
        Column to sort by (ascending, so highest appears at top).
    metrics : list of (col, color, display_name) tuples, optional
        Defaults to Combined / Median F1 / Nogueira.
    pretty_names : dict, optional
        ``{data_type: display_name}`` for y-axis labels.
    data_type_col : str
        Name of the column holding data-type identifiers.
    title : str
        Plot title.
    xaxis_title, yaxis_title : str
        Axis labels.
    xaxis_range : list, optional
        X-axis range.  Defaults to ``[0, 1.05]``.
    save_plot_path : str, optional
        If provided, saves the figure to this path.
    interactive : bool
        If True shows interactive Plotly; otherwise renders as PNG.
    dpi : int
        DPI for PNG rendering.
    width, height : int, optional
        Figure dimensions in pixels.
    margin : dict
        Plotly margin dict.
    base_font_size, title_font_size, axis_title_font_size,
    x_tick_font_size, y_tick_font_size, legend_font_size : int, optional
        Font size overrides.

    Returns
    -------
    go.Figure
    """
    print(f"\n--- Plotting: {title} ---")

    if metrics is None:
        metrics = [
            ("combined", "#AB63FA", "Combined"),
            ("median_f1", "#636EFA", "Median F1"),
            ("nogueira", "#EF553B", "Nogueira"),
        ]

    if xaxis_range is None:
        xaxis_range = [0, 1.05]

    df_sorted = df.sort_values(sort_by, ascending=True)
    y_labels = (
        df_sorted[data_type_col].map(pretty_names).tolist()
        if pretty_names
        else df_sorted[data_type_col].tolist()
    )

    fig = go.Figure()
    for col, color, display_name in metrics:
        vals = df_sorted[col].fillna(0)
        fig.add_trace(go.Bar(
            y=y_labels,
            x=vals,
            orientation="h",
            name=display_name,
            marker_color=color,
            text=vals.apply(lambda v: f"{v:.3f}"),
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        xaxis_range=xaxis_range,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    apply_dynamic_plotly_layout(
        fig,
        title=title,
        n_features=len(df_sorted),
        width=width,
        height=height,
        y_labels=y_labels,
        base_font_size=base_font_size,
        title_font_size=title_font_size,
        axis_title_font_size=axis_title_font_size,
        x_tick_font_size=x_tick_font_size,
        y_tick_font_size=y_tick_font_size,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        margin=margin,
    )

    if legend_font_size is not None:
        fig.update_layout(legend=dict(font=dict(size=legend_font_size)))

    if interactive:
        fig.show()
    else:
        fig.show(
            renderer="png",
            width=fig.layout.width,
            height=fig.layout.height,
            scale=_plotly_scale_from_dpi(dpi),
        )

    if save_plot_path:
        try:
            output_dir = os.path.dirname(save_plot_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            file_ext = os.path.splitext(save_plot_path)[1].lower().replace(".", "")
            if file_ext not in ["png", "jpeg", "webp", "svg", "pdf"]:
                save_plot_path = os.path.splitext(save_plot_path)[0] + ".png"
                file_ext = "png"
            fig.write_image(
                save_plot_path,
                width=fig.layout.width,
                height=fig.layout.height,
                format=file_ext,
                scale=_plotly_scale_from_dpi(dpi),
            )
            print(f"Plot saved to: {save_plot_path}")
        except ImportError:
            print("Error: 'kaleido' not installed. Run `pip install kaleido` to enable saving.")
        except Exception as e:
            print(f"An error occurred while saving the plot: {e}")

    return fig


def plot_precision_recall_curves(
    curves_data,
    title="Precision–Recall (PR) Curve",
    save_plot_path=None,
    interactive=True,
    dpi: int = 144,
    width: int | None = None,
    height: int | None = None,
    margin: dict = dict(t=80, b=60, l=80, r=40),
    marker_size: int = 2,
    base_font_size: int | None = None,
    title_font_size: int | None = None,
    axis_title_font_size: int | None = None,
    x_tick_font_size: int | None = None,
    y_tick_font_size: int | None = None,
    legend_font_size: int | None = None,
):
    """
    Plot one or more Precision–Recall curves with Average Precision (AP) annotations.

    Each entry in `curves_data` describes one curve. One curve can be designated as
    the “main” curve (solid/thicker) while others are drawn as dotted comparisons.
    A no-skill baseline is added at the positive-class prevalence (estimated from the
    main curve if present).

    Parameters
    ----------
    curves_data : list[dict]
        Each dict must contain:
        - y_true: array-like of binary labels
        - y_prob: array-like of predicted probabilities for the positive class
        - label: str, label displayed in legend
        - is_main_curve: bool, if True uses main styling and sets baseline prevalence
    title : str
        Plot title.
    interactive : bool, default=True
        If True, show interactive plot; else render PNG.
    dpi : int, default=144
        DPI used for PNG rendering and saving (96 -> scale=1).
    save_plot_path : str or None
        If provided, save the figure; format inferred from extension (requires kaleido).

    Dynamic sizing & fonts (optional)
    ---------------------------------
    width, height : int or None
        Override figure size (pixels). If None, fallbacks are used.
    margin : dict
        Margin dictionary for the figure layout.
    marker_size : int
        Marker line width for comparison curves.
    base_font_size, title_font_size, axis_title_font_size : int or None
        Layout font sizes. If None, fallbacks are used.
    x_tick_font_size, y_tick_font_size : int or None
        Tick label font sizes. If None, fallbacks are used.
    legend_font_size : int or None
        Legend font size. If None, derived from base font size.

    Returns
    -------
    plotly.graph_objects.Figure or None
        PR figure, or None if no valid curves are available.

    Notes
    -----
    - Curves whose y_true contains only one class are skipped.
    - Requires sklearn: `precision_recall_curve`, `average_precision_score`.
    """

    print("--- Plotting Precision–Recall Curve ---")

    if not isinstance(curves_data, list) or not curves_data:
        print("Cannot plot PR curve: 'curves_data' must be a non-empty list of dictionaries.")
        return None

    fig_pr = go.Figure()
    has_valid_curve = False

    comparison_colors = ["red", "green", "purple", "orange", "brown", "pink", "teal"]
    color_idx = 0

    baseline_prevalence = None

    for i, curve_info in enumerate(curves_data):
        y_true = curve_info.get("y_true")
        y_prob = curve_info.get("y_prob")
        label = curve_info.get("label", f"Curve {i+1}")
        is_main = curve_info.get("is_main_curve", False)

        if y_true is None or y_prob is None or len(y_true) == 0:
            print(f"Skipping '{label}': y_true or y_prob data is missing or empty.")
            continue
        if len(np.unique(y_true)) < 2:
            print(f"Skipping '{label}': y_true contains only one class.")
            continue

        has_valid_curve = True

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        if is_main and baseline_prevalence is None:
            baseline_prevalence = float(np.mean(y_true))

        if is_main:
            line_config = dict(color="steelblue", width=marker_size)
        else:
            line_config = dict(
                color=comparison_colors[color_idx % len(comparison_colors)],
                dash="dot",
                width=marker_size,
            )
            color_idx += 1

        fig_pr.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{label} (AP = {ap:.2f})",
                line=line_config,
            )
        )

    if not has_valid_curve:
        print("No valid curve data found to plot PR curve.")
        return None

    # Add no-skill baseline
    if baseline_prevalence is not None:
        fig_pr.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[baseline_prevalence, baseline_prevalence],
                mode="lines",
                name=f"No-skill (prevalence = {baseline_prevalence:.2f})",
                line=dict(color="grey", dash="dash", width=1),
            )
        )

    # Apply dynamic layout helper (PR doesn’t have a meaningful n_features)
    fig_pr = apply_dynamic_plotly_layout(
        fig_pr,
        title=title,
        n_features=None,
        width=width,
        height=height,
        base_font_size=base_font_size,
        title_font_size=title_font_size,
        axis_title_font_size=axis_title_font_size,
        x_tick_font_size=x_tick_font_size,
        y_tick_font_size=y_tick_font_size,
        xaxis_title="Recall",
        yaxis_title="Precision",
        reverse_y=False,
        margin=margin,
    )

    # Force axes ranges/grid
    fig_pr.update_xaxes(range=[0, 1], showgrid=True)
    fig_pr.update_yaxes(range=[0, 1], showgrid=True)

    # Legend settings (+ optional font control)
    final_base_font = int(fig_pr.layout.font.size) if fig_pr.layout.font and fig_pr.layout.font.size else 12
    legend_font_size = legend_font_size if legend_font_size is not None else _clamp(final_base_font, 9, 14)

    fig_pr.update_layout(
        showlegend=True,
        hovermode="closest",
        legend=dict(
            x=0.99,
            y=0.01,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="Black",
            borderwidth=1,
            font=dict(size=legend_font_size),
        ),
    )

    if interactive:
        fig_pr.show()
    else:
        fig_pr.show(
            renderer="png",
            width=fig_pr.layout.width,
            height=fig_pr.layout.height,
            scale=_plotly_scale_from_dpi(dpi),
        )

    if save_plot_path:
        try:
            output_dir = os.path.dirname(save_plot_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")

            file_extension = os.path.splitext(save_plot_path)[1].lower().replace(".", "")
            if file_extension not in ["png", "jpeg", "webp", "svg", "pdf"]:
                print(f"Warning: Unsupported file format '{file_extension}'. Saving as PNG.")
                save_plot_path = os.path.splitext(save_plot_path)[0] + ".png"
                file_extension = "png"

            fig_pr.write_image(
                save_plot_path,
                width=fig_pr.layout.width,
                height=fig_pr.layout.height,
                format=file_extension,
                scale=_plotly_scale_from_dpi(dpi),
            )
            print(f"PR curve saved to: {save_plot_path}")
        except ImportError:
            print("Error: 'kaleido' package not found. Please install it (`pip install kaleido`) to save plots.")
        except Exception as e:
            print(f"An error occurred while saving the plot: {e}")

    return fig_pr

def calculate_classification_metrics(
    y_true,
    y_pred,
    y_prob,
    y_true_constant: bool = False,
    selected_features_per_fold: Optional[Dict] = None,
    n_total_features: Optional[int] = None,
    stability_metrics: Optional[List[str]] = None,
):
    """
    Calculates a set of performance metrics for binary classification.

    Handles cases where y_true is constant by returning only applicable metrics.
    Optionally computes feature-selection stability across outer CV folds.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_pred : array-like
        Predicted binary labels (0 or 1) from the classifier.
    y_prob : array-like
        Predicted probabilities for the positive class.
    y_true_constant : bool, default=False
        If True, allows metric computation even when y_true has a single class.
    selected_features_per_fold : dict, optional
        Dictionary mapping fold index to list of selected feature names,
        e.g. ``results['selected_features_per_fold']``. When provided with
        ≥ 2 folds, stability metrics are computed and added to the output.
    n_total_features : int, optional
        Total number of input features (columns of X). Required for the
        Nogueira stability metric; ignored otherwise.
    stability_metrics : list of str, optional
        Which stability metrics to compute. Supported: ``"jaccard"``,
        ``"nogueira"``. Defaults to ``["jaccard", "nogueira"]`` when
        ``selected_features_per_fold`` is provided.

    Returns
    -------
    dict
        Dictionary containing calculated metrics. Missing metrics are skipped if not computable.
        When stability is requested, keys ``"stability_jaccard"`` and/or
        ``"stability_nogueira"`` are added.
    """
    print("\n--- Calculating Model Performance Metrics ---")

    if y_true is None or y_pred is None or len(y_true) == 0:
        print("Error: y_true or y_pred is missing or empty.")
        return {}

    metrics = {}

    unique_classes = np.unique(y_true)
    constant_labels = len(unique_classes) < 2

    # Handle constant y_true case
    if constant_labels and not y_true_constant:
        print("Error: y_true contains only one class. Set y_true_constant=True to compute limited metrics.")
        return {}

    # Accuracy is always computable
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    if constant_labels:
        # Only accuracy and precision (if possible) can be computed
        try:
            metrics['precision'] = precision_score(y_true, y_pred, pos_label=unique_classes[0])
        except Exception:
            metrics['precision'] = np.nan

        # AUC and recall are undefined for a single class
        metrics.update({
            'f1_score': np.nan,
            'sensitivity': np.nan,
            'specificity': np.nan,
            'auc_score': np.nan
        })
        print("Warning: y_true constant — only partial metrics computed.")
    else:
        # Normal binary case
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary', pos_label=1)
        metrics['sensitivity'] = recall_score(y_true, y_pred, pos_label=1)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) != 0 else 0.0

        try:
            metrics['auc_score'] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics['auc_score'] = np.nan

        metrics['precision'] = precision_score(y_true, y_pred, pos_label=1)

    # --- Outer-fold feature-selection stability ---
    if selected_features_per_fold is not None and len(selected_features_per_fold) >= 2:
        feature_sets = [set(fs) for fs in selected_features_per_fold.values()]
        if stability_metrics is None:
            stability_metrics = ["jaccard", "nogueira"]
        for sm in stability_metrics:
            if sm == "jaccard":
                metrics["stability_jaccard"] = stability_jaccard(feature_sets)
            elif sm == "nogueira":
                if n_total_features is not None:
                    metrics["stability_nogueira"] = stability_nogueira(feature_sets, n_total_features)
                else:
                    print("Warning: n_total_features not provided — skipping Nogueira stability.")

    # Print metrics
    for key, val in metrics.items():
        print(f"{key.capitalize()}: {val:.4f}" if not np.isnan(val) else f"{key.capitalize()}: N/A")

    print("\nMetrics calculated successfully.")
    return metrics

def apply_dynamic_plotly_layout(
        fig,
        *,
        title: str,
        n_features: Optional[int] = None,
        width: Optional[int] = None,
        y_labels: Optional[Sequence[str]] = None,
        height: Optional[int] = None,
        base_font_size: Optional[int] = None,
        title_font_size: Optional[int] = None,
        axis_title_font_size: Optional[int] = None,
        x_tick_font_size: Optional[int] = None,
        y_tick_font_size: Optional[int] = None,
        xaxis_title: Optional[str] = None,
        yaxis_title: Optional[str] = None,
        reverse_y: bool = False,
        margin: Optional[dict] = None,
    ):
    """
    Applies a dynamic Plotly layout to a given figure, automatically adjusting figure size and font sizes
    based on the number of features and label lengths for improved readability.

    Parameters
    ----------
    fig : plotly.graph_objs.Figure
        The Plotly figure object to update.
    title : str
        The title of the plot.
    n_features : int, optional
        Number of features (e.g., rows or categories) to scale height and font sizes automatically.
    width : int, optional
        Explicit figure width in pixels. If not provided, estimated from y_labels or defaults to 900.
    y_labels : Sequence[str], optional
        Sequence of y-axis labels used to estimate figure width if width is not provided.
    height : int, optional
        Explicit figure height in pixels. If not provided, auto-scaled based on n_features or defaults to 600.
    base_font_size : int, optional
        Base font size for the figure. Auto-scaled if n_features is provided.
    title_font_size : int, optional
        Font size for the plot title. Auto-scaled if n_features is provided.
    axis_title_font_size : int, optional
        Font size for axis titles. Auto-scaled if n_features is provided.
    x_tick_font_size : int, optional
        Font size for x-axis tick labels. Auto-scaled if n_features is provided.
    y_tick_font_size : int, optional
        Font size for y-axis tick labels. Auto-scaled if n_features is provided.
    xaxis_title : str, optional
        Custom title for the x-axis.
    yaxis_title : str, optional
        Custom title for the y-axis.
    reverse_y : bool, default False
        If True, reverses the y-axis direction.
    margin : dict, optional
        Custom margin dictionary for the figure layout.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        The updated Plotly figure object with the applied dynamic layout.
    """
    # --- WIDTH ---
    fallback_width = 900
    if width is None:
        if y_labels is not None:
            width = _clamp(_estimate_label_width_px(y_labels), 700, 1600)
        else:
            width = fallback_width

    # --- HEIGHT + FONTS ---
    fallback_height = 600
    fallback_base = 12
    fallback_title = 18
    fallback_axis_title = 14
    fallback_x_tick = 11
    fallback_y_tick = 11

    if n_features is not None:
        auto_height = _clamp(350 + 24 * n_features, 400, 1600)
        auto_y_tick = _clamp(int(14 - 0.12 * n_features), 8, 12)
        auto_x_tick = _clamp(auto_y_tick + 1, 9, 13)
        auto_axis_title = _clamp(auto_y_tick + 3, 11, 16)
        auto_title = _clamp(auto_axis_title + 3, 14, 22)
        auto_base = _clamp(auto_y_tick + 1, 9, 13)

        height = height if height is not None else auto_height
        base_font_size = base_font_size if base_font_size is not None else auto_base
        title_font_size = title_font_size if title_font_size is not None else auto_title
        axis_title_font_size = axis_title_font_size if axis_title_font_size is not None else auto_axis_title
        x_tick_font_size = x_tick_font_size if x_tick_font_size is not None else auto_x_tick
        y_tick_font_size = y_tick_font_size if y_tick_font_size is not None else auto_y_tick
    else:
        height = height if height is not None else fallback_height
        base_font_size = base_font_size if base_font_size is not None else fallback_base
        title_font_size = title_font_size if title_font_size is not None else fallback_title
        axis_title_font_size = axis_title_font_size if axis_title_font_size is not None else fallback_axis_title
        x_tick_font_size = x_tick_font_size if x_tick_font_size is not None else fallback_x_tick
        y_tick_font_size = y_tick_font_size if y_tick_font_size is not None else fallback_y_tick

    # Axis dicts
    xaxis = dict(
        tickfont=dict(size=x_tick_font_size),
        title=dict(font=dict(size=axis_title_font_size)),
    )
    if xaxis_title is not None:
        xaxis["title"]["text"] = xaxis_title

    yaxis = dict(
        tickfont=dict(size=y_tick_font_size),
        title=dict(font=dict(size=axis_title_font_size)),
    )
    if yaxis_title is not None:
        yaxis["title"]["text"] = yaxis_title
    if reverse_y:
        yaxis["autorange"] = "reversed"

    fig.update_layout(
        autosize=False,  # ensure width/height are respected
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font=dict(size=title_font_size),
        ),
        template="simple_white",
        width=width,
        height=height,
        font=dict(size=base_font_size),
        xaxis=xaxis,
        yaxis=yaxis,
        margin=margin,
    )
    return fig

def _clamp(x: int, lo: int, hi: int) -> int:
    """Clamp *x* to the closed interval [*lo*, *hi*].

    Parameters
    ----------
    x : int
        Value to clamp.
    lo : int
        Lower bound (inclusive).
    hi : int
        Upper bound (inclusive).

    Returns
    -------
    int
        ``max(lo, min(hi, x))``.
    """
    return max(lo, min(hi, x))

def _plotly_scale_from_dpi(dpi: int | None) -> float:
    """Convert a DPI value to a Plotly image-export scale factor.

    Parameters
    ----------
    dpi : int or None
        Target dots-per-inch.  If ``None``, a default scale of 1.5 is used.

    Returns
    -------
    float
        Scale factor (≥ 0.1) relative to the default 96 DPI baseline.
    """
    if dpi is None:
        return 1.5
    return max(float(dpi) / 96.0, 0.1)

def stability_jaccard(feature_sets: List[set]) -> float:
    """Average pairwise Jaccard index across feature sets.

    Parameters
    ----------
    feature_sets : list of set
        Each element is the set of feature names selected in one fold.

    Returns
    -------
    float
        Mean pairwise Jaccard index.  Returns 0.0 if fewer than 2
        non-empty sets exist.
    """
    non_empty = [s for s in feature_sets if len(s) > 0]
    if len(non_empty) < 2:
        return 0.0
    scores = []
    for si, sj in combinations(non_empty, 2):
        union = si | sj
        if len(union) == 0:
            continue
        scores.append(len(si & sj) / len(union))
    return float(np.mean(scores)) if scores else 0.0

def stability_nogueira(
    feature_sets: List[set], n_total_features: int
) -> float:
    """Nogueira et al. (2018) global stability index.

    Parameters
    ----------
    feature_sets : list of set
        Each element is the set of feature names selected in one fold.
    n_total_features : int
        Total number of candidate features (``len(X.columns)``).

    Returns
    -------
    float
        Nogueira stability index rescaled from [-1, 1] to [0, 1]
        via (raw + 1) / 2.  Returns 0.0 if fewer than 2 folds, or if
        k_bar is 0 or n (degenerate case).
    """
    m = len(feature_sets)
    n = n_total_features
    if m < 2 or n == 0:
        return 0.0

    # Build binary selection matrix Z (m x n)
    all_features = sorted(set().union(*feature_sets))
    feat_to_idx = {f: i for i, f in enumerate(all_features)}
    Z = np.zeros((m, n), dtype=np.float64)
    for i, s in enumerate(feature_sets):
        for f in s:
            Z[i, feat_to_idx[f]] = 1.0

    hatPF = np.mean(Z, axis=0)
    k_bar = np.sum(hatPF)
    if k_bar < 1e-12 or abs(k_bar - n) < 1e-12:
        return 0.0

    denom = (k_bar / n) * (1.0 - k_bar / n)
    if abs(denom) < 1e-12:
        return 0.0

    raw = float(1.0 - (m / (m - 1)) * np.mean(hatPF * (1.0 - hatPF)) / denom)
    return max(raw, 0.0)

def _estimate_label_width_px(labels: Sequence[str]) -> int:
    """Estimate the pixel width needed to display the longest label.

    Uses a rough heuristic of 8 px per character plus a fixed padding of
    220 px for axis ticks and margins.

    Parameters
    ----------
    labels : Sequence[str]
        Label strings to measure.

    Returns
    -------
    int
        Estimated width in pixels (0 if *labels* is empty).
    """
    if not labels:
        return 0
    max_chars = max(len(str(l)) for l in labels)
    return max_chars * 8 + 220

def clinical_binary_metrics(y_true, y_pred, y_proba):
    """Evaluate 0=oligo, 1=poly; retain binary F1 separately from macro-F1."""
    y_true, y_pred, y_proba = map(np.asarray, (y_true, y_pred, y_proba))
    if not (y_true.ndim == y_pred.ndim == y_proba.ndim == 1):
        raise ValueError("Expected one-dimensional prediction arrays.")
    if not len(y_true) or not (len(y_true) == len(y_pred) == len(y_proba)):
        raise ValueError("Prediction arrays must be non-empty and have equal lengths.")
    if set(np.unique(y_true)) != {0, 1} or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Both true classes and binary predictions are required.")
    if not np.isfinite(y_proba).all() or ((y_proba < 0) | (y_proba > 1)).any():
        raise ValueError("Positive-class probabilities must be finite and in [0, 1].")
    return dict(
        auc_score=roc_auc_score(y_true, y_proba),
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        f1_score=f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
        precision=precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        sensitivity=recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        specificity=recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        average_precision=average_precision_score(y_true, y_proba),
        brier_score=brier_score_loss(y_true, y_proba),
    )


def clinical_outer_splits(outer_cv, X, y):
    """Materialise splits and require exactly one held-out prediction per row."""
    splits = list(outer_cv.split(X, y) if hasattr(outer_cv, "split") else outer_cv)
    seen = np.zeros(len(X), dtype=int)
    universe = np.arange(len(X))
    for train, test in splits:
        train, test = np.asarray(train), np.asarray(test)
        for indices in (train, test):
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("Fold indices must be one-dimensional integers.")
            if not len(indices) or len(np.unique(indices)) != len(indices):
                raise ValueError("Empty fold or duplicate indices.")
            if ((indices < 0) | (indices >= len(X))).any():
                raise ValueError("Fold indices outside cohort.")
        if np.intersect1d(train, test).size or not np.array_equal(np.sort(np.r_[train, test]), universe):
            raise ValueError("Train/test must partition the cohort without overlap.")
        seen[test] += 1
    if not splits or not (seen == 1).all():
        raise ValueError("Every patient must appear in exactly one outer-test fold.")
    return splits


def validate_clinical_oof(predictions_df, model_name, cohort, outer_splits):
    """Validate patient/sample coverage, targets and frozen outer-fold membership."""
    keys = ["record_id", "sample_id"]
    required = keys + ["y_true", "y_pred", "y_proba", "outer_fold"]
    rows = predictions_df.loc[predictions_df.model == model_name].copy()
    if rows.empty or rows[required].isna().any().any():
        raise ValueError("Missing OOF rows or prediction fields.")
    if "dataset" in rows and not rows.dataset.eq("test").all():
        raise ValueError("Only outer-test predictions may enter pooled OOF evaluation.")
    for table in (rows, cohort):
        if table[keys].isna().any().any() or any(table[k].duplicated().any() for k in keys):
            raise ValueError("Expected one row per patient and per sample.")
    if len(rows) != len(cohort):
        raise ValueError("OOF coverage differs from the complete cohort.")
    expected = cohort[keys + ["Polymetastatic"]].reset_index(drop=True).copy()
    expected["expected_fold"] = 0
    for fold, (_, test) in enumerate(clinical_outer_splits(outer_splits, cohort, cohort.Polymetastatic), 1):
        expected.loc[test, "expected_fold"] = fold
    merged = expected.merge(rows, on=keys, how="left", validate="one_to_one", sort=False)
    if merged[required].isna().any().any():
        raise ValueError("OOF identifiers do not match the source cohort.")
    if not (merged.y_true == merged.Polymetastatic).all():
        raise ValueError("OOF targets do not match source targets.")
    if not (merged.outer_fold == merged.expected_fold).all():
        raise ValueError("OOF rows do not match the frozen outer-test splits.")
    clinical_binary_metrics(merged.y_true, merged.y_pred, merged.y_proba)
    return merged[rows.columns].reset_index(drop=True)


def rank_clinical_inner_scores(results_df, expected_models, n_outer_folds=5):
    """Rank only inner scores; alphabetical tie-break gives one winner per fold."""
    ranks = results_df[["outer_fold", "model", "best_inner_score"]].copy()
    expected = pd.MultiIndex.from_product([range(1, n_outer_folds + 1), list(expected_models)])
    actual = pd.MultiIndex.from_frame(ranks[["outer_fold", "model"]])
    if actual.has_duplicates or set(actual) != set(expected):
        raise ValueError("Expected one score per candidate in every outer fold.")
    if not np.isfinite(ranks.best_inner_score).all():
        raise ValueError("Inner-CV scores must be finite.")
    ranks = ranks.sort_values(["outer_fold", "best_inner_score", "model"], ascending=[True, False, True])
    ranks["rank"] = ranks.groupby("outer_fold").cumcount() + 1
    ranks["is_winner"] = ranks["rank"].eq(1)
    wins = ranks.loc[ranks.is_winner, "model"].value_counts().reindex(expected_models, fill_value=0)
    return ranks.reset_index(drop=True), wins.rename("outer_fold_wins").rename_axis("model").to_frame()


def clinical_flaml_settings(time_budget, scoring, random_state=42, n_jobs=1):
    """Shared protocol for outer-training and full-cohort clinical FLAML."""
    if time_budget is None or not np.isfinite(time_budget) or time_budget <= 0:
        raise ValueError("Set an explicit positive FLAML time budget in seconds.")
    if scoring != "f1_macro":
        raise ValueError("Clinical FLAML aligns to manual scoring='f1_macro'.")
    return dict(task="classification", time_budget=time_budget, eval_method="cv",
                split_type=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
                n_splits=5, metric="macro_f1", seed=random_state, n_jobs=n_jobs,
                retrain_full=True, auto_augment=False, skip_transform=True, verbose=0)


def _clinical_heatmap_text_colour(value, maximum):
    """Choose black/white by contrast against the actual Plotly Blues colour."""
    from plotly.colors import sample_colorscale
    rgb = sample_colorscale("Blues", [float(value) / max(float(maximum), 1.0)])[0]
    components = np.array([float(v) for v in rgb[4:-1].split(",")]) / 255.0
    linear = np.where(components <= 0.04045, components / 12.92, ((components + 0.055) / 1.055) ** 2.4)
    luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
    return "black" if (luminance + 0.05) / 0.05 >= 1.05 / (luminance + 0.05) else "white"


def plot_per_fold_lines(
    wide_df: "pd.DataFrame",
    title: str = "Metric per outer fold",
    xaxis_title: str = "Outer fold index",
    yaxis_title: str = "Score",
    colors: Optional[dict[str, str]] = None,
    interactive: bool = True,
    dpi: int = 144,
    width: int = 900,
    height: int = 350,
    margin: dict = dict(l=50, r=20, t=40, b=40),
    marker_size: int = 4,
    line_width: int = 1,
    base_font_size: int = 9,
    title_font_size: int = 11,
    legend_font_size: Optional[int] = None,
    save_plot_path: Optional[str] = None,
) -> "go.Figure":
    """Per-fold line chart comparing a metric across model variants.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Wide-format DataFrame where the index is the fold number and each
        column is a model variant (e.g. ``"Full Pipeline"``, ``"LR-only"``).
    title : str
        Plot title.
    xaxis_title, yaxis_title : str
        Axis labels.
    colors : dict, optional
        ``{variant_name: hex_color}``. Unmatched variants use Plotly defaults.
    interactive : bool
        If True shows interactive Plotly; otherwise renders as static PNG.
    dpi : int
        DPI for static PNG rendering.
    width, height : int
        Figure dimensions in pixels.
    margin : dict
        Plotly margin dict.
    marker_size : int
        Marker size for data points.
    line_width : int
        Line width.
    base_font_size : int
        Base font size for the figure.
    title_font_size : int
        Title font size.
    legend_font_size : int, optional
        Legend font size override.
    save_plot_path : str, optional
        If provided, saves the figure to this path.

    Returns
    -------
    go.Figure
    """
    if colors is None:
        colors = {}

    fig = go.Figure()
    for label in wide_df.columns:
        fig.add_trace(go.Scatter(
            x=wide_df.index,
            y=wide_df[label],
            mode="lines+markers",
            name=label,
            marker=dict(size=marker_size, color=colors.get(label)),
            line=dict(width=line_width, color=colors.get(label)),
        ))

    legend_kw = dict(orientation="h", y=-0.25)
    if legend_font_size is not None:
        legend_kw["font"] = dict(size=legend_font_size)

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=width,
        height=height,
        margin=margin,
        legend=legend_kw,
        title_font_size=title_font_size,
        font=dict(size=base_font_size),
        template="plotly_white",
    )

    if interactive:
        fig.show()
    else:
        fig.show(
            renderer="png",
            width=fig.layout.width,
            height=fig.layout.height,
            scale=_plotly_scale_from_dpi(dpi),
        )

    if save_plot_path:
        try:
            output_dir = os.path.dirname(save_plot_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            file_ext = os.path.splitext(save_plot_path)[1].lower().replace(".", "")
            if file_ext not in ["png", "jpeg", "webp", "svg", "pdf"]:
                save_plot_path = os.path.splitext(save_plot_path)[0] + ".png"
                file_ext = "png"
            fig.write_image(
                save_plot_path,
                width=fig.layout.width,
                height=fig.layout.height,
                format=file_ext,
                scale=_plotly_scale_from_dpi(dpi),
            )
            print(f"Plot saved to: {save_plot_path}")
        except ImportError:
            print("Error: 'kaleido' not installed. Run `pip install kaleido` to enable saving.")
        except Exception as e:
            print(f"An error occurred while saving the plot: {e}")

    return fig


def mutation_outer_splits(outer_cv, X, y):
    """Freeze and validate the mutation experiment's five outer partitions."""
    splits = clinical_outer_splits(outer_cv, X, y)
    if len(splits) != 5:
        raise ValueError("Mutation evaluation requires five outer folds.")
    return splits


def validate_mutation_oof(predictions_df, model_name, cohort, outer_splits):
    """Require one held-out prediction per unique sample, correct labels and folds.

    sample_id is the subject identifier available in this mutation input;
    duplicate identifiers are rejected rather than pooled as distinct subjects.
    Legacy prediction files without dataset/row_position are checked against
    the cohort and frozen outer-test membership in the same way as new files.
    """
    expected = cohort[["sample_id", "target"]].reset_index(drop=True).copy()
    if expected.isna().any().any() or expected.sample_id.duplicated().any():
        raise ValueError("Expected one nonmissing sample_id and target per subject.")
    required = ["sample_id", "outer_fold", "y_true", "y_pred", "y_proba"]
    rows = predictions_df.loc[predictions_df["model"] == model_name].copy()
    if rows.empty or rows[required].isna().any().any():
        raise ValueError("Missing OOF rows or prediction fields.")
    if rows.sample_id.duplicated().any() or len(rows) != len(expected):
        raise ValueError("Expected exactly one OOF prediction per subject and model.")
    if "dataset" in rows and not rows.dataset.eq("test").all():
        raise ValueError("Only outer-test predictions may enter OOF reporting.")
    expected["expected_fold"] = 0
    expected["expected_position"] = np.arange(len(expected))
    for fold, (_, test) in enumerate(mutation_outer_splits(outer_splits, cohort, cohort.target), 1):
        expected.loc[test, "expected_fold"] = fold
    merged = expected.merge(rows, on="sample_id", how="left", validate="one_to_one", sort=False)
    if merged[required].isna().any().any():
        raise ValueError("OOF identifiers differ from the mutation cohort.")
    if not merged.y_true.eq(merged.target).all():
        raise ValueError("OOF labels differ from cohort targets.")
    if not merged.outer_fold.eq(merged.expected_fold).all():
        raise ValueError("OOF predictions do not match the frozen outer-test folds.")
    if "row_position" in rows and not merged.row_position.eq(merged.expected_position).all():
        raise ValueError("OOF row positions differ from the source cohort.")
    clinical_binary_metrics(merged.y_true, merged.y_pred, merged.y_proba)
    return merged[rows.columns].reset_index(drop=True)


def mutation_pooled_oof_metrics(validated_oof):
    """Pooled binary metrics with mutation notebook column names."""
    m = clinical_binary_metrics(validated_oof.y_true, validated_oof.y_pred, validated_oof.y_proba)
    return {"roc_auc": m["auc_score"], "balanced_accuracy": m["balanced_accuracy"],
            "f1_macro": m["macro_f1"], "precision": m["precision"],
            "recall": m["sensitivity"], "specificity": m["specificity"],
            "average_precision": m["average_precision"], "brier_score": m["brier_score"]}


def rank_mutation_inner_scores(results_df, expected_models, n_outer_folds=5):
    """Fold ranks/wins using inner scores only, with alphabetical tie-breaking."""
    return rank_clinical_inner_scores(results_df, expected_models, n_outer_folds)


def mutation_flaml_settings(time_budget, random_state=42, n_jobs=1):
    """Share the thesis five-fold macro-F1 FLAML protocol and require a budget."""
    return clinical_flaml_settings(time_budget, "f1_macro", random_state, n_jobs)


def tune_full_cohort_flaml_mutation(X, y, time_budget, random_state=42, n_jobs=1):
    """Fresh FLAML search and full-data refit, exclusively for interpretation."""
    settings = mutation_flaml_settings(time_budget, random_state, n_jobs)
    from flaml import AutoML
    automl = AutoML()
    automl.fit(X_train=X, y_train=y, **settings)
    return automl


def extract_fitted_flaml_gene_rankings_mutation(automl, feature_names, top_n=20):
    """Extract gene scores from an already-fitted AutoML model; never refit it."""
    estimator = automl.model.estimator
    if hasattr(estimator, "feature_importances_"):
        scores = np.asarray(estimator.feature_importances_).ravel()
        column = "importance"
    elif hasattr(estimator, "coef_"):
        scores = np.asarray(estimator.coef_).ravel()
        column = "coefficient"
    else:
        raise ValueError("Selected FLAML estimator exposes no gene importance or coefficient.")
    if len(scores) != len(feature_names):
        raise ValueError("FLAML feature scores do not align with the mutation input columns.")
    frame = pd.DataFrame({"gene": feature_names, column: scores})
    order = np.argsort(-np.abs(scores), kind="stable")
    frame = frame.iloc[order].reset_index(drop=True)
    frame = annotate_genes_with_symbols(frame, verbose=False, fill_unmapped=False)
    return frame.head(top_n)


# Transcriptomics final benchmark: isolated from the clinical/mutation protocols.
def build_gene_level_tpm_classifiers(random_state=42, n_features=None):
    """Existing gene-level grids with mandatory, unchanged SelectKBest choices."""
    classifiers = build_representation_classifiers(
        random_state=random_state, n_features=n_features, use_feature_selection=True
    )
    for config in classifiers.values():
        selectors = [s for s in config["param_grid"]["selector"] if isinstance(s, SelectKBest)]
        if not selectors:
            raise ValueError("No existing SelectKBest candidates are available for this gene matrix.")
        config["param_grid"]["selector"] = selectors
        config["pipeline"].set_params(selector=clone(selectors[0]))
    return classifiers


def tpm_outer_splits(outer_cv, X, y):
    """Freeze five exhaustive, disjoint outer-test partitions."""
    splits = clinical_outer_splits(outer_cv, X, y)
    if len(splits) != 5:
        raise ValueError("Transcriptomics requires five outer folds.")
    return splits


def validate_tpm_representation_alignment(metadata, y, reference_metadata, reference_y):
    """Require identical patient order and targets before reusing frozen indices."""
    ids = metadata["sample_id_normalised"].reset_index(drop=True)
    expected = reference_metadata["sample_id_normalised"].reset_index(drop=True)
    if ids.isna().any() or ids.duplicated().any() or expected.isna().any() or expected.duplicated().any():
        raise ValueError("Expected one unique, nonmissing transcriptomics sample ID per patient.")
    if not ids.equals(expected) or not np.array_equal(np.asarray(y), np.asarray(reference_y)):
        raise ValueError("Representation patients/order/targets differ from the frozen transcriptomics cohort.")


def validate_tpm_oof(predictions_df, name, metadata, y, outer_splits):
    """Validate complete patient coverage, labels, probabilities and outer folds."""
    frame = predictions_df.loc[predictions_df.representation == name].copy()
    frame = frame.rename(columns={"sample_id_normalised": "sample_id", "y_prob_poly": "y_proba",
                                  "sample_index": "row_position"})
    frame["model"] = name
    cohort = pd.DataFrame({"sample_id": metadata.sample_id_normalised.to_numpy(), "target": np.asarray(y)})
    validate_mutation_oof(frame, name, cohort, tpm_outer_splits(outer_splits, cohort, y))
    if "best_inner_score" in frame and not np.isfinite(frame.best_inner_score).all():
        raise ValueError("Non-finite OOF inner-CV score.")
    return predictions_df.loc[predictions_df.representation == name].copy()


def tpm_pooled_oof_metrics(validated_oof):
    """Eight binary metrics from concatenated held-out transcriptomics rows."""
    return mutation_pooled_oof_metrics(validated_oof.rename(columns={"y_prob_poly": "y_proba"}))


def rank_tpm_inner_scores(results_df, combinations):
    """Rank all combinations only by each outer fold's inner score."""
    ranks, wins = rank_clinical_inner_scores(results_df.rename(columns={"representation": "model"}), combinations, 5)
    return ranks.rename(columns={"model": "representation"}), wins.rename_axis("representation")


def extract_fitted_tpm_feature_rankings(pipeline, feature_names):
    """Extract scores without fitting; track PCA, Tybalt and selector names in order."""
    names = np.asarray(feature_names, dtype=object)
    for step_name, step in pipeline.steps[:-1]:
        if step is None or isinstance(step, str):
            continue
        if step_name == "pca":
            names = np.asarray([f"PC{i+1}" for i in range(step.n_components_)], dtype=object)
        elif step_name == "tybalt":
            names = np.asarray([f"Tybalt_latent_{i+1}" for i in range(int(step.encoder_.output_shape[-1]))], dtype=object)
        elif hasattr(step, "get_support"):
            mask = step.get_support()
            if len(mask) != len(names):
                raise ValueError("Selector mask and incoming feature names differ.")
            names = names[mask]
    estimator = pipeline.steps[-1][1]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_).ravel()
        column = "importance"
    elif hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_).ravel()
        column = "coefficient"
    else:
        raise ValueError("Fitted estimator exposes no importances or coefficients.")
    if len(names) != len(values):
        raise ValueError("Feature names do not match the fitted classifier scores.")
    frame = pd.DataFrame({"feature": names, column: values, "feature_score": np.abs(values)})
    frame = frame.sort_values("feature_score", ascending=False, kind="stable").reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame


def evaluate_tpm_representation(X, y, representation_base_name, metadata_df, outer_cv, inner_cv,
                                scoring, output_dir, classifiers, registry, reference_metadata,
                                reference_y, run_nested_cv=False, n_jobs=-1):
    """Register unchanged pipelines; fit reviewed CV or load a complete saved bundle."""
    validate_tpm_representation_alignment(metadata_df, y, reference_metadata, reference_y)
    if len(X) != len(metadata_df):
        raise ValueError("Representation matrix and metadata lengths differ.")
    splits = tpm_outer_splits(outer_cv, X, y)
    results, predictions = {}, {}
    names = [f"{representation_base_name}_{family}" for family in classifiers]
    reviewed = [Path(output_dir) / f"transcriptomics_{name}_nested_cv_{kind}_reviewed.csv"
                for name in names for kind in ["results", "predictions"]]
    if not run_nested_cv and any(p.exists() for p in reviewed) and not all(p.exists() for p in reviewed):
        raise ValueError("Incomplete reviewed result bundle; do not mix old and new results.")
    use_reviewed = run_nested_cv or all(p.exists() for p in reviewed)
    for family, config in classifiers.items():
        name = f"{representation_base_name}_{family}"
        registry[name] = dict(X=X, y=y, metadata=metadata_df, pipeline=config["pipeline"],
                              param_grid=config["param_grid"], family=family, representation=representation_base_name,
                              reviewed=use_reviewed)
        if run_nested_cv:
            result, pred = run_nested_cv_for_tpm(X, y, name, config["pipeline"], config["param_grid"],
                splits, inner_cv, scoring, n_jobs=n_jobs, metadata_df=metadata_df, output_dir=output_dir)
        else:
            if use_reviewed:
                paths = [Path(output_dir) / f"transcriptomics_{name}_nested_cv_{kind}_reviewed.csv"
                         for kind in ["results", "predictions"]]
            else:
                paths = [Path(output_dir) / f"{name}_nested_cv_{kind}.csv" for kind in ["results", "predictions"]]
                print(f"Loading historical {name}; not evidence for mandatory-selection rerun.")
            result, pred = [pd.read_csv(p) for p in paths]
            # Enrich legacy provenance from the corresponding existing fold table, never refit.
            for field in ["best_inner_score", "best_params"]:
                if field not in pred:
                    pred = pred.merge(result[["outer_fold", field]], on="outer_fold", how="left", validate="many_to_one")
        pred = validate_tpm_oof(pred, name, metadata_df, y, splits)
        results[name], predictions[name] = result, pred
    return results, predictions


def tune_full_cohort_tpm(config, random_state=42, n_jobs=-1):
    """Fresh five-fold macro-F1 search of the registered selected combination only."""
    search = GridSearchCV(clone(config["pipeline"]), config["param_grid"], scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state), refit=True,
        n_jobs=n_jobs, error_score="raise")
    return search.fit(config["X"], config["y"])


def summarise_tpm_gene_stability(rankings, n_outer_folds=5):
    """Top-20 recurrence; absence after a fold's selector counts as no top appearance."""
    counts = rankings.groupby("feature").agg(
        n_folds_observed=("outer_fold", "nunique"), n_top_folds=("top_in_fold", "sum"),
        mean_score_when_retained=("feature_score", "mean"), best_rank=("rank", "min"))
    counts["top_fold_fraction"] = counts.n_top_folds / n_outer_folds
    return counts.sort_values(["n_top_folds", "mean_score_when_retained"], ascending=False).reset_index()


def tpm_flaml_settings(time_budget, random_state=42, n_jobs=1):
    """Same frozen-cohort protocol; preprocessing lives inside registered learners."""
    settings = clinical_flaml_settings(time_budget, "f1_macro", random_state, n_jobs)
    settings["sample"] = False
    return settings


def make_tpm_flaml_learner(config, learner_name):
    """Custom FLAML learner fits a complete gene-level pipeline inside each inner fold.

    The search domains are exactly the existing mandatory-selection gene-level
    classifier grids. Selectors and samplers are represented by integer choices
    for serialisation, and cloned anew for every fit. No full-cohort selector is
    fitted before FLAML's CV. A fit already in progress may exceed its time budget.
    """
    from flaml import tune
    from flaml.automl.model import SKLearnEstimator
    template = config["pipeline"]
    grid = config["param_grid"]
    if isinstance(grid, list):
        return make_conditional_tpm_flaml_learner(config, learner_name)
    domains = {key: list(values) for key, values in grid.items()}

    class TpmPipelineLearner(SKLearnEstimator):
        @classmethod
        def search_space(cls, data_size, task=None, **kwargs):
            return {key: {"domain": tune.choice(list(range(len(values)))) if key in ["selector", "sampler"]
                          else tune.choice(values), "init_value": 0 if key in ["selector", "sampler"] else values[0]}
                    for key, values in domains.items()}

        def fit(self, X_train, y_train, budget=None, free_mem_ratio=0, **kwargs):
            started = time.time()
            params = {}
            for key, values in domains.items():
                value = self.params.get(key, 0 if key in ["selector", "sampler"] else values[0])
                if key in ["selector", "sampler"]:
                    value = values[int(value)]
                    value = clone(value) if not isinstance(value, str) else value
                params[key] = value
            self._model = clone(template).set_params(**params)
            self._model.fit(X_train, y_train)
            return time.time() - started

    TpmPipelineLearner.__name__ = learner_name
    return TpmPipelineLearner


def fit_tpm_flaml(X, y, time_budget, random_state=42):
    """Fresh AutoML search/refit with fold-local gene selection for all learners."""
    settings = tpm_flaml_settings(time_budget, random_state)
    from flaml import AutoML
    configs = build_gene_level_tpm_classifiers(random_state, X.shape[1])
    automl = AutoML()
    names = []
    for family, config in configs.items():
        name = f"tpm_{family}"
        automl.add_learner(learner_name=name, learner_class=make_tpm_flaml_learner(config, name))
        names.append(name)
    automl.fit(X_train=X, y_train=y, estimator_list=names, **settings)
    return automl


def run_nested_cv_with_flaml_tpm(X, y, metadata, outer_splits, time_budget, random_state=42):
    """Five-fold outer evaluation of the inner-CV-safe gene-level AutoML comparator."""
    tpm_flaml_settings(time_budget, random_state)
    validate_tpm_representation_alignment(metadata, y, metadata, y)
    tests, trains, predictions = [], [], []
    for fold, (train, test) in enumerate(tpm_outer_splits(outer_splits, X, y), 1):
        automl = fit_tpm_flaml(X.iloc[train], y.iloc[train], time_budget, random_state)
        provenance = dict(representation="flaml_automl", outer_fold=fold, best_inner_score=1-automl.best_loss,
            best_params=json.dumps({"best_estimator": automl.best_estimator, "best_config": automl.best_config}, default=str),
            selection_metric="macro_f1", inner_n_splits=5, time_budget_seconds=time_budget)
        for indices, destination, dataset in [(train, trains, "train"), (test, tests, "test")]:
            pred = automl.predict(X.iloc[indices])
            probability = get_positive_class_probability(automl, X.iloc[indices])
            frame = pd.DataFrame({"y_true": y.iloc[indices].to_numpy(), "y_pred": pred, "y_prob_poly": probability})
            destination.append({**provenance, "dataset": dataset, **tpm_pooled_oof_metrics(frame)})
            if dataset == "test":
                for key, value in provenance.items(): frame[key] = value
                frame["sample_id_normalised"] = metadata.iloc[indices].sample_id_normalised.to_numpy()
                frame["sample_index"] = indices
                frame["dataset"] = "test"
                predictions.append(frame)
    oof = pd.concat(predictions, ignore_index=True)
    validate_tpm_oof(oof, "flaml_automl", metadata, y, outer_splits)
    return pd.DataFrame(tests), pd.DataFrame(trains), oof


def run_nested_cv_for_tpm(
    X: pd.DataFrame,
    y: pd.Series,
    representation_name: str,
    pipeline: Pipeline,
    param_grid: dict,
    outer_cv,
    inner_cv,
    scoring: str,
    n_jobs: int = -1,
    metadata_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reviewed TPM CV; retain outer-fit gene rankings without any additional fit."""

    if scoring != "f1_macro":
        raise ValueError("Transcriptomics selection requires f1_macro.")
    validate_tpm_representation_alignment(metadata_df, y, metadata_df, y)
    gene_rankings = []
    fold_results = []
    prediction_rows = []

    for outer_fold, (train_idx, test_idx) in enumerate(
        tpm_outer_splits(outer_cv, X, y),
        start=1,
    ):
        print(f"\n{representation_name} | Outer fold {outer_fold}")

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        grid_search = GridSearchCV(
            estimator=clone(pipeline),
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
            return_train_score=False
        )

        grid_search.fit(X_train, y_train)

        best_estimator = grid_search.best_estimator_
        if representation_name.startswith("gene_level_"):
            ranking = extract_fitted_tpm_feature_rankings(best_estimator, X.columns)
            ranking["representation"] = representation_name
            ranking["outer_fold"] = outer_fold
            ranking["top_in_fold"] = ranking["rank"].le(20)
            gene_rankings.append(ranking)

        y_pred_train = best_estimator.predict(X_train)
        y_prob_train = get_positive_class_probability(
            best_estimator,
            X_train,
        )

        y_pred_test = best_estimator.predict(X_test)
        y_prob_test = get_positive_class_probability(
            best_estimator,
            X_test,
        )

        train_metrics = calculate_binary_metrics(
            y_train,
            y_pred_train,
            y_prob_train,
        )

        test_metrics = calculate_binary_metrics(
            y_test,
            y_pred_test,
            y_prob_test,
        )

        row = {
            "representation": representation_name,
            "outer_fold": outer_fold,
            "best_inner_score": grid_search.best_score_,
            "best_params": json.dumps(grid_search.best_params_, default=str),
        }

        row.update({
            f"train_{key}": value
            for key, value in train_metrics.items()
        })

        row.update({
            f"test_{key}": value
            for key, value in test_metrics.items()
        })

        fold_results.append(row)

        for sample_position, sample_index in enumerate(test_idx):
            prediction_row = {
                "representation": representation_name,
                "outer_fold": outer_fold,
                "sample_index": sample_index,
                "dataset": "test",
                "best_inner_score": grid_search.best_score_,
                "best_params": json.dumps(grid_search.best_params_, default=str),
                "y_true": int(y_test.iloc[sample_position]),
                "y_pred": int(y_pred_test[sample_position]),
                "y_prob_poly": float(y_prob_test[sample_position]),
            }

            if (
                metadata_df is not None
                and "sample_id_normalised" in metadata_df.columns
            ):
                prediction_row["sample_id_normalised"] = metadata_df.iloc[
                    sample_index
                ]["sample_id_normalised"]

            prediction_rows.append(prediction_row)

        print("Best inner score:")
        print(grid_search.best_score_)

        print("\nBest params:")
        print(grid_search.best_params_)

        print("\nTest metrics:")
        print(test_metrics)

    results_df = pd.DataFrame(fold_results)
    predictions_df = pd.DataFrame(prediction_rows)

    validate_tpm_oof(predictions_df, representation_name, metadata_df, y, outer_cv)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        results_df.to_csv(
            os.path.join(
                output_dir,
                f"transcriptomics_{representation_name}_nested_cv_results_reviewed.csv",
            ),
            index=False,
        )

        predictions_df.to_csv(
            os.path.join(
                output_dir,
                f"transcriptomics_{representation_name}_nested_cv_predictions_reviewed.csv",
            ),
            index=False,
        )

    if output_dir is not None and gene_rankings:
        pd.concat(gene_rankings, ignore_index=True).to_csv(
            Path(output_dir) / f"transcriptomics_{representation_name}_outer_fold_feature_rankings_reviewed.csv", index=False)
    return results_df, predictions_df

# cBioPortal METABRIC revision: isolated additions, no changes to running TPM helpers.
def normalise_metabric_gene_identifier(value):
    """Normalise namespace syntax, preserving dots in ordinary gene symbols."""
    import re
    if pd.isna(value):
        return None
    token = str(value).strip().removeprefix("transcriptomics__")
    if token.lower() in {"", "nan", "na", "none"}:
        return None
    if re.fullmatch(r"ENSG\d+\.\d+", token):
        return token.rsplit(".", 1)[0]
    if re.fullmatch(r"\d+\.0", token):
        return token[:-2]
    return token


def audit_metabric_identifiers(feature_names, identifiers, mappings):
    """Resolve local identifiers and count duplicate groups without aggregating values."""
    import re
    if len(identifiers) != len(feature_names):
        raise ValueError("One identifier record is required per expression column.")
    aliases = {}
    for row in mappings.to_dict("records"):
        symbol = normalise_metabric_gene_identifier(row.get("gene_name"))
        if symbol is None or re.fullmatch(r"ENSG\d+|\d+", symbol):
            continue
        for col in ["gene_name", "ensembl_gene_id", "entrez_gene_id"]:
            token = normalise_metabric_gene_identifier(row.get(col))
            if token is not None:
                aliases.setdefault(token, set()).add(symbol)
    rows = []
    for position, row in enumerate(identifiers.to_dict("records")):
        tokens = [normalise_metabric_gene_identifier(row.get(c)) for c in ["gene_name", "ensembl_gene_id", "entrez_gene_id"]]
        tokens = [t for t in tokens if t is not None]
        symbols = set().union(*(aliases.get(t, set()) for t in tokens)) if tokens else set()
        symbol = next(iter(symbols)) if len(symbols) == 1 else None
        rows.append(dict(source_position=position, original_identifier=str(feature_names[position]),
            supplied_identifiers=";".join(tokens), gene_symbol=symbol,
            namespace="ensembl" if any(re.fullmatch(r"ENSG\d+", t) for t in tokens) else "entrez" if any(t.isdigit() for t in tokens) else "symbol",
            status="resolved" if symbol else "ambiguous" if symbols else "unresolved"))
    audit = pd.DataFrame(rows)
    keep = audit.gene_symbol.notna().to_numpy()
    counts = audit.loc[keep].groupby("gene_symbol").size()
    audit["duplicate_group_size"] = audit.gene_symbol.map(counts).fillna(0).astype(int)
    summary = dict(original_genes=len(audit), mapped_records=int(keep.sum()),
        harmonised_genes=len(counts), duplicate_groups=int(counts.gt(1).sum()),
        collapsed_records=int((counts-1).clip(lower=0).sum()),
        unresolved_ids=int(audit.status.eq("unresolved").sum()),
        ambiguous_ids=int(audit.status.eq("ambiguous").sum()))
    return audit, summary


def harmonise_metabric_expression_ids(expression, identifiers, mappings, duplicate_aggregation=None):
    """Harmonise expression only after the duplicate aggregation rule is explicit."""
    audit, summary = audit_metabric_identifiers(expression.columns, identifiers, mappings)
    if summary['duplicate_groups'] and duplicate_aggregation != 'mean':
        raise ValueError("Review duplicate counts and explicitly set duplicate_aggregation='mean' before preparing expression.")
    if duplicate_aggregation not in [None, 'mean']:
        raise ValueError("Supported duplicate-expression aggregation is arithmetic mean on the supplied log scale.")
    keep = audit.gene_symbol.notna().to_numpy()
    numeric = expression.iloc[:, keep].apply(pd.to_numeric, errors="raise").copy()
    numeric.columns = audit.loc[keep, "gene_symbol"].to_numpy()
    result = (numeric.T.groupby(level=0, sort=True).mean().T
              if summary["duplicate_groups"] else numeric.reindex(sorted(numeric.columns), axis=1))
    return result, audit, summary



def validate_cbioportal_metabric_source(expression_path, metadata_path, study_path=None):
    """Validate gene-row native expression and metadata before any transpose/export."""
    import csv
    expression_path, metadata_path = Path(expression_path), Path(metadata_path)
    study_path = Path(study_path) if study_path is not None else metadata_path.with_name('meta_study.txt')
    for path in [expression_path, metadata_path, study_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing canonical cBioPortal source: {path}; no reduced-source fallback.")
    def metadata(path):
        return {k.strip(): v.strip() for k, v in
                (line.split(':', 1) for line in path.read_text().splitlines() if ':' in line)}
    meta, study = metadata(metadata_path), metadata(study_path)
    if (meta.get('cancer_study_identifier') != 'brca_metabric' or study.get('cancer_study_identifier') != 'brca_metabric'
            or meta.get('stable_id') != 'mrna' or meta.get('genetic_alteration_type') != 'MRNA_EXPRESSION'
            or meta.get('datatype') != 'CONTINUOUS' or 'log2' not in meta.get('profile_description','').lower()
            or 'illumina ht-12 v3' not in meta.get('profile_description','').lower()
            or meta.get('data_filename') != expression_path.name):
        raise ValueError('Require matching brca_metabric Illumina HT-12 v3 native log2-intensity metadata.')
    with expression_path.open() as handle:
        header = next(csv.reader(handle, delimiter='\t'))
    if header[:2] != ['Hugo_Symbol', 'Entrez_Gene_Id'] or len(header[2:]) != len(set(header[2:])) or any(not x.strip() for x in header[2:]):
        raise ValueError('Expected gene identifiers followed by unique nonempty sample IDs.')
    raw = pd.read_csv(expression_path, sep='\t', low_memory=False)
    if len(raw) <= 1000:
        raise ValueError('Reduced expression extract is not the canonical full source.')
    numeric = raw.iloc[:, 2:].apply(pd.to_numeric, errors='raise')
    if numeric.empty or np.isinf(numeric.to_numpy()).any() or numeric.isna().all(axis=1).any() or numeric.isna().all(axis=0).any():
        raise ValueError('Empty, infinite or wholly missing expression row/sample.')
    raw.iloc[:, 2:] = numeric
    ids = raw.iloc[:, :2].rename(columns={'Hugo_Symbol':'gene_name', 'Entrez_Gene_Id':'entrez_gene_id'})
    report = dict(gene_rows=len(raw), samples=numeric.shape[1], total_columns=raw.shape[1],
        missing_values=int(numeric.isna().sum().sum()), infinite_values=0,
        constant_gene_rows=int(numeric.nunique(axis=1, dropna=True).le(1).sum()), duplicated_sample_ids=0)
    for column in ids:
        values = ids[column].map(normalise_metabric_gene_identifier)
        counts = values.value_counts()
        report[column] = dict(valid=int(values.notna().sum()), missing=int(values.isna().sum()),
            unique=len(counts), duplicate_groups=int(counts.gt(1).sum()),
            duplicate_excess_rows=int((counts-1).clip(lower=0).sum()))
    meta['study_metadata'] = study
    return raw, ids, meta, report


def load_cbioportal_metabric_expression(expression_path, metadata_path, study_path=None):
    """Transpose only a validated canonical source; preserve missing values for training-only imputation."""
    raw, ids, meta, report = validate_cbioportal_metabric_source(expression_path, metadata_path, study_path)
    expression = raw.iloc[:, 2:].T
    expression.columns = [f"source_row_{i}" for i in range(len(raw))]
    return expression, ids, meta


def audit_cbioportal_metabric_overlap(expression_path, metadata_path, study_path, institutional_expression, gene_mapping):
    """Data-only mapping/overlap audit: no expression aggregation, transpose or export."""
    raw, ids, meta, source_report = validate_cbioportal_metabric_source(expression_path, metadata_path, study_path)
    mappings = pd.concat([gene_mapping, ids], ignore_index=True)
    ext, ext_summary = audit_metabric_identifiers([f'source_row_{i}' for i in range(len(raw))], ids, mappings)
    inst, inst_summary = audit_metabric_identifiers(institutional_expression.columns,
        pd.DataFrame({'ensembl_gene_id':institutional_expression.columns}), mappings)
    common = sorted(set(ext.gene_symbol.dropna()) & set(inst.gene_symbol.dropna()))
    overlap = dict(common_genes=len(common),
        percent_original_institutional=100*len(common)/len(inst),
        percent_mapped_institutional=100*len(common)/inst_summary['harmonised_genes'],
        percent_harmonised_metabric=100*len(common)/ext_summary['harmonised_genes'],
        percent_original_metabric_rows=100*len(common)/len(ext))
    return dict(source=source_report, profile=meta, external_identifiers=ext,
        institutional_identifiers=inst, external_summary=ext_summary,
        institutional_summary=inst_summary, overlap=overlap, common_genes=common)



def metabric_source_fingerprint(paths):
    """Hash explicit source files without deserialising models or pickles."""
    import hashlib
    result = {}
    for path in paths:
        path = Path(path); digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
        result[str(path)] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    return result


def prepare_cbioportal_metabric_transfer(expression_path, metadata_path, institutional_expression,
        gene_mapping, output_dir, source_paths, n_top_genes=1000, random_state=42, validation_fraction=0.2,
        study_path=None, duplicate_aggregation=None):
    """Harmonise before overlap; preserve institutional input and select on external train.

    Institutional input is the unchanged canonical log2(TPM+1) matrix. No
    institutional labels are accepted. External training medians handle missing
    external values; no institutional feature filtering/scaling is fitted here.
    """
    from datetime import datetime, timezone
    import hashlib
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    external, external_ids, profile = load_cbioportal_metabric_expression(expression_path, metadata_path, study_path)
    mappings = pd.concat([gene_mapping, external_ids], ignore_index=True)
    institution_ids = pd.DataFrame({"ensembl_gene_id": institutional_expression.columns})
    ext, ext_audit, ext_summary = harmonise_metabric_expression_ids(external, external_ids, mappings, duplicate_aggregation)
    inst, inst_audit, inst_summary = harmonise_metabric_expression_ids(institutional_expression, institution_ids, mappings, duplicate_aggregation)
    common = sorted(set(ext.columns) & set(inst.columns))
    if len(common) != 16950 or ext.shape[0] != 1980 or inst.shape[0] != 139:
        raise ValueError("Canonical sample/common-gene contract changed.")
    if inst_summary['duplicate_groups']:
        raise ValueError("Institutional genes must not require aggregation.")
    if ext.index.has_duplicates or inst.index.has_duplicates:
        raise ValueError("Duplicate sample identifiers.")
    ordered_feature_sha256 = hashlib.sha256(('\n'.join(common)+'\n').encode()).hexdigest()
    missing_positions = np.argwhere(external.isna().to_numpy())
    missing_rows = []
    for sample_position, gene_position in missing_positions:
        identifier = ext_audit.iloc[gene_position]
        missing_rows.append(dict(sample_id=str(external.index[sample_position]),
            source_row=int(gene_position), supplied_identifiers=identifier.supplied_identifiers,
            gene_symbol=identifier.gene_symbol, mapping_status=identifier.status,
            in_common_space=identifier.gene_symbol in common))
    missing_table = pd.DataFrame(missing_rows)

    summary = pd.DataFrame([dict(dataset="METABRIC", **ext_summary), dict(dataset="institutional", **inst_summary)])
    summary["common_genes_after_harmonisation"] = len(common)
    summary["fraction_institutional_harmonised_genes_shared"] = len(common) / max(1, inst.shape[1])
    summary["fraction_original_institutional_genes_shared"] = len(common) / institutional_expression.shape[1]
    summary.to_csv(out / "metabric_cbioportal_gene_overlap_audit.csv", index=False)
    for name, audit in [("metabric", ext_audit), ("institutional", inst_audit)]:
        audit.to_csv(out / f"{name}_identifier_audit_cbioportal.csv", index=False)
        audit.loc[audit.status.ne("resolved")].to_csv(out / f"{name}_unresolved_identifiers_cbioportal.csv", index=False)
    if len(common) < n_top_genes:
        raise ValueError("Overlap is below the existing encoder gene budget. Review the saved mapping diagnostics before training.")
    train, val = train_test_split(np.arange(len(ext)), test_size=validation_fraction, random_state=random_state)
    medians = ext[common].iloc[train].median()
    filled = ext[common].fillna(medians)
    aligned = inst[common].copy()
    if not np.isfinite(filled.to_numpy()).all() or not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("Non-finite prepared expression; institutional imputation is not permitted.")
    if not filled.columns.equals(aligned.columns) or filled.columns.has_duplicates:
        raise ValueError("Aligned feature identities/order mismatch.")
    missing_table['remaining_missing_after_aggregation'] = [
        bool(pd.isna(ext.loc[row.sample_id, row.gene_symbol])) if row.in_common_space else False
        for row in missing_table.itertuples()]
    missing_table['training_median'] = [medians.get(row.gene_symbol, np.nan) for row in missing_table.itertuples()]
    missing_table.to_csv(out / 'metabric_canonical_missing_values.csv', index=False)
    medians.rename('external_train_median').to_csv(out / 'metabric_canonical_imputation_medians_16950.csv', index_label='gene')
    (out / 'metabric_canonical_ordered_genes_16950.txt').write_text('\n'.join(common)+'\n')
    save_metabric_expression_archive(out / 'metabric_canonical_harmonised_imputed_16950.npz', filled)
    save_metabric_expression_archive(out / 'institutional_canonical_aligned_16950.npz', aligned)

    variances = filled.iloc[train].var().dropna().sort_values(ascending=False, kind="stable")
    variances = variances[variances > 0]
    if len(variances) < n_top_genes:
        raise ValueError("Too few variable genes in external training data; inspect the source.")
    selected = variances.head(n_top_genes).index.tolist()
    selected_genes = pd.DataFrame({"gene": selected, "external_train_variance": variances.loc[selected].to_numpy(),
        "external_train_imputation_median": medians.loc[selected].to_numpy()})
    external_selected, institutional_selected = filled[selected], inst[selected]
    if not np.isfinite(institutional_selected.to_numpy()).all():
        raise ValueError("Nonfinite institutional values; no global data-dependent gene dropping is allowed.")
    scaler_data = scale_external_and_institutional_expression(
        external_selected, institutional_selected, validation_fraction, random_state, 'standard')
    if not scaler_data['external_train_df'].index.equals(external_selected.iloc[train].index):
        raise ValueError("Imputation/selection and scaling training partitions differ.")
    train_selected = external_selected.iloc[train]
    ranges = (train_selected.max() - train_selected.min()).replace(0, 1.0)
    pd.DataFrame({'gene': selected, 'rna_standard_mean': scaler_data['scaler'].mean_,
        'rna_standard_scale': scaler_data['scaler'].scale_, 'hvae_min': train_selected.min().to_numpy(),
        'hvae_range': ranges.to_numpy()}).to_csv(out / 'metabric_canonical_encoder_scaling_parameters.csv', index=False)
    selected_genes.to_csv(out / "metabric_cbioportal_selected_genes.csv", index=False)
    external_selected.to_csv(out / "metabric_cbioportal_selected_log2_expression.csv", index_label="sample_id")
    institutional_selected.to_csv(out / "institutional_cbioportal_aligned_log2_tpm.csv", index_label="sample_id_normalised")
    train_set = set(train)
    pd.DataFrame({"sample_id": ext.index, "partition": ["train" if i in train_set else "validation" for i in range(len(ext))]}).to_csv(out / "metabric_cbioportal_encoder_partitions.csv", index=False)
    provenance = dict(study_id="brca_metabric", profile_id="brca_metabric_mrna", profile_metadata=profile,
        source_url="https://www.cbioportal.org/study/summary?id=brca_metabric", source_files=metabric_source_fingerprint(source_paths),
        audited_at_utc=datetime.now(timezone.utc).isoformat(), download_date="not recorded locally",
        external_samples=len(ext), institutional_samples=len(inst), original_external_genes=external.shape[1],
        original_institutional_genes=institutional_expression.shape[1], common_genes=len(common), selected_genes=len(selected),
        random_state=random_state, validation_fraction=validation_fraction,
        duplicate_aggregation="arithmetic mean on the supplied log-expression scale",
        institutional_input="canonical preprocessed log2(TPM+1), with inherited X-only prevalence-filter caveat")
    provenance.update(ordered_feature_sha256=ordered_feature_sha256,
        selected_feature_sha256=hashlib.sha256(('\n'.join(selected)+'\n').encode()).hexdigest(),
        missing_source_values=len(missing_table),
        missing_common_after_aggregation=int(ext[common].isna().sum().sum()),
        missing_common_after_imputation=int(filled.isna().sum().sum()),
        imputation="per-gene median fitted on METABRIC training partition only",
        train_samples=len(train), validation_samples=len(val),
        harmonised_shape=list(filled.shape), institutional_aligned_shape=list(aligned.shape),
        encoder_input_shape=list(external_selected.shape),
        protocol_version="metabric_canonical_16950_train_median_v1")
    contract = {key: provenance[key] for key in ['source_files', 'ordered_feature_sha256',
        'selected_feature_sha256', 'random_state', 'validation_fraction', 'duplicate_aggregation', 'imputation', 'protocol_version']}
    provenance['preparation_fingerprint'] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    provenance['artifacts'] = metabric_source_fingerprint([
        out / filename for filename in ['metabric_canonical_ordered_genes_16950.txt',
        'metabric_canonical_harmonised_imputed_16950.npz', 'institutional_canonical_aligned_16950.npz',
        'metabric_canonical_missing_values.csv', 'metabric_canonical_imputation_medians_16950.csv',
        'metabric_cbioportal_selected_genes.csv', 'metabric_canonical_encoder_scaling_parameters.csv', 'metabric_cbioportal_selected_log2_expression.csv',
        'institutional_cbioportal_aligned_log2_tpm.csv', 'metabric_cbioportal_encoder_partitions.csv']])
    manifest = out / "metabric_cbioportal_source_provenance.json"
    temporary = manifest.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(provenance, indent=2))
    os.replace(temporary, manifest)
    return dict(external_selected=external_selected, institutional_selected=institutional_selected,
        selected_genes=selected_genes, overlap_summary=summary, provenance=provenance)


def summarise_metabric_encoder_history(history):
    """Recorded diagnostics at the minimum-val-loss epoch restored by early stopping."""
    if history.empty or "val_loss" not in history or not np.isfinite(history.val_loss).all():
        raise ValueError("Finite validation-loss history required.")
    best = int(np.argmin(history.val_loss.to_numpy()))
    return dict(best_epoch=best + 1, last_epoch_run=len(history), monitor="val_loss", restore_best_weights=True,
        metrics_at_restored_epoch={k: float(v) for k, v in history.iloc[best].items()},
        note="Recorded stochastic validation metrics at restored epoch, not a fresh deterministic evaluation.")


def align_metabric_latent_scores(scores, metadata, prefix):
    """Require exactly the canonical patients, then align fixed scores by sample ID."""
    ids = scores.sample_id_normalised
    if ids.isna().any() or ids.duplicated().any() or set(ids) != set(metadata.sample_id_normalised):
        raise ValueError("Representation patient subset differs from canonical TPM; frozen indices cannot be forced.")
    aligned = metadata[["sample_id_normalised"]].merge(scores, on="sample_id_normalised", how="left", validate="one_to_one", sort=False)
    cols = [c for c in aligned if c.startswith(prefix + "_latent_")]
    if not cols or not np.isfinite(aligned[cols].to_numpy()).all():
        raise ValueError("Missing or nonfinite latent features.")
    return aligned[cols].copy(), metadata.Polymetastatic.astype(int).copy(), metadata.copy()


def evaluate_metabric_latent_representation(X, y, name, metadata, reference_metadata, reference_y,
        outer_splits, inner_cv, output_dir, registry, score_path, run_nested_cv=False, random_state=42, configs=None):
    """Use established three-family grids; load only complete revised-source results."""
    validate_tpm_representation_alignment(metadata, y, reference_metadata, reference_y)
    configs = configs or build_representation_classifiers(random_state, X.shape[1], use_feature_selection=False)
    score_fingerprint = metabric_source_fingerprint([score_path])
    results, predictions = {}, {}
    for family, config in configs.items():
        full_name = f"{name}_{family}"
        registry[full_name] = dict(config, X=X, y=y, metadata=metadata, family=family, representation=name)
        paths = [Path(output_dir) / f"transcriptomics_{full_name}_nested_cv_{kind}_reviewed.csv" for kind in ["results", "predictions"]]
        manifest_path = Path(output_dir) / f"transcriptomics_{full_name}_provenance_reviewed.json"
        if run_nested_cv:
            result, pred = run_nested_cv_metabric_latent(X, y, full_name, config["pipeline"], config["param_grid"],
                outer_splits, inner_cv, "f1_macro", metadata_df=metadata, output_dir=output_dir)
            manifest_path.write_text(json.dumps({"score_source": score_fingerprint, "n_inner_splits": 5, "scoring": "f1_macro", "random_state": random_state}, indent=2))
        elif all(p.exists() for p in paths):
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("score_source") != score_fingerprint or manifest.get("n_inner_splits") != 5 or manifest.get("scoring") != "f1_macro" or manifest.get("random_state") != random_state:
                raise ValueError("Cached results belong to another encoder score file or protocol; rerun nested CV.")
            result, pred = [pd.read_csv(p) for p in paths]
        elif any(p.exists() for p in paths):
            raise ValueError(f"Incomplete revised results for {full_name}.")
        else:
            print(f"No revised {full_name} results; no historical fallback."); continue
        predictions[full_name] = validate_tpm_oof(pred, full_name, metadata, y, outer_splits)
        results[full_name] = result
    return results, predictions


def fit_metabric_latent_flaml(X, y, time_budget, random_state=42, configs=None):
    """FLAML fits the whole scaling/sampling/classifier pipeline inside each inner fold."""
    settings = tpm_flaml_settings(time_budget, random_state)
    automl = AutoML(); names = []
    for family, config in (configs or build_representation_classifiers(random_state, X.shape[1], use_feature_selection=False)).items():
        name = f"metabric_latent_{family}"
        automl.add_learner(learner_name=name, learner_class=make_tpm_flaml_learner(config, name))
        names.append(name)
    automl.fit(X_train=X, y_train=y, estimator_list=names, **settings)
    return automl


def load_metabric_reviewed_scores(path, provenance_path, source_paths, preparation_manifest_path=None):
    """Load fixed scores only if their revised-source manifest matches current inputs."""
    if not Path(path).exists():
        return None
    completion_path = Path(path).parent / "_completion.json"
    if not completion_path.is_file():
        raise ValueError("Cached encoder bundle is incomplete or predates operational provenance.")
    completion = json.loads(completion_path.read_text())
    for filename, expected in completion["files"].items():
        artifact = completion_path.parent / filename
        if metabric_source_fingerprint([artifact])[str(artifact)] != expected:
            raise ValueError(f"Cached encoder artifact failed integrity validation: {filename}")
    info = json.loads(Path(provenance_path).read_text())
    info = info.get("source_provenance", info)
    if info.get("study_id") != "brca_metabric" or info.get("source_files") != metabric_source_fingerprint(source_paths):
        raise ValueError("Cached scores do not match current cBioPortal/institutional input provenance.")
    if preparation_manifest_path is None:
        raise ValueError("Canonical preparation manifest is required for cached encoder scores.")
    current = json.loads(Path(preparation_manifest_path).read_text())
    for key in ['ordered_feature_sha256', 'selected_feature_sha256', 'preparation_fingerprint']:
        if not current.get(key) or info.get(key) != current[key]:
            raise ValueError(f"Cached encoder scores have incompatible {key}; recompute from canonical inputs.")
    return pd.read_csv(path)


def run_nested_cv_metabric_latent(
    X: pd.DataFrame,
    y: pd.Series,
    representation_name: str,
    pipeline: Pipeline,
    param_grid: dict,
    outer_cv,
    inner_cv,
    scoring: str,
    n_jobs: int = -1,
    metadata_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reviewed METABRIC latent CV; retain fold-specific rankings without additional fits."""

    if scoring != "f1_macro":
        raise ValueError("Transcriptomics selection requires f1_macro.")
    validate_tpm_representation_alignment(metadata_df, y, metadata_df, y)
    latent_rankings = []
    fold_results = []
    prediction_rows = []

    for outer_fold, (train_idx, test_idx) in enumerate(
        tpm_outer_splits(outer_cv, X, y),
        start=1,
    ):
        print(f"\n{representation_name} | Outer fold {outer_fold}")

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        grid_search = GridSearchCV(
            estimator=clone(pipeline),
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
            return_train_score=False
        )

        grid_search.fit(X_train, y_train)

        best_estimator = grid_search.best_estimator_
        ranking = extract_fitted_tpm_feature_rankings(best_estimator, X.columns)
        ranking["representation"] = representation_name
        ranking["outer_fold"] = outer_fold
        ranking["top_in_fold"] = ranking["rank"].le(20)
        latent_rankings.append(ranking)

        y_pred_train = best_estimator.predict(X_train)
        y_prob_train = get_positive_class_probability(
            best_estimator,
            X_train,
        )

        y_pred_test = best_estimator.predict(X_test)
        y_prob_test = get_positive_class_probability(
            best_estimator,
            X_test,
        )

        train_metrics = tpm_pooled_oof_metrics(pd.DataFrame({
            "y_true": y_train.to_numpy(), "y_pred": y_pred_train, "y_prob_poly": y_prob_train
        }))

        test_metrics = tpm_pooled_oof_metrics(pd.DataFrame({
            "y_true": y_test.to_numpy(), "y_pred": y_pred_test, "y_prob_poly": y_prob_test
        }))

        row = {
            "representation": representation_name,
            "outer_fold": outer_fold,
            "best_inner_score": grid_search.best_score_,
            "best_params": json.dumps(grid_search.best_params_, default=str),
        }

        row.update({
            f"train_{key}": value
            for key, value in train_metrics.items()
        })

        row.update({
            f"test_{key}": value
            for key, value in test_metrics.items()
        })

        fold_results.append(row)

        for sample_position, sample_index in enumerate(test_idx):
            prediction_row = {
                "representation": representation_name,
                "outer_fold": outer_fold,
                "sample_index": sample_index,
                "dataset": "test",
                "best_inner_score": grid_search.best_score_,
                "best_params": json.dumps(grid_search.best_params_, default=str),
                "y_true": int(y_test.iloc[sample_position]),
                "y_pred": int(y_pred_test[sample_position]),
                "y_prob_poly": float(y_prob_test[sample_position]),
            }

            if (
                metadata_df is not None
                and "sample_id_normalised" in metadata_df.columns
            ):
                prediction_row["sample_id_normalised"] = metadata_df.iloc[
                    sample_index
                ]["sample_id_normalised"]

            prediction_rows.append(prediction_row)

        print("Best inner score:")
        print(grid_search.best_score_)

        print("\nBest params:")
        print(grid_search.best_params_)

        print("\nTest metrics:")
        print(test_metrics)

    results_df = pd.DataFrame(fold_results)
    predictions_df = pd.DataFrame(prediction_rows)

    validate_tpm_oof(predictions_df, representation_name, metadata_df, y, outer_cv)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        results_df.to_csv(
            os.path.join(
                output_dir,
                f"transcriptomics_{representation_name}_nested_cv_results_reviewed.csv",
            ),
            index=False,
        )

        predictions_df.to_csv(
            os.path.join(
                output_dir,
                f"transcriptomics_{representation_name}_nested_cv_predictions_reviewed.csv",
            ),
            index=False,
        )

    if output_dir is not None and latent_rankings:
        pd.concat(latent_rankings, ignore_index=True).to_csv(
            Path(output_dir) / f"transcriptomics_{representation_name}_outer_fold_latent_rankings_reviewed.csv", index=False)
    return results_df, predictions_df


from sklearn.decomposition import PCA


def run_nested_flaml_metabric_latent(X, y, name, metadata, outer_splits, time_budget, random_state=42, configs=None):
    """Five-fold outer evaluation of the inner-CV-safe fixed-latent AutoML comparator."""
    tpm_flaml_settings(time_budget, random_state)
    validate_tpm_representation_alignment(metadata, y, metadata, y)
    tests, trains, predictions = [], [], []
    for fold, (train, test) in enumerate(tpm_outer_splits(outer_splits, X, y), 1):
        automl = fit_metabric_latent_flaml(X.iloc[train], y.iloc[train], time_budget, random_state, configs=configs)
        provenance = dict(representation=name, outer_fold=fold, best_inner_score=1-automl.best_loss,
            best_params=json.dumps({"best_estimator": automl.best_estimator, "best_config": automl.best_config}, default=str),
            selection_metric="macro_f1", inner_n_splits=5, time_budget_seconds=time_budget)
        for indices, destination, dataset in [(train, trains, "train"), (test, tests, "test")]:
            pred = automl.predict(X.iloc[indices])
            probability = get_positive_class_probability(automl, X.iloc[indices])
            frame = pd.DataFrame({"y_true": y.iloc[indices].to_numpy(), "y_pred": pred, "y_prob_poly": probability})
            destination.append({**provenance, "dataset": dataset, **tpm_pooled_oof_metrics(frame)})
            if dataset == "test":
                for key, value in provenance.items(): frame[key] = value
                frame["sample_id_normalised"] = metadata.iloc[indices].sample_id_normalised.to_numpy()
                frame["sample_index"] = indices
                frame["dataset"] = "test"
                predictions.append(frame)
    oof = pd.concat(predictions, ignore_index=True)
    validate_tpm_oof(oof, name, metadata, y, outer_splits)
    return pd.DataFrame(tests), pd.DataFrame(trains), oof


def make_conditional_tpm_flaml_learner(config, learner_name):
    """Keep conditional sampler/weight choices coupled in FLAML's search space.

    Mixed-type common domains use unordered integer tokens decoded at fit time.
    Other common domains retain their values; differing dimensions are
    enumerated jointly so the search cannot recreate excluded combinations.
    Dictionary-grid callers retain the existing adapter unchanged.
    """
    from flaml import tune
    from flaml.automl.model import SKLearnEstimator
    from sklearn.model_selection import ParameterGrid

    grids = config["param_grid"]
    keys = list(grids[0])
    if any(set(grid) != set(keys) for grid in grids):
        raise ValueError("Conditional FLAML grids must use the same parameter keys.")
    common = {key: grids[0][key] for key in keys
              if all(repr(grid[key]) == repr(grids[0][key]) for grid in grids)}
    coupled = [key for key in keys if key not in common]
    choices = list(ParameterGrid([{key: grid[key] for key in coupled} for grid in grids]))
    token = "conditional_branch"
    template = config["pipeline"]
    mixed = {key for key, values in common.items()
             if len({type(v.item() if isinstance(v, np.generic) else v)
                     for v in values}) > 1}
    indexed = mixed | {"selector", "sampler"}


    def canonical_choice(key, value, values):
        # FLAML/NumPy categorical sampling may coerce mixed numeric/string
        # domains to strings. Recover only an existing declared grid value.
        value = value.item() if isinstance(value, np.generic) else value
        candidates = [v.item() if isinstance(v, np.generic) else v for v in values]
        exact = [v for v in candidates if type(v) is type(value) and v == value]
        if len(exact) == 1:
            return exact[0]
        matches = [v for v in candidates if str(v) == str(value)]
        if len(matches) != 1:
            raise ValueError(
                f"FLAML parameter {key!r}={value!r} does not uniquely match its declared grid."
            )
        return matches[0]

    class ConditionalPipelineLearner(SKLearnEstimator):
        @classmethod
        def search_space(cls, data_size, task=None, **kwargs):
            space = {key: {"domain": tune.choice(list(range(len(values))), order=False)
                           if key in indexed else tune.choice(values),
                           "init_value": 0 if key in indexed else values[0]}
                     for key, values in common.items()}
            space[token] = {"domain": tune.choice(list(range(len(choices)))), "init_value": 0}
            return space

        def fit(self, X_train, y_train, budget=None, free_mem_ratio=0, **kwargs):
            started = time.time()
            params = dict(choices[int(self.params.get(token, 0))])
            for key, values in common.items():
                value = self.params.get(key, 0 if key in indexed else values[0])
                if key in indexed:
                    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                        raise ValueError(f"FLAML categorical token for {key!r} must be an integer.")
                    if not 0 <= int(value) < len(values):
                        raise ValueError(f"FLAML categorical token for {key!r} is out of range.")
                    value = values[int(value)]
                params[key] = (value if key in ["selector", "sampler"]
                               else canonical_choice(key, value, values))
            params = {key: clone(value) if hasattr(value, "get_params") else value
                      for key, value in params.items()}
            self._model = clone(template).set_params(**params)
            self._model.fit(X_train, y_train)
            return time.time() - started

    ConditionalPipelineLearner.__name__ = learner_name
    return ConditionalPipelineLearner


def save_metabric_expression_archive(path, frame):
    """Write canonical numeric expression with explicit ordered sample/gene arrays."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, expression=frame.to_numpy(dtype=np.float64),
                            sample_ids=frame.index.to_numpy(dtype=str), genes=frame.columns.to_numpy(dtype=str))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def begin_metabric_encoder_run(output_dir, representation_name, provenance):
    """Create an isolated attempt; never overwrite a published encoder bundle."""
    from datetime import datetime, timezone
    import uuid
    root = Path(output_dir)
    completed = root / f"{representation_name}_completed"
    if completed.exists():
        raise FileExistsError(f"Completed bundle already exists: {completed}; use verified cached scores or explicitly archive it before retraining.")
    attempt = root / "encoder-attempts" / f"{representation_name}_{uuid.uuid4().hex}"
    (attempt / "artifacts").mkdir(parents=True)
    state = dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
                 status="prepared", source_provenance=provenance,
                 representation_name=representation_name)
    (attempt / "run.json").write_text(json.dumps(state, indent=2))
    return dict(attempt=attempt, completed=completed, state=state)


def metabric_encoder_operational_callbacks(run):
    """Persist best validation weights, epoch history and timing without changing fitting."""
    import time
    from datetime import datetime, timezone
    class OperationalCheckpoint(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best = float("inf")
            self.started = None

        def record(self, **updates):
            run["state"].update(updates)
            target = run["attempt"] / "run.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(run["state"], indent=2))
            os.replace(temporary, target)

        def require_built(self):
            if not self.model.built:
                raise RuntimeError("Operational checkpoint requires a built model; initialise the VAE wrapper before fit().")

        def on_train_begin(self, logs=None):
            self.require_built()
            self.started = time.monotonic()
            self.record(status="training", training_started_at_utc=datetime.now(timezone.utc).isoformat())

        def on_epoch_end(self, epoch, logs=None):
            self.require_built()
            value = float((logs or {}).get("val_loss", float("nan")))
            if np.isfinite(value) and value < self.best:
                temporary = run["attempt"] / "best.tmp.weights.h5"
                self.model.save_weights(temporary)
                os.replace(temporary, run["attempt"] / "best.weights.h5")
                self.best = value
                self.record(best_epoch=int(epoch + 1), best_val_loss=value)
            self.record(last_completed_epoch=int(epoch + 1), elapsed_seconds=time.monotonic()-self.started,
                        last_epoch_at_utc=datetime.now(timezone.utc).isoformat())

        def on_train_end(self, logs=None):
            self.record(status="trained", training_ended_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.monotonic()-self.started)
    return [keras.callbacks.CSVLogger(str(run["attempt"] / "epoch_history.csv")), OperationalCheckpoint()]


def complete_metabric_encoder_run(run):
    """Publish an immutable, hashed artifact directory only after every save succeeds."""
    from datetime import datetime, timezone
    import shutil
    artifacts = run["attempt"] / "artifacts"
    name = run["state"]["representation_name"]
    required = [f"{name}_z_mean_encoder.keras", f"{name}_training_history.csv",
                f"{name}_metadata.json", f"transcriptomics_{name}_scores.csv"]
    if run["state"].get("status") != "trained":
        raise ValueError("Training did not finish successfully.")
    for filename in required:
        if not (artifacts / filename).is_file():
            raise ValueError(f"Missing completed artifact: {filename}")
    if run["completed"].exists():
        raise FileExistsError(run["completed"])
    for filename in ["best.weights.h5", "epoch_history.csv"]:
        shutil.copy2(run["attempt"] / filename, artifacts / filename)
    state = dict(run["state"], status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
    (artifacts / "run.json").write_text(json.dumps(state, indent=2))
    hashes = metabric_source_fingerprint(sorted(artifacts.iterdir()))
    completion = dict(source_provenance=state["source_provenance"],
                      files={Path(path).name: value for path, value in hashes.items()})
    (artifacts / "_completion.json").write_text(json.dumps(completion, indent=2))
    # Atomic directory rename on the same filesystem; a nonempty completed bundle cannot be replaced.
    os.rename(artifacts, run["completed"])
    return run["completed"] / f"transcriptomics_{name}_scores.csv"


def align_attention_pam50(sample_ids, label_path):
    """Join the original five-class PAM50 labels by exact METABRIC sample ID."""
    labels = pd.read_csv(label_path, usecols=['METABRIC_ID', 'Pam50Subtype'], dtype=str)
    ids = pd.Index(sample_ids).astype(str)
    if ids.has_duplicates or labels.METABRIC_ID.duplicated().any():
        raise ValueError('BLOCKED — PAM50 TARGET ALIGNMENT: duplicate IDs')
    series = labels.set_index('METABRIC_ID').Pam50Subtype.reindex(ids)
    classes = ['Basal', 'Her2', 'LumA', 'LumB', 'Normal']
    unexpected = set(series.dropna()) - set(classes) - {'?', ''}
    if unexpected or series.isna().any():
        raise ValueError('BLOCKED — PAM50 TARGET ALIGNMENT: unmatched IDs or unexpected labels')
    valid = series.isin(classes)
    report = dict(canonical_samples=len(ids), validated_samples=int(valid.sum()),
        unmatched_samples=[], duplicate_sample_ids=[], conflicting_labels=[],
        excluded_unknown_samples=ids[~valid].tolist(), classes=classes,
        class_counts=series[valid].value_counts().to_dict(),
        label_source=metabric_source_fingerprint([label_path]))
    return series[valid], report


def prepare_attention_canonical(expression_path, metadata_path, study_path, label_path,
        institutional, mapping, output_dir, source_paths, random_state=42):
    """Prepare the attention-specific stratified external split; no institutional fitting."""
    import hashlib
    external, identifiers, profile = load_cbioportal_metabric_expression(expression_path, metadata_path, study_path)
    labels, label_report = align_attention_pam50(external.index, label_path)
    mappings = pd.concat([mapping, identifiers], ignore_index=True)
    ext, _, _ = harmonise_metabric_expression_ids(external, identifiers, mappings, 'mean')
    inst, _, _ = harmonise_metabric_expression_ids(institutional,
        pd.DataFrame({'ensembl_gene_id': institutional.columns}), mappings, None)
    common = sorted(set(ext.columns) & set(inst.columns))
    if len(common) != 16950 or len(inst) != 139 or inst.index.has_duplicates:
        raise ValueError('Canonical gene/patient contract mismatch')
    ext = ext.loc[labels.index, common]; inst = inst[common]
    train, val = train_test_split(np.arange(len(ext)), test_size=.2, random_state=random_state,
                                  shuffle=True, stratify=labels)
    medians = ext.iloc[train].median(); ext = ext.fillna(medians)
    selected = ext.iloc[train].var().sort_values(ascending=False, kind='stable').head(1000).index.tolist()
    scaler = StandardScaler().fit(ext.iloc[train][selected])
    if not np.isfinite(ext.to_numpy()).all() or not np.isfinite(inst.to_numpy()).all():
        raise ValueError('Nonfinite canonical attention expression')
    fingerprint = lambda genes: hashlib.sha256(('\n'.join(genes)+'\n').encode()).hexdigest()
    provenance = dict(study_id='brca_metabric', source_files=metabric_source_fingerprint(source_paths),
        ordered_feature_sha256=fingerprint(common), selected_feature_sha256=fingerprint(selected),
        pam50=label_report, train_ids=ext.index[train].tolist(), validation_ids=ext.index[val].tolist(),
        random_state=random_state, validation_fraction=.2, selected_genes=selected,
        architecture=dict(hidden_dim=256, latent_dim=100, dropout=.3, gates='sigmoid', output='five-class softmax'),
        training=dict(optimizer='Adam', learning_rate=.001, batch_size=32, epochs=300, patience=30,
                      loss='sparse_categorical_crossentropy', reduce_lr_patience=10, reduce_lr_factor=.5, min_lr=.00001),
        imputation='METABRIC training median', gene_selection='METABRIC training variance', scaling='METABRIC training StandardScaler')
    provenance['preparation_fingerprint'] = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(scaler=scaler, medians=medians, selected=selected), out/'attention_preprocessing.joblib')
    gene_map = mapping.set_index('gene_name').loc[selected].reset_index()
    gene_map.to_csv(out/'attention_selected_genes.csv', index=False)
    pd.DataFrame({'sample_id': ext.index, 'pam50': labels,
        'partition': ['train' if i in set(train) else 'validation' for i in range(len(ext))]}).to_csv(out/'attention_external_partitions.csv', index=False)
    (out/'attention_preparation.json').write_text(json.dumps(provenance, indent=2))
    return dict(external=ext[selected], institutional=inst[selected], labels=labels,
        train=train, val=val, scaler=scaler, provenance=provenance, gene_map=gene_map)


def build_attention_classifiers(random_state=42, n_features=100):
    """Use established latent grids; deduplicate only balanced-SMOTE class weighting."""
    configs = build_representation_classifiers(random_state, n_features, False)
    if set(configs) != {'logreg_l1', 'random_forest', 'xgboost'}:
        raise ImportError('All three attention classifier families require XGBoost')
    for family in ['logreg_l1', 'random_forest']:
        grid = configs[family]['param_grid']
        configs[family]['param_grid'] = [dict(grid, sampler=['passthrough']),
            dict(grid, sampler=[SMOTE(random_state=random_state)], classifier__class_weight=[None])]
    return configs


def attention_integrated_gradients(output, values, baseline, steps=128):
    """Trapezoidal IG for one explicitly defined scalar output per patient."""
    x = np.asarray(values, dtype=np.float32); b = np.asarray(baseline, dtype=np.float32)
    if x.ndim != 2 or b.shape != (x.shape[1],) or steps < 2:
        raise ValueError('Invalid IG dimensions or integration steps')
    rows = []
    for row in x:
        path = tf.convert_to_tensor(b + np.linspace(0, 1, steps+1, dtype=np.float32)[:, None]*(row-b))
        with tf.GradientTape() as tape:
            tape.watch(path); predictions = output(path)
        gradients = tape.gradient(predictions, path)
        if gradients is None: raise ValueError('Disconnected attribution path')
        g = gradients.numpy(); rows.append((row-b)*((g[:-1]+g[1:])/2).mean(axis=0))
    attribution = np.asarray(rows)
    delta = np.asarray(output(tf.constant(x))) - np.asarray(output(tf.constant(b[None, :])))
    residual = attribution.sum(axis=1)-delta
    if not np.isfinite(attribution).all() or not np.isfinite(residual).all():
        raise ValueError('Nonfinite IG values')
    return attribution, residual


def canonicalise_ig_gene_mapping(mapping, genes):
    """Resolve one unique metadata key matching every model-facing gene identifier."""
    genes = pd.Index(genes)
    if genes.has_duplicates or genes.isna().any() or not mapping.columns.is_unique:
        raise ValueError('IG requires unique nonmissing input genes and unique mapping column names')
    candidates = [column for column in mapping.columns
                  if mapping[column].notna().all() and mapping[column].is_unique
                  and genes.isin(mapping[column]).all()]
    if 'gene_name' in candidates:
        key = 'gene_name'
    elif len(candidates) == 1 and 'gene_name' not in mapping.columns:
        key = candidates[0]
    else:
        raise ValueError(f'IG mapping must contain an unambiguous unique gene key covering all input genes; matching columns: {candidates}')
    result = mapping.copy()
    if key != 'gene_name':
        result['gene_name'] = result[key]  # Retain the original key and all metadata.
    return result


def save_attention_ig(output, raw, baseline, mapping, output_dir, target_name, steps=(64,128,256)):
    """Persist signed/absolute gene summaries and explicit numerical diagnostics."""
    mapping = canonicalise_ig_gene_mapping(mapping, raw.columns)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    previous = None; checks = []
    for count in steps:
        attr, residual = attention_integrated_gradients(output, raw, baseline, count)
        change = np.nan if previous is None else float(np.abs(attr-previous).mean())
        checks.append(dict(steps=count, max_absolute_completeness_residual=float(np.abs(residual).max()),
                           mean_absolute_change=change))
        previous = attr
    pd.DataFrame(checks).to_csv(out/f'{target_name}_ig_numerical_checks.csv', index=False)
    # Never silently publish attributions that failed completeness/convergence checks.
    if np.abs(residual).max() > .001 or checks[-1]['mean_absolute_change'] > .001:
        raise ValueError('IG numerical tolerance not met; inspect checks and increase integration steps')
    pd.DataFrame(attr, index=raw.index, columns=raw.columns).to_csv(out/f'{target_name}_ig_signed.csv', index_label='sample_id')
    pd.DataFrame(np.abs(attr), index=raw.index, columns=raw.columns).to_csv(out/f'{target_name}_ig_absolute.csv', index_label='sample_id')
    ranking = pd.DataFrame({'gene_name':raw.columns, 'mean_absolute_ig':np.abs(attr).mean(axis=0), 'mean_signed_ig':attr.mean(axis=0)})
    ranking = ranking.merge(mapping, on='gene_name', validate='one_to_one').sort_values('mean_absolute_ig', ascending=False)
    ranking.to_csv(out/f'{target_name}_ig_gene_ranking.csv', index=False)
    return ranking


def prepare_rostami_canonical(expression_path, metadata_path, study_path, label_path,
        institutional, mapping, output_dir, source_paths, random_state=42):
    """Prepare the Rostami-specific stratified external split; no institutional fitting."""
    import hashlib
    external, identifiers, profile = load_cbioportal_metabric_expression(expression_path, metadata_path, study_path)
    labels, label_report = align_attention_pam50(external.index, label_path)
    labels = labels[labels.isin(['Basal','Her2','LumA','LumB'])]
    label_report['rostami_class_counts'] = labels.value_counts().to_dict()
    label_report['rostami_eligible_samples'] = len(labels)
    mappings = pd.concat([mapping, identifiers], ignore_index=True)
    ext, _, _ = harmonise_metabric_expression_ids(external, identifiers, mappings, 'mean')
    inst, _, _ = harmonise_metabric_expression_ids(institutional,
        pd.DataFrame({'ensembl_gene_id': institutional.columns}), mappings, None)
    common = sorted(set(ext.columns) & set(inst.columns))
    if len(common) != 16950 or len(inst) != 139 or inst.index.has_duplicates:
        raise ValueError('Canonical gene/patient contract mismatch')
    ext = ext.loc[labels.index, common]; inst = inst[common]
    train, val = train_test_split(np.arange(len(ext)), test_size=.2, random_state=random_state,
                                  shuffle=True, stratify=labels)
    for subtype in ['Basal','Her2','LumA','LumB']:
        if (labels.iloc[train] == subtype).sum() < 2 or (labels.iloc[val] == subtype).sum() < 2:
            raise ValueError(f'Insufficient independent training/validation samples for {subtype}')
    medians = ext.iloc[train].median(); ext = ext.fillna(medians)
    selected = ext.iloc[train].var().sort_values(ascending=False, kind='stable').head(1000).index.tolist()
    scaler = StandardScaler().fit(ext.iloc[train][selected])
    if not np.isfinite(ext.to_numpy()).all() or not np.isfinite(inst.to_numpy()).all():
        raise ValueError('Nonfinite canonical attention expression')
    fingerprint = lambda genes: hashlib.sha256(('\n'.join(genes)+'\n').encode()).hexdigest()
    provenance = dict(study_id='brca_metabric', source_files=metabric_source_fingerprint(source_paths),
        ordered_feature_sha256=fingerprint(common), selected_feature_sha256=fingerprint(selected),
        pam50=label_report, train_ids=ext.index[train].tolist(), validation_ids=ext.index[val].tolist(),
        random_state=random_state, validation_fraction=.2, selected_genes=selected,
        architecture=dict(input_dim=1000, bottleneck_dim=500, hidden_activation='relu', output_activation='linear'),
        training=dict(optimizer='Adam', learning_rate=.001, batch_size=32, epochs=300, patience=30,
                      loss='mse', restore_best_weights=True, seeds=[42,101,202], reconstruction_seed=202, top_genes_per_subtype=300),
        imputation='METABRIC training median', gene_selection='METABRIC training variance', scaling='METABRIC training StandardScaler')
    provenance['preparation_fingerprint'] = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(scaler=scaler, medians=medians, selected=selected), out/'rostami_preprocessing.joblib')
    gene_map = mapping.set_index('gene_name').loc[selected].reset_index()
    gene_map.to_csv(out/'rostami_selected_genes.csv', index=False)
    pd.DataFrame({'sample_id': ext.index, 'pam50': labels,
        'partition': ['train' if i in set(train) else 'validation' for i in range(len(ext))]}).to_csv(out/'rostami_external_partitions.csv', index=False)
    (out/'rostami_preparation.json').write_text(json.dumps(provenance, indent=2))
    return dict(external=ext[selected], institutional=inst[selected], labels=labels,
        train=train, val=val, scaler=scaler, medians=medians, provenance=provenance, gene_map=gene_map)


def train_rostami_canonical(prepared, output_dir):
    """Train twelve external models with fixed partitions and isolated attempts."""
    import uuid
    root = Path(output_dir)
    completed = root / 'rostami_completed'
    if completed.exists():
        raise FileExistsError(f'Completed Rostami bundle exists: {completed}')
    attempt = root / 'encoder-attempts' / uuid.uuid4().hex
    artifacts = attempt / 'artifacts'; artifacts.mkdir(parents=True)
    models, score_tables, histories = {}, [], []
    scaler = prepared['scaler']; train = prepared['train']; val = prepared['val']
    labels = prepared['labels']; external = prepared['external']
    for subtype in ['Basal', 'Her2', 'LumA', 'LumB']:
        train_ids = [i for i in train if labels.iloc[i] == subtype]
        val_ids = [i for i in val if labels.iloc[i] == subtype]
        if len(train_ids) < 2 or len(val_ids) < 2:
            raise ValueError(f'Insufficient independent train/validation samples for {subtype}')
        X_train = scaler.transform(external.iloc[train_ids]).astype(np.float32)
        X_val = scaler.transform(external.iloc[val_ids]).astype(np.float32)
        for seed in [42, 101, 202]:
            set_global_seed(seed)
            unit = begin_metabric_encoder_run(attempt, f'{subtype}_seed{seed}', prepared['provenance'])
            autoencoder, encoder = build_rostami_autoencoder(1000, 500, .001)
            callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)]
            history = autoencoder.fit(X_train, X_train, validation_data=(X_val, X_val),
                epochs=300, batch_size=32, shuffle=True,
                callbacks=callbacks + metabric_encoder_operational_callbacks(unit), verbose=1)
            frame = pd.DataFrame(history.history); frame['subtype'] = subtype
            frame['seed'] = seed; frame['epoch'] = np.arange(1, len(frame)+1)
            histories.append(frame)
            scores = compute_encoder_gene_scores(autoencoder, external.columns.tolist())
            scores['subtype'] = subtype; scores['seed'] = seed
            scores['rank_within_seed'] = np.arange(1, len(scores)+1); score_tables.append(scores)
            autoencoder.save(artifacts / f'{subtype}_seed{seed}_autoencoder.keras')
            frame.to_csv(artifacts / f'{subtype}_seed{seed}_history.csv', index=False)
            import shutil
            for filename in ['best.weights.h5', 'epoch_history.csv', 'run.json']:
                shutil.copy2(unit['attempt']/filename, artifacts/f'{subtype}_seed{seed}_{filename}')
            if seed == 202: models[subtype] = autoencoder
    return dict(attempt=attempt, artifacts=artifacts, completed=completed, models=models,
                scores=pd.concat(score_tables, ignore_index=True), histories=pd.concat(histories, ignore_index=True))


def summarise_rostami_markers(scores, top_n=300):
    """Mean weight-norm selection with separate top-set recurrence and rank agreement."""
    frame = scores.copy(); frame['in_top_set'] = frame.rank_within_seed.le(top_n)
    summary = frame.groupby(['subtype','gene'], as_index=False).agg(
        mean_gene_score=('gene_score','mean'), std_gene_score=('gene_score','std'),
        mean_rank=('rank_within_seed','mean'), top_set_recurrence=('in_top_set','sum'), n_seeds=('seed','nunique'))
    selected = summary.sort_values(['subtype','mean_gene_score'], ascending=[True,False]).groupby('subtype').head(top_n)
    agreement = []
    for subtype, group in frame.groupby('subtype'):
        ranks = group.pivot(index='gene', columns='seed', values='rank_within_seed')
        for first, second in [(42,101),(42,202),(101,202)]:
            agreement.append(dict(subtype=subtype, seed_a=first, seed_b=second,
                spearman_rank_agreement=ranks[first].corr(ranks[second], method='spearman')))
    return summary, selected, sorted(selected.gene.unique()), pd.DataFrame(agreement)


def rostami_reconstruction_scores(models, scaled_expression):
    """One MSE per subtype, using only the fixed seed-202 model for each subtype."""
    x = np.asarray(scaled_expression, dtype=np.float32)
    return np.column_stack([np.square(x-models[subtype].predict(x,batch_size=32,verbose=0)).mean(axis=1)
                            for subtype in ['Basal','Her2','LumA','LumB']])


def publish_rostami_bundle(run, prepared, markers, marker_scores, reconstruction_scores):
    """Publish all twelve models and both representations together, without overwrite."""
    from datetime import datetime, timezone
    artifacts = run['artifacts']
    for frame in [marker_scores, reconstruction_scores]:
        if len(frame) != 139 or frame.sample_id_normalised.duplicated().any() or set(frame.sample_id_normalised) != set(prepared['institutional'].index):
            raise ValueError('Incomplete institutional representation')
        if not np.isfinite(frame.drop(columns='sample_id_normalised').to_numpy()).all():
            raise ValueError('Nonfinite institutional representation')
    if reconstruction_scores.shape[1] != 5:
        raise ValueError('Expected exactly four subtype reconstruction-error features')
    for subtype in ['Basal','Her2','LumA','LumB']:
        for seed in [42,101,202]:
            if not (artifacts/f'{subtype}_seed{seed}_autoencoder.keras').is_file():
                raise ValueError('Incomplete twelve-model Rostami bundle')
    if run['completed'].exists(): raise FileExistsError(run['completed'])
    run['scores'].to_csv(artifacts/'gene_scores_by_seed.csv',index=False)
    run['histories'].to_csv(artifacts/'training_history.csv',index=False)
    joblib.dump(dict(scaler=prepared['scaler'], medians=prepared['medians'],
        selected=prepared['external'].columns.tolist(), markers=markers), artifacts/'preprocessing.joblib')
    prepared['gene_map'].to_csv(artifacts/'selected_gene_mapping.csv', index=False)
    marker_scores.to_csv(artifacts/'marker_scores.csv',index=False)
    reconstruction_scores.to_csv(artifacts/'reconstruction_scores.csv',index=False)
    (artifacts/'provenance.json').write_text(json.dumps(prepared['provenance'],indent=2))
    hashes = metabric_source_fingerprint(sorted(artifacts.iterdir()))
    (artifacts/'_completion.json').write_text(json.dumps(dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        files={Path(k).name:v for k,v in hashes.items()}),indent=2))
    os.rename(artifacts,run['completed'])


def load_rostami_bundle(output_dir, source_paths):
    """Reject incomplete, corrupted, historical or mismatched canonical artifacts."""
    root = Path(output_dir)/'rostami_completed'
    if not root.exists(): return None
    completion = json.loads((root/'_completion.json').read_text())
    required = {'marker_scores.csv','reconstruction_scores.csv','preprocessing.joblib','provenance.json','selected_gene_mapping.csv'}
    required.update(f'{s}_seed{seed}_autoencoder.keras' for s in ['Basal','Her2','LumA','LumB'] for seed in [42,101,202])
    if not required.issubset(completion['files']): raise ValueError('Incomplete canonical bundle')
    for filename, expected in completion['files'].items():
        if metabric_source_fingerprint([root/filename])[str(root/filename)] != expected:
            raise ValueError(f'Rostami artifact hash mismatch: {filename}')
    provenance = json.loads((root/'provenance.json').read_text())
    current = json.loads((Path(output_dir)/'rostami_preparation.json').read_text())
    if provenance['source_files'] != metabric_source_fingerprint(source_paths): raise ValueError('Rostami source mismatch')
    for key in ['preparation_fingerprint','ordered_feature_sha256','selected_feature_sha256']:
        if not current.get(key) or provenance.get(key)!=current[key]: raise ValueError(f'Rostami {key} mismatch')
    return dict(root=root, marker=pd.read_csv(root/'marker_scores.csv'), reconstruction=pd.read_csv(root/'reconstruction_scores.csv'))


def prepare_metabric_vae_ig_output(encoder, offset, scale, downstream=None, coordinate=None):
    """Fixed affine external preprocessing and inference-only encoder; optional L1 risk."""
    offset = np.asarray(offset, dtype=np.float32); scale = np.asarray(scale, dtype=np.float32)
    if not np.isfinite(offset).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Invalid frozen external scaling parameters')
    if downstream is not None:
        classifier = downstream.named_steps['classifier']
        if not isinstance(classifier, LogisticRegression) or classifier.penalty != 'l1' or not np.array_equal(classifier.classes_, [0,1]):
            raise ValueError('Predictive IG requires the selected binary L1 pipeline')
        if downstream.named_steps.get('selector', 'passthrough') != 'passthrough':
            raise ValueError('Unexpected latent selector in IG pathway')
        latent_scaler = downstream.named_steps['scaler']
    def output(raw):
        z = encoder((tf.cast(raw, tf.float32)-tf.constant(offset))/tf.constant(scale), training=False)
        if downstream is None:
            if coordinate is None: raise ValueError('Representation attribution requires a named coordinate')
            return z[:,coordinate]
        z = (z-tf.constant(latent_scaler.mean_,tf.float32))/tf.constant(latent_scaler.scale_,tf.float32)
        return tf.math.sigmoid(tf.linalg.matvec(z,tf.constant(classifier.coef_[0],tf.float32))+float(classifier.intercept_[0]))
    return output


def run_metabric_vae_ig(config, fitted_model, completed_dir, representation, kind,
        preparation_dir, source_paths, mapping_path):
    """Interpret an already selected/refitted combination, never train or select models."""
    import uuid
    completed_dir = Path(completed_dir); preparation_dir = Path(preparation_dir)
    manifest_path = preparation_dir/'metabric_cbioportal_source_provenance.json'
    score_path = completed_dir/f'transcriptomics_{representation}_scores.csv'
    cached = load_metabric_reviewed_scores(score_path, completed_dir/f'{representation}_metadata.json', source_paths, manifest_path)
    if cached is None: raise ValueError('Completed canonical encoder required for IG')
    manifest = json.loads(manifest_path.read_text())
    filenames = ['institutional_cbioportal_aligned_log2_tpm.csv', 'metabric_canonical_encoder_scaling_parameters.csv']
    for filename in filenames:
        path = preparation_dir/filename
        if metabric_source_fingerprint([path])[str(path)] != manifest['artifacts'].get(str(path)):
            raise ValueError(f'IG preparation artifact mismatch: {filename}')
    raw = pd.read_csv(preparation_dir/filenames[0],index_col=0)
    ids = config['metadata'].sample_id_normalised.astype(str)
    raw.index = raw.index.astype(str)
    if raw.index.has_duplicates or set(raw.index) != set(ids): raise ValueError('IG patient alignment mismatch')
    raw = raw.loc[ids]
    params = pd.read_csv(preparation_dir/filenames[1]).set_index('gene')
    if raw.shape[1]!=1000 or list(params.index)!=list(raw.columns): raise ValueError('IG selected-gene order mismatch')
    baseline = params.rna_standard_mean.to_numpy()  # External training mean for BOTH branches.
    if kind == 'rna':
        scaler = joblib.load(completed_dir/f'{representation}_scaler.joblib')
        offset, scale = scaler.mean_, scaler.scale_
        if not np.allclose(offset,baseline): raise ValueError('RNA training-mean baseline mismatch')
    elif kind == 'hvae':
        frozen = pd.read_csv(completed_dir/f'{representation}_metabric_minmax_scaling_params.csv').set_index('gene')
        if list(frozen.index)!=list(raw.columns): raise ValueError('HVAE gene order mismatch')
        offset, scale = frozen.metabric_train_min.to_numpy(), frozen.metabric_train_range.to_numpy()
    else: raise ValueError('Unknown external encoder kind')
    encoder = keras.models.load_model(completed_dir/f'{representation}_z_mean_encoder.keras',compile=False)
    latent = encoder(((raw.to_numpy()-offset)/scale).astype(np.float32),training=False).numpy()
    cached = cached.set_index('sample_id_normalised').loc[ids]
    if not np.allclose(latent,cached.to_numpy(),atol=1e-4,rtol=1e-4) or not np.allclose(latent,config['X'].to_numpy(),atol=1e-4,rtol=1e-4):
        raise ValueError('Frozen IG encoder does not reproduce selected latent features')
    mapping = pd.read_csv(mapping_path)
    mapping = mapping.set_index('gene_name').loc[raw.columns].reset_index()
    root = completed_dir/'interpretation_ig'; root.mkdir(exist_ok=True)
    token = uuid.uuid4().hex; pending = root/f'.pending_{token}'; pending.mkdir()
    diagnostics = dict(representation=representation, family=config['family'],
        preparation_fingerprint=manifest['preparation_fingerprint'],
        baseline='METABRIC external training mean in supplied log-expression space',
        external_transform=kind, max_latent_difference=float(np.abs(latent-config['X'].to_numpy()).max()))
    if config['family']=='logreg_l1':
        output = prepare_metabric_vae_ig_output(encoder,offset,scale,downstream=fitted_model)
        actual = output(tf.constant(raw.to_numpy(),tf.float32)).numpy()
        expected = fitted_model.predict_proba(config['X'])[:,1]
        diagnostics['max_probability_difference'] = float(np.abs(actual-expected).max())
        (pending/'agreement.json').write_text(json.dumps(diagnostics,indent=2))
        if not np.allclose(actual,expected,atol=1e-5,rtol=1e-5): raise ValueError('TensorFlow/sklearn IG probability mismatch')
        targets = [('polymetastatic_probability',output)]
    elif config['family'] in ['random_forest','xgboost']:
        targets = [(f'latent_{i+1}_representation', prepare_metabric_vae_ig_output(encoder,offset,scale,coordinate=i)) for i in range(latent.shape[1])]
        (pending/'agreement.json').write_text(json.dumps(diagnostics,indent=2))
    else: raise ValueError('Unregistered IG family')
    pd.DataFrame({'gene':raw.columns,'external_training_mean':baseline,'transformed_baseline':(baseline-offset)/scale}).to_csv(pending/'baseline.csv',index=False)
    for name, output in targets:
        save_attention_ig(output,raw,baseline,mapping,pending,name)
    # Publish interpretation only after every target passes all numerical checks.
    destination = root/f'validated_{token}'
    os.rename(pending,destination)
    return destination


def resolve_metabric_ig_refit(output_dir, metadata, y, outer_splits, source_paths):
    """Validate reviewed selection and decode the saved full-cohort grid point; no fitting."""
    from sklearn.model_selection import ParameterGrid
    root = Path(output_dir)
    summary = pd.read_csv(root/'metabric_cbioportal_model_selection_summary_reviewed.csv').set_index('representation')
    folds = pd.read_csv(root/'metabric_cbioportal_all_nested_cv_results_reviewed.csv')
    representations = ['metabric_cbioportal_integrativevae_rna_only_latent100',
                       'metabric_cbioportal_integrativevae_hvae_rna_ds256_latent128']
    expected = {f'{r}_{f}' for r in representations for f in ['logreg_l1','random_forest','xgboost']}
    if summary.index.has_duplicates or set(summary.index)!=expected:
        raise ValueError('IG restoration requires the complete six-combination selection summary')
    rank_tpm_inner_scores(folds, sorted(expected))
    means = folds.groupby('representation').best_inner_score.mean().reindex(summary.index)
    if not np.isfinite(summary.mean_inner_cv_score).all() or not np.allclose(means,summary.mean_inner_cv_score):
        raise ValueError('Reviewed selection summary disagrees with fold inner scores')
    selected = summary.sort_index().sort_values('mean_inner_cv_score',ascending=False,kind='stable').index[0]
    registry = {}
    for representation in representations:
        parent = root if representation==representations[0] else root/'Original-HVAE-RNA-Branch'
        completed = parent/f'{representation}_completed'
        score_path = completed/f'transcriptomics_{representation}_scores.csv'
        scores = load_metabric_reviewed_scores(score_path,completed/f'{representation}_metadata.json',source_paths,
                                               root/'metabric_cbioportal_source_provenance.json')
        if scores is None: raise ValueError('Missing completed encoder scores for IG restoration')
        if scores.sample_id_normalised.duplicated().any() or set(scores.sample_id_normalised)!=set(metadata.sample_id_normalised):
            raise ValueError('IG restoration patient mismatch')
        scores = scores.set_index('sample_id_normalised').loc[metadata.sample_id_normalised]
        X = scores.reset_index(drop=True)
        results, predictions = evaluate_metabric_latent_representation(X,y,representation,metadata,metadata,y,
            outer_splits,StratifiedKFold(5,shuffle=True,random_state=42),root,registry,score_path,
            run_nested_cv=False,random_state=42)
        if len(results)!=3: raise ValueError('Missing reviewed per-family results or provenance')
        for name, frame in results.items():
            actual = frame.set_index('outer_fold').best_inner_score.sort_index()
            reference = folds.loc[folds.representation==name].set_index('outer_fold').best_inner_score.sort_index()
            if not np.allclose(actual,reference): raise ValueError('Aggregate/per-family inner scores disagree')
    config = registry[selected]
    if config['family']!='logreg_l1':
        raise ValueError('Saved-model reconstruction is supported only for selected L1; no tree refit is permitted here')
    info = json.loads((root/f'{selected}_params_full_cohort_tuned.json').read_text())
    if info.get('representation')!=selected or info.get('scoring')!='f1_macro' or info.get('n_splits')!=5 or info.get('interpretation_only') is not True:
        raise ValueError('Missing/incompatible fresh full-cohort tuning metadata')
    saved = info['best_params']
    matching = [p for p in ParameterGrid(config['param_grid']) if json.loads(json.dumps(p,default=str))==saved]
    if not matching: raise ValueError('Saved full-cohort parameters are not in the registered grid')
    return selected, config, matching[0]


def restore_metabric_ig_l1_model(output_dir, metadata, y, outer_splits, source_paths):
    """One fixed-parameter interpretation fit; never rerun search, CV or encoder fitting."""
    selected, config, params = resolve_metabric_ig_refit(output_dir,metadata,y,outer_splits,source_paths)
    estimator = clone(config['pipeline']).set_params(**params)
    estimator.fit(config['X'],config['y'])
    return selected, config, estimator


# Preserved-result synthesis and reporting helpers (no fitting or metric computation).


def select_mean_inner_combination(summary):
    """Preserve alphabetical/stable tie handling for an indexed mean-inner summary."""
    return summary.sort_index().sort_values('mean_inner_cv_score', ascending=False, kind='stable').index[0]


def prepare_tybalt_ig_inputs(aligned_path, cohort_path, training_reference):
    """Recover the benchmark's exact ordered patient/gene input; no representation fit."""
    aligned = pd.read_csv(aligned_path)
    cohort = pd.read_csv(cohort_path)
    key = 'sample_id_normalised'
    genes = list(pd.read_csv(training_reference, sep='\t', index_col=0, nrows=0).columns)
    if list(aligned.columns) != [key] + genes or len(genes) != 5000 or len(set(genes)) != 5000:
        raise ValueError('Aligned Tybalt columns do not match the pretrained 5000-gene order')
    for frame in [aligned, cohort]:
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError('Tybalt IG requires unique nonmissing patient identifiers')
    if set(aligned[key]) != set(cohort[key]) or len(cohort) != 139:
        raise ValueError('Tybalt IG cohort does not match the benchmark')
    merged = cohort[[key, 'Polymetastatic']].merge(aligned, on=key, how='inner', validate='one_to_one', sort=False)
    if merged[key].tolist() != cohort[key].tolist():
        raise ValueError('Patient order changed during Tybalt alignment')
    X = merged[genes].copy()
    y = merged.Polymetastatic.astype(int)
    if not np.isfinite(X.to_numpy()).all() or set(y) != {0, 1}:
        raise ValueError('Nonfinite expression or invalid binary target')
    return X, y, merged[key].copy()


def compose_tybalt_ig_probability(pipeline):
    """Frozen MinMax/clip -> twohidden encoder -> StandardScaler -> L1 probability."""
    tybalt = pipeline.named_steps['tybalt']
    scaler = pipeline.named_steps['scaler']
    classifier = pipeline.named_steps['classifier']
    if pipeline.named_steps['selector'] != 'passthrough' or classifier.penalty != 'l1':
        raise ValueError('Expected the benchmark L1/no-selector Tybalt chain')
    if not np.array_equal(classifier.classes_, [0, 1]):
        raise ValueError('Class 1 must be polymetastatic probability')
    if tybalt.encoder_.input_shape[-1] != 5000 or tybalt.encoder_.output_shape[-1] != 100:
        raise ValueError('Wrong Tybalt encoder dimensions')
    if classifier.coef_.shape != (1, 100) or scaler.n_features_in_ != 100:
        raise ValueError('Latent/classifier coordinate dimensions do not match')
    def latent(raw):
        # sklearn MinMax uses float64 for the saved aligned CSV, then the transformer casts.
        x = tf.cast(raw, tf.float64)
        x = x * tf.constant(tybalt.scaler_.scale_, tf.float64) + tf.constant(tybalt.scaler_.min_, tf.float64)
        if tybalt.clip:
            x = tf.clip_by_value(x, 0., 1.)
        return tybalt.encoder_(tf.cast(x, tf.float32), training=False)
    def probability(raw):
        z = tf.cast(latent(raw), tf.float64)
        z = (z - tf.constant(scaler.mean_, tf.float64)) / tf.constant(scaler.scale_, tf.float64)
        return tf.math.sigmoid(tf.linalg.matvec(z, tf.constant(classifier.coef_[0], tf.float64))
                               + tf.constant(classifier.intercept_[0], tf.float64))
    return probability, latent


def save_tybalt_ig_checks(output, raw, baseline, destination, steps=(64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)):
    """Reuse trapezoidal IG, retain diagnostics, publish arrays only after convergence."""
    destination = Path(destination)
    previous = None
    checks = []
    for count in steps:
        print(f'Tybalt IG: {count} steps, {len(raw)} patients', flush=True)
        attr, residual = attention_integrated_gradients(output, raw, baseline, steps=count)
        change = np.nan if previous is None else float(np.abs(attr - previous).mean())
        maximum_change = np.nan if previous is None else float(np.abs(attr - previous).max())
        checks.append(dict(steps=count, max_absolute_completeness_residual=float(np.abs(residual).max()),
                           mean_absolute_completeness_residual=float(np.abs(residual).mean()),
                           mean_absolute_change=change, max_absolute_change=maximum_change))
        pd.DataFrame(checks).to_csv(destination/'polymetastatic_probability_ig_numerical_checks.csv', index=False)
        print(checks[-1], flush=True)
        previous = attr
        if count >= 256 and np.abs(residual).max() <= .001 and change <= .001 and maximum_change <= .001:
            break
    else:
        raise ValueError('Tybalt IG failed completeness/convergence: attempt retained, not published')
    pd.DataFrame(attr, index=raw.index, columns=raw.columns).to_csv(destination/'polymetastatic_probability_ig_signed.csv', index_label='sample_id')
    pd.DataFrame(np.abs(attr), index=raw.index, columns=raw.columns).to_csv(destination/'polymetastatic_probability_ig_absolute.csv', index_label='sample_id')
    ranking = pd.DataFrame({'gene_name': raw.columns, 'mean_absolute_ig': np.abs(attr).mean(axis=0),
                            'mean_signed_ig': attr.mean(axis=0)}).sort_values('mean_absolute_ig', ascending=False)
    ranking.to_csv(destination/'polymetastatic_probability_ig_gene_ranking.csv', index=False)
    return ranking, pd.DataFrame(checks)


def run_tybalt_twohidden100_ig(aligned_path, cohort_path, encoder_path, training_reference, output_root):
    """Scoped 5-fold full-cohort tuning and gene IG; never rerun the predictive benchmark."""
    from datetime import datetime, timezone
    from uuid import uuid4
    from sklearn.model_selection import ParameterGrid
    import hashlib
    import shutil
    import platform
    import sklearn
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output_root = Path(output_root)
    attempt = output_root / ('attempt_' + stamp + '_' + uuid4().hex[:8])
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def write_json(name, data):
        temporary = attempt / (name + '.tmp')
        temporary.write_text(json.dumps(data, indent=2, default=str))
        os.replace(temporary, attempt/name)
    status = dict(status='running', interpretation_only=True, started_utc=stamp)
    write_json('run.json', status)
    try:
        X, y, patients = prepare_tybalt_ig_inputs(aligned_path, cohort_path, training_reference)
        sources = metabric_source_fingerprint([aligned_path, cohort_path, encoder_path, training_reference])
        ordered_hash = lambda values: hashlib.sha256(json.dumps(list(values), separators=(',', ':')).encode()).hexdigest()
        gene_hash, patient_hash = ordered_hash(X.columns), ordered_hash(patients)
        config = wrap_with_tybalt(build_representation_classifiers(random_state=42, n_features=5000,
                                   use_feature_selection=False), encoder_path)['logreg_l1']
        config.update(X=X, y=y)
        grid = [{k: str(v) if k in ('sampler', 'selector') else v for k,v in row.items()}
                for row in ParameterGrid(config['param_grid'])]
        metadata = dict(interpretation_only=True, representation='tybalt_twohidden100', classifier='logreg_l1',
                        selection_metric='f1_macro', n_splits=5, shuffle=True, random_state=42,
                        sources=sources, gene_order_sha256=gene_hash, patient_order_sha256=patient_hash,
                        input_shape=list(X.shape), latent_dim=100, class_1='Polymetastatic', grid=grid,
                        baseline='Transformed-space zero: fitted per-gene minima; model reference, not healthy tissue or biological zero',
                        preprocessing='Aligned log2(TPM+1) gene-symbol values; missing genes zero-filled upstream; fitted MinMaxScaler, clip [0,1], float32 encoder, latent StandardScaler; sampler only during fit',
                        versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, tensorflow=tf.__version__),
                        integration_steps=[64,128,256,512,1024,2048,4096,8192,16384], min_steps=256,
                        tolerance=dict(probability=1e-5, latent=1e-4, completeness=.001, mean_change=.001, max_change=.001))
        write_json('provenance.json', metadata)
        pd.DataFrame({'sample_id':patients, 'Polymetastatic':y}).to_csv(attempt/'patient_order.csv', index=False)
        pd.DataFrame({'gene_name':X.columns}).to_csv(attempt/'gene_order.csv', index=False)
        print(f'Tybalt interpretation search: {len(grid)} candidates x 5 folds; {attempt}', flush=True)
        search = tune_full_cohort_tpm(config, random_state=42, n_jobs=1)
        fitted = search.best_estimator_
        tybalt = fitted.named_steps['tybalt']
        if list(tybalt.scaler_.feature_names_in_) != list(X.columns):
            raise ValueError('Fitted gene order mismatch')
        params = dict(interpretation_only=True, representation='tybalt_twohidden100_logreg_l1',
                      scoring='f1_macro', n_splits=5, best_inner_cv_score=float(search.best_score_),
                      best_params={k: str(v) if k in ('sampler','selector') else v for k,v in search.best_params_.items()})
        write_json('interpretation_only_params.json', params)
        pd.DataFrame(search.cv_results_).to_csv(attempt/'interpretation_only_cv_results.csv', index=False)
        joblib.dump(tybalt.scaler_, attempt/'input_scaler.joblib')
        joblib.dump(fitted.named_steps['scaler'], attempt/'latent_scaler.joblib')
        joblib.dump(fitted.named_steps['classifier'], attempt/'classifier.joblib')
        joblib.dump(fitted, attempt/'interpretation_pipeline.joblib')
        shutil.copy2(encoder_path, attempt/'encoder_twohidden100_vae.hdf5')
        baseline = tybalt.scaler_.data_min_.copy()
        transformed = tybalt.scaler_.transform(pd.DataFrame([baseline], columns=X.columns))[0]
        if not np.allclose(transformed, 0, atol=1e-12):
            raise ValueError('Baseline does not map to transformed zero')
        pd.DataFrame({'gene_name':X.columns, 'aligned_expression_baseline':baseline,
                      'transformed_baseline':np.clip(transformed,0,1)}).to_csv(attempt/'baseline.csv', index=False)
        probability, latent = compose_tybalt_ig_probability(fitted)
        reference = fitted.predict_proba(X)[:,1]
        reconstructed = probability(tf.constant(X.to_numpy(), tf.float64)).numpy()
        # Also check the float32 coordinates used by the shared IG integrator.
        reconstructed32 = probability(tf.constant(X.to_numpy(), tf.float32)).numpy()
        z = tybalt.transform(X)
        ztf = latent(tf.constant(X.to_numpy(), tf.float32)).numpy()
        agreement = dict(max_probability_difference=float(np.abs(reference-reconstructed).max()),
                         max_ig_float32_probability_difference=float(np.abs(reference-reconstructed32).max()),
                         max_latent_difference=float(np.abs(z-ztf).max()),
                         patient_order_sha256=patient_hash, gene_order_sha256=gene_hash,
                         input_dim=5000, latent_dim=100, classifier_coordinate_order=list(range(100)),
                         classes=fitted.named_steps['classifier'].classes_.tolist(), target='P(polymetastatic = 1)',
                         baseline_probability=float(probability(tf.constant(baseline[None,:],tf.float32)).numpy()[0]))
        write_json('agreement.json', agreement)
        pd.DataFrame({'sample_id':patients, 'pipeline_probability':reference,
                      'tensorflow_probability':reconstructed, 'ig_float32_probability':reconstructed32}).to_csv(attempt/'prediction_agreement.csv', index=False)
        if max(agreement['max_probability_difference'],agreement['max_ig_float32_probability_difference']) > 1e-5 or agreement['max_latent_difference'] > 1e-4:
            raise ValueError('Tybalt prediction agreement failed; IG not published')
        print('Probability agreement PASSED:', agreement, flush=True)
        raw = X.copy(); raw.index = pd.Index(patients, name='sample_id')
        ranking, checks = save_tybalt_ig_checks(probability, raw, baseline, attempt)
        if sources != metabric_source_fingerprint([aligned_path, cohort_path, encoder_path, training_reference]):
            raise ValueError('Source files changed during interpretation')
        status.update(status='complete', ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
        write_json('run.json', status)
        hashes = metabric_source_fingerprint([p for p in attempt.iterdir() if p.is_file()])
        write_json('_completion.json', dict(interpretation_only=True, files={Path(k).name:v for k,v in hashes.items()}))
        complete = output_root / attempt.name.replace('attempt_', 'validated_', 1)
        os.rename(attempt, complete)
        print('TYBALT IG COMPLETE:', complete, flush=True)
        print(ranking.head(20).to_string(index=False), flush=True)
        return complete
    except Exception as exc:
        status.update(status='failed', error=str(exc), elapsed_seconds=time.monotonic()-started)
        write_json('run.json', status)
        raise


def resume_tybalt_twohidden100_ig(attempt_path, steps=(2048, 4096, 8192, 16384)):
    """Continue numerical refinement from preserved fitted objects, without any refit."""
    import shutil
    from datetime import datetime, timezone
    from uuid import uuid4
    old = Path(attempt_path)
    provenance = json.loads((old/'provenance.json').read_text())
    if json.loads((old/'run.json').read_text())['status'] != 'failed':
        raise ValueError('Resume only a stopped, unpublished interpretation attempt')
    sources = provenance['sources']
    if metabric_source_fingerprint(sources) != sources:
        raise ValueError('Source fingerprint mismatch; cannot resume Tybalt IG')
    files = list(sources)
    X, y, patients = prepare_tybalt_ig_inputs(files[0], files[1], files[3])
    if patients.tolist() != pd.read_csv(old/'patient_order.csv').sample_id.tolist() or list(X.columns) != pd.read_csv(old/'gene_order.csv').gene_name.tolist():
        raise ValueError('Gene or patient order mismatch on resume')
    pipeline = joblib.load(old/'interpretation_pipeline.joblib')
    probability, latent = compose_tybalt_ig_probability(pipeline)
    reference = pd.read_csv(old/'prediction_agreement.csv').pipeline_probability.to_numpy()
    if not np.allclose(pipeline.predict_proba(X)[:,1], reference, atol=1e-10, rtol=0):
        raise ValueError('Restored fitted model differs from original interpretation predictions')
    if np.max(np.abs(probability(tf.constant(X.to_numpy(),tf.float32)).numpy()-reference)) > 1e-5:
        raise ValueError('Restored differentiable path disagrees with fitted predictions')
    destination = old.parent / ('attempt_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    shutil.copytree(old, destination)
    start = time.monotonic()
    state = dict(status='running_ig_refinement', interpretation_only=True, resumed_from=str(old), no_refit=True,
                 started_utc=datetime.now(timezone.utc).isoformat())
    (destination/'run.json').write_text(json.dumps(state, indent=2))
    baseline = pd.read_csv(old/'baseline.csv').aligned_expression_baseline.to_numpy()
    raw = X.copy(); raw.index = pd.Index(patients,name='sample_id')
    earlier = pd.read_csv(old/'polymetastatic_probability_ig_numerical_checks.csv')
    try:
        ranking, checks = save_tybalt_ig_checks(probability, raw, baseline, destination, steps=steps)
        # First repeated resolution establishes the previous attribution array; keep its original change diagnostic.
        for column in ['mean_absolute_change','max_absolute_change']:
            if checks.iloc[0].steps in earlier.steps.values:
                checks.loc[checks.index[0],column] = earlier.set_index('steps').loc[checks.iloc[0].steps,column]
        combined = pd.concat([earlier.loc[earlier.steps < checks.iloc[0].steps],checks],ignore_index=True)
        combined.to_csv(destination/'polymetastatic_probability_ig_numerical_checks.csv',index=False)
        provenance.update(resumed_from=str(old), integration_steps=combined.steps.astype(int).tolist(), no_refit_on_resume=True)
        (destination/'provenance.json').write_text(json.dumps(provenance,indent=2))
        state.update(status='complete',elapsed_seconds=time.monotonic()-start,ended_utc=datetime.now(timezone.utc).isoformat())
        (destination/'run.json').write_text(json.dumps(state,indent=2))
        if metabric_source_fingerprint(sources) != sources:
            raise ValueError('Source files changed during IG refinement')
        hashes = metabric_source_fingerprint([p for p in destination.iterdir() if p.is_file()])
        (destination/'_completion.json').write_text(json.dumps({'interpretation_only':True,
            'files':{Path(k).name:v for k,v in hashes.items()}},indent=2))
        final = destination.with_name(destination.name.replace('attempt_','validated_',1))
        os.rename(destination,final)
        print('TYBALT IG COMPLETE:',final,flush=True)
        print(combined.to_string(index=False),flush=True)
        print(ranking.head(20).to_string(index=False),flush=True)
        return final
    except Exception as exc:
        state.update(status='failed',error=str(exc),elapsed_seconds=time.monotonic()-start)
        (destination/'run.json').write_text(json.dumps(state,indent=2))
        raise


def load_tybalt_ig_results(bundle_path):
    """Read an integrity-checked completed Tybalt IG bundle, without loading a model."""
    root = Path(bundle_path)
    if not root.name.startswith('validated_'):
        raise ValueError('Only a validated Tybalt interpretation bundle may be displayed')
    completion = json.loads((root/'_completion.json').read_text())
    if completion.get('interpretation_only') is not True:
        raise ValueError('Missing interpretation-only completion contract')
    for filename, expected in completion['files'].items():
        if Path(filename).name != filename:
            raise ValueError('Completion manifest must use local filenames')
        path = root/filename
        if metabric_source_fingerprint([path])[str(path)] != expected:
            raise ValueError(f'Tybalt interpretation artifact integrity mismatch: {filename}')
    params = json.loads((root/'interpretation_only_params.json').read_text())
    agreement = json.loads((root/'agreement.json').read_text())
    provenance = json.loads((root/'provenance.json').read_text())
    if params.get('representation') != 'tybalt_twohidden100_logreg_l1' or params.get('interpretation_only') is not True:
        raise ValueError('Unexpected interpretation model')
    checks = pd.read_csv(root/'polymetastatic_probability_ig_numerical_checks.csv')
    last = checks.iloc[-1]
    if last.max_absolute_completeness_residual > .001 or last.mean_absolute_change > .001 or last.max_absolute_change > .001:
        raise ValueError('Published IG does not satisfy its numerical contract')
    ranking = pd.read_csv(root/'polymetastatic_probability_ig_gene_ranking.csv')
    if len(ranking) != 5000 or ranking.gene_name.duplicated().any():
        raise ValueError('Invalid input-gene ranking')
    return dict(params=params, agreement=agreement, checks=checks, ranking=ranking, provenance=provenance)


# Presentation helpers consolidated from utils_notebook_display
"""Presentation-only numeric formatting; never modifies analytical objects."""
import builtins
import numbers
import re
import numpy as np
import pandas as pd
from IPython.display import display as _notebook_ipython_display

def notebook_label(key):
    return '_'.join(map(str, key)) if isinstance(key, tuple) else str(key)

def notebook_kind(key):
    key = notebook_label(key).lower().replace(' ', '_').replace('-', '_')
    if any((s in key for s in ('param', 'config', 'learning_rate', 'tolerance', 'epsilon', 'threshold'))):
        return 'exact'
    if key in {'c', 'k', 'alpha', 'beta', 'seed', 'random_state'} or key.startswith(('classifier__', 'clf__', 'selector__')) or key in {'max_features', 'max_samples', 'max_depth', 'min_samples_leaf', 'min_samples_split', 'n_estimators', 'scale_pos_weight', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'min_child_weight', 'max_leaves'}:
        return 'exact'
    if any((s in key for s in ('p_value', 'pvalue', 'q_value', 'qvalue', 'p_adjust', 'padj', 'fdr'))) or re.search('(^|_)(p|q|pval|qval)($|_)', key):
        return 'significant'
    if any((s in key for s in ('additivity_error', 'residual', 'max_probability_difference', 'max_latent_difference'))):
        return 'significant'
    if any((s in key for s in ('pct', 'percent'))):
        return 'percentage'
    if not any((s in key for s in ('mean', 'std', 'fraction', 'ratio', 'proportion', 'frequency', 'score'))):
        if key.startswith(('n_', 'num_')) or key in {'n', 'count', 'counts', 'rank', 'outer_fold', 'fold', 'events', 'patients', 'features', 'genes', 'n_splits', 'candidate_count', 'background_n'} or key.endswith(('_count', '_counts', '_rank', '_folds')):
            return 'count'
    return 'default'

def notebook_number(value, key='', precision=3):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        return value
    if pd.isna(value):
        return 'NaN'
    if not np.isfinite(value):
        return str(value)
    kind = notebook_kind(key)
    if kind == 'exact':
        return str(value)
    if kind == 'significant':
        return f'{value:.3g}'
    if kind == 'percentage':
        return f'{value:.1f}'
    if kind == 'count' and float(value).is_integer():
        return str(int(value))
    if isinstance(value, numbers.Integral):
        return str(value)
    return f'{value:.{precision}f}'

def notebook_object(value, key=''):
    if notebook_kind(key) == 'exact':
        return value
    if isinstance(value, dict):
        return {k: notebook_object(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)((notebook_object(v, key) for v in value))
    return notebook_number(value, key)

def notebook_table(frame):
    """Return a formatted display copy, leaving values/dtypes of frame intact."""
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    shown = frame.copy(deep=True).astype(object)
    for j, col in enumerate(frame.columns):
        values = frame.iloc[:, j]
        precision = 3
        if any((s in notebook_label(col).lower() for s in ('shap', 'importance'))) and pd.api.types.is_numeric_dtype(values):
            finite = values.replace([np.inf, -np.inf], np.nan).dropna()
            if finite.nunique() > 1 and finite.nunique() - finite.round(3).nunique() >= 2 and (finite.round(4).nunique() > finite.round(3).nunique()):
                precision = 4
        keys = [col] * len(frame)
        if str(col).lower() == 'value' and any(('param' in str(c).lower() for c in frame.columns)):
            keys = ['parameter_value'] * len(frame)
        else:
            keys = [idx if notebook_kind(idx) in {'count', 'significant', 'percentage'} and notebook_kind(col) == 'default' else col for idx in frame.index]
        shown.iloc[:, j] = [notebook_object(v, key) if isinstance(v, (dict, list, tuple)) else notebook_number(v, key, precision) for v, key in zip(values, keys)]
    shown = shown.rename(columns={'primary_only': 'common_cohort_only', 'primary_outer_fold_union': 'common_cohort_outer_fold_union', 'in_primary_outer_fold_input_union': 'in_common_cohort_outer_fold_input_union'})
    return shown

def thesis_display(*objects, **kwargs):
    """Format only display copies; forward figures/Markdown and other objects."""
    formatted = []
    for obj in objects:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            obj = notebook_table(obj)
        elif isinstance(obj, dict):
            obj = notebook_object(obj)
        formatted.append(obj)
    return _notebook_ipython_display(*formatted, **kwargs)

def thesis_print(*objects, **kwargs):
    """Format notebook print arguments without altering their source objects."""
    key = objects[0] if objects and isinstance(objects[0], str) else ''
    values = []
    for obj in objects:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            obj = notebook_table(obj).to_string()
        elif isinstance(obj, (dict, list, tuple)):
            obj = notebook_object(obj, key)
        else:
            obj = notebook_number(obj, key)
        values.append(obj)
    return builtins.print(*values, **kwargs)


# Presentation helpers consolidated from utils_gene_display
"""Exact, symbol-first display labels; analytical objects are never modified."""
from functools import lru_cache
from pathlib import Path
import re
import warnings
from contextlib import contextmanager
import pandas as pd

@lru_cache(maxsize=1)
def gene_symbols():
    """Use preserved project annotations only; ambiguous IDs retain their ID."""
    base = Path('/mnt/data3/MCPP-Breast')
    paths = [base / 'Transcriptomic-Data/Preprocessed/transcriptomics_feature_name_mapping.csv']
    paths += sorted((base / 'Machine-Learning-Models-poly_vs_oligo/Mutation-Data').glob('*.csv'))
    pairs = {}
    for path in paths:
        columns = pd.read_csv(path, nrows=0).columns
        keys = ['ensembl_gene_id', 'gene_name'] if {'ensembl_gene_id', 'gene_name'} <= set(columns) else ['gene', 'gene_symbol'] if {'gene', 'gene_symbol'} <= set(columns) else None
        if keys:
            frame = pd.read_csv(path, usecols=keys, dtype=str).dropna()
            for identifier, symbol in frame[keys].itertuples(index=False, name=None):
                pairs.setdefault(identifier, set()).add(symbol)
    import json
    groups = Path(__file__).with_name('common_cohort_mutation_identifier_groups.json')
    if groups.exists():
        for symbol, identifiers in json.loads(groups.read_text())['groups'].items():
            for identifier in identifiers:
                pairs.setdefault(identifier, set()).add(symbol)
    return {identifier: next(iter(symbols)) for identifier, symbols in pairs.items() if len(symbols) == 1}


@contextmanager
def quiet_candidate_f_tests():
    """Only the two known sklearn ANOVA warnings; no score/data changes."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='^Features \\[.*\\] are constant\\.$', category=UserWarning, module='^sklearn\\.feature_selection\\._univariate_selection$')
        warnings.filterwarnings('ignore', message='^invalid value encountered in divide$', category=RuntimeWarning, module='^sklearn\\.feature_selection\\._univariate_selection$')
        yield

def plot_saved_shap_bar(path):
    """Plot preserved aggregate SHAP values without recalculating attributions."""
    import matplotlib.pyplot as plt
    ranking = pd.read_csv(path).head(20).iloc[::-1]
    labels = [interpretation_label(f) if interpretation_label(f) != f else interpretation_label(d) for f, d in zip(ranking.feature, ranking.display_name)]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(range(len(ranking)), ranking.mean_abs_shap, color='#008bfb', height=0.7)
    ax.set_yticks(range(len(ranking)), labels=labels)
    ax.set_xlabel('Mean |SHAP| in probability units')
    ax.set_title('Strict N=110 — probability-scale SHAP\nInterpretation-only full-cohort RF', fontsize=11)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.tight_layout()
    return fig
_CLINICAL_LABELS = {'age_at_diagnosis': 'Age at diagnosis', 'abc_age_at_diagnosis': 'Age at diagnosis', 'abc_ajcc_x_n_N0': 'AJCC N0'}

def interpretation_label(value):
    """Symbol-only table label from exact annotations, retaining unresolved IDs."""
    if not isinstance(value, str):
        return value
    latent = re.fullmatch('(?:metabric_cbioportal_\\w+_)?latent_(\\d+)', value)
    if latent:
        return f'Latent {latent.group(1)}'
    modality_label = re.fullmatch('(.+) \\[(?:mutation|transcriptomics)\\]', value)
    if modality_label and modality_label.group(1) in gene_symbols().values():
        return modality_label.group(1)
    match = re.fullmatch('.+ \\(([^()]+)\\)', value)
    original = match.group(1) if match else value
    identifier = re.sub('^(?:mutation__?|transcriptomics__?)', '', original)
    symbol = gene_symbols().get(identifier)
    if symbol:
        return symbol
    marker_prefix = 'metabric_cbioportal_rostami_marker_expression__'
    if original.startswith(marker_prefix):
        symbol = original[len(marker_prefix):]
        if symbol in gene_symbols().values():
            return symbol
    clinical = re.sub('^clinical__?', '', original)
    if clinical in _CLINICAL_LABELS:
        return _CLINICAL_LABELS[clinical]
    if match:
        return original
    if re.fullmatch('ENSG\\d+(?:\\.\\d+)?|\\d+', identifier):
        return identifier
    return value

def interpretation_table(frame, *, multimodal=False):
    """Simplify a reader-facing table copy only, never its underlying artifact."""
    if not isinstance(frame, pd.DataFrame):
        return frame
    shown = frame.copy(deep=True)
    shown = shown.drop(columns=['original_feature_name', 'model_feature_name'], errors='ignore')
    label_columns = ['feature', 'gene', 'gene_name', 'gene_symbol', 'display_name', 'ensembl_gene_id']
    for column in label_columns:
        if column in shown:
            shown[column] = shown[column].map(interpretation_label)
    primary = next((c for c in ['feature', 'gene', 'gene_symbol', 'gene_name'] if c in shown), None)
    if primary is not None:
        for column in ['gene_name', 'gene_symbol', 'ensembl_gene_id', 'display_name']:
            if column == primary or column not in shown:
                continue
            values = shown[column]
            missing = values.isna() | values.astype(str).isin(['', 'nan', 'NaN', 'None', '<NA>'])
            if (missing | values.eq(shown[primary])).all():
                shown = shown.drop(columns=column)
    if 'shared_genes' in shown:
        shown['shared_genes'] = shown['shared_genes'].map(lambda values: [interpretation_label(v) for v in values] if isinstance(values, list) else values)
    if shown.index.name in ['feature', 'gene', 'gene_name']:
        shown.index = shown.index.map(interpretation_label)
    if not multimodal and any((c in shown for c in ['feature', 'gene', 'gene_name'])):
        for column in ['feature_modality', 'modality']:
            if column in shown and shown[column].nunique(dropna=True) <= 1:
                shown = shown.drop(columns=column)
    return shown

def display_interpretation(*objects, **kwargs):
    """Display single-modality rankings without implementation-name columns."""
    thesis_display(*(interpretation_table(obj) for obj in objects), **kwargs)

def display_multimodal_interpretation(*objects, **kwargs):
    """Keep modality labels in multimodal rankings, including top-N subsets."""
    thesis_display(*(interpretation_table(obj, multimodal=True) for obj in objects), **kwargs)

def symbol_only_saved_plot(source):
    """Relabel a recognised saved PNG margin; preserve its full original bytes.

    Layouts are keyed by PNG SHA-256 so a label mask cannot be applied to a
    different figure. Unrecognised images are displayed unchanged. No SHAP
    values, model, axes, points, bars or colour scales are reconstructed.
    """
    import base64
    import hashlib
    import json
    from html import escape
    from IPython.display import Image, SVG
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    layouts = json.loads(Path(__file__).with_name('gene_plot_display_layouts.json').read_text())
    layout = layouts.get(hashlib.sha256(raw).hexdigest())
    if layout is None:
        return Image(data=raw)
    if layout.get('mode') == 'raster_panel_reflow':
        return _reflow_saved_raster(raw, layout)
    width, height = layout['size']
    x, y, box_width, box_height = layout['label_region']
    encoded = base64.b64encode(raw).decode('ascii')
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', f'<image width="{width}" height="{height}" xlink:href="data:image/png;base64,{encoded}"/>', f'<rect x="{x}" y="{y}" width="{box_width}" height="{box_height}" fill="white"/>']
    for name, centre in zip(layout['labels'], layout['y_centres']):
        svg.append(f'''<text x="{x + box_width - 5}" y="{centre}" text-anchor="end" dominant-baseline="central" font-family="DejaVu Sans, sans-serif" font-size="{layout['font_size']}" fill="#333333">{escape(name)}</text>''')
    return SVG(data=''.join(svg) + '</svg>')

def _reflow_saved_raster(raw, layout):
    """Relocate/resize preserved raster panels; never reconstruct SHAP data.

    Caller verifies the original image hash. Each source rectangle is displayed
    by an affine mapping; numeric ticks and labels follow that same mapping.
    The complete original PNG remains embedded for provenance.
    """
    import base64
    from html import escape
    from IPython.display import SVG
    width, height = layout['canvas_size']
    iw, ih = layout['size']
    encoded = base64.b64encode(raw).decode('ascii')
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<defs><image id="original-shap" width="{iw}" height="{ih}" xlink:href="data:image/png;base64,{encoded}"/></defs>']
    for panel in layout['panels']:
        sx, sy, sw, sh = panel['source']
        dx, dy, dw, dh = panel['destination']
        parts.append(f'<svg x="{dx}" y="{dy}" width="{dw}" height="{dh}" viewBox="{sx} {sy} {sw} {sh}" preserveAspectRatio="none" overflow="hidden"><use xlink:href="#original-shap"/></svg>')
    for text in layout['texts']:
        x, y = (text['x'], text['y'])
        rotation = f''' transform="rotate({text['rotate']} {x} {y})"''' if 'rotate' in text else ''
        parts.append(f'''<text x="{x}" y="{y}" text-anchor="{text.get('anchor', 'middle')}" dominant-baseline="{text.get('baseline', 'auto')}" font-family="DejaVu Sans, sans-serif" font-size="{text['size']}" fill="#222222"{rotation}>{escape(text['text'])}</text>''')
    return SVG(data=''.join(parts) + '</svg>')


# Presentation helpers consolidated from utils_latent_display
"""Presentation-only labels and plots for preserved latent-model results."""
import matplotlib.pyplot as plt

def latent_model_label(name):
    """Keep representation and classifier identity without internal prefixes."""
    family = next((label for suffix, label in [('logreg_l1', 'Logistic Regression (L1)'), ('random_forest', 'Random Forest'), ('xgboost', 'XGBoost')] if str(name).endswith(suffix)), str(name))
    if 'hvae' in str(name):
        return 'HVAE — ' + family
    if 'rna_only' in str(name):
        return 'RNA-only VAE — ' + family
    return family

def plot_latent_outer_scores(wide):
    """Plot supplied inner scores, with one discrete tick per outer fold."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for name in wide:
        ax.plot(wide.index, wide[name], marker='o', label=latent_model_label(name))
    ax.set(xticks=[1, 2, 3, 4, 5], xlabel='Outer fold', ylabel='Inner-CV macro-F1', title='Inner-CV macro-F1 by outer fold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9)
    fig.tight_layout()
    return fig

def plot_latent_train_test(means, sd=None):
    """Display existing means/SDs; do not calculate performance or gap values."""
    columns = ['train_roc_auc', 'test_roc_auc']
    labels = {'train_roc_auc': 'Outer-train AUC (apparent)', 'test_roc_auc': 'Outer-test AUC (held-out)'}
    shown = means[columns].rename(columns=labels, index=latent_model_label)
    errors = None if sd is None else sd[columns].rename(columns=labels, index=latent_model_label)
    ax = shown.plot.bar(yerr=errors, capsize=3, figsize=(10, 4.5), color=['#7d99ad', '#246a91'], rot=0)
    ax.set(xlabel='', ylabel='ROC-AUC', title='Apparent outer-training vs held-out outer-test AUC')
    ax.set_xticklabels([label.replace(' — ', '\n').replace('Logistic Regression ', 'Logistic Regression\n') for label in shown.index], fontsize=8)
    ax.legend(fontsize=9, loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2)
    ax.set_title('Apparent outer-training vs held-out outer-test AUC', pad=38)
    ax.figure.tight_layout()
    return ax.figure

def plot_latent_train_test_folds(results):
    """Display five preserved train/test fold values per model, without refitting."""
    import numpy as np
    from matplotlib.patches import Patch
    groups = list(results.groupby('representation', sort=True))
    for name, frame in groups:
        if len(frame) != 5 or set(frame.outer_fold) != {1, 2, 3, 4, 5}:
            raise ValueError(f'{name}: expected exactly one row for each outer fold 1–5.')
        if not np.isfinite(frame[['train_roc_auc', 'test_roc_auc']].to_numpy()).all():
            raise ValueError(f'{name}: train/test AUC values must be finite.')
    fig, ax = plt.subplots(figsize=(12, 5.5))
    positions = np.arange(len(groups))
    handles = []
    for column, offset, color, label in [('train_roc_auc', -0.18, '#7d99ad', 'Outer-train AUC (apparent)'), ('test_roc_auc', 0.18, '#246a91', 'Outer-test AUC (held-out)')]:
        values = [frame.sort_values('outer_fold')[column].to_numpy() for _, frame in groups]
        ax.boxplot(values, positions=positions + offset, widths=0.3, patch_artist=True, manage_ticks=False, showfliers=False, boxprops={'facecolor': color, 'edgecolor': color}, medianprops={'color': '#222222', 'linewidth': 1.5})
        for x, fold_values in zip(positions + offset, values):
            ax.scatter(x + np.linspace(-0.055, 0.055, 5), fold_values, s=18, color='#222222', alpha=0.7, zorder=3)
        handles.append(Patch(facecolor=color, label=label))
    ax.set_xticks(positions, [latent_model_label(name) for name, _ in groups], rotation=35, ha='right', fontsize=9)
    ax.set(ylabel='ROC-AUC', title='Train vs test — five outer folds')
    ax.legend(handles=handles, loc='lower left', fontsize=9)
    fig.tight_layout()
    return fig


# Presentation helpers consolidated from utils_common_cohort_presentation
"""Presentation of preserved common-cohort results; no fitting or metric computation."""
import base64
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
MODALITY_NAMES = {'clinical': 'Clinical', 'mutation': 'Mutation', 'transcriptomics': 'Transcriptomics', 'clinical_mutation': 'Clinical + Mutation', 'clinical_transcriptomics': 'Clinical + Transcriptomics', 'mutation_transcriptomics': 'Mutation + Transcriptomics', 'clinical_mutation_transcriptomics': 'Clinical + Mutation + Transcriptomics'}
FAMILY_NAMES = {'logreg_l1': 'Logistic Regression (L1)', 'logreg_l2': 'Logistic Regression (L2)', 'random_forest': 'Random Forest', 'xgboost': 'XGBoost'}

def plot_common_cohort_comparison(selected):
    """Plot the seven preserved selected-family pooled AUCs, in modality order."""
    frame = selected.set_index('configuration').loc[list(MODALITY_NAMES)]
    if not frame.N.eq(110).all():
        raise ValueError('This comparison requires N=110 in every saved row.')
    fig, ax = plt.subplots(figsize=(10, 5.2))
    positions = np.arange(len(frame))
    ax.scatter(frame.roc_auc, positions, color='#246a91', s=55)
    for y, value in zip(positions, frame.roc_auc):
        ax.annotate(f'{value:.3f}', (value, y), xytext=(7, 0), textcoords='offset points', va='center')
    ax.set_yticks(positions, list(MODALITY_NAMES.values()))
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel='Pooled OOF ROC-AUC', title='Predictive performance across modality combinations\nCommon cohort (N=110); identical frozen outer folds')
    ax.axvline(0.5, color='grey', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.2)
    fig.tight_layout()
    return fig

def plot_clinical_apparent_heldout(comparison):
    """Use saved means and SDs verbatim; no gap or inner-score bars."""
    frame = comparison.set_index('model')
    positions = np.arange(len(frame))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for offset, split, label, color in [(-width / 2, 'train', 'Outer-train AUC (apparent)', '#7d99ad'), (width / 2, 'test', 'Outer-test AUC (held-out)', '#246a91')]:
        ax.bar(positions + offset, frame[f'auc_score_{split}'], width, yerr=frame[f'auc_score_{split}_std'], capsize=4, color=color, label=label)
    ax.set_xticks(positions, [FAMILY_NAMES.get(x, x) for x in frame.index], rotation=45, ha='right')
    ax.set(ylabel='Mean ROC-AUC ± outer-fold SD', ylim=(0, 1.08), title='Outer-training apparent AUC vs held-out outer-test AUC\nClinical full-availability analysis (N=149)')
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig


# ==============================================================================
# TCGA-BRCA prognostic association helpers
# ==============================================================================
"""Public-cohort prognostic association of a prespecified SHAP gene set."""
import gzip as _tcga_prognostic_gzip
import warnings as _tcga_prognostic_warnings
import numpy as _tcga_prognostic_np
import pandas as _tcga_prognostic_pd
from lifelines import CoxPHFitter as _tcga_prognostic_CoxPHFitter
from lifelines.exceptions import ConvergenceWarning as _tcga_prognostic_ConvergenceWarning
from lifelines.statistics import proportional_hazard_test as _tcga_prognostic_proportional_hazard_test
from statsmodels.stats.multitest import multipletests as _tcga_prognostic_multipletests


def load_tcga_gene_cohort(expression_path, clinical_path, genes):
    """Validate repeated CDR records and stream only the frozen expression rows."""
    clinical = _tcga_prognostic_pd.read_csv(clinical_path, sep='\t', dtype=str)
    clinical = clinical.loc[clinical['cancer type abbreviation'].eq('BRCA')].copy()
    if not clinical['sample'].str[:12].eq(clinical['_PATIENT']).all():
        raise ValueError('CDR sample/patient identifiers disagree.')
    numeric = ['age_at_initial_pathologic_diagnosis'] + [k for e in ['PFI', 'DFI', 'OS', 'DSS'] for k in [e, e + '.time']]
    for col in numeric:
        clinical[col] = _tcga_prognostic_pd.to_numeric(clinical[col], errors='coerce')
    checked = numeric + ['ajcc_pathologic_tumor_stage']
    consistency = clinical.groupby('_PATIENT')[checked].nunique(dropna=False)
    if (consistency > 1).any().any():
        raise ValueError('Conflicting repeated CDR endpoint/covariate records.')
    clinical = clinical.sort_values('sample').drop_duplicates('_PATIENT').set_index('_PATIENT')
    ids = set(genes['ensembl_gene_id'])
    rows = {}
    with _tcga_prognostic_gzip.open(expression_path, 'rt') as stream:
        samples = stream.readline().rstrip('\n').split('\t')[1:]
        primary = sorted(((s, i) for i, s in enumerate(samples) if s[:12] in clinical.index and s[13:15] == '01'))
        chosen = {}
        for sample, index in primary:
            chosen.setdefault(sample[:12], (sample, index))
        positions = [index for sample, index in chosen.values()]
        for line in stream:
            versioned, _, values = line.partition('\t')
            key = versioned.split('.')[0]
            if key in ids:
                if key in rows:
                    raise ValueError('Duplicate expression row for ' + key)
                values = values.rstrip('\n').split('\t')
                rows[key] = [float(values[i]) for i in positions]
    if set(rows) != ids:
        raise ValueError('Prespecified genes missing from expression: ' + str(ids - set(rows)))
    expression = _tcga_prognostic_pd.DataFrame(rows, index=list(chosen)).loc[:, genes['ensembl_gene_id']]
    if not _tcga_prognostic_np.isfinite(expression.to_numpy()).all():
        raise ValueError('Non-finite expression; no silent patient-specific omission allowed.')
    cohort = clinical.loc[expression.index].join(expression, validate='one_to_one')
    summary = [{'cohort': 'BRCA primary expression matched to CDR', 'N': len(cohort), 'events': _tcga_prognostic_np.nan, 'censored': _tcga_prognostic_np.nan, 'expression_samples': len(primary), 'duplicate_primary_samples_removed': len(primary) - len(chosen)}]
    for endpoint in ['PFI', 'DFI', 'OS', 'DSS']:
        eligible = endpoint_cohort(cohort, endpoint)
        summary.append({'cohort': endpoint, 'N': len(eligible), 'events': int(eligible[endpoint].sum()), 'censored': int((eligible[endpoint] == 0).sum())})
    adjusted = endpoint_cohort(cohort, 'PFI', adjusted=True)
    summary.append({'cohort': 'PFI age/stage adjusted', 'N': len(adjusted), 'events': int(adjusted.PFI.sum()), 'censored': int((adjusted.PFI == 0).sum())})
    return (cohort, _tcga_prognostic_pd.DataFrame(summary))

def endpoint_cohort(cohort, endpoint, adjusted=False):
    mask = cohort[endpoint].isin([0, 1]) & _tcga_prognostic_np.isfinite(cohort[endpoint + '.time']) & (cohort[endpoint + '.time'] > 0)
    data = cohort.loc[mask].copy()
    if adjusted:
        stages = {'Stage ' + suffix: stage for stage, suffixes in {'I': ['I', 'IA', 'IB', 'IC'], 'II': ['II', 'IIA', 'IIB', 'IIC'], 'III': ['III', 'IIIA', 'IIIB', 'IIIC'], 'IV': ['IV', 'IVA', 'IVB', 'IVC']}.items() for suffix in suffixes}
        data['stage'] = data['ajcc_pathologic_tumor_stage'].map(stages)
        data['age'] = data['age_at_initial_pathologic_diagnosis']
        data = data.loc[data.stage.notna() & _tcga_prognostic_np.isfinite(data.age)].copy()
        data['stage'] = _tcga_prognostic_pd.Categorical(data['stage'], categories=['I', 'II', 'III', 'IV'])
    return data



# ==============================================================================
# MCPP expression and overall-survival helpers
# ==============================================================================
"""Prespecified MCPP gene-expression characterization and metastatic-origin OS.

These analyses do not select predictive models or alter their artifacts.
"""
from pathlib import Path as _mcpp_biological_Path
import hashlib as _mcpp_biological_hashlib
import json as _mcpp_biological_json
import numpy as _mcpp_biological_np
import pandas as _mcpp_biological_pd
import matplotlib.pyplot as _mcpp_biological_plt
from lifelines import CoxPHFitter as _mcpp_biological_CoxPHFitter, KaplanMeierFitter as _mcpp_biological_KaplanMeierFitter
from lifelines.statistics import logrank_test as _mcpp_biological_logrank_test, proportional_hazard_test as _mcpp_biological_proportional_hazard_test
from lifelines.plotting import add_at_risk_counts as _mcpp_biological_add_at_risk_counts
from lifelines.utils import median_survival_times as _mcpp_biological_median_survival_times
_mcpp_biological_SEED = 42
_mcpp_biological_GROUPS = {0: 'Oligometastatic', 1: 'Polymetastatic'}
_mcpp_biological_COLORS = {0: '#2878a5', 1: '#c55a3e'}

def sha256(path):
    return _mcpp_biological_hashlib.sha256(_mcpp_biological_Path(path).read_bytes()).hexdigest()

def inputs(base):
    base = _mcpp_biological_Path(base)
    return {'clinical': base / 'Clinical-Data/20260406-mcpp-breast-clinical-data.csv'}

def save_table(frame, path):
    """Refuse to replace an existing result; analyses publish only new files."""
    path = _mcpp_biological_Path(path)
    with path.open('x') as handle:
        frame.to_csv(handle, index=False)

def save_json(value, path):
    with _mcpp_biological_Path(path).open('x') as handle:
        _mcpp_biological_json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')

def save_figure(fig, path):
    with _mcpp_biological_Path(path).open('xb') as handle:
        fig.savefig(handle, format='png', dpi=180, bbox_inches='tight')

def _mcpp_biological_normalise_id(value):
    if _mcpp_biological_pd.isna(value):
        raise ValueError('Missing sample identifier.')
    value = str(value).strip().upper().replace('#', '')
    return value[:-2] + 'B' if value.endswith('BC') else value

def plot_expression(expression, labels, genes):
    fig, axes = _mcpp_biological_plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
    rng = _mcpp_biological_np.random.default_rng(_mcpp_biological_SEED)
    for ax, g in zip(axes.flat, genes.itertuples()):
        values = [expression.loc[labels.eq(k), g.ensembl_gene_id].values for k in [0, 1]]
        ax.boxplot(values, positions=[0, 1], widths=0.48, showfliers=False, medianprops={'color': 'black'}, zorder=3)
        for k in [0, 1]:
            ax.scatter(k + rng.uniform(-0.17, 0.17, len(values[k])), values[k], s=15, alpha=0.65, color=_mcpp_biological_COLORS[k], edgecolors='none', zorder=2)
        ax.set(title=g.gene_symbol, xticks=[0, 1], xticklabels=['Oligo (43)', 'Poly (67)'], ylabel='log2(TPM + 1)')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Prespecified genes: expression by baseline metastatic phenotype')
    return fig


def prepare_endpoint(paths, cohort):
    c = _mcpp_biological_pd.read_csv(paths['clinical'], sep=';')
    c = c.loc[c.FFPE_type.eq('Breast')].copy()
    c['sample_key'] = c.sample_id.map(_mcpp_biological_normalise_id)
    c = c.loc[c.sample_key.isin(cohort.index)].set_index('sample_key')
    if c.index.duplicated().any() or set(c.index) != set(cohort.index):
        raise ValueError('Clinical linkage must be one-to-one.')
    c = c.loc[cohort.index]
    if not _mcpp_biological_np.array_equal(c.record_id, cohort.record_id) or not _mcpp_biological_np.array_equal(c.Oligo_or_Pluri.map({'Oligometastatic': 0, 'Plurimetastatic': 1}), cohort.y_true):
        raise ValueError('Clinical identity or baseline phenotype mismatch.')
    cols = ['abc_m1_diagnosis', 'death_date', 'last_fu', 'abc_date_of_birth', 'abc_date_of_diagnosis', 'oligometastatic_diag_date', 'oligo_to_pluri_date']
    d = {k: _mcpp_biological_pd.to_datetime(c[k], format='%d/%m/%Y', errors='coerce') for k in cols}
    end = d['death_date'].combine_first(d['last_fu'])
    duration = (end - d['abc_m1_diagnosis']).dt.days
    invalid = sum((int((c[k].notna() & d[k].isna()).sum()) for k in cols))
    checks = dict(invalid_nonmissing_dates=invalid, missing_duration=int(duration.isna().sum()), nonpositive_duration=int(duration.le(0).sum()), birth_after_primary=int(d['abc_date_of_birth'].gt(d['abc_date_of_diagnosis']).sum()), primary_after_metastatic=int(d['abc_date_of_diagnosis'].gt(d['abc_m1_diagnosis']).sum()), both_death_and_followup=int((d['death_date'].notna() & d['last_fu'].notna()).sum()), oligo_origin_mismatch=int((cohort.y_true.eq(0) & d['oligometastatic_diag_date'].ne(d['abc_m1_diagnosis'])).sum()), progression_outside_followup=int((d['oligo_to_pluri_date'].lt(d['abc_m1_diagnosis']) | d['oligo_to_pluri_date'].gt(end)).sum()))
    if any(checks.values()):
        raise ValueError(f'Endpoint QC needs review: {checks}')
    data = _mcpp_biological_pd.DataFrame({'duration_days': duration, 'event': d['death_date'].notna().astype(int), 'baseline_polymetastatic': cohort.y_true, 'age_at_metastatic_diagnosis': (d['abc_m1_diagnosis'] - d['abc_date_of_birth']).dt.days / 365.25})
    if len(data) != 110 or data.event.sum() != 84 or (not _mcpp_biological_np.isfinite(data.values).all()) or data.age_at_metastatic_diagnosis.le(0).any():
        raise ValueError('Endpoint does not match the approved 110 patients / 84 deaths.')
    summary = data.groupby('baseline_polymetastatic').agg(N=('event', 'size'), deaths=('event', 'sum')).reset_index()
    summary['censored'] = summary.N - summary.deaths
    summary['group'] = summary.baseline_polymetastatic.map(_mcpp_biological_GROUPS)
    checks['N'] = 110
    checks['deaths'] = 84
    checks['censored'] = 26
    checks['baseline_oligo_later_progressed'] = int((cohort.y_true.eq(0) & c.oligo_to_pluri.eq(1)).sum())
    checks['reference_group'] = 'baseline oligometastatic; later progression does not change group'
    return (data, summary, checks)

def fit_cox(data, covariates, model_label):
    frame = data[['duration_days', 'event', *covariates]].copy()
    fitter = _mcpp_biological_CoxPHFitter().fit(frame, duration_col='duration_days', event_col='event')
    ph = _mcpp_biological_proportional_hazard_test(fitter, frame, time_transform='rank')
    rows = []
    for term in covariates:
        s = fitter.summary.loc[term]
        rows.append(dict(model=model_label, term=term, N=len(frame), events=int(frame.event.sum()), coefficient=s['coef'], HR=s['exp(coef)'], CI_lower=s['exp(coef) lower 95%'], CI_upper=s['exp(coef) upper 95%'], p_value=s['p'], PH_p_value=float(ph.summary.loc[term, 'p']), PH_violation_flag=bool(ph.summary.loc[term, 'p'] < 0.05)))
    return (fitter, _mcpp_biological_pd.DataFrame(rows))

def plot_km(data):
    fig, ax = _mcpp_biological_plt.subplots(figsize=(16, 7))
    fitters = []
    rows = []
    for group in [0, 1]:
        f = data.loc[data.baseline_polymetastatic.eq(group)]
        km = _mcpp_biological_KaplanMeierFitter(label=_mcpp_biological_GROUPS[group]).fit(f.duration_days / 30.4375, event_observed=f.event)
        km.plot_survival_function(ax=ax, ci_show=True, show_censors=True, color=_mcpp_biological_COLORS[group])
        ci = _mcpp_biological_median_survival_times(km.confidence_interval_survival_function_).iloc[0].values
        rows.append(dict(group=_mcpp_biological_GROUPS[group], N=len(f), events=int(f.event.sum()), median_OS_months=km.median_survival_time_, median_CI_lower=float(min(ci)), median_CI_upper=float(max(ci))))
        fitters.append(km)
    ticks = _mcpp_biological_np.arange(0, float(data.duration_days.max() / 30.4375) + 1, 12)
    ax.set(xticks=ticks, xlabel='Months from metastatic diagnosis', ylabel='Overall survival probability', title='Baseline metastatic phenotype and overall survival')
    _mcpp_biological_add_at_risk_counts(*fitters, ax=ax, xticks=ticks, rows_to_show=['At risk'], at_risk_count_from_start_of_period=True, fontsize=8)
    fig.tight_layout()
    a = data.loc[data.baseline_polymetastatic.eq(0)]
    b = data.loc[data.baseline_polymetastatic.eq(1)]
    test = _mcpp_biological_logrank_test(a.duration_days, b.duration_days, event_observed_A=a.event, event_observed_B=b.event)
    return (fig, _mcpp_biological_pd.DataFrame(rows), _mcpp_biological_pd.DataFrame([{'test': 'two-sided log-rank', 'chi_squared': test.test_statistic, 'p_value': test.p_value, 'N': len(data), 'events': int(data.event.sum())}]))

def plot_ph(fitter, data):
    frame = data[['duration_days', 'event', 'baseline_polymetastatic']]
    residuals = fitter.compute_residuals(frame, kind='scaled_schoenfeld')
    months = frame.loc[residuals.index, 'duration_days'] / 30.4375
    fig, ax = _mcpp_biological_plt.subplots(figsize=(8, 4))
    ax.scatter(months, residuals.baseline_polymetastatic, s=18, alpha=0.65)
    ax.axhline(0, color='grey', ls='--')
    ax.set(xlabel='Months from metastatic diagnosis', ylabel='Scaled Schoenfeld residual', title='Baseline phenotype: proportional-hazards diagnostic')
    fig.tight_layout()
    return fig

def plot_gene_hr(results):
    fig, ax = _mcpp_biological_plt.subplots(figsize=(8, 5))
    y = _mcpp_biological_np.arange(len(results))
    ax.hlines(y, results.CI_lower, results.CI_upper, color='#2878a5')
    ax.scatter(results.HR, y, color='#2878a5')
    ax.axvline(1, color='grey', ls='--')
    ax.set_xscale('log')
    ax.set(yticks=y, yticklabels=results.gene_symbol, xlabel='OS hazard ratio per +1 SD expression (95% CI)', title='Secondary prognostic analysis: eight prespecified genes')
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ==============================================================================
# Primary-origin overall-survival endpoint
# ==============================================================================
"""Complementary survival endpoints and apparent association-model diagnostics."""
import numpy as _survival_supplement_np
import pandas as _survival_supplement_pd
pass
pass

def primary_origin_endpoint(clinical_path, cohort, metastatic):
    """Keep the same patients, events, phenotype and adjustment; change origin only."""
    raw = _survival_supplement_pd.read_csv(clinical_path, sep=';')
    raw = raw.loc[raw.FFPE_type.eq('Breast')].copy()
    raw['sample_key'] = raw.sample_id.map(_mcpp_biological_normalise_id)
    raw = raw.loc[raw.sample_key.isin(cohort.index)].set_index('sample_key')
    if raw.index.duplicated().any() or set(raw.index) != set(cohort.index):
        raise ValueError('Nonunique or incomplete clinical linkage')
    raw = raw.loc[cohort.index]
    dates = {}
    for key in ['abc_date_of_diagnosis', 'abc_m1_diagnosis', 'death_date', 'last_fu']:
        dates[key] = _survival_supplement_pd.to_datetime(raw[key], format='%d/%m/%Y', errors='coerce')
        if (raw[key].notna() & dates[key].isna()).any():
            raise ValueError('Uninterpretable date: ' + key)
    end = dates['death_date'].combine_first(dates['last_fu'])
    duration = (end - dates['abc_date_of_diagnosis']).dt.days
    if duration.isna().any() or duration.le(0).any() or dates['abc_date_of_diagnosis'].gt(dates['abc_m1_diagnosis']).any():
        raise ValueError('Invalid primary-origin chronology')
    result = metastatic.copy()
    result['duration_days'] = duration
    if not _survival_supplement_np.array_equal(result.event, dates['death_date'].notna().astype(int)):
        raise ValueError('Event mismatch')
    return result



# ==============================================================================
# Common-cohort (N=110) comparison
# ==============================================================================
"""Strict training-partition preprocessing and common-patient modality comparison."""
from pathlib import Path as _common_cohort_Path
from functools import partial as _common_cohort_partial
import hashlib as _common_cohort_hashlib
import json as _common_cohort_json
import os as _common_cohort_os
import time as _common_cohort_time
from datetime import datetime as _common_cohort_datetime, timezone as _common_cohort_timezone
import numpy as _common_cohort_np
import pandas as _common_cohort_pd
from scipy.stats import fisher_exact as _common_cohort_fisher_exact
from sklearn.base import BaseEstimator as _common_cohort_BaseEstimator, TransformerMixin as _common_cohort_TransformerMixin
from sklearn.preprocessing import OneHotEncoder as _common_cohort_OneHotEncoder, StandardScaler as _common_cohort_StandardScaler
from sklearn.impute import SimpleImputer as _common_cohort_SimpleImputer
from sklearn.feature_selection import SelectKBest as _common_cohort_SelectKBest, f_classif as _common_cohort_f_classif, mutual_info_classif as _common_cohort_mutual_info_classif
from sklearn.pipeline import Pipeline as _common_cohort_Pipeline
from sklearn.model_selection import StratifiedKFold as _common_cohort_StratifiedKFold, ParameterGrid as _common_cohort_ParameterGrid
from sklearn.linear_model import LogisticRegression as _common_cohort_LogisticRegression
from sklearn.ensemble import RandomForestClassifier as _common_cohort_RandomForestClassifier
from sklearn.metrics import roc_auc_score as _common_cohort_roc_auc_score, average_precision_score as _common_cohort_average_precision_score, balanced_accuracy_score as _common_cohort_balanced_accuracy_score, f1_score as _common_cohort_f1_score, precision_score as _common_cohort_precision_score, recall_score as _common_cohort_recall_score, brier_score_loss as _common_cohort_brier_score_loss, roc_curve as _common_cohort_roc_curve, precision_recall_curve as _common_cohort_precision_recall_curve
from sklearn.calibration import calibration_curve as _common_cohort_calibration_curve
from xgboost import XGBClassifier as _common_cohort_XGBClassifier
from joblib import Parallel as _common_cohort_Parallel, delayed as _common_cohort_delayed
from threadpoolctl import threadpool_limits as _common_cohort_threadpool_limits
import matplotlib.pyplot as _common_cohort_plt
_common_cohort_CONFIGURATIONS = {'clinical': ('clinical',), 'mutation': ('mutation',), 'transcriptomics': ('transcriptomics',), 'clinical_mutation': ('clinical', 'mutation'), 'clinical_transcriptomics': ('clinical', 'transcriptomics'), 'mutation_transcriptomics': ('mutation', 'transcriptomics'), 'clinical_mutation_transcriptomics': ('clinical', 'mutation', 'transcriptomics')}
_common_cohort_CLINICAL_SOURCES = ['age_at_diagnosis', 'abc_smoker', 'abc_alcohol', 'histo_result___1', 'abc_breast_side', 'abc_breast_quadrant', 'abc_grade', 'abc_multicentric', 'abc_multifocal', 'abc_re', 'abc_rp', 'abc_her2', 'luminal_a', 'luminal_b', 'tnbc', 'her2', 'abc_p53', 'BRCA', 'abc_hist', 'abc_ajcc_m', 'menopausal_status', 'abc_lnd_n_excised', 'abc_lnd_n_positive', 'abc_ajcc_x_t', 'abc_ajcc_x_n']

def normalise_id(value):
    if _common_cohort_pd.isna(value):
        raise ValueError('Missing patient/sample identifier.')
    value = str(value).strip().upper().replace('#', '')
    return value[:-2] + 'B' if value.endswith('BC') else value

def file_hash(path):
    h = _common_cohort_hashlib.sha256()
    with _common_cohort_Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(value, path):
    path = _common_cohort_Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(_common_cohort_json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    _common_cohort_os.replace(temp, path)

def load_common_sources(base, repo):
    """Recover unfiltered inputs and the authoritative patient/fold mapping."""
    base, repo = (_common_cohort_Path(base), _common_cohort_Path(repo))
    old = base / 'Machine-Learning-Models-poly_vs_oligo'
    fold_path = old / 'Clinical-Mutation-Transcriptomics-Combined/Reviewed/gene_level/clinical_mutation_transcriptomics_nested_cv_predictions_reviewed.csv'
    paths = {'folds': fold_path, 'clinical': base / 'Clinical-Data/20260406-mcpp-breast-clinical-data.csv', 'clinical_linkage': base / 'Clinical-Data/20260401-mcpp-breast-clinical-all-metastatic-data-preprocessing-without-treatment.csv', 'mutation_oligo': base / 'Mutation-Data/gene_presence_data_gnomad_1_0.csv', 'mutation_poly': base / 'Mutation-Data/primpluri_gene_new_220925.csv', 'mutation_schema': repo / 'scripts/OligoPolyThesis/common_cohort_mutation_identifier_groups.json', 'transcriptomics': base / 'Transcriptomic-Data/Preprocessed/transcriptomics_oligo_poly_raw_tpm_common_genes.csv'}
    oof = _common_cohort_pd.read_csv(fold_path)
    oof = oof.loc[oof.dataset.eq('test')].copy()
    for _, frame in oof.groupby('model'):
        if len(frame) != 110 or frame.sample_id.duplicated().any():
            raise ValueError('Authoritative OOF coverage is incomplete.')
    if (oof.groupby('sample_id')[['outer_fold', 'y_true', 'record_id']].nunique() != 1).any().any():
        raise ValueError('Family-specific outer assignments or labels disagree.')
    cohort = oof.sort_values('row_position').drop_duplicates('sample_id')[['sample_id', 'record_id', 'y_true', 'outer_fold']].reset_index(drop=True)
    cohort.sample_id = cohort.sample_id.map(normalise_id)
    if cohort.sample_id.duplicated().any() or len(cohort) != 110:
        raise ValueError('Expected 110 unique patients.')
    counts = _common_cohort_pd.crosstab(cohort.outer_fold, cohort.y_true)
    if counts.values.tolist() != [[8, 14], [8, 14], [9, 13], [9, 13], [9, 13]]:
        raise ValueError('Frozen fold/class counts changed.')
    ids = cohort.sample_id.tolist()
    labels = cohort.set_index('sample_id').y_true
    c = _common_cohort_pd.read_csv(paths['clinical'], sep=';')
    c = c.loc[c.FFPE_type.eq('Breast')].copy()
    c['key'] = c.sample_id.map(normalise_id)
    c = c.loc[c.key.isin(ids)].set_index('key').copy()
    if c.index.duplicated().any() or set(c.index) != set(ids):
        raise ValueError('Clinical primary-sample linkage is not one-to-one.')
    clinical_y = c.Oligo_or_Pluri.map({'Oligometastatic': 0, 'Plurimetastatic': 1})
    if not clinical_y.loc[ids].equals(labels.loc[ids].rename('Oligo_or_Pluri')):
        if not _common_cohort_np.array_equal(clinical_y.loc[ids], labels.loc[ids]):
            raise ValueError('Raw clinical labels disagree with the frozen cohort.')
    dates = {k: _common_cohort_pd.to_datetime(c[k], format='%d/%m/%Y', errors='coerce') for k in ['abc_date_of_diagnosis', 'abc_date_of_birth']}
    c['age_at_diagnosis'] = (dates['abc_date_of_diagnosis'] - dates['abc_date_of_birth']).dt.days // 365
    c['abc_ajcc_x_t'] = c.abc_ajcc_p_t_2.combine_first(c.abc_ajcc_c_t)
    c['abc_ajcc_x_n'] = c.abc_ajcc_p_n.combine_first(c.abc_ajcc_c_n)
    clinical = c.loc[ids, _common_cohort_CLINICAL_SOURCES].copy()
    clinical['BRCA'] = clinical['BRCA'].replace({'Unknown': _common_cohort_np.nan})
    for col in clinical:
        if col != 'BRCA':
            clinical[col] = _common_cohort_pd.to_numeric(clinical[col], errors='raise')
    mutation_parts = []
    for key, target in [('mutation_oligo', 0), ('mutation_poly', 1)]:
        frame = _common_cohort_pd.read_csv(paths[key]).rename(columns={'Unnamed: 0': 'sample_id'})
        frame['sample_id'] = frame.sample_id.map(normalise_id)
        frame = frame.set_index('sample_id').drop(columns=['Progress'], errors='ignore')
        common = frame.index.intersection(ids)
        if not (labels.loc[common] == target).all():
            raise ValueError('Raw mutation labels disagree.')
        mutation_parts.append(frame)
    m = _common_cohort_pd.concat(mutation_parts, axis=0, sort=False).fillna(0)
    if m.index.duplicated().any() or not set(ids).issubset(m.index):
        raise ValueError('Mutation sample linkage is ambiguous/incomplete.')
    m = m.loc[ids].astype(float)
    if not _common_cohort_np.isin(m.values, [0, 1]).all():
        raise ValueError('Mutation inputs must be binary.')
    groups = _common_cohort_json.loads(paths['mutation_schema'].read_text())['groups']
    mutation_names = list(m.columns)
    position = {name: i for i, name in enumerate(mutation_names)}
    values = m.to_numpy(dtype=float, copy=True)
    dropped = set()
    for cols in groups.values():
        if not set(cols).issubset(position):
            raise ValueError('Mutation identifier harmonization schema changed.')
        ensembl = [x for x in cols if x.startswith('ENSG')]
        keep = ensembl[0] if ensembl else cols[0]
        values[:, position[keep]] = values[:, [position[x] for x in cols]].max(axis=1)
        dropped.update((x for x in cols if x != keep))
    keep_indices = [i for i, name in enumerate(mutation_names) if name not in dropped]
    m = _common_cohort_pd.DataFrame(values[:, keep_indices], index=m.index, columns=[mutation_names[i] for i in keep_indices])
    t = _common_cohort_pd.read_csv(paths['transcriptomics'])
    t['sample_id_normalised'] = t.sample_id_normalised.map(normalise_id)
    t = t.set_index('sample_id_normalised')
    if t.index.duplicated().any() or not set(ids).issubset(t.index):
        raise ValueError('TPM sample linkage is ambiguous/incomplete.')
    if not _common_cohort_np.array_equal(t.loc[ids, 'Polymetastatic'], labels.loc[ids]):
        raise ValueError('Raw TPM labels disagree.')
    t = t.loc[ids, [x for x in t.columns if x.startswith('transcriptomics__')]].astype(float)
    if not _common_cohort_np.isfinite(t.values).all() or (t.values < 0).any():
        raise ValueError('Expected finite nonnegative raw TPM.')
    clinical.columns = ['clinical__' + x for x in clinical.columns]
    m.columns = ['mutation__' + x for x in m.columns]
    blocks = {'clinical': clinical.reset_index(drop=True), 'mutation': m.reset_index(drop=True), 'transcriptomics': t.reset_index(drop=True)}
    return (blocks, cohort, paths)

class ClinicalFoldPreprocessor(_common_cohort_BaseEstimator, _common_cohort_TransformerMixin):
    """Fit original missingness and frequency rules before categorical encoding."""

    def fit(self, X, y):
        X = X.copy()
        missing = X.isna()
        delta = missing.groupby(_common_cohort_np.asarray(y)).mean().max() - missing.groupby(_common_cohort_np.asarray(y)).mean().min()
        keep = []
        for col in X:
            if missing[col].mean() > 0.8 or delta[col] > 0.3:
                continue
            if col == 'clinical__age_at_diagnosis':
                keep.append(col)
            elif X[col].nunique(dropna=True) > 1:
                frequency = X[col].value_counts(dropna=True) / len(X)
                if ((frequency > 0.2) & (frequency < 0.8)).any():
                    keep.append(col)
        self.columns_ = keep
        numeric = [c for c in keep if c == 'clinical__age_at_diagnosis']
        self.numeric_ = numeric
        self.categorical_ = [c for c in keep if c not in numeric]
        self.imputer_ = _common_cohort_SimpleImputer(strategy='median', keep_empty_features=True)
        if numeric:
            self.imputer_.fit(X[numeric])
        self.encoder_ = _common_cohort_OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        if self.categorical_:
            self.encoder_.fit(self._categorical(X))
        names = numeric + (self.encoder_.get_feature_names_out(self.categorical_).tolist() if self.categorical_ else [])
        self.mask_ = _common_cohort_np.array([not name.endswith('_missing') for name in names])
        self.names_ = _common_cohort_np.array(names, dtype=object)[self.mask_]
        raw = self._transform(X)
        self.variable_ = _common_cohort_np.ptp(raw, axis=0) > 0
        self.names_ = self.names_[self.variable_]
        if not len(self.names_):
            raise ValueError('No clinical features survive training-fold preprocessing.')
        return self

    def _categorical(self, X):
        frame = X[self.categorical_].copy()
        groups = {'abc_grade': {1: '1-2', 2: '1-2', 4: 'missing'}, 'abc_her2': {1: 'negative_low', 5: 'negative_low', 2: 'positive', 3: 'positive', 4: 'positive', 6: 'missing'}, 'abc_ajcc_x_n': {1: 'N0', 2: 'N1', **{i: 'N2plus' for i in range(3, 10)}, 11: 'missing', 12: 'missing'}, 'abc_ajcc_m': {1: 'M0like', 2: 'M0like', 4: 'missing'}, 'abc_ajcc_x_t': {i: 'T1' for i in range(2, 6)}, 'abc_re': {3: 'missing'}, 'abc_rp': {3: 'missing'}, 'menopausal_status': {4: 'missing'}}
        for col in frame:
            mapping = groups.get(col.removeprefix('clinical__'), {})
            frame[col] = frame[col].map(lambda v: 'missing' if _common_cohort_pd.isna(v) else str(mapping.get(v, v)))
        return frame

    def _transform(self, X):
        arrays = []
        if self.numeric_:
            arrays.append(self.imputer_.transform(X[self.numeric_]))
        if self.categorical_:
            arrays.append(self.encoder_.transform(self._categorical(X)))
        return _common_cohort_np.concatenate(arrays, axis=1)[:, self.mask_]

    def transform(self, X):
        return self._transform(X)[:, self.variable_]

    def get_feature_names_out(self, input_features=None):
        return self.names_

class MutationFoldPreprocessor(_common_cohort_BaseEstimator, _common_cohort_TransformerMixin):
    """Train-only 5% prevalence, constants and |phi|>.85/Fisher p<.05 redundancy."""

    def fit(self, X, y=None):
        a = X.to_numpy(dtype=_common_cohort_np.float64)
        counts = a.sum(axis=0)
        retained = _common_cohort_np.flatnonzero((counts >= _common_cohort_np.ceil(0.05 * len(a))) & (counts < len(a)))
        a = a[:, retained]
        freq = counts[retained]
        n = len(a)
        denominator = _common_cohort_np.sqrt(freq * (n - freq))
        dropped = set()
        for start in range(0, a.shape[1], 256):
            joint = a[:, start:start + 256].T @ a
            phi = (n * joint - freq[start:start + 256, None] * freq[None, :]) / (denominator[start:start + 256, None] * denominator[None, :])
            left, right = _common_cohort_np.where(_common_cohort_np.abs(phi) > 0.85)
            for i, j in zip(left + start, right):
                if i >= j:
                    continue
                both = int(joint[i - start, j])
                fi = int(freq[i])
                fj = int(freq[j])
                if _common_cohort_fisher_exact([[both, fi - both], [fj - both, n - fi - fj + both]]).pvalue < 0.05:
                    dropped.add(i if fi < fj else j)
        self.indices_ = retained[[i not in dropped for i in range(len(retained))]]
        self.names_ = _common_cohort_np.array(X.columns, dtype=object)[self.indices_]
        return self

    def transform(self, X):
        return X.iloc[:, self.indices_].to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return self.names_

class TPMFoldPreprocessor(_common_cohort_BaseEstimator, _common_cohort_TransformerMixin):

    def fit(self, X, y=None):
        a = X.to_numpy(dtype=float)
        self.mask_ = ((a >= 1).sum(axis=0) >= 5) & (_common_cohort_np.ptp(a, axis=0) > 0)
        self.names_ = _common_cohort_np.array(X.columns, dtype=object)[self.mask_]
        return self

    def transform(self, X):
        return _common_cohort_np.log2(X.to_numpy(dtype=float)[:, self.mask_] + 1)

    def get_feature_names_out(self, input_features=None):
        return self.names_

class FitLocalXGBClassifier(_common_cohort_XGBClassifier):

    def fit(self, X, y, **kwargs):
        requested = self.scale_pos_weight
        if requested == 'balanced':
            self.scale_pos_weight = float(_common_cohort_np.sum(_common_cohort_np.asarray(y) == 0) / _common_cohort_np.sum(_common_cohort_np.asarray(y) == 1))
        self.effective_scale_pos_weight_ = self.scale_pos_weight
        try:
            return super().fit(X, y, **kwargs)
        finally:
            self.scale_pos_weight = requested

def compact_grids(configuration):
    selectors = ['passthrough'] if configuration == 'clinical' else [_common_cohort_SelectKBest(_common_cohort_f_classif, k=200), _common_cohort_SelectKBest(_common_cohort_f_classif, k=500), _common_cohort_SelectKBest(_common_cohort_partial(_common_cohort_mutual_info_classif, random_state=42), k=200)]
    if configuration in ['mutation', 'clinical_mutation']:
        selectors = ['passthrough'] + selectors
    common = {'selector': selectors}
    grids = {'logreg_l1': {**common, 'classifier__C': [0.0001, 0.001, 0.01, 0.1, 1], 'classifier__class_weight': [None, 'balanced']}, 'random_forest': {**common, 'classifier__n_estimators': [100, 300], 'classifier__max_depth': [2, 3], 'classifier__min_samples_leaf': [5, 10], 'classifier__max_features': [0.1, 'sqrt'], 'classifier__class_weight': [None, 'balanced']}, 'xgboost': {**common, 'classifier__n_estimators': [50, 100], 'classifier__max_depth': [1, 2], 'classifier__learning_rate': [0.01, 0.05], 'classifier__min_child_weight': [5, 10], 'classifier__reg_lambda': [1, 5], 'classifier__scale_pos_weight': [1.0, 'balanced']}}
    if configuration == 'clinical':
        grids['logreg_l2'] = {**common, 'classifier__C': [0.001, 0.01, 0.1, 1, 10], 'classifier__class_weight': [None, 'balanced']}
    return grids

def compact_pipeline(family):
    if family.startswith('logreg'):
        estimator = _common_cohort_LogisticRegression(penalty='l1' if family == 'logreg_l1' else 'l2', solver='liblinear' if family == 'logreg_l1' else 'lbfgs', max_iter=5000, random_state=42)
    elif family == 'random_forest':
        estimator = _common_cohort_RandomForestClassifier(bootstrap=True, max_samples=None, min_samples_split=2, n_jobs=1, random_state=42)
    else:
        estimator = FitLocalXGBClassifier(subsample=0.8, colsample_bytree=0.1, reg_alpha=0, objective='binary:logistic', eval_metric='logloss', tree_method='hist', n_jobs=1, random_state=42)
    return _common_cohort_Pipeline([('selector', 'passthrough'), ('scaler', _common_cohort_StandardScaler()), ('classifier', estimator)])

def prepare_training_partitions(blocks, cohort, cache_dir):
    """Fit preprocessing separately for each outer/inner training partition.

    Reuse only identical training partitions across classifiers/input combinations;
    never fit preprocessing on an outer-training set for an inner validation fold.
    """
    import joblib as joblib
    cache_dir = _common_cohort_Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    y = cohort.y_true.to_numpy()
    summary = []
    partitions = {}
    transformers = {'clinical': ClinicalFoldPreprocessor, 'mutation': MutationFoldPreprocessor, 'transcriptomics': TPMFoldPreprocessor}
    for outer in range(1, 6):
        tr = _common_cohort_np.flatnonzero(cohort.outer_fold.to_numpy() != outer)
        te = _common_cohort_np.flatnonzero(cohort.outer_fold.to_numpy() == outer)
        inner = list(_common_cohort_StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y[tr]))
        units = [('outer', tr, te)] + [(f'inner_{i}', tr[a], tr[b]) for i, (a, b) in enumerate(inner, 1)]
        partitions[outer] = {'train': tr, 'test': te, 'inner': inner}
        for unit, train, valid in units:
            print(f'PREPROCESS outer={outer} partition={unit}', flush=True)
            sizes = {}
            for modality, frame in blocks.items():
                transformer = transformers[modality]().fit(frame.iloc[train], y[train])
                a = transformer.transform(frame.iloc[train])
                b = transformer.transform(frame.iloc[valid])
                names = transformer.get_feature_names_out()
                if not _common_cohort_np.isfinite(a).all() or not _common_cohort_np.isfinite(b).all():
                    raise ValueError('Non-finite fold preprocessing.')
                sizes[modality] = a.shape[1]
                path = cache_dir / f'fold_{outer}_{unit}_{modality}.joblib'
                temp = path.with_suffix('.tmp')
                joblib.dump({'train': a, 'valid': b, 'names': names, 'train_idx': train, 'valid_idx': valid}, temp)
                _common_cohort_os.replace(temp, path)
                summary.append({'outer_fold': outer, 'partition': unit, 'modality': modality, 'n_training': len(train), 'retained_features': len(names)})
            for config, modalities in _common_cohort_CONFIGURATIONS.items():
                count = sum((sizes[m] for m in modalities))
                if config != 'clinical' and count < 500:
                    raise ValueError(f'SELECTOR INFEASIBLE: input={config}, outer_fold={outer}, partition={unit}, retained_features={count}, required_k=500')
    return (partitions, _common_cohort_pd.DataFrame(summary))

def binary_metrics(y, pred, prob):
    return {'roc_auc': _common_cohort_roc_auc_score(y, prob), 'average_precision': _common_cohort_average_precision_score(y, prob), 'balanced_accuracy': _common_cohort_balanced_accuracy_score(y, pred), 'macro_f1': _common_cohort_f1_score(y, pred, average='macro'), 'precision': _common_cohort_precision_score(y, pred, zero_division=0), 'sensitivity': _common_cohort_recall_score(y, pred, zero_division=0), 'specificity': _common_cohort_recall_score(y, pred, pos_label=0, zero_division=0), 'brier_score': _common_cohort_brier_score_loss(y, prob)}

def selector_label(selector):
    if isinstance(selector, str):
        return selector
    return ('mi' if isinstance(selector.score_func, _common_cohort_partial) else 'f') + str(selector.k)

def prepare_selected_partitions(config, outer, cohort, cache_dir):
    """Cache deterministic training-only selectors/scalers across grid candidates."""
    import joblib as joblib
    from sklearn.base import clone as clone
    selectors = compact_grids(config)['logreg_l1']['selector']
    selected = {}
    y = cohort.y_true.to_numpy()
    for selector in selectors:
        label = selector_label(selector)
        selected[label] = {}
        for unit in ['outer'] + [f'inner_{i}' for i in range(1, 6)]:
            blocks = [joblib.load(_common_cohort_Path(cache_dir) / f'fold_{outer}_{unit}_{m}.joblib') for m in _common_cohort_CONFIGURATIONS[config]]
            train = _common_cohort_np.concatenate([b['train'] for b in blocks], axis=1)
            valid = _common_cohort_np.concatenate([b['valid'] for b in blocks], axis=1)
            names = _common_cohort_np.concatenate([b['names'] for b in blocks])
            if not isinstance(selector, str):
                fitted = clone(selector).fit(train, y[blocks[0]['train_idx']])
                train = fitted.transform(train)
                valid = fitted.transform(valid)
                names = names[fitted.get_support()]
            scaler = _common_cohort_StandardScaler().fit(train)
            selected[label][unit] = {'train': scaler.transform(train), 'valid': scaler.transform(valid), 'train_y': y[blocks[0]['train_idx']], 'valid_y': y[blocks[0]['valid_idx']], 'train_idx': blocks[0]['train_idx'], 'valid_idx': blocks[0]['valid_idx'], 'names': names}
    return selected

def _common_cohort_score_candidate(family, params, selected):
    from sklearn.base import clone as clone
    label = selector_label(params['selector'])
    scores = []
    with _common_cohort_threadpool_limits(1):
        for i in range(1, 6):
            data = selected[label][f'inner_{i}']
            model = clone(compact_pipeline(family).named_steps['classifier'])
            model.set_params(**{k.removeprefix('classifier__'): v for k, v in params.items() if k != 'selector'})
            model.fit(data['train'], data['train_y'])
            scores.append(_common_cohort_f1_score(data['valid_y'], model.predict(data['valid']), average='macro'))
    return float(_common_cohort_np.mean(scores))

def run_common_comparison(cohort, output_dir, n_jobs=8):
    """Exhaustive five-fold inner search on separately preprocessed inner partitions.

    All candidate means are computed before the outer refit. Cached transforms
    depend only on their exact training partition and never on outer outcomes.
    """
    from sklearn.base import clone as clone
    output_dir = _common_cohort_Path(output_dir)
    cache = output_dir / 'private/partition_cache'
    units = output_dir / 'private/completed_units'
    units.mkdir(exist_ok=True)
    protocol_hash = file_hash(output_dir / 'common_cohort_protocol.json')
    for config in _common_cohort_CONFIGURATIONS:
        for outer in range(1, 6):
            selected = None
            for family, grid in compact_grids(config).items():
                path = units / f'{config}_{family}_fold{outer}.json'
                if path.exists():
                    saved = _common_cohort_json.loads(path.read_text())
                    if saved.get('protocol_hash') != protocol_hash:
                        raise ValueError('Incompatible completed unit.')
                    print(f'RESTORED {config} {family} fold={outer}', flush=True)
                    continue
                start = _common_cohort_time.time()
                print(f'START {config} {family} fold={outer} {_common_cohort_datetime.now(_common_cohort_timezone.utc).isoformat()}', flush=True)
                if selected is None:
                    with _common_cohort_threadpool_limits(1):
                        selected = prepare_selected_partitions(config, outer, cohort, cache)
                candidates = list(_common_cohort_ParameterGrid(grid))
                scores = _common_cohort_Parallel(n_jobs=n_jobs, backend='loky', pre_dispatch=n_jobs)((_common_cohort_delayed(_common_cohort_score_candidate)(family, p, selected) for p in candidates))
                if not _common_cohort_np.isfinite(scores).all():
                    raise ValueError('Non-finite inner score.')
                best = int(_common_cohort_np.argmax(scores))
                params = candidates[best]
                data = selected[selector_label(params['selector'])]['outer']
                model = clone(compact_pipeline(family).named_steps['classifier'])
                model.set_params(**{k.removeprefix('classifier__'): v for k, v in params.items() if k != 'selector'})
                with _common_cohort_threadpool_limits(1):
                    model.fit(data['train'], data['train_y'])
                trprob = model.predict_proba(data['train'])[:, 1]
                teprob = model.predict_proba(data['valid'])[:, 1]
                trpred = model.predict(data['train'])
                tepred = model.predict(data['valid'])
                predictions = cohort.iloc[data['valid_idx']][['sample_id', 'record_id', 'y_true', 'outer_fold']].copy()
                predictions['configuration'] = config
                predictions['family'] = family
                predictions['y_pred'] = tepred
                predictions['y_proba'] = teprob
                saved = {'protocol_hash': protocol_hash, 'configuration': config, 'family': family, 'outer_fold': outer, 'best_inner_score': scores[best], 'best_params': {k: selector_label(v) if k == 'selector' else v for k, v in params.items()}, 'inner_candidate_scores': scores, 'n_candidates': len(candidates), 'selected_features': list(data['names']), 'n_selected_features': len(data['names']), 'train': binary_metrics(data['train_y'], trpred, trprob), 'test': binary_metrics(data['valid_y'], tepred, teprob), 'predictions': predictions.to_dict('records'), 'elapsed_seconds': _common_cohort_time.time() - start, 'completed_utc': _common_cohort_datetime.now(_common_cohort_timezone.utc).isoformat()}
                atomic_json(saved, path)
                print(f'COMPLETE {config} {family} fold={outer} inner={scores[best]:.6f} elapsed={_common_cohort_time.time() - start:.1f}s', flush=True)
    return summarise_common_results(cohort, output_dir)

def summarise_common_results(cohort, output_dir):
    output_dir = _common_cohort_Path(output_dir)
    metrics = []
    predictions = []
    scores = []
    for config in _common_cohort_CONFIGURATIONS:
        for family in compact_grids(config):
            for outer in range(1, 6):
                p = output_dir / 'private/completed_units' / f'{config}_{family}_fold{outer}.json'
                unit = _common_cohort_json.loads(p.read_text())
                predictions.extend(unit['predictions'])
                scores.append({'configuration': config, 'family': family, 'outer_fold': outer, 'best_inner_score': unit['best_inner_score'], 'n_selected_features': unit['n_selected_features']})
                for split in ['train', 'test']:
                    metrics.append({'configuration': config, 'family': family, 'outer_fold': outer, 'split': split, **unit[split]})
    pred = _common_cohort_pd.DataFrame(predictions)
    fold = _common_cohort_pd.DataFrame(metrics)
    scores = _common_cohort_pd.DataFrame(scores)
    expected = cohort.set_index('sample_id')
    for _, frame in pred.groupby(['configuration', 'family']):
        if len(frame) != 110 or frame.sample_id.duplicated().any() or set(frame.sample_id) != set(expected.index):
            raise ValueError('Incomplete OOF coverage.')
        truth = expected.loc[frame.sample_id]
        if not _common_cohort_np.array_equal(truth.outer_fold, frame.outer_fold) or not _common_cohort_np.array_equal(truth.y_true, frame.y_true):
            raise ValueError('OOF folds/labels disagree.')
        if not _common_cohort_np.isfinite(frame.y_proba).all() or not frame.y_proba.between(0, 1).all():
            raise ValueError('Invalid probabilities.')
    ranking = scores.groupby(['configuration', 'family']).best_inner_score.agg(['mean', 'std']).reset_index().rename(columns={'mean': 'mean_inner_macro_f1', 'std': 'std_inner_macro_f1'})
    ranking = ranking.sort_values(['configuration', 'mean_inner_macro_f1', 'family'], ascending=[True, False, True])
    winners = ranking.drop_duplicates('configuration').set_index('configuration').family.to_dict()
    pooled = []
    for (config, family), frame in pred.groupby(['configuration', 'family']):
        pooled.append({'configuration': config, 'family': family, 'N': len(frame), 'selected': winners[config] == family, **binary_metrics(frame.y_true, frame.y_pred, frame.y_proba)})
    pooled = _common_cohort_pd.DataFrame(pooled)
    metric_cols = list(binary_metrics([0, 1], [0, 1], [0.1, 0.9]))
    mean_sd = fold.groupby(['configuration', 'family', 'split'])[metric_cols].agg(['mean', 'std'])
    mean_sd.columns = ['_'.join(c) for c in mean_sd.columns]
    mean_sd = mean_sd.reset_index()
    train = fold[fold.split == 'train']
    test = fold[fold.split == 'test']
    gap = train.merge(test, on=['configuration', 'family', 'outer_fold'], suffixes=('_train', '_test'), validate='one_to_one')
    for m in metric_cols:
        gap[m + '_gap'] = gap[m + '_train'] - gap[m + '_test']
    diagnostics = gap.groupby(['configuration', 'family'])[[m + s for m in metric_cols for s in ['_train', '_test', '_gap']]].agg(['mean', 'std'])
    diagnostics.columns = ['_'.join(c) for c in diagnostics.columns]
    diagnostics = diagnostics.reset_index()
    pred.to_csv(output_dir / 'private/common_cohort_oof_predictions.csv', index=False)
    for name, table in [('common_cohort_pooled_metrics', pooled), ('common_cohort_outer_fold_metrics', fold), ('common_cohort_outer_fold_mean_sd', mean_sd), ('common_cohort_train_test_diagnostics', diagnostics), ('common_cohort_inner_model_ranking', ranking), ('common_cohort_selected_feature_counts', scores)]:
        table.to_csv(output_dir / (name + '.csv'), index=False)
    selected = pooled[pooled.selected].set_index('configuration').loc[list(_common_cohort_CONFIGURATIONS)].reset_index()
    selected.to_csv(output_dir / 'common_cohort_primary_comparison.csv', index=False)
    selected[['configuration', 'family', 'brier_score']].to_csv(output_dir / 'common_cohort_brier_summary.csv', index=False)
    for kind in ['roc', 'pr', 'calibration']:
        fig, ax = _common_cohort_plt.subplots(figsize=(9, 6))
        for config in _common_cohort_CONFIGURATIONS:
            frame = pred[(pred.configuration == config) & (pred.family == winners[config])]
            label = config.replace('_', ' + ') + ' (' + winners[config] + ')'
            if kind == 'roc':
                a, b, _ = _common_cohort_roc_curve(frame.y_true, frame.y_proba)
                ax.plot(a, b, label=label)
            elif kind == 'pr':
                precision, recall, _ = _common_cohort_precision_recall_curve(frame.y_true, frame.y_proba)
                ax.plot(recall, precision, label=label)
            else:
                fraction, mean = _common_cohort_calibration_curve(frame.y_true, frame.y_proba, n_bins=5, strategy='uniform')
                ax.plot(mean, fraction, 'o-', label=label)
        if kind in ['roc', 'calibration']:
            ax.plot([0, 1], [0, 1], '--', color='grey', linewidth=1)
        else:
            ax.axhline(67 / 110, linestyle='--', color='grey', linewidth=1)
        labels = {'roc': ('False-positive rate', 'True-positive rate'), 'pr': ('Recall', 'Precision'), 'calibration': ('Mean predicted probability', 'Observed polymetastatic proportion')}
        ax.set(xlabel=labels[kind][0], ylabel=labels[kind][1], title='Common cohort: pooled OOF ' + kind.upper())
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f'common_cohort_{kind}.png', dpi=180)
        _common_cohort_plt.close(fig)
    return (selected, ranking, diagnostics)


def initialise_common_output(cohort, paths, output_dir):
    """Publish the outcome-independent protocol and private cohort manifest."""
    output_dir = _common_cohort_Path(output_dir)
    private = output_dir / 'private'
    private.mkdir(parents=True, exist_ok=True)
    grids = {config: {family: {key: [selector_label(v) for v in values] if key == 'selector' else values for key, values in grid.items()} for family, grid in compact_grids(config).items()} for config in _common_cohort_CONFIGURATIONS}
    protocol = {'version': 1, 'seed': 42, 'cohort_N': 110, 'classes': {'oligo': 43, 'poly': 67}, 'outer_folds': 5, 'inner_folds': 5, 'scoring': 'f1_macro', 'sampler': 'passthrough', 'sources': {k: {'path': str(v), 'sha256': file_hash(v)} for k, v in paths.items()}, 'clinical_sources': _common_cohort_CLINICAL_SOURCES, 'grids': grids, 'preprocessing': {'clinical': 'train-only >80% missing or >30% target-group missingness gap; nonconstant category frequency strictly 20–80%; fitted OHE; median age', 'mutation': 'fixed identifier OR; train-only 5% prevalence; nonconstant; |phi|>.85 and Fisher p<.05 redundancy; lower frequency removed, second source column on tie', 'transcriptomics': 'raw TPM; train-only TPM>=1 in >=5 patients; nonconstant; log2(TPM+1)'}, 'selection': 'seeded MI42 or F-test; scaler fitted after selection; exact-partition caching; mean inner macro-F1; alphabetical family ties'}
    path = output_dir / 'common_cohort_protocol.json'
    # A historical 'code_sha256' field in an existing protocol is ignored and the file is not rewritten.
    existing = None
    if path.exists():
        existing = {k: v for k, v in _common_cohort_json.loads(path.read_text()).items() if k != 'code_sha256'}
    if (private / 'completed_units').exists() and any((private / 'completed_units').iterdir()):
        if existing != protocol:
            raise ValueError('Protocol changed after completed work; refusing reuse.')
    if existing != protocol:
        atomic_json(protocol, path)
    cohort.to_csv(private / 'common_patients.csv', index=False)
    cohort[['sample_id', 'record_id', 'y_true', 'outer_fold']].to_csv(private / 'frozen_outer_fold_assignments.csv', index=False)
    return protocol

# ==============================================================================
# Common-cohort (N=110) reporting
# ==============================================================================
"""Read-only primary-cohort result validation and presentation; no model fitting."""
from pathlib import Path as _primary110_reporting_Path
import hashlib as _primary110_reporting_hashlib
import json as _primary110_reporting_json
import numpy as _primary110_reporting_np
import pandas as _primary110_reporting_pd
import matplotlib.pyplot as _primary110_reporting_plt
from sklearn.metrics import roc_curve as _primary110_reporting_roc_curve, precision_recall_curve as _primary110_reporting_precision_recall_curve, confusion_matrix as _primary110_reporting_confusion_matrix
from sklearn.calibration import calibration_curve as _primary110_reporting_calibration_curve
_primary110_reporting_CONFIG = {'clinical': ('Clinical-Data', 'clinical', 'xgboost', 149), 'mutation': ('Mutation-Data', 'mutation', 'random_forest', 112), 'transcriptomics': ('Transcriptomics-Representation-Learning', 'transcriptomics', 'gene_level_xgboost', 139)}

def _primary110_reporting_hash(path):
    digest = _primary110_reporting_hashlib.sha256()
    with _primary110_reporting_Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def load_primary110(root):
    """Verify source/protocol hashes and complete held-out coverage before reporting."""
    root = _primary110_reporting_Path(root)
    protocol_path = root / 'common_cohort_protocol.json'
    protocol = _primary110_reporting_json.loads(protocol_path.read_text())
    for item in protocol['sources'].values():
        if _primary110_reporting_hash(item['path']) != item['sha256']:
            raise ValueError(f"Source fingerprint mismatch: {item['path']}")
    cohort = _primary110_reporting_pd.read_csv(root / 'private/common_patients.csv')
    if len(cohort) != 110 or not cohort.sample_id.is_unique or cohort.y_true.value_counts().to_dict() != {1: 67, 0: 43}:
        raise ValueError('Invalid primary cohort.')
    predictions = _primary110_reporting_pd.read_csv(root / 'private/common_cohort_oof_predictions.csv')
    ranking = _primary110_reporting_pd.read_csv(root / 'common_cohort_inner_model_ranking.csv')
    selected = _primary110_reporting_pd.read_csv(root / 'common_cohort_primary_comparison.csv')
    units = []
    fingerprint = _primary110_reporting_hash(protocol_path)
    for config, family in ranking[['configuration', 'family']].itertuples(index=False, name=None):
        for fold in range(1, 6):
            unit = _primary110_reporting_json.loads((root / 'private/completed_units' / f'{config}_{family}_fold{fold}.json').read_text())
            if unit['protocol_hash'] != fingerprint or (unit['configuration'], unit['family'], unit['outer_fold']) != (config, family, fold):
                raise ValueError('Invalid completed unit.')
            scores = unit['inner_candidate_scores']
            if len(scores) != unit['n_candidates'] or not _primary110_reporting_np.isfinite(scores).all() or unit['best_inner_score'] != max(scores):
                raise ValueError('Invalid inner search record.')
            if len(set(unit['selected_features'])) != unit['n_selected_features']:
                raise ValueError('Invalid selected-feature identities.')
            units.append(unit)
    checkpoint_predictions = _primary110_reporting_pd.DataFrame([row for u in units for row in u['predictions']])
    keys = ['configuration', 'family', 'outer_fold', 'sample_id']
    _primary110_reporting_pd.testing.assert_frame_equal(predictions.sort_values(keys).reset_index(drop=True), checkpoint_predictions[predictions.columns].sort_values(keys).reset_index(drop=True), check_dtype=False)
    for _, frame in predictions.groupby(['configuration', 'family']):
        if len(frame) != 110 or not frame.sample_id.is_unique or set(frame.sample_id) != set(cohort.sample_id):
            raise ValueError('Incomplete OOF coverage.')
        ref = cohort.set_index('sample_id').loc[frame.sample_id]
        if not _primary110_reporting_np.array_equal(ref.y_true, frame.y_true) or not _primary110_reporting_np.array_equal(ref.outer_fold, frame.outer_fold):
            raise ValueError('Patient labels or frozen folds differ.')
        if not _primary110_reporting_np.isfinite(frame.y_proba).all() or not frame.y_proba.between(0, 1).all():
            raise ValueError('Invalid probability.')
    for row in selected.itertuples():
        best = ranking[ranking.configuration.eq(row.configuration)].sort_values(['mean_inner_macro_f1', 'family'], ascending=[False, True]).iloc[0]
        if row.family != best.family:
            raise ValueError('Selected family differs from mean inner-CV criterion.')
    return dict(root=root, cohort=cohort, predictions=predictions, ranking=ranking, selected=selected, units=units, retained=_primary110_reporting_pd.read_csv(root / 'common_cohort_retained_feature_counts.csv'), pooled=_primary110_reporting_pd.read_csv(root / 'common_cohort_pooled_metrics.csv'), diagnostics=_primary110_reporting_pd.read_csv(root / 'common_cohort_train_test_diagnostics.csv'))


def plot_primary110(bundle, configuration):
    """Build figures from preserved OOF predictions; return figures without saving."""
    family = bundle['selected'].set_index('configuration').loc[configuration, 'family']
    data = bundle['predictions']
    data = data[data.configuration.eq(configuration) & data.family.eq(family)]
    figures = {}
    title = f"{configuration.replace('_', ' + ')} | {family} | common cohort N=110"
    for kind in ['roc', 'pr', 'confusion_matrix', 'calibration']:
        fig, ax = _primary110_reporting_plt.subplots(figsize=(7, 5))
        if kind == 'roc':
            x, y, _ = _primary110_reporting_roc_curve(data.y_true, data.y_proba)
            ax.plot(x, y)
            ax.plot([0, 1], [0, 1], '--', color='grey')
            ax.set(xlabel='False-positive rate', ylabel='Sensitivity')
        elif kind == 'pr':
            precision, recall, _ = _primary110_reporting_precision_recall_curve(data.y_true, data.y_proba)
            ax.plot(recall, precision)
            ax.axhline(67 / 110, linestyle='--', color='grey')
            ax.set(xlabel='Recall', ylabel='Precision')
        elif kind == 'calibration':
            observed, predicted = _primary110_reporting_calibration_curve(data.y_true, data.y_proba, n_bins=5, strategy='uniform')
            ax.plot(predicted, observed, 'o-')
            ax.plot([0, 1], [0, 1], '--', color='grey')
            ax.set(xlabel='Mean predicted probability', ylabel='Observed polymetastatic proportion')
        else:
            matrix = _primary110_reporting_confusion_matrix(data.y_true, data.y_pred, labels=[0, 1])
            ax.imshow(matrix, cmap='Blues')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Oligo', 'Poly'], yticklabels=['Oligo', 'Poly'], xlabel='Predicted', ylabel='Observed')
        ax.set_title(title + '\n' + kind.replace('_', ' '))
        fig.tight_layout()
        figures[kind] = fig
    return figures

# ==============================================================================
# Common-cohort (N=110) interpretation
# ==============================================================================
"""Interpretation-only tuning of the five locked strict common-cohort families.

Predictive evaluation remains in utils_common_cohort and is never called here.
"""
from pathlib import Path as _primary110_interpretation_Path
from datetime import datetime as _primary110_interpretation_datetime, timezone as _primary110_interpretation_timezone
from itertools import combinations as _primary110_interpretation_combinations
import hashlib as _primary110_interpretation_hashlib
import json as _primary110_interpretation_json
import os as _primary110_interpretation_os
import uuid as _primary110_interpretation_uuid
import importlib.metadata
import importlib as _primary110_interpretation_importlib
import joblib as _primary110_interpretation_joblib
import numpy as _primary110_interpretation_np
import pandas as _primary110_interpretation_pd
import matplotlib.pyplot as _primary110_interpretation_plt
import shap as _primary110_interpretation_shap
from sklearn.base import BaseEstimator as _primary110_interpretation_BaseEstimator, TransformerMixin as _primary110_interpretation_TransformerMixin
from sklearn.pipeline import Pipeline as _primary110_interpretation_Pipeline
from sklearn.model_selection import GridSearchCV as _primary110_interpretation_GridSearchCV, StratifiedKFold as _primary110_interpretation_StratifiedKFold, ParameterGrid as _primary110_interpretation_ParameterGrid
from joblib import Memory as _primary110_interpretation_Memory, parallel_config as _primary110_interpretation_parallel_config
from threadpoolctl import threadpool_limits as _primary110_interpretation_threadpool_limits
pass
_primary110_interpretation_LOCKED = {'clinical': 'logreg_l2', 'mutation': 'random_forest', 'transcriptomics': 'random_forest', 'clinical_mutation': 'random_forest', 'clinical_mutation_transcriptomics': 'random_forest'}
_primary110_interpretation_SEED = 42
_primary110_interpretation_PROBABILITY_TOLERANCE = 1e-12
_primary110_interpretation_SHAP_TOLERANCE = 1e-06
_primary110_interpretation_LABEL = 'Strict harmonized N=110 — interpretation-only full-cohort fit'

def _primary110_interpretation_now():
    return _primary110_interpretation_datetime.now(_primary110_interpretation_timezone.utc).isoformat()

def _primary110_interpretation_write_json(data, path):
    with _primary110_interpretation_Path(path).open('x') as handle:
        _primary110_interpretation_json.dump(data, handle, indent=2, allow_nan=False, default=str)
        handle.write('\n')

def _primary110_interpretation_table(frame, path):
    with _primary110_interpretation_Path(path).open('x') as handle:
        frame.to_csv(handle, index=False)

def _primary110_interpretation_figure(fig, path):
    with _primary110_interpretation_Path(path).open('xb') as handle:
        fig.savefig(handle, format='png', dpi=180, bbox_inches='tight')
    _primary110_interpretation_plt.close(fig)

def _primary110_interpretation_fit_modality(modality, X, y):
    classes = {'clinical': ClinicalFoldPreprocessor, 'mutation': MutationFoldPreprocessor, 'transcriptomics': TPMFoldPreprocessor}
    return classes[modality]().fit(X, y)

class StrictModalityPreprocessor(_primary110_interpretation_BaseEstimator, _primary110_interpretation_TransformerMixin):
    """Reuse exact source rules, fitting only rows supplied by the CV pipeline.

    Optional caching is keyed on each modality's training X and y. Neither
    validation data nor full-cohort transforms enter an inner-training fit.
    """

    def __init__(self, configuration, cache_dir=None):
        self.configuration = configuration
        self.cache_dir = cache_dir

    def fit(self, X, y):
        self.blocks_ = []
        fit = _primary110_interpretation_Memory(self.cache_dir, verbose=0).cache(_primary110_interpretation_fit_modality) if self.cache_dir else _primary110_interpretation_fit_modality
        for modality in _common_cohort_CONFIGURATIONS[self.configuration]:
            columns = [c for c in X if c.startswith(modality + '__')]
            transformer = fit(modality, X[columns], _primary110_interpretation_np.asarray(y))
            self.blocks_.append((modality, columns, transformer))
        self.names_ = _primary110_interpretation_np.concatenate([t.get_feature_names_out() for _, _, t in self.blocks_])
        if len(set(self.names_)) != len(self.names_):
            raise ValueError('Duplicate transformed feature names.')
        if self.configuration != 'clinical' and len(self.names_) < 500:
            raise ValueError(f'{self.configuration}: fewer than 500 retained inputs; no selector adjustment allowed.')
        self.training_N_ = len(X)
        return self

    def transform(self, X):
        values = _primary110_interpretation_np.concatenate([t.transform(X[c]) for _, c, t in self.blocks_], axis=1)
        if not _primary110_interpretation_np.isfinite(values).all():
            raise ValueError('Nonfinite preprocessed input.')
        return values

    def get_feature_names_out(self, input_features=None):
        return self.names_

def load_locked_inputs(base, repo):
    base, repo = (_primary110_interpretation_Path(base), _primary110_interpretation_Path(repo))
    common = base / 'Machine-Learning-Models-poly_vs_oligo/Common-Cohort-Modality-Comparison'
    protocol = _primary110_interpretation_json.loads((common / 'common_cohort_protocol.json').read_text())
    selected = _primary110_interpretation_pd.read_csv(common / 'common_cohort_primary_comparison.csv').set_index('configuration')
    for config, family in _primary110_interpretation_LOCKED.items():
        if selected.loc[config, 'family'] != family:
            raise ValueError('Locked family mismatch.')
    blocks, cohort, paths = load_common_sources(base, repo)
    if len(cohort) != 110 or cohort.y_true.value_counts().to_dict() != {1: 67, 0: 43}:
        raise ValueError('Common cohort changed.')
    for key, path in paths.items():
        if file_hash(path) != protocol['sources'][key]['sha256']:
            raise ValueError(f'Source changed: {key}')
    paths = {**paths, 'gene_mapping': base / 'Transcriptomic-Data/Preprocessed/transcriptomics_feature_name_mapping.csv'}
    return (blocks, cohort, paths, selected.reset_index())

def _primary110_interpretation_serial_params(params):
    return {k: selector_label(v) if k == 'selector' else v for k, v in params.items()}

def _primary110_interpretation_fingerprint(configuration, cohort, paths):
    payload = {'configuration': configuration, 'family': _primary110_interpretation_LOCKED[configuration], 'sources': {k: file_hash(v) for k, v in paths.items()}, 'cohort': cohort.to_dict('records'), 'grid': [_primary110_interpretation_serial_params(p) for p in _primary110_interpretation_ParameterGrid(compact_grids(configuration)[_primary110_interpretation_LOCKED[configuration]])], 'cv': {'n_splits': 5, 'shuffle': True, 'random_state': 42, 'scoring': 'f1_macro'}, 'versions': {k: _primary110_interpretation_importlib.metadata.version(k) for k in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'shap', 'joblib']}}
    digest = _primary110_interpretation_hashlib.sha256(_primary110_interpretation_json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return (digest, payload)

def _primary110_interpretation_feature_metadata(model, names, mapping_path):
    mapping = _primary110_interpretation_pd.read_csv(mapping_path)
    pairs = mapping[['ensembl_gene_id', 'gene_name']].dropna().drop_duplicates()
    if pairs.groupby('ensembl_gene_id').gene_name.nunique().gt(1).any():
        raise ValueError('Ambiguous gene-symbol mapping.')
    symbols = pairs.drop_duplicates('ensembl_gene_id').set_index('ensembl_gene_id').gene_name
    clinical = {}
    for modality, columns, t in model.named_steps['preprocessor'].blocks_:
        if modality == 'clinical':
            for feature in t.get_feature_names_out():
                if feature in t.numeric_:
                    clinical[feature] = (feature.removeprefix('clinical__'), None)
                else:
                    matches = [v for v in t.categorical_ if feature.startswith(v + '_')]
                    original = max(matches, key=len)
                    clinical[feature] = (original.removeprefix('clinical__'), feature[len(original) + 1:])
    rows = []
    for i, name in enumerate(names):
        modality, identifier = name.split('__', 1)
        symbol = symbols.get(identifier, _primary110_interpretation_np.nan) if modality != 'clinical' else _primary110_interpretation_np.nan
        original, level = clinical.get(name, (None, None))
        rows.append({'feature_position': i, 'feature': name, 'modality': modality, 'ensembl_gene_id': identifier if identifier.startswith('ENSG') else None, 'gene_symbol': symbol, 'original_clinical_variable': original, 'encoded_level': level, 'display_name': f'{symbol} [{modality}]' if _primary110_interpretation_pd.notna(symbol) else name})
    result = _primary110_interpretation_pd.DataFrame(rows)
    dup = result.display_name.duplicated(keep=False)
    result.loc[dup, 'display_name'] = result.loc[dup, 'display_name'] + ' (' + result.loc[dup, 'feature'] + ')'
    return result

def classifier_input(model, X):
    values = model.named_steps['preprocessor'].transform(X)
    names = model.named_steps['preprocessor'].get_feature_names_out()
    selector = model.named_steps['selector']
    if not isinstance(selector, str):
        names = names[selector.get_support()]
        values = selector.transform(values)
    values = model.named_steps['scaler'].transform(values)
    if values.shape != (len(X), len(names)):
        raise ValueError('Classifier input/name shape mismatch.')
    if model.named_steps['classifier'].n_features_in_ != len(names):
        raise ValueError('Classifier feature count mismatch.')
    return (_primary110_interpretation_np.asarray(values), _primary110_interpretation_np.asarray(names))

def _primary110_interpretation_clinical_interpretation(model, metadata, out):
    clf = model.named_steps['classifier']
    beta = clf.coef_[0]
    scale = model.named_steps['scaler'].scale_
    table = metadata.copy()
    table['scaler_SD'] = scale
    table['standardized_coefficient'] = beta
    table['odds_multiplier_per_scaler_SD'] = _primary110_interpretation_np.exp(beta)
    age = table.original_clinical_variable.eq('age_at_diagnosis')
    table['raw_unit_coefficient'] = _primary110_interpretation_np.where(age, beta / scale, _primary110_interpretation_np.nan)
    table['raw_unit_odds_multiplier'] = _primary110_interpretation_np.where(age, _primary110_interpretation_np.exp(beta / scale), _primary110_interpretation_np.nan)
    table['raw_unit'] = _primary110_interpretation_np.where(age, 'one year at primary diagnosis', 'not a standalone named-category contrast')
    table['direction'] = _primary110_interpretation_np.where(beta > 0, 'toward poly', _primary110_interpretation_np.where(beta < 0, 'toward oligo', 'zero'))
    order = _primary110_interpretation_np.argsort(-_primary110_interpretation_np.abs(beta), kind='stable')
    rank = _primary110_interpretation_np.empty(len(beta), int)
    rank[order] = _primary110_interpretation_np.arange(1, len(beta) + 1)
    table['absolute_standardized_coefficient_rank'] = rank
    contrasts = []
    for variable, g in table.loc[~age].groupby('original_clinical_variable', sort=False):
        for a, b in _primary110_interpretation_combinations(g.itertuples(), 2):
            contrast = b.standardized_coefficient / b.scaler_SD - a.standardized_coefficient / a.scaler_SD
            contrasts.append({'original_clinical_variable': variable, 'reference_level': a.encoded_level, 'comparison_level': b.encoded_level, 'log_odds_contrast': contrast, 'odds_multiplier': float(_primary110_interpretation_np.exp(contrast)), 'interpretation': 'fitted category A to B, other source variables held fixed; no inferential CI'})
    _primary110_interpretation_table(table.sort_values('absolute_standardized_coefficient_rank'), out / 'clinical_coefficients.csv')
    _primary110_interpretation_table(_primary110_interpretation_pd.DataFrame(contrasts), out / 'clinical_category_contrasts.csv')
    top = table.iloc[order[:20]].iloc[::-1]
    fig, ax = _primary110_interpretation_plt.subplots(figsize=(10, 8))
    ax.barh(top.feature, top.standardized_coefficient, color=_primary110_interpretation_np.where(top.standardized_coefficient > 0, '#c55a3e', '#2878a5'))
    ax.axvline(0, color='grey', lw=0.8)
    ax.set(xlabel='Coefficient per fitted scaler SD (including one-hot columns)', title='Strict N=110 clinical L2 — predictive coefficients, no inferential CI')
    fig.tight_layout()
    _primary110_interpretation_figure(fig, out / 'clinical_coefficients.png')

def _primary110_interpretation_tree_interpretation(model, values, metadata, probabilities, out):
    clf = model.named_steps['classifier']
    positive = _primary110_interpretation_np.flatnonzero(clf.classes_ == 1)
    if len(positive) != 1:
        raise ValueError('Positive class is ambiguous.')
    positive = int(positive[0])
    masker = _primary110_interpretation_shap.maskers.Independent(values, max_samples=len(values))
    if masker.data.shape != values.shape or not _primary110_interpretation_np.array_equal(masker.data, values):
        raise ValueError('SHAP background was subsampled or reordered.')
    explainer = _primary110_interpretation_shap.TreeExplainer(clf, data=masker, feature_perturbation='interventional', model_output='probability')
    if explainer.data.shape != values.shape or not _primary110_interpretation_np.array_equal(explainer.data, values):
        raise ValueError('Explainer did not preserve the complete 110-patient background.')
    raw = explainer.shap_values(values, check_additivity=False)
    if isinstance(raw, list):
        attribution = _primary110_interpretation_np.asarray(raw[positive])
        output_shape = [len(raw), *_primary110_interpretation_np.asarray(raw[0]).shape]
    else:
        raw = _primary110_interpretation_np.asarray(raw)
        output_shape = list(raw.shape)
        if raw.ndim != 3 or raw.shape[2] != len(clf.classes_):
            raise ValueError(f'Unexpected RF SHAP output {raw.shape}')
        attribution = raw[:, :, positive]
    base = float(_primary110_interpretation_np.asarray(explainer.expected_value)[positive])
    if attribution.shape != values.shape or not _primary110_interpretation_np.isfinite(attribution).all():
        raise ValueError('Invalid SHAP dimensions/values.')
    residual = base + attribution.sum(axis=1) - probabilities
    agreement = {'model_output': 'P(poly=1)', 'positive_class': int(clf.classes_[positive]), 'positive_class_index': positive, 'feature_perturbation': 'interventional', 'background_N': len(explainer.data), 'RF_output_shape': output_shape, 'positive_class_SHAP_shape': list(attribution.shape), 'base_value': base, 'max_absolute_additivity_error': float(_primary110_interpretation_np.abs(residual).max()), 'mean_absolute_additivity_error': float(_primary110_interpretation_np.abs(residual).mean()), 'absolute_tolerance': _primary110_interpretation_SHAP_TOLERANCE, 'feature_names_order_validated': True, 'background_subsampled': False}
    if _primary110_interpretation_np.abs(residual).max() > _primary110_interpretation_SHAP_TOLERANCE:
        raise ValueError(f'Probability-scale additivity failed: {agreement}')
    ranking = metadata.assign(mean_abs_shap=_primary110_interpretation_np.abs(attribution).mean(axis=0), mean_signed_shap=attribution.mean(axis=0))
    ranking = ranking.sort_values('mean_abs_shap', ascending=False, kind='stable').reset_index(drop=True)
    ranking['rank'] = _primary110_interpretation_np.arange(1, len(ranking) + 1)
    _primary110_interpretation_table(ranking, out / 'shap_ranking.csv')
    native = metadata.assign(native_impurity_importance=clf.feature_importances_).sort_values('native_impurity_importance', ascending=False, kind='stable')
    native['native_rank'] = _primary110_interpretation_np.arange(1, len(native) + 1)
    _primary110_interpretation_table(native, out / 'native_importance.csv')
    _primary110_interpretation_write_json(agreement, out / 'shap_agreement.json')
    for kind in ['beeswarm', 'bar']:
        kwargs = {'plot_type': 'bar'} if kind == 'bar' else {}
        _primary110_interpretation_shap.summary_plot(attribution, values, feature_names=metadata.display_name.tolist(), max_display=20, show=False, plot_size=(11, 8), **kwargs)
        fig = _primary110_interpretation_plt.gcf()
        _primary110_interpretation_plt.title('Strict N=110 — probability-scale SHAP\nInterpretation-only full-cohort RF', fontsize=11)
        _primary110_interpretation_plt.xlabel('Mean |SHAP| in probability units' if kind == 'bar' else 'SHAP contribution to P(poly=1)')
        fig.tight_layout()
        _primary110_interpretation_figure(fig, out / f'shap_{kind}.png')
    return agreement

def _primary110_interpretation_modality_summary(ranking, modalities, out):
    table = ranking.groupby('modality').agg(n_classifier_inputs=('feature', 'size'), total_mean_absolute_SHAP=('mean_abs_shap', 'sum'), mean_absolute_SHAP_per_input=('mean_abs_shap', 'mean')).reindex(modalities)
    table['n_classifier_inputs'] = table.n_classifier_inputs.fillna(0).astype(int)
    table['total_mean_absolute_SHAP'] = table.total_mean_absolute_SHAP.fillna(0)
    total = table.total_mean_absolute_SHAP.sum()
    if total <= 0:
        raise ValueError('No nonzero SHAP attribution to summarize.')
    table['proportion_of_total'] = table.total_mean_absolute_SHAP / total
    table = table.rename_axis('modality').reset_index()
    _primary110_interpretation_table(table, out / 'modality_shap_summary.csv')
    fig, ax = _primary110_interpretation_plt.subplots(figsize=(7, 4))
    ax.bar(table.modality, table.proportion_of_total, color='#2878a5')
    ax.set(ylabel='Proportion of summed feature-level mean |SHAP|', ylim=(0, 1), title='Strict N=110 — model attribution by modality')
    fig.tight_layout()
    _primary110_interpretation_figure(fig, out / 'modality_attribution.png')

def _primary110_interpretation_validate_completed(directory, fingerprint, payload):
    p = directory / 'provenance.json'
    if not p.exists():
        raise ValueError('Incomplete interpretation directory; do not reuse.')
    v = _primary110_interpretation_json.loads(p.read_text())
    stored = v.get('protocol', {})
    dump = lambda value: _primary110_interpretation_json.dumps(value, sort_keys=True, default=str)
    # Earlier bundles also recorded a hash of an archived source snapshot ('utility_sha256'); it is ignored.
    matches = v['fingerprint'] == fingerprint or (
        _primary110_interpretation_hashlib.sha256(dump(stored).encode()).hexdigest() == v['fingerprint']
        and dump({k: x for k, x in stored.items() if k != 'utility_sha256'}) == dump(payload))
    if not matches or not v['complete']:
        raise ValueError('Incompatible interpretation bundle.')
    for name, h in v['output_hashes'].items():
        if file_hash(directory / name) != h:
            raise ValueError('Published artifact hash mismatch.')
    return v

def fit_interpretation(configuration, blocks, cohort, paths, output_root, repo, n_jobs=4):
    """Tune one locked family; validate; publish a complete interpretation bundle."""
    if configuration not in _primary110_interpretation_LOCKED:
        raise ValueError('Interpretation not authorized for this input.')
    output_root, repo = (_primary110_interpretation_Path(output_root), _primary110_interpretation_Path(repo))
    output_root.mkdir(parents=True, exist_ok=True)
    fingerprint, protocol = _primary110_interpretation_fingerprint(configuration, cohort, paths)
    final = output_root / configuration
    if final.exists():
        v = _primary110_interpretation_validate_completed(final, fingerprint, protocol)
        print(f'LOAD completed interpretation: {configuration}', flush=True)
        return final
    started = _primary110_interpretation_now()
    attempt = _primary110_interpretation_uuid.uuid4().hex[:12]
    private = output_root / 'private' / configuration / attempt
    private.mkdir(parents=True)
    pending = output_root / f'.pending_{configuration}_{attempt}'
    pending.mkdir()
    X = _primary110_interpretation_pd.concat([blocks[m] for m in _common_cohort_CONFIGURATIONS[configuration]], axis=1)
    y = cohort.y_true.to_numpy()
    if len(X) != 110 or X.columns.duplicated().any():
        raise ValueError('Invalid interpretation inputs.')
    family = _primary110_interpretation_LOCKED[configuration]
    base = compact_pipeline(family)
    model = _primary110_interpretation_Pipeline([('preprocessor', StrictModalityPreprocessor(configuration, str(output_root / 'private/modality_cache'))), *base.steps], memory=_primary110_interpretation_Memory(private / 'pipeline_cache', verbose=0))
    grid = compact_grids(configuration)[family]
    cv = list(_primary110_interpretation_StratifiedKFold(5, shuffle=True, random_state=_primary110_interpretation_SEED).split(X, y))
    search = _primary110_interpretation_GridSearchCV(model, grid, scoring='f1_macro', cv=cv, refit=True, n_jobs=n_jobs, pre_dispatch=n_jobs, error_score='raise', return_train_score=False)
    _primary110_interpretation_write_json({'started_utc': started, 'fingerprint': fingerprint, 'interpretation_only': True, 'N': len(X), 'candidates': len(_primary110_interpretation_ParameterGrid(grid))}, private / 'run.json')
    print(f'START {configuration}: {len(_primary110_interpretation_ParameterGrid(grid))} candidates x 5 folds; {started}', flush=True)
    with _primary110_interpretation_threadpool_limits(1), _primary110_interpretation_parallel_config(backend='loky', inner_max_num_threads=1):
        search.fit(X, y)
    fitted = search.best_estimator_
    values, names = classifier_input(fitted, X)
    clf = fitted.named_steps['classifier']
    pos = _primary110_interpretation_np.flatnonzero(clf.classes_ == 1)
    if len(pos) != 1:
        raise ValueError('Positive class missing.')
    pos = int(pos[0])
    probs = fitted.predict_proba(X)[:, pos]
    direct = clf.predict_proba(values)[:, pos]
    if not _primary110_interpretation_np.allclose(probs, direct, rtol=0, atol=_primary110_interpretation_PROBABILITY_TOLERANCE):
        raise ValueError('Pipeline/classifier probability mismatch.')
    fitted.memory = None
    pipeline_path = private / 'fitted_pipeline.joblib'
    _primary110_interpretation_joblib.dump(fitted, pipeline_path)
    reloaded = _primary110_interpretation_joblib.load(pipeline_path)
    reloaded_values, reloaded_names = classifier_input(reloaded, X)
    reload_probs = reloaded.predict_proba(X)[:, pos]
    if not _primary110_interpretation_np.array_equal(probs, reload_probs) or not _primary110_interpretation_np.array_equal(names, reloaded_names) or (not _primary110_interpretation_np.array_equal(values, reloaded_values)):
        raise ValueError('Persisted pipeline does not reproduce exact probabilities/features.')
    metadata = _primary110_interpretation_feature_metadata(fitted, names, _primary110_interpretation_Path(paths['transcriptomics']).parent / 'transcriptomics_feature_name_mapping.csv')
    _primary110_interpretation_table(metadata, pending / 'classifier_input_features.csv')
    candidates = _primary110_interpretation_pd.DataFrame(search.cv_results_)
    candidates['params'] = candidates.params.map(lambda p: _primary110_interpretation_json.dumps(_primary110_interpretation_serial_params(p), sort_keys=True))
    keep = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score', *[f'split{i}_test_score' for i in range(5)], 'mean_fit_time', 'mean_score_time']
    _primary110_interpretation_table(candidates[keep], pending / 'tuning_results.csv')
    params = {'configuration': configuration, 'family': family, 'best_params': _primary110_interpretation_serial_params(search.best_params_), 'best_inner_macro_f1': float(search.best_score_), 'interpretation_only': True, 'N': 110, 'n_splits': 5, 'scoring': 'f1_macro', 'random_state': 42, 'n_retained_before_selection': len(fitted.named_steps['preprocessor'].names_), 'n_classifier_inputs': len(names), 'candidate_count': len(candidates)}
    _primary110_interpretation_write_json(params, pending / 'best_params.json')
    if family == 'logreg_l2':
        _primary110_interpretation_clinical_interpretation(fitted, metadata, pending)
    else:
        _primary110_interpretation_tree_interpretation(fitted, values, metadata, probs, pending)
        if len(_common_cohort_CONFIGURATIONS[configuration]) > 1:
            _primary110_interpretation_modality_summary(_primary110_interpretation_pd.read_csv(pending / 'shap_ranking.csv'), _common_cohort_CONFIGURATIONS[configuration], pending)
    v = {'label': _primary110_interpretation_LABEL, 'complete': True, 'fingerprint': fingerprint, 'protocol': protocol, 'started_utc': started, 'completed_utc': _primary110_interpretation_now(), 'private_pipeline': str(pipeline_path), 'private_pipeline_sha256': file_hash(pipeline_path), 'reload_probabilities_identical': True, 'max_reload_probability_difference': float(_primary110_interpretation_np.max(_primary110_interpretation_np.abs(probs - reload_probs))), 'max_pipeline_classifier_probability_difference': float(_primary110_interpretation_np.max(_primary110_interpretation_np.abs(probs - direct))), 'probability_absolute_tolerance': _primary110_interpretation_PROBABILITY_TOLERANCE, 'ordered_classifier_features_sha256': _primary110_interpretation_hashlib.sha256(_primary110_interpretation_json.dumps(names.tolist()).encode()).hexdigest(), 'preprocessing': 'learned entirely within each inner-training partition; final refit on full 110 only after selection', 'patient_level_SHAP_exported': False, 'output_hashes': {p.name: file_hash(p) for p in pending.iterdir() if p.is_file()}}
    _primary110_interpretation_write_json(v, pending / 'provenance.json')
    _primary110_interpretation_os.replace(pending, final)
    print(f'COMPLETE {configuration}: {len(names)} classifier inputs; {_primary110_interpretation_now()}', flush=True)
    return final



# Reusable helper from 20260601-modeling-mutation.ipynb
def extract_selected_hyperparameters(results_df):
    """
    Extract selected hyperparameters from the nested cross-validation results.

    The best parameters selected in each outer fold are expanded into separate
    columns to make model comparison and regularization diagnostics easier.
    """
    selected_rows = []

    for _, row in results_df.iterrows():
        best_params = parse_best_params_mutation(row["best_params"])

        selected_row = {
            "model": row["model"],
            "outer_fold": row["outer_fold"],
            "roc_auc": row["roc_auc"],
            "balanced_accuracy": row["balanced_accuracy"],
            "f1_macro": row["f1_macro"],
            "best_params": row["best_params"]
        }

        for param_name, param_value in best_params.items():
            clean_param_name = param_name.replace("clf__", "selected_")
            selected_row[clean_param_name] = param_value

        selected_rows.append(selected_row)

    return pd.DataFrame(selected_rows)


# Reusable helper from 20260710-representation-learning-tpm-oligopolypatients.ipynb
def extract_sampler_from_best_params(best_params):
    params = json.loads(best_params)

    sampler = str(params.get("sampler", "passthrough"))

    if "SMOTE" in sampler:
        return "SMOTE"

    return "passthrough"


# Reusable helper from 20260707-preprocessing-tpm-oligopolypatients.ipynb
def normalise_transcriptomics_sample_id(sample_id):
    """
    Normalise transcriptomics sample IDs.

    Examples
    --------
    '101B.WTS_S57' -> '101B'
    '100BC' -> '100B'
    """

    sample_id = str(sample_id).strip().upper()

    if "." in sample_id:
        sample_id = sample_id.split(".")[0]

    if sample_id.endswith("BC"):
        sample_id = f"{sample_id[:-2]}B"

    return sample_id


# Reusable helper from 20260707-preprocessing-tpm-oligopolypatients.ipynb
def check_duplicate_genes(
    df,
    gene_id_col,
    gene_name_col,
    dataset_name
):
    """
    Check duplicated Ensembl gene IDs and gene names.
    """

    return {
        "dataset": dataset_name,
        "n_rows": df.shape[0],
        "n_unique_gene_ids": df[gene_id_col].nunique(dropna=True),
        "n_duplicated_gene_ids": df[gene_id_col].duplicated().sum(),
        "n_missing_gene_ids": df[gene_id_col].isna().sum(),
        "n_unique_gene_names": df[gene_name_col].nunique(dropna=True),
        "n_duplicated_gene_names": df[gene_name_col].duplicated().sum(),
        "n_missing_gene_names": df[gene_name_col].isna().sum(),
    }


# Reusable helper from 20260707-preprocessing-tpm-oligopolypatients.ipynb
def summarise_expression_values(
    df,
    gene_id_col,
    gene_name_col,
    dataset_name
):
    """
    Summarise expression value distribution for a transcriptomics matrix.
    """

    expression_df = df.drop(columns=[gene_id_col, gene_name_col]).copy()

    expression_df = expression_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    values = expression_df.values.ravel()
    values = values[~np.isnan(values)]

    return {
        "dataset": dataset_name,
        "n_values": len(values),
        "min": np.min(values),
        "q1": np.quantile(values, 0.25),
        "median": np.median(values),
        "mean": np.mean(values),
        "q3": np.quantile(values, 0.75),
        "q95": np.quantile(values, 0.95),
        "q99": np.quantile(values, 0.99),
        "max": np.max(values),
        "n_zero": np.sum(values == 0),
        "zero_fraction": np.mean(values == 0),
        "n_missing": np.isnan(expression_df.values).sum(),
    }


# Reusable helper from 20260707-preprocessing-tpm-oligopolypatients.ipynb
def convert_to_sample_by_gene_matrix(
    df,
    gene_id_col,
    gene_name_col,
    common_gene_ids,
    dataset_name,
    target_value,
    target_label
):
    """
    Convert transcriptomics matrix from genes x samples to samples x genes.
    """

    df_common = df[
        df[gene_id_col].astype(str).isin(common_gene_ids)
    ].copy()

    df_common[gene_id_col] = df_common[gene_id_col].astype(str)

    df_common = (
        df_common
        .sort_values(by=gene_id_col)
        .reset_index(drop=True)
    )

    expression_df = df_common.drop(
        columns=[gene_id_col, gene_name_col]
    ).copy()

    expression_df = expression_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    expression_df.index = df_common[gene_id_col].astype(str)

    sample_by_gene_df = expression_df.T.reset_index()

    sample_by_gene_df = sample_by_gene_df.rename(
        columns={"index": "sample_id_original"}
    )

    sample_by_gene_df["sample_id_normalised"] = (
        sample_by_gene_df["sample_id_original"]
        .apply(normalise_transcriptomics_sample_id)
    )

    sample_by_gene_df["transcriptomics_source"] = dataset_name
    sample_by_gene_df["metastatic_group"] = target_label
    sample_by_gene_df["Polymetastatic"] = target_value

    gene_feature_cols = [
        col for col in sample_by_gene_df.columns
        if col not in [
            "sample_id_original",
            "sample_id_normalised",
            "transcriptomics_source",
            "metastatic_group",
            "Polymetastatic"
        ]
    ]

    sample_by_gene_df = sample_by_gene_df.rename(
        columns={
            col: f"transcriptomics__{col}"
            for col in gene_feature_cols
        }
    )

    thesis_print(f"{dataset_name} sample-by-gene matrix shape:")
    thesis_print(sample_by_gene_df.shape)

    thesis_print(f"\nTarget distribution for {dataset_name}:")
    thesis_display(sample_by_gene_df["Polymetastatic"].value_counts())

    return sample_by_gene_df


from statsmodels.stats.multitest import multipletests


def adjust_p_values(
    df_results: pd.DataFrame,
    method: str = "fdr_bh",
    alpha: float = 0.05,
    p_value_column: str = "p_value",
    new_p_value_column: str = "p_value_adjusted",
    reject_column: str = "reject_hypothesis",
    print_results: bool = True,
) -> pd.DataFrame:
    """
    Adjust p-values in a results DataFrame to control for multiple comparisons.

    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame with at least a column of raw p-values.
    method : str, default='fdr_bh'
        See statsmodels.multipletests. E.g., 'bonferroni', 'holm', 'fdr_bh', 'fdr_by'.
    alpha : float, default=0.05
        Family-wise (Bonferroni/Holm) or FDR level depending on method.
    p_value_column : str, default='p_value'
        Column name containing raw p-values.
    new_p_value_column : str, default='p_value_adjusted'
        Column to store adjusted p-values.
    reject_column : str, default='reject_hypothesis'
        Column to store boolean rejection indicators.
    print_results : bool, default=True
        If True, prints a short summary table.

    Returns
    -------
    pd.DataFrame
        Input df with added `new_p_value_column` and `reject_column`.
    """
    if print_results:
        print(f"\n--- Adjusting p-values using '{method}' (alpha={alpha}) ---")

    if not isinstance(df_results, pd.DataFrame) or df_results.empty:
        if print_results:
            print("Error: Input must be a non-empty DataFrame.")
        return df_results

    if p_value_column not in df_results.columns:
        raise ValueError(f"'{p_value_column}' column not found in df_results.")

    df_valid = df_results.dropna(subset=[p_value_column]).copy()
    valid_idx = df_valid.index

    if df_valid.empty:
        df_results[new_p_value_column] = np.nan
        df_results[reject_column] = False
        if print_results:
            print("Warning: No valid p-values to adjust.")
        return df_results

    reject, pvals_adj, _, _ = multipletests(
        df_valid[p_value_column].values, alpha=alpha, method=method
    )

    df_results.loc[valid_idx, new_p_value_column] = pvals_adj
    df_results.loc[valid_idx, reject_column] = reject

    missing_idx = df_results.index.difference(valid_idx)
    if len(missing_idx) > 0:
        df_results.loc[missing_idx, new_p_value_column] = np.nan
        df_results.loc[missing_idx, reject_column] = False

    if print_results:
        cols = [c for c in ["feature", p_value_column, new_p_value_column, reject_column] if c in df_results.columns]
        print(
            df_results.sort_values(by=new_p_value_column, na_position="last")
                     .head(20)[cols]
                     .to_string(index=False)
        )
    return df_results


def plot_frequency(
    non_zero_coef_counts,
    id_to_symbol=None,
    title="Frequency of Non-zero Coefficients Across Lasso Models",
    save_plot_path=None,
    interactive=True,
    dpi: int = 144,
    y_axis_title="Features",
    x_axis_title="Frequency",
    bar_color="green",
    width: int | None = None,
    height: int | None = None,
    margin: dict = dict(t=80, b=60, l=120, r=40),
    base_font_size: int | None = None,
    title_font_size: int | None = None,
    axis_title_font_size: int | None = None,
    x_tick_font_size: int | None = None,
    y_tick_font_size: int | None = None,
):
    """
    Plot feature selection frequency as a horizontal bar chart.

    This function takes a mapping of feature -> count of non-zero coefficients across
    multiple fitted models (e.g., Lasso across CV folds) and visualizes counts in
    descending order. Optional ID-to-symbol mapping can be applied for readability.

    Parameters
    ----------
    non_zero_coef_counts : dict[str, int]
        Feature identifier -> frequency count.
    id_to_symbol : dict[str, str] or None
        Optional mapping from IDs to display labels (e.g., gene symbol).
    title : str
        Plot title.
    y_axis_title : str
        Y-axis title (e.g., "Features", "Genes").
    x_axis_title : str
        X-axis title (e.g., "Frequency", "Count").
    bar_color : str or list[str]
        Either a single color (applied to all bars) or a list of per-bar colors.
    interactive : bool, default=True
        If True, show interactive plot; else render PNG.
    dpi : int, default=144
        DPI used for PNG rendering and saving (96 -> scale=1).
    save_plot_path : str or None
        If provided, save the figure; format inferred from extension (requires kaleido).

    Dynamic sizing & fonts (optional)
    ---------------------------------
    width, height : int or None
        If None, height is auto-scaled using n_features=len(counts); width can be
        inferred from label length if labels are available.
    base_font_size, title_font_size, axis_title_font_size : int or None
        Layout font sizes; auto-scaled when n_features is available unless overridden.
    x_tick_font_size, y_tick_font_size : int or None
        Tick label font sizes; auto-scaled when n_features is available unless overridden.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Figure, or None if `non_zero_coef_counts` is empty.

    Notes
    -----
    - Uses `apply_dynamic_plotly_layout` with n_features available.
    - If `bar_color` is a list with wrong length, the first color is used as fallback.
    """

    print(f"\n--- Plotting: {title} ---")

    if not non_zero_coef_counts:
        print("No data available to plot.")
        return None

    # Sort by frequency (descending)
    sorted_items = sorted(non_zero_coef_counts.items(), key=lambda item: item[1], reverse=True)
    feature_ids = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    # Map to symbols if provided
    if id_to_symbol is not None:
        labels = [id_to_symbol.get(fid, fid) for fid in feature_ids]
    else:
        labels = feature_ids

    # --- Validate color input ---
    if isinstance(bar_color, list):
        if len(bar_color) != len(counts):
            print("Warning: bar_color list length does not match number of features; using first color only.")
            bar_color = bar_color[0] if bar_color else "green"
    if not isinstance(bar_color, list):
        bar_color = [bar_color] * len(counts)

    # Create bar plot
    fig = go.Figure(
        go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker_color=bar_color,
        )
    )

    # Apply dynamic layout helper (this plot DOES have n_features)
    fig = apply_dynamic_plotly_layout(
        fig,
        title=title,
        n_features=len(counts),
        width=width,                # if None -> inferred from y_labels length or fallback
        y_labels=labels,            # enables auto-width based on longest label
        height=height,              # if None -> scales with n_features
        base_font_size=base_font_size,
        title_font_size=title_font_size,
        axis_title_font_size=axis_title_font_size,
        x_tick_font_size=x_tick_font_size,
        y_tick_font_size=y_tick_font_size,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        reverse_y=True,
        margin=margin,
    )

    # Keep your background setting
    fig.update_layout(plot_bgcolor="white")

    if interactive:
        fig.show()
    else:
        fig.show(
            renderer="png",
            width=fig.layout.width,
            height=fig.layout.height,
            scale=_plotly_scale_from_dpi(dpi),
        )

    if save_plot_path:
        try:
            output_dir = os.path.dirname(save_plot_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")

            file_ext = os.path.splitext(save_plot_path)[1].lower().replace(".", "")
            if file_ext not in ["png", "jpeg", "webp", "svg", "pdf"]:
                print(f"Warning: Unsupported format '{file_ext}'. Saving as PNG.")
                save_plot_path = os.path.splitext(save_plot_path)[0] + ".png"
                file_ext = "png"

            fig.write_image(
                save_plot_path,
                width=fig.layout.width,
                height=fig.layout.height,
                format=file_ext,
                scale=_plotly_scale_from_dpi(dpi),
            )
            print(f"Plot saved to: {save_plot_path}")
        except ImportError:
            print("Error: 'kaleido' not installed. Run `pip install kaleido` to enable saving.")
        except Exception as e:
            print(f"An error occurred while saving the plot: {e}")

    return fig


def plot_grouped_histogram(
    df,
    value_col,
    group_col,
    groups=None,
    bins=30,
    alpha=0.6,
    figsize=(8, 5),
    title=None,
    xlabel=None,
    ylabel="Number of samples",
    legend_title=None
):
    """
    Plot overlaid histograms for a numeric variable stratified by group.

    This function is useful for comparing the distribution of a sample-level
    numeric variable across cohorts or classes, such as the number of mutated
    genes per sample in oligo and poly cohorts.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the numeric variable and grouping column.

    value_col : str
        Name of the numeric column to plot.

    group_col : str
        Name of the column defining the groups to compare.

    groups : list or None, default=None
        Groups to plot. If None, all non-missing groups in ``group_col`` are
        plotted in sorted order.

    bins : int, default=30
        Number of histogram bins.

    alpha : float, default=0.6
        Transparency level for each histogram.

    figsize : tuple, default=(8, 5)
        Figure size.

    title : str or None, default=None
        Plot title.

    xlabel : str or None, default=None
        X-axis label. If None, ``value_col`` is used.

    ylabel : str, default="Number of samples"
        Y-axis label.

    legend_title : str or None, default=None
        Optional title for the legend.

    Returns
    -------
    None
        Displays a matplotlib histogram.
    """

    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found in dataframe.")

    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in dataframe.")

    plot_df = df[[value_col, group_col]].dropna().copy()

    if groups is None:
        groups = sorted(plot_df[group_col].unique())

    plt.figure(figsize=figsize)

    for group in groups:
        values = plot_df.loc[
            plot_df[group_col] == group,
            value_col
        ]

        plt.hist(
            values,
            bins=bins,
            alpha=alpha,
            label=str(group)
        )

    if title is not None:
        plt.title(title)

    plt.xlabel(xlabel if xlabel is not None else value_col)
    plt.ylabel(ylabel)

    if legend_title is not None:
        plt.legend(title=legend_title)
    else:
        plt.legend()

    plt.show()


def validate_allowed_values(
    df: pd.DataFrame,
    allowed_values: dict = None,
    id_cols: list = None,
    binary_prefixes: list = None,
    auto_add_binary_prefixes: bool = True,
    verbose: bool = True
):
    """
    Validate whether selected columns contain only predefined allowed values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    allowed_values : dict, optional
        Dictionary mapping column names to lists of allowed values.
        If None, a default dictionary is used.

    id_cols : list, optional
        Identifier columns to include in the invalid entries output.
        Default: ['record_id', 'sample_id'] if present in the dataframe.

    binary_prefixes : list, optional
        Prefixes for checkbox-style binary columns that should automatically
        receive allowed values [0, 1].

    auto_add_binary_prefixes : bool, optional
        Whether to automatically assign [0, 1] to columns matching the prefixes.

    verbose : bool, optional
        Whether to print summary information.

    Returns
    -------
    invalid_values_df : pd.DataFrame
        DataFrame with invalid entries found.

    summary_invalid : pd.DataFrame
        Summary table with counts by invalid column and invalid value.

    allowed_values : dict
        Final dictionary of allowed values used in the validation.
    """

    def canonical_value(x):
        """
        Normalize values so that 1, 1.0, '1', and '1.0' are treated equivalently.
        """
        if pd.isna(x):
            return np.nan

        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return np.nan
            try:
                fx = float(x)
                if fx.is_integer():
                    return str(int(fx))
                return str(fx)
            except Exception:
                return x

        if isinstance(x, (int, np.integer)):
            return str(int(x))

        if isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return np.nan
            if float(x).is_integer():
                return str(int(x))
            return str(float(x))

        return str(x)

    if allowed_values is None:
        allowed_values = {
            'abc_smoker': [1, 2, 3, 4],
            'abc_alcohol': [1, 2, 3, 4, 5],
            'histo_result___1': [1, 2],

            'abc_breast_side': [1, 2, 3, 4],
            'abc_breast_quadrant': [1, 2, 3, 4, 5],
            'abc_grade': [1, 2, 3, 4],

            'abc_multicentric': [0, 1],
            'abc_multifocal': [0, 1],

            'abc_re': [1, 2, 3],
            'abc_rp': [1, 2, 3],
            'abc_her2': [1, 2, 3, 4, 5, 6],

            'luminal_a': [0, 1],
            'luminal_b': [0, 1],
            'tnbc': [0, 1],
            'her2': [0, 1],

            'abc_p53': [1, 2, 3],
            'abc_hist': [1, 2, 3, 4],

            'abc_ajcc_c_t': list(range(1, 15)),
            'abc_ajcc_p_t_2': list(range(1, 15)),

            'abc_neoadj_rt': [0, 1],
            'abc_neoadj_chemo': [0, 1],
            'abc_neoadj_hormone': [0, 1],
            'abc_neoadj_her2': [0, 1],
            'abc_surg_primary': [0, 1],
            'abc_adj_rt': [0, 1],

            'tumor2': [0, 1],
            'tumor2_breast_side': [1, 2],
            'tumor2_breast_quadrant': [1, 2, 3, 4, 5],
            'tumor2_grade': [1, 2, 3, 4],
            'tumor2_multicentric': [0, 1],
            'tumor2_multifocal': [0, 1],
            'tumor2_re': [1, 2, 3],
            'tumor2_pr': [1, 2, 3],
            'tumor2_her2': [1, 2, 3, 4, 5, 6],
            'tumor2_la': [0, 1],
            'tumor2_lb': [0, 1],
            'tumor2_tnbc': [0, 1],
            'tumor2_her2like': [0, 1],
            'tumor2_neoadj_rt': [0, 1],
            'tumor2_neoadj_qt': [0, 1],
        }

    if binary_prefixes is None:
        binary_prefixes = [
            'abc_neoadj_hormone_scheme___',
            'abc_neoadj_her2_scheme___',
            'abc_adj___',
            'abc_adj_chemo___',
            'abc_adj_ht___',
            'abc_adj_her2___',
            'tumour2_type___',
            'abc_m1_diagnose___'
        ]

    allowed_values = allowed_values.copy()

    if auto_add_binary_prefixes:
        for col in df.columns:
            if any(col.startswith(prefix) for prefix in binary_prefixes):
                allowed_values[col] = [0, 1]

    if id_cols is None:
        id_cols = [col for col in ['record_id', 'sample_id'] if col in df.columns]

    invalid_entries = []

    for col, valid_options in allowed_values.items():
        if col not in df.columns:
            continue

        valid_set = {canonical_value(v) for v in valid_options}
        normalized_col = df[col].apply(canonical_value)

        invalid_mask = normalized_col.notna() & ~normalized_col.isin(valid_set)

        if invalid_mask.any():
            temp = df.loc[invalid_mask, id_cols + [col]].copy()
            temp['invalid_column'] = col
            temp['invalid_value'] = df.loc[invalid_mask, col].values
            temp['normalized_invalid_value'] = normalized_col.loc[invalid_mask].values
            temp['allowed_values'] = ", ".join(sorted(valid_set))
            invalid_entries.append(temp)

    if invalid_entries:
        invalid_values_df = pd.concat(invalid_entries, ignore_index=True)

        summary_invalid = (
            invalid_values_df
            .groupby(['invalid_column', 'normalized_invalid_value'])
            .size()
            .reset_index(name='count')
            .sort_values(['invalid_column', 'count'], ascending=[True, False])
        )

        if verbose:
            print(f"Number of invalid entries found: {len(invalid_values_df)}")

        return invalid_values_df, summary_invalid, allowed_values

    if verbose:
        print("No invalid entries found for the specified columns.")

    invalid_values_df = pd.DataFrame()
    summary_invalid = pd.DataFrame(columns=['invalid_column', 'normalized_invalid_value', 'count'])

    return invalid_values_df, summary_invalid, allowed_values


import math
from itertools import islice
from scipy.stats import fisher_exact

def get_2x2_table(df, col1, col2):
    table = pd.crosstab(df[col1], df[col2])

    for row_val in [0, 1]:
        if row_val not in table.index:
            table.loc[row_val] = 0
    for col_val in [0, 1]:
        if col_val not in table.columns:
            table[col_val] = 0

    table = table.sort_index().sort_index(axis=1)
    return table

def phi_coefficient_from_table(table):
    a = table.loc[1, 1]
    b = table.loc[1, 0]
    c = table.loc[0, 1]
    d = table.loc[0, 0]

    denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denominator == 0:
        return np.nan

    return (a * d - b * c) / denominator

def compute_pair(df, col1, col2, phi_save_threshold):
    table = get_2x2_table(df, col1, col2)
    phi = phi_coefficient_from_table(table)

    if np.isnan(phi) or abs(phi) < phi_save_threshold:
        return None

    _, p_value = fisher_exact(table)

    return {
        "Feature1": col1,
        "Feature2": col2,
        "Phi": phi,
        "P_value": p_value,
        "n11": int(table.loc[1, 1]),
        "n10": int(table.loc[1, 0]),
        "n01": int(table.loc[0, 1]),
        "n00": int(table.loc[0, 0]),
        "freq_feature1": int(df[col1].sum()),
        "freq_feature2": int(df[col2].sum())
    }

def process_chunk(df, pairs_chunk, phi_save_threshold):
    results = []
    for col1, col2 in pairs_chunk:
        res = compute_pair(df, col1, col2, phi_save_threshold)
        if res is not None:
            results.append(res)
    return results

def chunked_iterable(iterable, size):
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            break
        yield chunk


def clean_gene_symbol(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    if x == "":
        return np.nan

    return x


def expression_shap_direction(expression, attribution, feature_names, genes):
    """Descriptive direction: Spearman plus monotonic quantile-bin medians.

    No hypothesis test or significance cutoff is used. Mixed bin trends remain
    unclear even when the global rank correlation has a nonzero sign.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    values = np.asarray(attribution)
    names = list(feature_names)
    if values.shape != (len(expression), len(names)) or not np.isfinite(values).all():
        raise ValueError('SHAP/expression alignment mismatch')
    rows, bins = [], []
    for gene in genes.itertuples():
        name = 'transcriptomics__' + gene.ensembl_gene_id
        row = {'Gene': gene.gene_symbol, 'feature': name, 'SHAP_direction': 'unclear/non-monotonic', 'Spearman_expression_SHAP': np.nan}
        if name not in names:
            row['notes'] = 'Not a selected input to this fitted classifier; no SHAP direction assigned.'
            rows.append(row)
            continue
        x = expression[name].to_numpy(dtype=float)
        s = values[:, names.index(name)]
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite expression')
        if np.unique(x).size < 2 or np.unique(s).size < 2:
            row['notes'] = 'Expression or attribution is constant.'
            rows.append(row)
            continue
        rho = float(spearmanr(x, s).statistic)
        frame = pd.DataFrame({'expression': x, 'SHAP': s})
        frame['bin'] = pd.qcut(frame.expression, q=5, duplicates='drop')
        summary = frame.groupby('bin', observed=True).agg(expression_median=('expression', 'median'), shap_median=('SHAP', 'median'), N=('SHAP', 'size')).reset_index(drop=True)
        summary['Gene'] = gene.gene_symbol
        summary['bin_number'] = np.arange(1, len(summary)+1)
        bins.append(summary)
        delta = np.diff(summary.shap_median)
        tol = np.finfo(float).eps * 100 * max(1., float(np.max(np.abs(s))))
        if len(delta) and rho > 0 and np.all(delta >= -tol) and np.any(delta > tol):
            row['SHAP_direction'] = 'higher expression -> PMD'
        elif len(delta) and rho < 0 and np.all(delta <= tol) and np.any(delta < -tol):
            row['SHAP_direction'] = 'higher expression -> OMD'
        row['Spearman_expression_SHAP'] = rho
        row['notes'] = 'Exploratory sign and monotonic equal-frequency-bin median check; no significance threshold; heterogeneous individual contributions remain possible.'
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(bins, ignore_index=True) if bins else pd.DataFrame()


def fit_frozen_family_gene_cox(cohort, genes, endpoint='PFI', adjusted=False):
    """One unpenalized continuous Cox fit per frozen gene; separate BH family."""
    data = endpoint_cohort(cohort, endpoint, adjusted=adjusted)
    results = []
    for _, gene in genes.iterrows():
        values = data[gene.ensembl_gene_id]
        sd = values.std(ddof=1)
        if not _tcga_prognostic_np.isfinite(sd) or sd <= 0:
            raise ValueError('Gene has no usable expression variation: ' + gene.gene_symbol)
        frame = _tcga_prognostic_pd.DataFrame({'duration': data[endpoint + '.time'], 'event': data[endpoint], 'expression_z': (values - values.mean()) / sd})
        formula = 'expression_z'
        if adjusted:
            frame['age'] = data.age
            frame['stage'] = data.stage
            formula += ' + age + C(stage)'
        with _tcga_prognostic_warnings.catch_warnings():
            _tcga_prognostic_warnings.simplefilter('error', _tcga_prognostic_ConvergenceWarning)
            model = _tcga_prognostic_CoxPHFitter(penalizer=0.0).fit(frame, 'duration', 'event', formula=formula, fit_options={'step_size': 0.25} if adjusted else None)
        row = model.summary.loc['expression_z']
        ph = _tcga_prognostic_proportional_hazard_test(model, frame, time_transform='rank').summary.loc['expression_z', 'p']
        results.append({**gene.to_dict(), 'endpoint': endpoint, 'adjusted': adjusted, 'N': len(frame), 'events': int(frame.event.sum()), 'coefficient': row['coef'], 'HR_per_1SD': row['exp(coef)'], 'CI_lower': row['exp(coef) lower 95%'], 'CI_upper': row['exp(coef) upper 95%'], 'p_value': row['p'], 'PH_assumption_p': ph, 'PH_violation_flag': bool(ph < 0.05), 'expression_mean': values.mean(), 'expression_sd': sd})
    results = _tcga_prognostic_pd.DataFrame(results)
    if len(results) != len(genes) or not _tcga_prognostic_np.isfinite(results[['p_value', 'PH_assumption_p', 'HR_per_1SD', 'CI_lower', 'CI_upper']]).all().all():
        raise ValueError('Incomplete/non-finite frozen-gene Cox result family.')
    results['q_value'] = _tcga_prognostic_multipletests(results.p_value, method='fdr_bh')[1]
    return results


def plot_thesis_saved_figure(figure_kind, configuration=None):
    """Return a live matplotlib Figure from preserved data; never fit or save artifacts.

    Common-cohort curves use the existing reporting helper and its unchanged
    calibration bins. Forest plots read saved estimates; train/test reads saved SDs.
    """
    from pathlib import Path
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    u = sys.modules[__name__]
    R = Path(__file__).resolve().parents[2]
    B = Path('/mnt/data3/MCPP-Breast')
    M = B / 'Machine-Learning-Models-poly_vs_oligo'
    C = M / 'Common-Cohort-Modality-Comparison'
    D = M / 'MCPP-Biological-Characterization/common-110/harmonised_main8'
    read = pd.read_csv
    figures = []
    def save(fig, original, priority):
        # Match export typography while retaining a live Figure for plt.show().
        for ax in fig.axes:
            ax.title.set_fontsize(12)
            ax.tick_params(labelsize=10)
        figures.append(fig)
    families = {'random_forest':'Random Forest','logreg_l1':'Logistic Regression (L1)','logreg_l2':'Logistic Regression (L2)','xgboost':'XGBoost'}
    priority = {}
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,'savefig.dpi':300,'pdf.fonttype':42}):
        if figure_kind in {'roc','pr','confusion_matrix','calibration'}:
            bundle={'selected':read(C/'common_cohort_primary_comparison.csv'),'predictions':read(C/'private/common_cohort_oof_predictions.csv')}
            families={'random_forest':'Random Forest','logreg_l1':'Logistic Regression (L1)','logreg_l2':'Logistic Regression (L2)','xgboost':'XGBoost'}
            priority={'mutation':{'roc','pr','calibration'},'clinical_mutation':{'roc','pr','calibration'},'transcriptomics':{'roc','pr','confusion_matrix','calibration'},'clinical_mutation_transcriptomics':{'roc','pr','confusion_matrix','calibration'}}
            for config in [configuration]:
             name=' + '.join({'clinical':'Clinical','mutation':'Mutation','transcriptomics':'Transcriptomics'}[x] for x in config.split('_'))
             family=bundle['selected'].set_index('configuration').loc[config,'family']
             # Existing reporting helper preserves ROC/PR coordinates, five uniform calibration bins and saved labels.
             for kind,fig in u.plot_primary110(bundle,config).items():
              if kind != figure_kind:
               plt.close(fig)
               continue
              ax=fig.axes[0];title={'roc':'ROC curve','pr':'Precision–recall curve','confusion_matrix':'Confusion matrix','calibration':'Calibration'}[kind]
              ax.set_title(f'{name}\n{families[family]} | {title} | N=110',pad=12)
              fig.set_size_inches(7.2,5.6);fig.tight_layout()
              save(fig,C/f'primary_110/{config}/{config}_primary_110_{kind}.png',kind in priority.get(config,set()))
        elif figure_kind == 'expression':
            g=read(D/'harmonised_only_main8_manifest.csv');cohort=read(C/'private/common_patients.csv');cohort['key']=cohort.sample_id.map(u._mcpp_biological_normalise_id);cohort=cohort.set_index('key')
            x=read(B/'Transcriptomic-Data/Preprocessed/transcriptomics_oligo_poly_log2_tpm_preprocessed.csv');x['key']=x.sample_id_normalised.map(u._mcpp_biological_normalise_id);x=x.set_index('key').loc[cohort.index];assert np.array_equal(x.Polymetastatic,cohort.y_true)
            e=x[['transcriptomics__'+v for v in g.ensembl_gene_id]].copy();e.columns=g.ensembl_gene_id
            fig=u.plot_expression(e,cohort.y_true,g);fig.suptitle('Eight model-prioritised genes | Expression by metastatic phenotype');save(fig,D/'harmonised_only_main8_expression.png',True)
        elif figure_kind == 'cox':
            for origin in [configuration]:
             fig=u.plot_gene_hr(read(D/f'harmonised_only_main8_{origin}_OS_cox.csv'));fig.axes[0].set_title(f'Eight model-prioritised genes | {origin.capitalize()}-origin OS');save(fig,D/f'harmonised_only_main8_{origin}_OS_forest.png',origin=='metastatic')
        elif figure_kind == 'train_test':
            diagnostics=read(C/'common_cohort_train_test_diagnostics.csv')
            for config,part in [(configuration, diagnostics.loc[diagnostics.configuration.eq(configuration)])]:
             name=' + '.join({'clinical':'Clinical','mutation':'Mutation','transcriptomics':'Transcriptomics'}[x] for x in config.split('_'))
             fig,ax=plt.subplots(figsize=(8,5.8));positions=np.arange(len(part));width=.36
             for offset,split,label,color in [(-width/2,'train','Outer-train AUC (apparent)','#397ca5'),(width/2,'test','Outer-test AUC (held-out)','#d88745')]:
              ax.bar(positions+offset,part[f'roc_auc_{split}_mean'],width,yerr=part[f'roc_auc_{split}_std'],capsize=4,label=label,color=color)
             ax.set(xticks=positions,xticklabels=[families[f] for f in part.family],ylabel='ROC-AUC',title=f'{name} | Train/test performance\nCommon cohort N=110')
             ax.tick_params(axis='x',rotation=20);ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),frameon=False)
             fig.text(.5,.015,'Means ± SD across five outer folds; training performance is apparent.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.09,1,1))
             original=C/f'primary_110/{config}/{config}_primary_110_train_test.png'
             # Unique filenames for the pairwise train_test.png sources.
             save(fig,original.with_name(config+'_train_test.png'),False)

        else:
            raise ValueError(f'Unknown saved figure: {figure_kind}')
    if len(figures) != 1:
        raise ValueError('Expected exactly one saved-data figure.')
    return figures[0]
