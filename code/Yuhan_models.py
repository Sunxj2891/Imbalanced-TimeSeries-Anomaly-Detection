"""Train Yuhan's Random Forest and XGBoost models.

The input files are already engineered and split into train/dev/test sets.
This script does not redo preprocessing, create a new split, or refit a scaler.
It uses the dev set for hyperparameter tuning and the test set only for final
evaluation.
"""

from __future__ import annotations

import logging
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET_COLUMN = "class"

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "outputs" / "processed_data"
TABLES_DIR = PROJECT_DIR / "outputs" / "tables"
PREDICTIONS_DIR = PROJECT_DIR / "outputs" / "predictions"
MODELS_DIR = PROJECT_DIR / "outputs" / "models"
LOGS_DIR = PROJECT_DIR / "outputs" / "logs"


def setup_outputs_and_logging() -> None:
    """Create output folders and configure logging."""
    for folder in [TABLES_DIR, PREDICTIONS_DIR, MODELS_DIR, LOGS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=LOGS_DIR / "Yuhan_models.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )


def log_progress(message: str) -> None:
    """Print and log progress messages."""
    print(message, flush=True)
    logging.info(message)


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into features and target."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' was not found.")

    x = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN].astype(int)
    return x, y


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train, dev, and test engineered datasets."""
    log_progress("Loading engineered train/dev/test data")

    train_df = pd.read_csv(DATA_DIR / "train_engineered.csv")
    dev_df = pd.read_csv(DATA_DIR / "dev_engineered.csv")
    test_df = pd.read_csv(DATA_DIR / "test_engineered.csv")

    x_train, y_train = split_features_target(train_df)
    x_dev, y_dev = split_features_target(dev_df)
    x_test, y_test = split_features_target(test_df)

    log_progress(f"Train shape: {x_train.shape}")
    log_progress(f"Dev shape: {x_dev.shape}")
    log_progress(f"Test shape: {x_test.shape}")
    return x_train, y_train, x_dev, y_dev, x_test, y_test


def scale_pos_weight(y: pd.Series) -> float:
    """Compute negative/positive class ratio for XGBoost."""
    negative_count = int((y == 0).sum())
    positive_count = int((y == 1).sum())
    return negative_count / positive_count if positive_count > 0 else 1.0


def positive_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    """Return the positive-class probability."""
    return model.predict_proba(x)[:, 1]


def tune_with_dev_set(
    model_name: str,
    base_params: dict[str, Any],
    param_grid: dict[str, list[Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_dev: pd.DataFrame,
    y_dev: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune hyperparameters by training on train and scoring on dev."""
    log_progress(f"Tuning {model_name} on the dev set")

    rows: list[dict[str, Any]] = []
    best_score = -1.0
    best_params: dict[str, Any] | None = None

    for params in ParameterGrid(param_grid):
        model_params = {**base_params, **params}
        if model_name == "Random Forest":
            model = RandomForestClassifier(**model_params)
        else:
            model = XGBClassifier(**model_params)

        model.fit(x_train, y_train)
        dev_proba = positive_probability(model, x_dev)
        dev_pred = model.predict(x_dev)
        dev_score = average_precision_score(y_dev, dev_proba)

        row = {
            "model": model_name,
            "dev_pr_auc_average_precision": dev_score,
            "dev_roc_auc": roc_auc_score(y_dev, dev_proba),
            "dev_f1_score": f1_score(y_dev, dev_pred, zero_division=0),
            "dev_recall": recall_score(y_dev, dev_pred, zero_division=0),
            "dev_precision": precision_score(y_dev, dev_pred, zero_division=0),
            "params": params,
        }
        rows.append(row)

        if dev_score > best_score:
            best_score = dev_score
            best_params = params

    tuning_results = pd.DataFrame(rows)
    tuning_results["rank_dev_pr_auc"] = (
        tuning_results["dev_pr_auc_average_precision"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    tuning_results = tuning_results.sort_values(
        ["model", "rank_dev_pr_auc"]
    ).reset_index(drop=True)

    if best_params is None:
        raise RuntimeError(f"No valid parameter setting found for {model_name}.")

    log_progress(f"Best {model_name} parameters: {best_params}")
    log_progress(f"Best {model_name} dev PR-AUC: {best_score:.4f}")
    return best_params, tuning_results


def fit_final_model(
    model_name: str,
    base_params: dict[str, Any],
    best_params: dict[str, Any],
    x_train_dev: pd.DataFrame,
    y_train_dev: pd.Series,
) -> Any:
    """Fit the final model on train + dev using the selected parameters."""
    final_params = {**base_params, **best_params}

    if model_name == "Random Forest":
        model = RandomForestClassifier(**final_params)
    else:
        final_params["scale_pos_weight"] = scale_pos_weight(y_train_dev)
        model = XGBClassifier(**final_params)

    model.fit(x_train_dev, y_train_dev)
    return model


def evaluate_model(
    model_name: str,
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate the final model on the held-out test set."""
    y_pred = model.predict(x_test)
    y_proba = positive_probability(model, x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc_average_precision": average_precision_score(y_test, y_proba),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }
    return metrics, y_pred, y_proba


def main() -> None:
    setup_outputs_and_logging()
    x_train, y_train, x_dev, y_dev, x_test, y_test = load_data()

    x_train_dev = pd.concat([x_train, x_dev], ignore_index=True)
    y_train_dev = pd.concat([y_train, y_dev], ignore_index=True)

    rf_base_params = {
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    rf_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    xgb_base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight(y_train),
    }
    xgb_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    rf_best_params, rf_tuning = tune_with_dev_set(
        "Random Forest", rf_base_params, rf_grid, x_train, y_train, x_dev, y_dev
    )
    xgb_best_params, xgb_tuning = tune_with_dev_set(
        "XGBoost", xgb_base_params, xgb_grid, x_train, y_train, x_dev, y_dev
    )

    log_progress("Training final models on train + dev")
    rf_model = fit_final_model(
        "Random Forest", rf_base_params, rf_best_params, x_train_dev, y_train_dev
    )
    xgb_model = fit_final_model(
        "XGBoost", xgb_base_params, xgb_best_params, x_train_dev, y_train_dev
    )

    log_progress("Evaluating final models on test set")
    rf_metrics, rf_pred, rf_proba = evaluate_model(
        "Random Forest", rf_model, x_test, y_test
    )
    xgb_metrics, xgb_pred, xgb_proba = evaluate_model(
        "XGBoost", xgb_model, x_test, y_test
    )

    results_df = pd.DataFrame([rf_metrics, xgb_metrics])
    tuning_df = pd.concat([rf_tuning, xgb_tuning], ignore_index=True)
    predictions_df = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "random_forest_pred": rf_pred,
            "random_forest_proba": rf_proba,
            "xgboost_pred": xgb_pred,
            "xgboost_proba": xgb_proba,
        }
    )

    log_progress("Saving outputs")
    results_df.to_csv(TABLES_DIR / "Yuhan_model_results.csv", index=False)
    tuning_df.to_csv(TABLES_DIR / "Yuhan_tuning_results.csv", index=False)
    predictions_df.to_csv(
        PREDICTIONS_DIR / "Yuhan_test_predictions.csv", index=False
    )
    joblib.dump(rf_model, MODELS_DIR / "random_forest_model.joblib")
    joblib.dump(xgb_model, MODELS_DIR / "boosting_model.joblib")

    print("\nFinal test results:")
    print(results_df.round(4).to_string(index=False))
    log_progress("Yuhan model training complete")


if __name__ == "__main__":
    main()
