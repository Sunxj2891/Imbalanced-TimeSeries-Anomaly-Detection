"""Train and evaluate an MLP neural network for HTRU2 pulsar detection.

This script uses the already engineered train/dev/test CSV files. It does not
redo raw preprocessing or create a new split.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 7
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
        filename=LOGS_DIR / "mlp_model.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )


def log_progress(message: str) -> None:
    """Print and log progress messages."""
    print(message, flush=True)
    logging.info(message)


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split one data frame into features and target."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' was not found.")
    x = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN].astype(int)
    return x, y


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load final engineered train/dev/test data."""
    log_progress("Loading engineered train/dev/test data")

    train_df = pd.read_csv(DATA_DIR / "train_engineered.csv")
    dev_df = pd.read_csv(DATA_DIR / "dev_engineered.csv")
    test_df = pd.read_csv(DATA_DIR / "test_engineered.csv")

    x_train, y_train = split_features_target(train_df)
    x_dev, y_dev = split_features_target(dev_df)
    x_test, y_test = split_features_target(test_df)

    for split_name, x, y in [
        ("Train", x_train, y_train),
        ("Dev", x_dev, y_dev),
        ("Test", x_test, y_test),
    ]:
        counts = y.value_counts().sort_index().to_dict()
        log_progress(f"{split_name} shape: {x.shape}")
        log_progress(f"{split_name} class distribution: {counts}")

    log_progress(f"Feature columns: {list(x_train.columns)}")
    return x_train, y_train, x_dev, y_dev, x_test, y_test


def build_mlp(params: dict[str, Any]) -> MLPClassifier:
    """Create an MLPClassifier with fixed base settings."""
    return MLPClassifier(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        alpha=params["alpha"],
        learning_rate_init=params["learning_rate_init"],
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=RANDOM_STATE,
        early_stopping=False,
    )


def evaluate_predictions(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, Any]:
    """Calculate standard binary classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc_average_precision": average_precision_score(y_true, y_proba),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def try_import_random_oversampler() -> Any | None:
    """Import RandomOverSampler if imblearn is available."""
    try:
        from imblearn.over_sampling import RandomOverSampler

        log_progress("imblearn is available; including RandomOverSampler MLP variant")
        return RandomOverSampler
    except Exception as error:
        log_progress(f"imblearn is not available; skipping oversampling variant: {error}")
        return None


def tune_mlp(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_dev: pd.DataFrame,
    y_dev: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune MLP hyperparameters using the dev set."""
    param_grid = {
        "hidden_layer_sizes": [(8,), (16,), (32,), (16, 8), (32, 16)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.0005],
    }

    RandomOverSampler = try_import_random_oversampler()
    imbalance_methods = ["none"]
    if RandomOverSampler is not None:
        imbalance_methods.append("random_oversampling")

    rows: list[dict[str, Any]] = []
    best_score = -1.0
    best_config: dict[str, Any] | None = None

    total_configs = len(list(ParameterGrid(param_grid))) * len(imbalance_methods)
    config_number = 0

    for imbalance_method in imbalance_methods:
        for params in ParameterGrid(param_grid):
            config_number += 1
            log_progress(
                f"Tuning MLP config {config_number}/{total_configs}: "
                f"imbalance_method={imbalance_method}, params={params}"
            )

            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_dev_scaled = scaler.transform(x_dev)

            y_train_fit = y_train
            x_train_fit = x_train_scaled
            if imbalance_method == "random_oversampling":
                sampler = RandomOverSampler(random_state=RANDOM_STATE)
                x_train_fit, y_train_fit = sampler.fit_resample(
                    x_train_scaled, y_train
                )

            model = build_mlp(params)
            model.fit(x_train_fit, y_train_fit)

            dev_pred = model.predict(x_dev_scaled)
            dev_proba = model.predict_proba(x_dev_scaled)[:, 1]
            metrics = evaluate_predictions("MLP", y_dev, dev_pred, dev_proba)

            row = {
                "model": "MLP",
                "imbalance_method": imbalance_method,
                "hidden_layer_sizes": params["hidden_layer_sizes"],
                "alpha": params["alpha"],
                "learning_rate_init": params["learning_rate_init"],
                **{f"dev_{key}": value for key, value in metrics.items() if key != "model"},
            }
            rows.append(row)

            dev_score = metrics["pr_auc_average_precision"]
            if dev_score > best_score:
                best_score = dev_score
                best_config = {
                    **params,
                    "imbalance_method": imbalance_method,
                }

    if best_config is None:
        raise RuntimeError("No MLP configuration was evaluated.")

    tuning_results = pd.DataFrame(rows)
    tuning_results["rank_dev_pr_auc"] = (
        tuning_results["dev_pr_auc_average_precision"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    tuning_results = tuning_results.sort_values("rank_dev_pr_auc").reset_index(
        drop=True
    )

    log_progress(f"Best MLP hyperparameters: {best_config}")
    log_progress(f"Best MLP dev PR-AUC: {best_score:.4f}")
    return best_config, tuning_results


def fit_final_model(
    best_config: dict[str, Any],
    x_train_dev: pd.DataFrame,
    y_train_dev: pd.Series,
) -> Pipeline:
    """Fit the final scaler + MLP pipeline on train + dev."""
    scaler = StandardScaler()
    x_train_dev_scaled = scaler.fit_transform(x_train_dev)
    y_train_fit = y_train_dev
    x_train_fit = x_train_dev_scaled

    sampler = None
    if best_config["imbalance_method"] == "random_oversampling":
        RandomOverSampler = try_import_random_oversampler()
        if RandomOverSampler is None:
            raise RuntimeError("Best config needs RandomOverSampler, but it is unavailable.")
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
        x_train_fit, y_train_fit = sampler.fit_resample(
            x_train_dev_scaled, y_train_dev
        )

    params = {
        "hidden_layer_sizes": best_config["hidden_layer_sizes"],
        "alpha": best_config["alpha"],
        "learning_rate_init": best_config["learning_rate_init"],
    }
    mlp = build_mlp(params)
    mlp.fit(x_train_fit, y_train_fit)

    # Store sampler as an attribute for transparency. Prediction only needs scaler + MLP.
    pipeline = Pipeline([("scaler", scaler), ("mlp", mlp)])
    pipeline.imbalance_method = best_config["imbalance_method"]
    pipeline.random_oversampler = sampler
    return pipeline


def main() -> None:
    setup_outputs_and_logging()
    x_train, y_train, x_dev, y_dev, x_test, y_test = load_data()

    best_config, tuning_results = tune_mlp(x_train, y_train, x_dev, y_dev)
    tuning_results.to_csv(TABLES_DIR / "mlp_tuning_results.csv", index=False)

    log_progress("Training final MLP on train + dev")
    x_train_dev = pd.concat([x_train, x_dev], ignore_index=True)
    y_train_dev = pd.concat([y_train, y_dev], ignore_index=True)
    final_pipeline = fit_final_model(best_config, x_train_dev, y_train_dev)

    log_progress("Evaluating final MLP on test set")
    test_pred = final_pipeline.predict(x_test)
    test_proba = final_pipeline.predict_proba(x_test)[:, 1]
    final_metrics = evaluate_predictions("MLP", y_test, test_pred, test_proba)
    final_metrics.update(
        {
            "imbalance_method": best_config["imbalance_method"],
            "best_hidden_layer_sizes": best_config["hidden_layer_sizes"],
            "best_alpha": best_config["alpha"],
            "best_learning_rate_init": best_config["learning_rate_init"],
        }
    )

    results_df = pd.DataFrame([final_metrics])
    predictions_df = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "mlp_pred": test_pred,
            "mlp_proba": test_proba,
        }
    )

    log_progress("Saving MLP outputs")
    results_df.to_csv(TABLES_DIR / "mlp_model_results.csv", index=False)
    predictions_df.to_csv(PREDICTIONS_DIR / "mlp_test_predictions.csv", index=False)
    joblib.dump(final_pipeline, MODELS_DIR / "mlp_model.joblib")

    log_progress(f"Final MLP test metrics: {final_metrics}")
    print("\nFinal MLP test results:")
    print(results_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
