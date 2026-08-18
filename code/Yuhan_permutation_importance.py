"""Run permutation importance for the group's models.

Models included:
1. Logistic Regression
2. SVM
3. Random Forest
4. XGBoost
5. MLP Neural Network

The script uses the existing engineered train/dev/test files. Hyperparameters
are selected on the dev set using average precision, then final models are fit
on train + dev. The MLP model is loaded from its saved final pipeline.
Permutation importance is calculated on the test set.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET_COLUMN = "class"

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "outputs" / "processed_data"
TABLES_DIR = PROJECT_DIR / "outputs" / "tables"
MODELS_DIR = PROJECT_DIR / "outputs" / "models"
LOGS_DIR = PROJECT_DIR / "outputs" / "logs"


def setup() -> None:
    """Create output folders and configure logging."""
    for folder in [TABLES_DIR, MODELS_DIR, LOGS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=LOGS_DIR / "Yuhan_permutation_importance.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )


def log_progress(message: str) -> None:
    """Print and log progress."""
    print(message, flush=True)
    logging.info(message)


def load_split(file_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load one engineered split and separate features from target."""
    data = pd.read_csv(DATA_DIR / file_name)
    x = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN].astype(int)
    return x, y


def scale_pos_weight(y: pd.Series) -> float:
    """Compute negative/positive ratio for XGBoost."""
    negative_count = int((y == 0).sum())
    positive_count = int((y == 1).sum())
    return negative_count / positive_count if positive_count > 0 else 1.0


def positive_probability(model: Any, x: pd.DataFrame) -> Any:
    """Return positive-class probability for model selection."""
    return model.predict_proba(x)[:, 1]


def tune_on_dev(
    model_name: str,
    model_factory: Any,
    param_grid: dict[str, list[Any]],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_dev: pd.DataFrame,
    y_dev: pd.Series,
) -> tuple[Any, dict[str, Any], float]:
    """Train each parameter setting on train and select by dev AP."""
    best_score = -1.0
    best_params: dict[str, Any] | None = None
    best_model = None

    for params in ParameterGrid(param_grid):
        model = model_factory(params)
        model.fit(x_train, y_train)
        score = average_precision_score(y_dev, positive_probability(model, x_dev))

        if score > best_score:
            best_score = score
            best_params = dict(params)
            best_model = model

    if best_model is None or best_params is None:
        raise RuntimeError(f"No valid parameter setting found for {model_name}.")

    log_progress(f"Best {model_name} dev average precision: {best_score:.4f}")
    log_progress(f"Best {model_name} parameters: {best_params}")
    return best_model, best_params, best_score


def build_logistic_regression(_: dict[str, Any]) -> Pipeline:
    """Create the Logistic Regression baseline pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=1000,
                    C=1.0,
                ),
            ),
        ]
    )


def build_svm_candidates() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Create SVM variants from the teammate notebook."""
    # A compact grid keeps the permutation-importance run practical while
    # staying inside the same C/gamma choices used in the SVM notebook.
    hyperparameters = {
        "svm__C": [1, 10, 100],
        "svm__gamma": ["scale", 0.001, 0.01],
    }

    def plain_svm(params: dict[str, Any]) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(kernel="rbf", probability=True, random_state=7),
                ),
            ]
        ).set_params(**params)

    def class_balanced_svm(params: dict[str, Any]) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=7,
                    ),
                ),
            ]
        ).set_params(**params)

    def smote_svm(params: dict[str, Any]) -> ImbPipeline:
        return ImbPipeline(
            [
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=7)),
                (
                    "svm",
                    SVC(kernel="rbf", probability=True, random_state=7),
                ),
            ]
        ).set_params(**params)

    return {
        "SVM": (plain_svm, hyperparameters),
        "SVM Class Balanced": (class_balanced_svm, hyperparameters),
        "SVM SMOTE": (smote_svm, hyperparameters),
    }


def build_random_forest(params: dict[str, Any]) -> RandomForestClassifier:
    """Create Random Forest with imbalance-aware class weights."""
    base_params = {
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    return RandomForestClassifier(**{**base_params, **params})


def build_xgboost(params: dict[str, Any], y_train: pd.Series) -> XGBClassifier:
    """Create XGBoost with scale_pos_weight."""
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight(y_train),
    }
    return XGBClassifier(**{**base_params, **params})


def refit_model(
    model_name: str,
    best_params: dict[str, Any],
    x_train_dev: pd.DataFrame,
    y_train_dev: pd.Series,
) -> Any:
    """Refit the selected model on train + dev."""
    if model_name == "Logistic Regression":
        model = build_logistic_regression(best_params)
    elif model_name == "SVM":
        svm_candidates = build_svm_candidates()
        model = svm_candidates["SVM"][0](best_params)
    elif model_name == "SVM Class Balanced":
        svm_candidates = build_svm_candidates()
        model = svm_candidates["SVM Class Balanced"][0](best_params)
    elif model_name == "SVM SMOTE":
        svm_candidates = build_svm_candidates()
        model = svm_candidates["SVM SMOTE"][0](best_params)
    elif model_name == "Random Forest":
        model = build_random_forest(best_params)
    elif model_name == "XGBoost":
        model = build_xgboost(best_params, y_train_dev)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model.fit(x_train_dev, y_train_dev)
    return model


def run_permutation_importance(
    model_name: str,
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Run permutation importance and return a tidy dataframe."""
    log_progress(f"Running permutation importance for {model_name}")
    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "model": model_name,
            "feature": x_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    importance["rank"] = (
        importance["importance_mean"].rank(ascending=False, method="min").astype(int)
    )
    return importance.sort_values(["rank", "feature"]).reset_index(drop=True)


def load_mlp_summary() -> tuple[Any, float | None, dict[str, Any]]:
    """Load the saved final MLP pipeline and its recorded metadata."""
    mlp_path = MODELS_DIR / "mlp_model.joblib"
    if not mlp_path.exists():
        raise FileNotFoundError(
            "MLP model was not found. Run code/mlp_model.py before permutation importance."
        )

    model = joblib.load(mlp_path)
    dev_score = None
    best_params: dict[str, Any] = {}

    results_path = TABLES_DIR / "mlp_model_results.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
        if not results.empty:
            row = results.iloc[0]
            best_params = {
                "imbalance_method": row.get("imbalance_method", "missing"),
                "hidden_layer_sizes": row.get("best_hidden_layer_sizes", "missing"),
                "alpha": float(row["best_alpha"]) if "best_alpha" in row else "missing",
                "learning_rate_init": (
                    float(row["best_learning_rate_init"])
                    if "best_learning_rate_init" in row
                    else "missing"
                ),
            }

    tuning_path = TABLES_DIR / "mlp_tuning_results.csv"
    if tuning_path.exists():
        tuning = pd.read_csv(tuning_path)
        if "rank_dev_pr_auc" in tuning.columns and not tuning.empty:
            best_row = tuning.sort_values("rank_dev_pr_auc").iloc[0]
            dev_score = float(best_row["dev_pr_auc_average_precision"])

    return model, dev_score, best_params


def main() -> None:
    setup()
    log_progress("Loading engineered train/dev/test data")
    x_train, y_train = load_split("train_engineered.csv")
    x_dev, y_dev = load_split("dev_engineered.csv")
    x_test, y_test = load_split("test_engineered.csv")
    x_train_dev = pd.concat([x_train, x_dev], ignore_index=True)
    y_train_dev = pd.concat([y_train, y_dev], ignore_index=True)

    selected_models: dict[str, tuple[Any, dict[str, Any], float]] = {}

    log_progress("Training Logistic Regression")
    lr_model, lr_params, lr_score = tune_on_dev(
        "Logistic Regression",
        build_logistic_regression,
        [{}],
        x_train,
        y_train,
        x_dev,
        y_dev,
    )
    selected_models["Logistic Regression"] = (lr_model, lr_params, lr_score)

    log_progress("Training SVM variants and selecting the best one")
    svm_selection_rows = []
    best_svm_name = ""
    best_svm_score = -1.0
    best_svm_params: dict[str, Any] = {}
    for svm_name, (factory, grid) in build_svm_candidates().items():
        _, params, score = tune_on_dev(
            svm_name, factory, grid, x_train, y_train, x_dev, y_dev
        )
        svm_selection_rows.append(
            {"svm_variant": svm_name, "dev_average_precision": score, "params": params}
        )
        if score > best_svm_score:
            best_svm_name = svm_name
            best_svm_score = score
            best_svm_params = params

    log_progress(f"Selected SVM representative: {best_svm_name}")
    selected_models[best_svm_name] = (None, best_svm_params, best_svm_score)
    pd.DataFrame(svm_selection_rows).to_csv(
        TABLES_DIR / "Yuhan_svm_variant_selection.csv", index=False
    )

    log_progress("Training Random Forest")
    rf_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    _, rf_params, rf_score = tune_on_dev(
        "Random Forest",
        build_random_forest,
        rf_grid,
        x_train,
        y_train,
        x_dev,
        y_dev,
    )
    selected_models["Random Forest"] = (None, rf_params, rf_score)

    log_progress("Training XGBoost")
    xgb_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    _, xgb_params, xgb_score = tune_on_dev(
        "XGBoost",
        lambda params: build_xgboost(params, y_train),
        xgb_grid,
        x_train,
        y_train,
        x_dev,
        y_dev,
    )
    selected_models["XGBoost"] = (None, xgb_params, xgb_score)

    all_importance = []
    summary_rows = []
    for model_name, (_, best_params, dev_score) in selected_models.items():
        log_progress(f"Refitting final {model_name} on train + dev")
        final_model = refit_model(model_name, best_params, x_train_dev, y_train_dev)
        test_ap = average_precision_score(y_test, positive_probability(final_model, x_test))
        summary_rows.append(
            {
                "model": model_name,
                "dev_average_precision": dev_score,
                "test_average_precision": test_ap,
                "best_params": best_params,
            }
        )
        joblib.dump(
            final_model,
            MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_permutation_model.joblib",
        )
        all_importance.append(
            run_permutation_importance(model_name, final_model, x_test, y_test)
        )

    log_progress("Loading final MLP model")
    mlp_model, mlp_dev_score, mlp_params = load_mlp_summary()
    mlp_test_ap = average_precision_score(y_test, positive_probability(mlp_model, x_test))
    summary_rows.append(
        {
            "model": "MLP",
            "dev_average_precision": mlp_dev_score,
            "test_average_precision": mlp_test_ap,
            "best_params": mlp_params,
        }
    )
    all_importance.append(run_permutation_importance("MLP", mlp_model, x_test, y_test))

    importance_df = pd.concat(all_importance, ignore_index=True)
    top_features_df = (
        importance_df.sort_values(["model", "rank"])
        .groupby("model", as_index=False)
        .head(5)
        .reset_index(drop=True)
    )

    importance_df.to_csv(
        TABLES_DIR / "Yuhan_permutation_importance_all_models.csv", index=False
    )
    top_features_df.to_csv(
        TABLES_DIR / "Yuhan_permutation_importance_top_features.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        TABLES_DIR / "Yuhan_permutation_model_summary.csv", index=False
    )

    print("\nTop 5 permutation importance features per model:")
    print(top_features_df.round(5).to_string(index=False))
    log_progress("Permutation importance complete")


if __name__ == "__main__":
    main()
