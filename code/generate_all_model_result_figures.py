"""Generate report-ready result figures for all final models.

Included models:
- Logistic Regression
- SVM
- Class Balanced SVM
- SMOTE SVM
- Random Forest
- XGBoost
- MLP
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "outputs" / "processed_data"
MODELS_DIR = PROJECT_DIR / "outputs" / "models"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = pd.read_csv(DATA_DIR / "train_engineered.csv")
    dev = pd.read_csv(DATA_DIR / "dev_engineered.csv")
    test = pd.read_csv(DATA_DIR / "test_engineered.csv")

    x_train = train.drop(columns=["class"])
    y_train = train["class"].astype(int)
    x_dev = dev.drop(columns=["class"])
    y_dev = dev["class"].astype(int)
    x_test = test.drop(columns=["class"])
    y_test = test["class"].astype(int)
    return x_train, y_train, x_dev, y_dev, x_test, y_test


def fit_logistic_regression(
    x_train_dev: pd.DataFrame,
    y_train_dev: pd.Series,
    x_test: pd.DataFrame,
) -> dict[str, object]:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(random_state=42, max_iter=2000, C=0.1)),
        ]
    )
    pipeline.fit(x_train_dev, y_train_dev)
    return {
        "pred": pipeline.predict(x_test),
        "proba": pipeline.predict_proba(x_test)[:, 1],
    }


def fit_svm_models(
    x_train_dev: pd.DataFrame,
    y_train_dev: pd.Series,
    x_test: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    """Recreate the three final SVM variants with their best parameters."""
    svm_params = {"C": 10, "gamma": 0.001}
    models = {
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=7,
                        **svm_params,
                    ),
                ),
            ]
        ),
        "Class Balanced SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=7,
                        **svm_params,
                    ),
                ),
            ]
        ),
        "SMOTE SVM": ImbPipeline(
            [
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=7)),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        random_state=7,
                        **svm_params,
                    ),
                ),
            ]
        ),
    }

    outputs = {}
    for name, model in models.items():
        model.fit(x_train_dev, y_train_dev)
        proba = model.predict_proba(x_test)[:, 1]
        outputs[name] = {
            "pred": (proba >= 0.5).astype(int),
            "proba": proba,
        }
    return outputs


def load_saved_model_predictions(x_test: pd.DataFrame) -> dict[str, dict[str, object]]:
    models = {
        "Random Forest": joblib.load(MODELS_DIR / "random_forest_model.joblib"),
        "XGBoost": joblib.load(MODELS_DIR / "boosting_model.joblib"),
        "MLP": joblib.load(MODELS_DIR / "mlp_model.joblib"),
    }

    outputs = {}
    for name, model in models.items():
        outputs[name] = {
            "pred": model.predict(x_test),
            "proba": model.predict_proba(x_test)[:, 1],
        }
    return outputs


def plot_confusion_matrices(y_test: pd.Series, outputs: dict[str, dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes_flat = axes.ravel()

    for ax, (name, output) in zip(axes_flat, outputs.items()):
        cm = confusion_matrix(y_test, output["pred"], labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

    for ax in axes_flat[len(outputs) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "all_models_confusion_matrices.png", dpi=300)
    plt.close()


def plot_pr_curves(y_test: pd.Series, outputs: dict[str, dict[str, object]]) -> None:
    plt.figure(figsize=(9, 7))
    for name, output in outputs.items():
        proba = output["proba"]
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.5f})", linewidth=2)

    baseline = y_test.mean()
    plt.axhline(
        y=baseline,
        linestyle="--",
        label=f"Baseline positive rate = {baseline:.5f}",
    )
    plt.title("PR Curve on Test set")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "all_models_pr_curves.png", dpi=300)
    plt.close()


def plot_roc_curves(y_test: pd.Series, outputs: dict[str, dict[str, object]]) -> None:
    plt.figure(figsize=(9, 7))
    for name, output in outputs.items():
        proba = output["proba"]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (ROC-AUC={auc:.5f})", linewidth=2)

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.title("ROC Curve on Test set")
    plt.xlabel("FP(False Positive) Rate")
    plt.ylabel("TP(True Positive) Rate")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "all_models_roc_curves.png", dpi=300)
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_dev, y_dev, x_test, y_test = load_data()
    x_train_dev = pd.concat([x_train, x_dev], ignore_index=True)
    y_train_dev = pd.concat([y_train, y_dev], ignore_index=True)

    outputs = {
        "Logistic Regression": fit_logistic_regression(
            x_train_dev, y_train_dev, x_test
        ),
        **fit_svm_models(x_train_dev, y_train_dev, x_test),
        **load_saved_model_predictions(x_test),
    }

    plot_confusion_matrices(y_test, outputs)
    plot_pr_curves(y_test, outputs)
    plot_roc_curves(y_test, outputs)

    print("Saved report-ready figures:")
    print(FIGURES_DIR / "all_models_confusion_matrices.png")
    print(FIGURES_DIR / "all_models_pr_curves.png")
    print(FIGURES_DIR / "all_models_roc_curves.png")


if __name__ == "__main__":
    main()
