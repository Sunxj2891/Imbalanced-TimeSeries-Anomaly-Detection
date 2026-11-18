import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"


def read_rows(name: str):
    with (TABLES / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_close(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > 1e-9:
        raise RuntimeError(f"{label}: expected {expected}, got {actual}")


def validate_confusion_counts(row, label: str) -> None:
    counts = [int(row[key]) for key in ("TN", "FP", "FN", "TP")]
    if min(counts) < 0 or sum(counts) != 2685:
        raise RuntimeError(f"{label}: invalid confusion-matrix counts {counts}")


def main() -> None:
    model_rows = {row["model"]: row for row in read_rows("Yuhan_model_results.csv")}
    mlp = read_rows("mlp_model_results.csv")[0]
    svm_rows = {row["model_name"]: row for row in read_rows("svm_final_results.csv")}

    for name in ("Random Forest", "XGBoost"):
        validate_confusion_counts(model_rows[name], name)
    validate_confusion_counts(mlp, "MLP")
    for name in ("SVM", "SVM_Class_balanced", "SVM_SMOTE"):
        validate_confusion_counts(svm_rows[name], name)

    assert_close(float(mlp["pr_auc_average_precision"]), 0.9240319607266901, "MLP PR-AUC")
    assert_close(float(mlp["f1_score"]), 0.896551724137931, "MLP F1")
    assert_close(float(svm_rows["SVM_SMOTE"]["test_recall_class_1"]), 0.9065, "SMOTE SVM recall")

    print("Validated HTRU2 result artifacts:")
    print(f"MLP PR-AUC={float(mlp['pr_auc_average_precision']):.4f}, F1={float(mlp['f1_score']):.4f}")
    print(f"SMOTE SVM recall={float(svm_rows['SVM_SMOTE']['test_recall_class_1']):.4f}")


if __name__ == "__main__":
    main()
