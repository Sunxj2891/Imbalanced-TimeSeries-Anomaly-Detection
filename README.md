# Imbalanced Signal Anomaly Detection & Multi-Model Benchmark

Benchmark study on binary classification under severe class imbalance (1:10 positive-to-negative ratio) using the HTRU2 dataset. This repository contains code for domain-specific feature engineering, Mutual Information dependency analysis, and performance comparisons across 7 machine learning models.

[Full Report (PDF)](docs/group4_iml_project2_report.pdf)

---

## Benchmark Results

Models were evaluated on engineered train/dev/test splits (70/15/15) with primary focus on **PR-AUC (Average Precision)** and **Minority-Class Recall**:

| Model | PR-AUC (Average Precision) | Precision | Recall | F1-Score | FP | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression (Baseline) | 0.9131 | 0.9598 | 0.7764 | 0.8584 | 8 | 55 |
| Standard RBF SVM | 0.9134 | 0.9563 | 0.8008 | 0.8717 | 9 | 49 |
| Class-Balanced SVM | 0.9158 | 0.9524 | 0.8130 | 0.8772 | 10 | 46 |
| SMOTE Resampled SVM | 0.9113 | 0.7908 | **0.9065** | 0.8447 | 59 | **23** |
| Random Forest (Balanced) | 0.9154 | 0.9034 | 0.8740 | 0.8884 | 23 | 31 |
| Extreme Gradient Boosting (XGBoost) | 0.9139 | 0.7893 | 0.8984 | 0.8403 | 59 | 25 |
| **Multi-Layer Perceptron (MLP)** | **0.9240** | **0.9541** | **0.8455** | **0.8966** | 10 | 38 |

### Summary of Findings
1. **SMOTE vs. Cost-Sensitive Weighting**: SMOTE yielded the highest recall (90.65%, FN=23), but introduced synthetic noise near decision boundaries, increasing false positives by ~6x (FP=59).
2. **Optimal Trade-off**: The Multi-Layer Perceptron (MLP) achieved the highest overall PR-AUC (0.9240) and F1-score (0.8966) while maintaining low false positive rates (FP=10).
3. **Feature Contributions**: Engineered dimensionless features (`Profile_shape_score` and `Profile_cv`) achieved top Mutual Information scores (0.225 and 0.206), outperforming standard dispersion metrics.

---

## Feature Formulations

$$\text{Profile\_cv} = \frac{\text{Profile\_stdev}}{\text{Profile\_mean} + 10^{-6}}$$

$$\text{Profile\_shape\_score} = |\text{Profile\_skewness}| + |\text{Profile\_kurtosis}|$$

---

## Reproduction

```bash
pip install -r requirements.txt
python code/generate_all_model_result_figures.py
```

---

## Repository Structure

```text
Imbalanced-TimeSeries-Anomaly-Detection/
├── README.md                   # Benchmark documentation
├── requirements.txt            # Dependencies
├── docs/                       # Project report PDF
├── code/                       # Preprocessing and model training scripts
└── outputs/
    ├── figures/                # PR curves and confusion matrices
    └── tables/                 # Model performance tables (CSV)
```


---

## Artifact validation

The headline metrics are checked-in experiment outputs, not values copied from the course PDF. Run the following command to validate table schemas, confusion-matrix totals, and the reported PR-AUC/F1 values:

~~~bash
python3 scripts/verify_results.py
~~~

The verifier does not retrain models and does not modify outputs. Rebuilding the figures from the processed train/dev/test split is available with:

~~~bash
python3 code/generate_all_model_result_figures.py
~~~

## Scope

HTRU2 is a tabular pulsar-candidate benchmark with signal-summary features. This repository demonstrates imbalanced binary classification and model selection; it should not be described as a production anomaly-detection service or as a deployed time-series system.

---

## Contributors
Xingjian Sun, Yuhan Wang, Yunhong Huang (The University of Melbourne)
