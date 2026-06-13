# Fraud Detection ML Pipeline

## Project Overview

This project is a modular machine learning pipeline for detecting fraudulent credit card transactions. It demonstrates an end-to-end classification workflow, including data loading, preprocessing, model training, evaluation, and saving results for reproducibility.

The project is designed as a GitHub portfolio piece for Data Science, Machine Learning, and AI internship applications.

## Problem Statement

Credit card fraud detection is a highly imbalanced binary classification problem. Fraudulent transactions are rare compared with normal transactions, so model performance must be evaluated with metrics that reflect minority-class detection, not accuracy alone.

The goal is to identify fraudulent transactions while keeping the pipeline simple, readable, and easy to extend.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- joblib
- XGBoost, optional if installed

## Dataset Note

The pipeline expects the dataset at:

```text
data/creditcard.csv
```

The target column must be named `Class`:

- `0` = normal transaction
- `1` = fraudulent transaction

The dataset file is not committed to GitHub because it may be large and may have licensing restrictions.

## Workflow

1. Load transaction data from `data/creditcard.csv`
2. Validate that the target column exists
3. Split the dataset into features and target
4. Create a stratified train-test split
5. Apply preprocessing inside scikit-learn pipelines
6. Tune hyperparameters and select the best model with stratified k-fold cross-validation on the training set
7. Train all models on the full training set
8. Evaluate each model once on the held-out test set using fraud-relevant metrics
9. Save cross-validation results, test metrics, confusion matrix, and the best model artifact

## Model Training Approach

The pipeline trains at least two models:

- Logistic Regression
- Random Forest

If `xgboost` is installed, the pipeline also trains:

- XGBoost Classifier

Class imbalance is handled using:

- `class_weight="balanced"` for Logistic Regression and Random Forest
- `scale_pos_weight` for XGBoost

Preprocessing is included inside model pipelines to keep training and evaluation consistent.

## Evaluation Metrics

The project evaluates models using:

- Confusion matrix
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

For fraud detection, recall and PR-AUC are especially important because the positive class is rare.

## Hyperparameter Tuning and Model Selection

Each model's hyperparameters are tuned with **RandomizedSearchCV** using **5-fold
stratified cross-validation on the training set only**, scored by PR-AUC. The best
configuration per model is compared, and the overall best model is selected by mean
cross-validated PR-AUC. The test set is never used for tuning or selection; it is
held out and scored a single time for the final results below. This avoids leaking
the test set into the selection decision and gives a variance estimate (standard
deviation across folds) for each model.

`RandomizedSearchCV` is preferred over an exhaustive grid search because it samples
the parameter space, covering more ground for a fixed compute budget.

Best cross-validated results per model (training set):

| Model | Mean PR-AUC | Std PR-AUC | Best Hyperparameters |
|---|---:|---:|---|
| Random Forest | 0.8458 | 0.0184 | n_estimators=200, max_depth=None, min_samples_split=5, min_samples_leaf=1, max_features=sqrt |
| Logistic Regression | 0.7568 | 0.0518 | C=0.1 |

## Results

Final results on the held-out test set (default 0.5 threshold):

| Model | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0611 | 0.9184 | 0.1145 | 0.9720 | 0.7107 |
| Random Forest | 0.9610 | 0.7551 | 0.8457 | 0.9566 | 0.8633 |

PR-AUC is used as the selection metric because it is better suited for imbalanced
classification than accuracy. Random Forest wins on both cross-validation and the
held-out test set.

## Threshold Tuning

The metrics above use the default 0.5 probability cutoff, which is rarely the right
operating point for imbalanced fraud detection. For the best model, a decision
threshold is selected from **cross-validated predictions on the training set** (the
test set is not used to pick the threshold) and then applied once to the test set.

Test-set metrics for the best model (Random Forest) at different thresholds:

| Strategy | Threshold | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Default (0.5) | 0.50 | 0.9610 | 0.7551 | 0.8457 |
| Max F1 | 0.20 | 0.8300 | 0.8469 | 0.8384 |
| Recall >= 0.90 | 0.01 | 0.1080 | 0.8980 | 0.1928 |

Key takeaways:

- The **max-F1 threshold (0.20)** trades precision for recall: it lifts recall from
  0.76 to 0.85, but on this test set its F1 (0.838) is essentially tied with the
  default (0.846). The right choice depends on whether catching more fraud is worth
  the extra false positives.
- The max-F1 threshold is chosen on training cross-validation, so it does not always
  beat the default on the test set. This small gap is itself a useful illustration
  of how an operating point selected on one split generalizes to unseen data.
- Forcing **90% recall** requires such a low threshold (0.01) that precision
  collapses, which illustrates the precision-recall tradeoff and why the threshold
  should be chosen from the business cost of false negatives vs false positives.

Generated outputs:

```text
results/tuning_results.csv
results/metrics.csv
results/threshold_metrics.csv
results/confusion_matrix.png
results/precision_recall_curve.png
models/best_model.pkl
```

## How to Run

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python main.py
```

## Folder Structure

```text
fraud-detection-ml-pipeline/
|-- data/
|   |-- README.md
|   `-- creditcard.csv
|-- models/
|   |-- README.md
|   `-- best_model.pkl
|-- results/
|   |-- tuning_results.csv
|   |-- metrics.csv
|   |-- threshold_metrics.csv
|   |-- confusion_matrix.png
|   `-- precision_recall_curve.png
|-- evaluate_model.py
|-- main.py
|-- train_model.py
|-- requirements.txt
`-- README.md
```

## Future Improvements

- Compare additional imbalance techniques such as SMOTE or undersampling
- Track experiments with MLflow or Weights & Biases
- Add unit tests for data loading, model training, and evaluation
- Package the trained model behind a simple API or dashboard
