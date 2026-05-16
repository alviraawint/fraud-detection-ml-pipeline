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
6. Train multiple classification models
7. Evaluate each model using fraud-relevant metrics
8. Save metrics, confusion matrix, and the best model artifact

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

## Results

Current results from the included pipeline run:

| Model | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0610 | 0.9184 | 0.1144 | 0.9721 | 0.7189 |
| Random Forest | 0.9605 | 0.7449 | 0.8391 | 0.9529 | 0.8542 |

The best model is selected using PR-AUC because it is better suited for imbalanced classification than accuracy.

Generated outputs:

```text
results/metrics.csv
results/confusion_matrix.png
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
|   |-- metrics.csv
|   `-- confusion_matrix.png
|-- evaluate_model.py
|-- main.py
|-- train_model.py
|-- requirements.txt
`-- README.md
```

## Future Improvements

- Add cross-validation and hyperparameter tuning
- Add threshold tuning to improve the precision-recall tradeoff
- Compare additional imbalance techniques such as SMOTE or undersampling
- Track experiments with MLflow or Weights & Biases
- Add unit tests for data loading, model training, and evaluation
- Package the trained model behind a simple API or dashboard
