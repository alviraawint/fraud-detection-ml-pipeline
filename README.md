# Fraud Detection ML Pipeline

This project is a beginner-friendly machine learning pipeline for credit card fraud detection. It trains multiple classification models, handles class imbalance, evaluates fraud-focused metrics, and saves portfolio-ready results.

## Project Structure

```text
fraud-detection-ml-pipeline/
├── data/
│   ├── creditcard.csv
│   └── README.md
├── models/
│   └── README.md
├── results/
│   ├── metrics.csv
│   └── confusion_matrix.png
├── main.py
├── train_model.py
├── evaluate_model.py
├── requirements.txt
└── README.md
```

## Dataset

The pipeline expects the dataset at:

```text
data/creditcard.csv
```

The target column must be named `Class`:

- `0` = normal transaction
- `1` = fraudulent transaction

This project is designed for the popular credit card fraud dataset from Kaggle, but it will work with any CSV that follows the same target-column format.

## Models

The pipeline trains:

- Logistic Regression
- Random Forest
- XGBoost, only if `xgboost` is already installed

Class imbalance is handled with:

- `class_weight="balanced"` for Logistic Regression and Random Forest
- `scale_pos_weight` for XGBoost

## Evaluation Metrics

The pipeline reports:

- Confusion matrix
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

For fraud detection, recall and PR-AUC are especially useful because fraud cases are rare.

## How to Run

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python main.py
```

## Outputs

After running the pipeline, these files will be created:

```text
results/metrics.csv
results/confusion_matrix.png
models/best_model.pkl
```

The best model is selected using PR-AUC because it is a strong metric for imbalanced fraud detection problems.

## Portfolio Notes

This project demonstrates:

- Clean Python project structure
- Data loading and validation
- Train-test splitting with stratification
- Preprocessing with scikit-learn pipelines
- Class imbalance handling
- Multiple model training
- Model evaluation using fraud-relevant metrics
- Saving reproducible results and trained models
