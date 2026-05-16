# Fraud Detection ML Pipeline

## Project Overview

This project is a modular machine learning pipeline for detecting potentially fraudulent credit card transactions. It demonstrates an end-to-end classification workflow, including data loading, validation, preprocessing, stratified train-test splitting, model training, evaluation, and saving results for reproducibility.

The project is designed as a GitHub portfolio piece for Data Science, Machine Learning, and AI internship applications.

## Business Problem

Credit card fraud detection is a risk analytics problem where fraudulent transactions are rare compared with normal transactions. Because the positive class is small, accuracy alone can be misleading: a model could appear accurate while still missing many fraud cases.

The goal is to identify suspicious transactions while balancing two business risks:

- **False negatives:** fraudulent transactions that the model misses.
- **False positives:** legitimate transactions that are incorrectly flagged as fraud.

In practice, fraud teams often care about recall, precision, F1-score, and PR-AUC because these metrics describe how well the model detects the rare fraud class and how many false alerts it creates.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- joblib
- XGBoost, optional if installed

## Dataset

The pipeline expects the dataset at:

```text
data/creditcard.csv
```

The target column must be named `Class`:

- `0` = normal transaction
- `1` = fraudulent transaction

The dataset file is not committed to GitHub because it may be large and may have licensing restrictions. It is intentionally excluded from version control, so users should download the dataset separately and place it in the `data/` folder before running the pipeline.

Recommended dataset details to update after downloading the data:

| Item | Value |
|---|---|
| Dataset source | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| Number of rows | 284807 |
| Number of features | 30 |
| Target column | `Class` |
| Fraud class | `1` |
| Fraud rate | 0.1727% |

## Pipeline Workflow

```text
Raw CSV Data
     ↓
Validate Target Column
     ↓
Split Features and Target
     ↓
Stratified Train/Test Split
     ↓
Preprocessing with scikit-learn Pipelines
     ↓
Train Logistic Regression, Random Forest, and optional XGBoost
     ↓
Evaluate with Fraud-Relevant Metrics
     ↓
Save Metrics, Confusion Matrix, and Best Model
```

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

Because fraud detection is highly imbalanced, accuracy alone is not a reliable metric. This project evaluates models using metrics that better reflect rare-class detection:

| Metric | Why It Matters |
|---|---|
| Confusion matrix | Shows true negatives, false positives, false negatives, and true positives |
| Precision | Of the transactions flagged as fraud, how many were actually fraud |
| Recall | Of all real fraud cases, how many the model detected |
| F1-score | Balances precision and recall |
| ROC-AUC | Measures ranking quality across classification thresholds |
| PR-AUC | Especially useful when the positive class is rare |

The best model is selected using PR-AUC because it is better suited for imbalanced classification than accuracy.

## Results

Current results from the included pipeline run:

| Model | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0610 | 0.9184 | 0.1144 | 0.9721 | 0.7189 |
| Random Forest | 0.9605 | 0.7449 | 0.8391 | 0.9529 | 0.8542 |

### Key Takeaways

- Logistic Regression achieved higher recall in this run, meaning it identified more fraud cases, but its low precision indicates many false positives.
- Random Forest achieved stronger precision, F1-score, and PR-AUC in this run, making it the best model under the current PR-AUC selection rule.
- These results should be interpreted as baseline model results. Threshold tuning, cross-validation, and hyperparameter tuning could change the precision-recall tradeoff.

### Confusion Matrix

The confusion matrix below is generated for the best model selected by PR-AUC:

![Confusion Matrix](results/confusion_matrix.png)

## Generated Outputs

After running the pipeline, the following files are generated or updated:

```text
results/metrics.csv              # model comparison metrics
results/confusion_matrix.png      # confusion matrix for the best model
models/best_model.pkl             # saved best model artifact
```

The trained model artifact is not committed to GitHub because model files are generated outputs and can become large.

## How to Run

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional: install XGBoost if you want to train the XGBoost model too:

```bash
pip install xgboost
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
|   `-- creditcard.csv          # not committed; user-provided dataset
|-- models/
|   |-- README.md
|   `-- best_model.pkl          # generated after running the pipeline
|-- results/
|   |-- metrics.csv             # saved model metrics
|   `-- confusion_matrix.png    # saved confusion matrix image
|-- evaluate_model.py           # evaluation metrics and plotting
|-- main.py                     # pipeline entry point
|-- train_model.py              # data loading, splitting, and model training
|-- requirements.txt
`-- README.md
```

## Skills Demonstrated

- Python scripting for end-to-end ML workflows
- Data loading and validation with pandas
- Train-test splitting with stratification
- Preprocessing with scikit-learn pipelines
- Handling class imbalance with class weights
- Training Logistic Regression and Random Forest models
- Optional XGBoost model training
- Fraud-focused model evaluation with precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrix
- Saving model artifacts with joblib
- Organizing a machine learning repository for GitHub portfolio presentation

## Future Improvements

- Add cross-validation and hyperparameter tuning
- Add threshold tuning to improve the precision-recall tradeoff
- Compare additional imbalance techniques such as SMOTE or undersampling
- Track experiments with MLflow or Weights & Biases
- Add unit tests for data loading, model training, and evaluation
- Package the trained model behind a simple API or dashboard
