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
Tune Hyperparameters + Select Best Model (Stratified K-Fold CV, PR-AUC)
     ↓
Train Best Models on Full Training Set
     ↓
Evaluate Once on Held-Out Test Set (Fraud-Relevant Metrics)
     ↓
Tune Decision Threshold (from Training CV)
     ↓
Save CV Results, Metrics, Threshold Comparison, Plots, and Best Model
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

### Key Takeaways

- Logistic Regression achieved higher recall at the default threshold, meaning it identified more fraud cases, but its low precision indicates many false positives.
- Random Forest achieved stronger precision, F1-score, and PR-AUC, making it the best model under the PR-AUC selection rule.
- Cross-validation and hyperparameter tuning are already applied (see the section above); the decision threshold is tuned below to adjust the precision-recall tradeoff.

### Confusion Matrix

The confusion matrix below is generated for the best model selected by PR-AUC:

![Confusion Matrix](results/confusion_matrix.png)

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

## Generated Outputs

After running the pipeline, the following files are generated or updated:

```text
results/tuning_results.csv         # cross-validated tuning results per model
results/metrics.csv                # test-set model comparison metrics
results/threshold_metrics.csv      # test-set metrics at different thresholds
results/confusion_matrix.png       # confusion matrix for the best model
results/precision_recall_curve.png # precision-recall curve for the best model
models/best_model.pkl              # saved best model artifact
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
|   |-- tuning_results.csv      # cross-validated tuning results
|   |-- metrics.csv             # saved model metrics
|   |-- threshold_metrics.csv   # metrics at different thresholds
|   |-- confusion_matrix.png    # saved confusion matrix image
|   `-- precision_recall_curve.png # saved precision-recall curve
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
- Leak-free model selection with stratified k-fold cross-validation
- Hyperparameter tuning with RandomizedSearchCV
- Decision-threshold tuning from the precision-recall curve
- Saving model artifacts with joblib
- Organizing a machine learning repository for GitHub portfolio presentation

## Future Improvements

- Compare additional imbalance techniques such as SMOTE or undersampling
- Track experiments with MLflow or Weights & Biases
- Add unit tests for data loading, model training, and evaluation
- Package the trained model behind a simple API or dashboard
