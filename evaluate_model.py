from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_single_model(model, X_test, y_test):
    """Calculate fraud detection metrics for one trained model."""
    y_pred = model.predict(X_test)
    y_probability = get_positive_class_probabilities(model, X_test)

    metrics = {
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": calculate_auc(y_test, y_probability, metric="roc"),
        "pr_auc": calculate_auc(y_test, y_probability, metric="pr"),
    }

    return metrics


def evaluate_models(models, X_test, y_test):
    """Evaluate all trained models and return a metrics table plus raw results."""
    rows = []
    evaluation_results = {}

    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        result = evaluate_single_model(model, X_test, y_test)
        evaluation_results[model_name] = result

        rows.append(
            {
                "model": model_name,
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
                "roc_auc": result["roc_auc"],
                "pr_auc": result["pr_auc"],
            }
        )

    metrics_df = pd.DataFrame(rows)
    return metrics_df, evaluation_results


def get_positive_class_probabilities(model, X_test):
    """Return fraud probabilities when available.

    Most classifiers expose predict_proba. This helper keeps evaluation readable
    and gives a clear error if a future model does not support probabilities.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            f"{model.__class__.__name__} must support predict_proba for AUC metrics."
        )

    return model.predict_proba(X_test)[:, 1]


def calculate_auc(y_true, y_probability, metric):
    """Calculate ROC-AUC or PR-AUC, returning NaN if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return np.nan

    if metric == "roc":
        return roc_auc_score(y_true, y_probability)

    if metric == "pr":
        return average_precision_score(y_true, y_probability)

    raise ValueError("metric must be either 'roc' or 'pr'.")


def save_metrics(metrics_df, output_path):
    """Save model metrics to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)


def plot_confusion_matrix(confusion_matrix_values, model_name, output_path):
    """Save a confusion matrix plot for the selected model."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix_values,
        display_labels=["Normal", "Fraud"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
