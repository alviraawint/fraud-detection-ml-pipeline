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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


CV_FOLDS = 5


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


def cross_val_pr_curve(model, X_train, y_train, n_splits=CV_FOLDS, random_state=42):
    """Build a precision-recall curve from cross-validated training predictions.

    The decision threshold must be chosen without looking at the test set, just
    like model selection. We generate out-of-fold fraud probabilities on the
    training data and return the precision-recall curve so an operating
    threshold can be selected from training data alone.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_train, oof_probabilities)
    return precisions, recalls, thresholds


def select_threshold(precisions, recalls, thresholds, policy="f1", recall_target=0.90):
    """Pick a decision threshold from a precision-recall curve.

    precision_recall_curve returns one more precision/recall value than
    thresholds (the final point has no threshold), so we align by dropping it.

    Policies:
      - "f1": the threshold that maximizes F1 on the curve.
      - "recall_target": among thresholds reaching at least recall_target,
        the one with the highest precision; falls back to best recall if the
        target is unreachable.
    """
    aligned_precisions = precisions[:-1]
    aligned_recalls = recalls[:-1]

    if policy == "f1":
        denominator = aligned_precisions + aligned_recalls
        f1_scores = np.where(
            denominator > 0,
            2 * aligned_precisions * aligned_recalls / denominator,
            0.0,
        )
        best_index = int(np.argmax(f1_scores))
    elif policy == "recall_target":
        eligible = np.where(aligned_recalls >= recall_target)[0]
        if eligible.size == 0:
            best_index = int(np.argmax(aligned_recalls))
        else:
            best_index = int(eligible[np.argmax(aligned_precisions[eligible])])
    else:
        raise ValueError("policy must be either 'f1' or 'recall_target'.")

    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(aligned_precisions[best_index]),
        "recall": float(aligned_recalls[best_index]),
    }


def evaluate_at_threshold(model, X_test, y_test, threshold):
    """Score a model on the test set using a custom probability threshold."""
    y_probability = get_positive_class_probabilities(model, X_test)
    y_pred = (y_probability >= threshold).astype(int)

    return {
        "threshold": threshold,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }


def build_threshold_comparison(model, X_test, y_test, strategies):
    """Compare test-set metrics across named (strategy, threshold) operating points."""
    rows = []
    for strategy_name, threshold in strategies:
        result = evaluate_at_threshold(model, X_test, y_test, threshold)
        rows.append(
            {
                "strategy": strategy_name,
                "threshold": result["threshold"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1_score"],
            }
        )

    return pd.DataFrame(rows)


def plot_precision_recall_curve(
    precisions, recalls, model_name, output_path, marked_points=None
):
    """Save the precision-recall curve with optional highlighted operating points."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(recalls, precisions, label=f"{model_name} (training CV)")

    if marked_points:
        for label, recall_value, precision_value in marked_points:
            plt.scatter([recall_value], [precision_value], zorder=5, label=label)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


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
