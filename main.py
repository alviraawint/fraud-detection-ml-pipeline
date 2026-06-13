from pathlib import Path

import joblib

from evaluate_model import (
    build_threshold_comparison,
    cross_val_pr_curve,
    evaluate_models,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    save_metrics,
    select_threshold,
)
from train_model import (
    build_models,
    load_data,
    prepare_train_test_data,
    tune_models,
)


DATA_PATH = Path("data/creditcard.csv")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
METRICS_PATH = RESULTS_DIR / "metrics.csv"
TUNING_RESULTS_PATH = RESULTS_DIR / "tuning_results.csv"
THRESHOLD_METRICS_PATH = RESULTS_DIR / "threshold_metrics.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
PR_CURVE_PATH = RESULTS_DIR / "precision_recall_curve.png"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
RECALL_TARGET = 0.90


def main():
    """Run the complete fraud detection machine learning pipeline."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data(DATA_PATH)

    print("Preparing train and test sets...")
    X_train, X_test, y_train, y_test = prepare_train_test_data(data)

    print("Building models...")
    models = build_models(y_train)

    # Hyperparameters are tuned with RandomizedSearchCV using stratified k-fold
    # cross-validation on the TRAINING data only, scored by PR-AUC. This single
    # step both tunes each model and selects the best one; the test set is left
    # untouched until the final evaluation below so it cannot leak into the
    # selection. RandomizedSearchCV refits the best estimator on the full
    # training set, so the returned models are ready to evaluate.
    print("Tuning hyperparameters and selecting the best model...")
    trained_models, tuning_results = tune_models(models, X_train, y_train)
    save_metrics(tuning_results, TUNING_RESULTS_PATH)
    best_model_name = tuning_results.iloc[0]["model"]
    print(
        f"Best model by mean CV PR-AUC: {best_model_name} "
        f"({tuning_results.iloc[0]['pr_auc_mean']:.4f} "
        f"+/- {tuning_results.iloc[0]['pr_auc_std']:.4f})"
    )
    print(f"Best hyperparameters: {tuning_results.iloc[0]['best_params']}")

    print("Evaluating models on the held-out test set...")
    metrics_df, evaluation_results = evaluate_models(trained_models, X_test, y_test)
    save_metrics(metrics_df, METRICS_PATH)

    best_model = trained_models[best_model_name]
    best_result = evaluation_results[best_model_name]

    # Choose a decision threshold for the best model from cross-validated
    # training predictions, then report how each operating point behaves on the
    # untouched test set. The default 0.5 cutoff is rarely right for imbalanced
    # fraud detection, so we compare it against business-driven thresholds.
    print("Tuning decision threshold on the training set...")
    precisions, recalls, thresholds = cross_val_pr_curve(best_model, X_train, y_train)
    f1_choice = select_threshold(precisions, recalls, thresholds, policy="f1")
    recall_choice = select_threshold(
        precisions,
        recalls,
        thresholds,
        policy="recall_target",
        recall_target=RECALL_TARGET,
    )

    threshold_strategies = [
        ("Default (0.5)", 0.5),
        ("Max F1", f1_choice["threshold"]),
        (f"Recall >= {RECALL_TARGET:.2f}", recall_choice["threshold"]),
    ]
    threshold_df = build_threshold_comparison(
        best_model, X_test, y_test, threshold_strategies
    )
    save_metrics(threshold_df, THRESHOLD_METRICS_PATH)
    print("Threshold comparison on the test set:")
    print(threshold_df.to_string(index=False))

    print(f"Saving precision-recall curve to {PR_CURVE_PATH}")
    plot_precision_recall_curve(
        precisions,
        recalls,
        model_name=best_model_name,
        output_path=PR_CURVE_PATH,
        marked_points=[
            ("Max F1", f1_choice["recall"], f1_choice["precision"]),
            (
                f"Recall >= {RECALL_TARGET:.2f}",
                recall_choice["recall"],
                recall_choice["precision"],
            ),
        ],
    )

    print(f"Saving confusion matrix for best model: {best_model_name}")
    plot_confusion_matrix(
        best_result["confusion_matrix"],
        model_name=best_model_name,
        output_path=CONFUSION_MATRIX_PATH,
    )

    print(f"Saving best model to {BEST_MODEL_PATH}")
    joblib.dump(best_model, BEST_MODEL_PATH)

    print("Pipeline complete.")
    print(f"Metrics saved to {METRICS_PATH}")
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
