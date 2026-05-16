from pathlib import Path

import joblib

from evaluate_model import evaluate_models, plot_confusion_matrix, save_metrics
from train_model import load_data, prepare_train_test_data, train_models


DATA_PATH = Path("data/creditcard.csv")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
METRICS_PATH = RESULTS_DIR / "metrics.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"


def main():
    """Run the complete fraud detection machine learning pipeline."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_data(DATA_PATH)

    print("Preparing train and test sets...")
    X_train, X_test, y_train, y_test = prepare_train_test_data(data)

    print("Training models...")
    trained_models = train_models(X_train, y_train)

    print("Evaluating models...")
    metrics_df, evaluation_results = evaluate_models(trained_models, X_test, y_test)
    save_metrics(metrics_df, METRICS_PATH)

    best_model_name = metrics_df.sort_values("pr_auc", ascending=False).iloc[0]["model"]
    best_model = trained_models[best_model_name]
    best_result = evaluation_results[best_model_name]

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
