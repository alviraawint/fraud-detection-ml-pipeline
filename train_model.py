import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGET_COLUMN = "Class"
CV_FOLDS = 5
TUNING_ITERATIONS = 6


def load_data(file_path):
    """Load the credit card fraud dataset from a CSV file."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. Place creditcard.csv inside the data folder."
        )

    data = pd.read_csv(file_path)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must include a '{TARGET_COLUMN}' target column.")

    return data


def prepare_train_test_data(data, test_size=0.2):
    """Split features and target, then create stratified train and test sets."""
    data = data.copy()

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_models(y_train):
    """Create baseline fraud detection models.

    Class imbalance is handled with class weights for scikit-learn models.
    XGBoost is added only when the package is installed in the environment.
    """
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    xgboost_model = build_xgboost_model(y_train)
    if xgboost_model is not None:
        models["XGBoost"] = xgboost_model
    else:
        print("XGBoost is not installed. Skipping XGBoost model.")

    return models


def build_xgboost_model(y_train):
    """Return an XGBoost pipeline if xgboost is installed, otherwise return None."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    class_counts = y_train.value_counts()
    normal_count = class_counts.get(0, 0)
    fraud_count = class_counts.get(1, 0)
    scale_pos_weight = normal_count / fraud_count if fraud_count > 0 else 1

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=0,
                ),
            ),
        ]
    )


def fit_models(models, X_train, y_train):
    """Fit a dictionary of already-built models on the training data."""
    trained_models = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        trained_models[model_name] = model

    return trained_models


def train_models(X_train, y_train):
    """Build and fit all configured models and return them in a dictionary."""
    models = build_models(y_train)
    return fit_models(models, X_train, y_train)


def build_param_distributions(models):
    """Hyperparameter search spaces for each model, keyed by pipeline step.

    Keys are prefixed with ``model__`` because every estimator lives in the
    ``"model"`` step of a scikit-learn Pipeline. Only distributions for models
    that are actually present are returned, so XGBoost is skipped when it is not
    installed.
    """
    known_distributions = {
        "Logistic Regression": {
            "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        },
        "Random Forest": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__max_features": ["sqrt", 0.5],
        },
        "XGBoost": {
            "model__n_estimators": [100, 200, 300, 400],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        },
    }

    return {name: known_distributions[name] for name in models if name in known_distributions}


def tune_models(
    models,
    X_train,
    y_train,
    n_iter=TUNING_ITERATIONS,
    n_splits=CV_FOLDS,
    random_state=RANDOM_STATE,
):
    """Tune each model with RandomizedSearchCV and select on cross-validated PR-AUC.

    The search runs stratified k-fold cross-validation on the training data only
    and scores candidates by average precision (PR-AUC). With ``refit=True`` the
    returned best estimator is already refit on the full training set, so this
    single step both tunes hyperparameters and selects the best model without
    ever touching the test set.

    The search itself runs candidates sequentially (``n_jobs=1``) while each
    estimator keeps its own internal parallelism. This avoids nested parallelism
    (many parallel workers each training a parallel forest), which can exhaust
    memory on large datasets.
    """
    param_distributions = build_param_distributions(models)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    tuned_models = {}
    rows = []
    for model_name, model in models.items():
        if model_name not in param_distributions:
            raise ValueError(f"No hyperparameter grid defined for '{model_name}'.")

        print(
            f"Tuning {model_name} with RandomizedSearchCV "
            f"({n_iter} candidates, {n_splits}-fold)..."
        )
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[model_name],
            n_iter=n_iter,
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
            random_state=random_state,
            refit=True,
        )
        search.fit(X_train, y_train)

        tuned_models[model_name] = search.best_estimator_
        best_index = search.best_index_
        rows.append(
            {
                "model": model_name,
                "pr_auc_mean": search.cv_results_["mean_test_score"][best_index],
                "pr_auc_std": search.cv_results_["std_test_score"][best_index],
                "best_params": json.dumps(search.best_params_),
            }
        )

    tuning_results = pd.DataFrame(rows).sort_values(
        "pr_auc_mean", ascending=False, ignore_index=True
    )
    return tuned_models, tuning_results
