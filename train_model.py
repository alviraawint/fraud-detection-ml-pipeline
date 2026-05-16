from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGET_COLUMN = "Class"


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


def train_models(X_train, y_train):
    """Fit all configured models and return them in a dictionary."""
    models = build_models(y_train)
    trained_models = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        trained_models[model_name] = model

    return trained_models
