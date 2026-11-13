
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "raw" / "drug.csv"
MODEL_DIR = ROOT_DIR / "artifacts" / "models"
REPORT_DIR = ROOT_DIR / "artifacts" / "reports"


# --------------------------------------------------
# DATASET CONFIGURATION
# --------------------------------------------------

TARGET = "Drug"

NUMERIC_FEATURES = ["Age", "Na_to_K"]
CATEGORICAL_FEATURES = ["Sex", "BP", "Cholesterol"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

MODELS = {
    "DRUG_CLASSIFIER_RF": {
        "algorithm": "Random Forest",
        "filename": "drug_rf.joblib",
        "estimator": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    },
    "DRUG_CLASSIFIER_DT": {
        "algorithm": "Decision Tree",
        "filename": "drug_dt.joblib",
        "estimator": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        )
    },
    "DRUG_CLASSIFIER_LR": {
        "algorithm": "Logistic Regression",
        "filename": "drug_lr.joblib",
        "estimator": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    }
}


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

def create_preprocessor():

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = ALL_FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    X = data[ALL_FEATURES].copy()
    y = data[TARGET].copy()

    if y.isnull().any():
        raise ValueError(
            "Target column contains missing values. "
            "Please clean the Drug column before training."
        )

    # Use one common split for all three algorithms.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Save common test data for model evaluation.
    test_data_path = REPORT_DIR / "test_data.joblib"

    joblib.dump(
        (X_test, y_test),
        test_data_path
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # Train each algorithm using the same training data.
    for model_name, config in MODELS.items():

        pipeline = Pipeline([
            ("preprocessor", create_preprocessor()),
            ("classifier", config["estimator"])
        ])

        pipeline.fit(X_train, y_train)

        model_path = MODEL_DIR / config["filename"]

        joblib.dump(
            pipeline,
            model_path
        )

        print(
            f"Trained {config['algorithm']} "
            f"and saved to {model_path}"
        )

    # Save sample input for Snowflake Model Registry.
    sample_input_path = REPORT_DIR / "sample_input.csv"

    X_train.head(5).to_csv(
        sample_input_path,
        index=False
    )

    print("\nAll three models trained successfully.")
    print(f"Models directory: {MODEL_DIR}")
    print(f"Reports directory: {REPORT_DIR}")


if __name__ == "__main__":
    main()
