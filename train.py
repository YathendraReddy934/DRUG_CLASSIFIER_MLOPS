
import os
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


DATA_PATH = "Data/drug.csv"
MODEL_DIR = "Model"
RESULTS_DIR = "Results"

TARGET = "Drug"

NUMERIC_FEATURES = ["Age", "Na_to_K"]
CATEGORICAL_FEATURES = ["Sex", "BP", "Cholesterol"]

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


def create_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    data = pd.read_csv(DATA_PATH)

    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = data[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Save the same test data for evaluating all three models.
    joblib.dump(
        (X_test, y_test),
        os.path.join(RESULTS_DIR, "test_data.joblib")
    )

    for model_name, config in MODELS.items():
        pipeline = Pipeline([
            ("preprocessor", create_preprocessor()),
            ("classifier", config["estimator"])
        ])

        pipeline.fit(X_train, y_train)

        model_path = os.path.join(MODEL_DIR, config["filename"])
        joblib.dump(pipeline, model_path)

        print(
            f"Trained {config['algorithm']} "
            f"and saved to {model_path}"
        )

    # Save sample input for Snowflake Registry signatures.
    X_train.head(5).to_csv(
        os.path.join(RESULTS_DIR, "sample_input.csv"),
        index=False
    )

    print("\nAll three models trained successfully.")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")


if __name__ == "__main__":
    main()
