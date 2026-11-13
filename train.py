
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Data" / "drug.csv"
MODEL_DIR = ROOT / "Model"
RESULTS_DIR = ROOT / "Results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "Age", "Sex", "BP",
        "Cholesterol", "Na_to_K", "Drug"
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing dataset columns: {sorted(missing)}"
        )

    df = df[required_columns].copy()
    df = df.dropna(subset=["Drug"])

    X = df.drop(columns=["Drug"])
    y = df["Drug"]

    numeric_features = ["Age", "Na_to_K"]
    categorical_features = [
        "Sex", "BP", "Cholesterol"
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1
                )
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    model_path = MODEL_DIR / "drug_pipeline.joblib"
    test_path = RESULTS_DIR / "test_data.joblib"

    joblib.dump(pipeline, model_path)

    joblib.dump(
        {
            "X_test": X_test,
            "y_test": y_test
        },
        test_path
    )

    metadata = {
        "algorithm": "RandomForestClassifier",
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "features": required_columns[:-1],
        "target": "Drug",
        "classes": sorted(y.unique().tolist())
    }

    with open(
        RESULTS_DIR / "training_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("Training completed successfully.")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
