
import os
import json
import joblib
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from snowflake.snowpark import Session
from snowflake.ml.registry import Registry


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")


def get_required_env(name):
    """Read and validate required environment variables."""
    value = os.getenv(name)

    if not value:
        raise ValueError(
            f"Missing required environment variable: {name}"
        )

    return value


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    # File paths
    model_path = ROOT / "Model" / "drug_pipeline.joblib"
    metrics_path = ROOT / "Results" / "metrics.json"
    dataset_path = ROOT / "Data" / "drug.csv"

    # Validate files
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Evaluation metrics not found: {metrics_path}"
        )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    # Load trained model
    model = joblib.load(model_path)

    # Load evaluation metrics
    with open(metrics_path, encoding="utf-8") as file:
        metrics = json.load(file)

    # Load sample input
    df = pd.read_csv(dataset_path)

    required_columns = [
        "Age",
        "Sex",
        "BP",
        "Cholesterol",
        "Na_to_K"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    sample_input = (
        df[required_columns]
        .head(5)
        .copy()
    )

    if sample_input.empty:
        raise ValueError("No sample input records available.")

    # --------------------------------------------------
    # SNOWFLAKE CONFIGURATION
    # --------------------------------------------------

    account = get_required_env("SNOWFLAKE_ACCOUNT")
    user = get_required_env("SNOWFLAKE_USER")
    password = get_required_env("SNOWFLAKE_PASSWORD")
    role = get_required_env("SNOWFLAKE_ROLE")
    warehouse = get_required_env("SNOWFLAKE_WAREHOUSE")

    database = os.getenv(
        "SNOWFLAKE_DATABASE",
        "AGENT_QUICKSTART"
    )

    model_schema = os.getenv(
        "SNOWFLAKE_MODEL_SCHEMA",
        "DRUG_CLASSIFIER_ML"
    )

    results_schema = os.getenv(
        "SNOWFLAKE_RESULTS_SCHEMA",
        "DRUG_CLASSIFIER_RESULTS"
    )

    # Create Snowpark session
    session = Session.builder.configs({
        "account": account,
        "user": user,
        "password": password,
        "role": role,
        "warehouse": warehouse,
        "database": database,
        "schema": model_schema
    }).create()

    try:

        # --------------------------------------------------
        # SET EXPLICIT DATABASE AND SCHEMA CONTEXT
        # --------------------------------------------------

        session.use_database(database)

        # The model registry is stored in this schema
        session.use_schema(model_schema)

        print("Snowflake connection successful.")
        print(f"Database: {database}")
        print(f"Model schema: {model_schema}")

        # --------------------------------------------------
        # INITIALIZE MODEL REGISTRY
        # --------------------------------------------------

        registry = Registry(
            session=session,
            database_name=database,
            schema_name=model_schema
        )

        # Generate model version
        commit = os.getenv("GITHUB_SHA", "LOCAL")

        version_name = (
            "V" + commit[:12].upper()
            if commit != "LOCAL"
            else "V" + datetime.now(
                timezone.utc
            ).strftime("%Y%m%d%H%M%S")
        )

        # --------------------------------------------------
        # REGISTER MODEL
        # --------------------------------------------------

        model_version = registry.log_model(
            model=model,
            model_name="DRUG_CLASSIFIER_RF",
            version_name=version_name,
            comment="Random Forest drug classification pipeline",
            metrics=metrics,
            sample_input_data=sample_input,
            pip_requirements=[
                "scikit-learn==1.5.2",
                "pandas==2.2.3",
                "numpy==2.1.3"
            ],
            target_platforms=["WAREHOUSE"]
        )

        print("Model registered successfully.")
        print("Model: DRUG_CLASSIFIER_RF")
        print(f"Version: {version_name}")

        # --------------------------------------------------
        # MODEL INFERENCE TEST
        # --------------------------------------------------

        # Set an explicit schema for Snowpark's
        # internal pandas DataFrame upload.
        session.use_database(database)
        session.use_schema(results_schema)

        print("Running Snowflake model inference...")

        result = model_version.run(
            sample_input,
            function_name="predict"
        )

        print("Snowflake inference test successful.")
        print(result.head())

        # --------------------------------------------------
        # SAVE METRICS TO SNOWFLAKE
        # --------------------------------------------------

        required_metrics = [
            "accuracy",
            "weighted_f1",
            "macro_f1"
        ]

        missing_metrics = [
            key for key in required_metrics
            if key not in metrics
        ]

        if missing_metrics:
            raise ValueError(
                f"Missing evaluation metrics: {missing_metrics}"
            )

        metrics_row = pd.DataFrame([{
            "MODEL_NAME": "DRUG_CLASSIFIER_RF",
            "MODEL_VERSION": version_name,
            "ACCURACY": metrics["accuracy"],
            "WEIGHTED_F1": metrics["weighted_f1"],
            "MACRO_F1": metrics["macro_f1"],
            "GITHUB_COMMIT": commit
        }])

        # Explicit database and schema are important
        # when using write_pandas.
        session.write_pandas(
            metrics_row,
            table_name="MODEL_METRICS",
            database=database,
            schema=results_schema,
            auto_create_table=False,
            overwrite=False
        )

        print("Metrics saved successfully.")
        print(
            f"Table: {database}.{results_schema}.MODEL_METRICS"
        )

        print("-----------------------------------")
        print("DRUG CLASSIFICATION MLOPS SUCCESS")
        print("-----------------------------------")

    finally:

        session.close()

        print("Snowflake session closed.")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
