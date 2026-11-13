
import os
import json
import joblib
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from snowflake.snowpark import Session
from snowflake.ml.registry import Registry

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

def main():
    model_path = ROOT / "Model" / "drug_pipeline.joblib"
    metrics_path = ROOT / "Results" / "metrics.json"
    dataset_path = ROOT / "Data" / "drug.csv"

    if not model_path.exists():
        raise FileNotFoundError("Trained model not found.")

    if not metrics_path.exists():
        raise FileNotFoundError("Evaluation metrics not found.")

    model = joblib.load(model_path)

    with open(metrics_path, encoding="utf-8") as file:
        metrics = json.load(file)

    df = pd.read_csv(dataset_path)

    sample_input = df[
        ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
    ].head(5)

    session = Session.builder.configs({
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "role": os.environ["SNOWFLAKE_ROLE"],
        "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
        "database": os.environ["SNOWFLAKE_DATABASE"]
    }).create()

    try:
        database = os.getenv(
            "SNOWFLAKE_DATABASE",
            "AGENT_QUICKSTART"
        )
        schema = os.getenv(
            "SNOWFLAKE_MODEL_SCHEMA",
            "DRUG_CLASSIFIER_ML"
        )

        registry = Registry(
            session=session,
            database_name=database,
            schema_name=schema
        )

        commit = os.getenv("GITHUB_SHA", "LOCAL")
        version_name = (
            "V" + commit[:12].upper()
            if commit != "LOCAL"
            else "V" + datetime.now(timezone.utc).strftime(
                "%Y%m%d%H%M%S"
            )
        )

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

        result = model_version.run(
            sample_input,
            function_name="predict"
        )

        print("Snowflake inference test:")
        print(result.head())

        results_schema = os.getenv(
            "SNOWFLAKE_RESULTS_SCHEMA",
            "DRUG_CLASSIFIER_RESULTS"
        )

        metrics_row = pd.DataFrame([{
            "MODEL_NAME": "DRUG_CLASSIFIER_RF",
            "MODEL_VERSION": version_name,
            "ACCURACY": metrics["accuracy"],
            "WEIGHTED_F1": metrics["weighted_f1"],
            "MACRO_F1": metrics["macro_f1"],
            "GITHUB_COMMIT": commit
        }])

        session.write_pandas(
            metrics_row,
            table_name="MODEL_METRICS",
            database=database,
            schema=results_schema,
            auto_create_table=False,
            overwrite=False
        )

        print("Metrics saved to Snowflake.")

    finally:
        session.close()

if __name__ == "__main__":
    main()