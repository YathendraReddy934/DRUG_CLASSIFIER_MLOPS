
import os
import json
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from snowflake.snowpark import Session
from snowflake.ml.registry import Registry


load_dotenv()

DATABASE = os.getenv("SNOWFLAKE_DATABASE", "AGENT_QUICKSTART")
MODEL_SCHEMA = os.getenv(
    "SNOWFLAKE_MODEL_SCHEMA", "DRUG_CLASSIFIER_ML"
)
RESULTS_SCHEMA = os.getenv(
    "SNOWFLAKE_RESULTS_SCHEMA", "DRUG_CLASSIFIER_RESULTS"
)

MODELS = {
    "DRUG_CLASSIFIER_RF": {
        "algorithm": "Random Forest",
        "filename": "drug_rf.joblib"
    },
    "DRUG_CLASSIFIER_DT": {
        "algorithm": "Decision Tree",
        "filename": "drug_dt.joblib"
    },
    "DRUG_CLASSIFIER_LR": {
        "algorithm": "Logistic Regression",
        "filename": "drug_lr.joblib"
    }
}


def create_session():
    connection_parameters = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "role": os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        "database": DATABASE,
        "schema": MODEL_SCHEMA
    }

    return Session.builder.configs(connection_parameters).create()


def main():
    with open("Results/metrics.json", "r", encoding="utf-8") as file:
        all_metrics = json.load(file)

    sample_input = pd.read_csv("Results/sample_input.csv")

    commit_sha = os.getenv("GITHUB_SHA", "LOCAL")
    version_name = (
        "V" + commit_sha[:12].upper()
        if commit_sha != "LOCAL"
        else "V" + datetime.now().strftime("%Y%m%d%H%M%S")
    )

    session = create_session()

    try:
        session.sql(f"USE DATABASE {DATABASE}").collect()
        session.sql(f"USE SCHEMA {MODEL_SCHEMA}").collect()

        registry = Registry(
            session=session,
            database_name=DATABASE,
            schema_name=MODEL_SCHEMA
        )

        metric_rows = []

        for model_name, config in MODELS.items():
            metric = all_metrics[model_name]

            model_path = os.path.join("Model", config["filename"])

            import joblib
            model = joblib.load(model_path)

            print(f"\nRegistering {model_name}...")

            model_version = registry.log_model(
                model=model,
                model_name=model_name,
                version_name=version_name,
                comment=(
                    f"{config['algorithm']} drug classification model. "
                    f"Git commit: {commit_sha}"
                ),
                metrics={
                    "accuracy": metric["accuracy"],
                    "weighted_f1": metric["weighted_f1"],
                    "macro_f1": metric["macro_f1"]
                },
                sample_input_data=sample_input,
                pip_requirements=[
                    "scikit-learn==1.5.2",
                    "pandas==2.2.3",
                    "numpy==2.1.3"
                ],
                target_platforms=["WAREHOUSE"]
            )

            # Confirm that the registered model can perform inference.
            test_result = model_version.run(
                sample_input.head(1),
                function_name="predict"
            )

            print(f"Registered version: {version_name}")
            print("Test inference completed.")
            print(test_result)

            metric_rows.append({
                "MODEL_NAME": model_name,
                "MODEL_VERSION": version_name,
                "ALGORITHM": config["algorithm"],
                "ACCURACY": metric["accuracy"],
                "WEIGHTED_F1": metric["weighted_f1"],
                "MACRO_F1": metric["macro_f1"],
                "GITHUB_COMMIT": commit_sha
            })

        metrics_df = pd.DataFrame(metric_rows)

        session.write_pandas(
            metrics_df,
            table_name="MODEL_METRICS",
            database=DATABASE,
            schema=RESULTS_SCHEMA,
            auto_create_table=False,
            overwrite=False
        )

        print("\nAll three models registered successfully.")
        print("Three model metric rows written to Snowflake.")

    finally:
        session.close()


if __name__ == "__main__":
    main()
