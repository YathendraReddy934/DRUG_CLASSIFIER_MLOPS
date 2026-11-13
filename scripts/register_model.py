
from pathlib import Path
from datetime import datetime
import json
import os

import joblib
import pandas as pd
from dotenv import load_dotenv

from snowflake.snowpark import Session
from snowflake.ml.registry import Registry


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT_DIR / "artifacts" / "models"
REPORT_DIR = ROOT_DIR / "artifacts" / "reports"


# --------------------------------------------------
# SNOWFLAKE CONFIGURATION
# --------------------------------------------------

DATABASE = os.getenv(
    "SNOWFLAKE_DATABASE",
    "AGENT_QUICKSTART"
)

MODEL_SCHEMA = os.getenv(
    "SNOWFLAKE_MODEL_SCHEMA",
    "DRUG_CLASSIFIER_ML"
)

RESULTS_SCHEMA = os.getenv(
    "SNOWFLAKE_RESULTS_SCHEMA",
    "DRUG_CLASSIFIER_RESULTS"
)


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

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


# --------------------------------------------------
# SNOWFLAKE CONNECTION
# --------------------------------------------------

def create_session():

    required_variables = [
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD"
    ]

    missing_variables = [
        variable
        for variable in required_variables
        if not os.getenv(variable)
    ]

    if missing_variables:
        raise EnvironmentError(
            "Missing required Snowflake environment variables: "
            + ", ".join(missing_variables)
        )

    connection_parameters = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "role": os.getenv(
            "SNOWFLAKE_ROLE",
            "ACCOUNTADMIN"
        ),
        "warehouse": os.getenv(
            "SNOWFLAKE_WAREHOUSE",
            "COMPUTE_WH"
        ),
        "database": DATABASE,
        "schema": MODEL_SCHEMA
    }

    return (
        Session.builder
        .configs(connection_parameters)
        .create()
    )


# --------------------------------------------------
# VERSION NAME
# --------------------------------------------------

def get_version_name():

    commit_sha = os.getenv(
        "GITHUB_SHA",
        "LOCAL"
    )

    if commit_sha == "LOCAL":
        version_name = (
            "V" + datetime.now().strftime("%Y%m%d%H%M%S")
        )
    else:
        version_name = (
            "V" + commit_sha[:12].upper()
        )

    return commit_sha, version_name


# --------------------------------------------------
# INPUT FILE VALIDATION
# --------------------------------------------------

def load_registration_inputs():

    metrics_path = REPORT_DIR / "metrics.json"
    sample_input_path = REPORT_DIR / "sample_input.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}. "
            "Run scripts/evaluate.py first."
        )

    if not sample_input_path.exists():
        raise FileNotFoundError(
            f"Sample input file not found: {sample_input_path}. "
            "Run scripts/train.py first."
        )

    with open(
        metrics_path,
        "r",
        encoding="utf-8"
    ) as file:
        all_metrics = json.load(file)

    sample_input = pd.read_csv(sample_input_path)

    for model_name, config in MODELS.items():

        if model_name not in all_metrics:
            raise ValueError(
                f"Metrics missing for {model_name}"
            )

        model_path = MODEL_DIR / config["filename"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

    return all_metrics, sample_input


# --------------------------------------------------
# MODEL REGISTRATION
# --------------------------------------------------

def main():

    all_metrics, sample_input = load_registration_inputs()

    commit_sha, version_name = get_version_name()

    session = create_session()

    try:

        session.sql(
            f"USE DATABASE {DATABASE}"
        ).collect()

        session.sql(
            f"USE SCHEMA {MODEL_SCHEMA}"
        ).collect()

        registry = Registry(
            session=session,
            database_name=DATABASE,
            schema_name=MODEL_SCHEMA
        )

        metric_rows = []

        # Register each of the three models.
        for model_name, config in MODELS.items():

            metric = all_metrics[model_name]

            model_path = MODEL_DIR / config["filename"]

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

            # Verify that the registered model can predict.
            test_result = model_version.run(
                sample_input.head(1),
                function_name="predict"
            )

            print(f"Registered version: {version_name}")
            print("Test inference completed.")
            print(test_result)

            # Prepare the metrics table record.
            metric_rows.append({
                "MODEL_NAME": model_name,
                "MODEL_VERSION": version_name,
                "ALGORITHM": config["algorithm"],
                "ACCURACY": metric["accuracy"],
                "WEIGHTED_F1": metric["weighted_f1"],
                "MACRO_F1": metric["macro_f1"],
                "GITHUB_COMMIT": commit_sha
            })

        # --------------------------------------------------
        # WRITE MODEL METRICS
        # --------------------------------------------------

        metrics_df = pd.DataFrame(metric_rows)

        session.write_pandas(
            metrics_df,
            table_name="MODEL_METRICS",
            database=DATABASE,
            schema=RESULTS_SCHEMA,
            auto_create_table=False,
            overwrite=False
        )

        print(
            "\nThree model metric records written to Snowflake."
        )

        # --------------------------------------------------
        # CHAMPION MODEL SELECTION
        # --------------------------------------------------

        # Select the model with the highest Macro F1.
        champion = max(
            metric_rows,
            key=lambda row: row["MACRO_F1"]
        )

        champion_df = pd.DataFrame([{
            "RUN_COMMIT": commit_sha,
            "MODEL_NAME": champion["MODEL_NAME"],
            "ALGORITHM": champion["ALGORITHM"],
            "MODEL_VERSION": champion["MODEL_VERSION"],
            "SELECTION_METRIC": "MACRO_F1",
            "METRIC_VALUE": champion["MACRO_F1"]
        }])

        session.write_pandas(
            champion_df,
            table_name="MODEL_CHAMPION_HISTORY",
            database=DATABASE,
            schema=RESULTS_SCHEMA,
            auto_create_table=False,
            overwrite=False
        )

        print("\nChampion model selected successfully.")
        print(f"Champion model: {champion['MODEL_NAME']}")
        print(f"Algorithm: {champion['ALGORITHM']}")
        print(f"Model version: {champion['MODEL_VERSION']}")
        print(
            f"Champion Macro F1: {champion['MACRO_F1']:.4f}"
        )

        print("\nAll three models registered successfully.")

    finally:
        session.close()


if __name__ == "__main__":
    main()
