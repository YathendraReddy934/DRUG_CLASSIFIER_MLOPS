
import uuid

import pandas as pd
import streamlit as st

from snowflake.snowpark.context import get_active_session
from snowflake.ml.registry import Registry


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATABASE = "AGENT_QUICKSTART"
MODEL_SCHEMA = "DRUG_CLASSIFIER_ML"
RESULTS_SCHEMA = "DRUG_CLASSIFIER_RESULTS"
WAREHOUSE = "COMPUTE_WH"

MODEL_CONFIG = {
    "Random Forest": {
        "name": "DRUG_CLASSIFIER_RF",
        "algorithm": "Random Forest"
    },
    "Decision Tree": {
        "name": "DRUG_CLASSIFIER_DT",
        "algorithm": "Decision Tree"
    },
    "Logistic Regression": {
        "name": "DRUG_CLASSIFIER_LR",
        "algorithm": "Logistic Regression"
    }
}


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Drug Classification MLOps",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Drug Classification MLOps Dashboard")

st.warning(
    "Educational demonstration only. This model is not validated "
    "for medical diagnosis or prescribing."
)


# --------------------------------------------------
# SNOWFLAKE SESSION
# --------------------------------------------------

@st.cache_resource
def get_snowflake_session():
    return get_active_session()


session = get_snowflake_session()

session.sql(
    f"USE WAREHOUSE {WAREHOUSE}"
).collect()


# --------------------------------------------------
# MODEL VERSION LOADING
# --------------------------------------------------

@st.cache_resource(ttl=300)
def get_registered_model_version(model_name):

    registry = Registry(
        session=session,
        database_name=DATABASE,
        schema_name=MODEL_SCHEMA
    )

    model = registry.get_model(model_name)

    versions = session.sql(
        f"""
        SHOW VERSIONS IN MODEL
        {DATABASE}.{MODEL_SCHEMA}.{model_name}
        """
    ).to_pandas()

    if versions.empty:
        raise ValueError(
            f"No registered versions found for {model_name}"
        )

    # Normalize Snowflake result column names.
    versions.columns = [
        str(column).replace('"', '').strip().upper()
        for column in versions.columns
    ]

    if "NAME" not in versions.columns:
        raise ValueError(
            f"NAME column missing. Available columns: "
            f"{list(versions.columns)}"
        )

    if "CREATED_ON" in versions.columns:
        versions["CREATED_ON"] = pd.to_datetime(
            versions["CREATED_ON"],
            errors="coerce"
        )

        versions = versions.sort_values(
            "CREATED_ON",
            ascending=False
        )

    latest_version_name = str(
        versions.iloc[0]["NAME"]
    )

    return (
        model.version(latest_version_name),
        latest_version_name
    )


# --------------------------------------------------
# MODEL METRICS
# --------------------------------------------------

def get_model_metrics(model_name):

    return session.sql(
        f"""
        SELECT
            MODEL_NAME,
            MODEL_VERSION,
            ALGORITHM,
            ACCURACY,
            WEIGHTED_F1,
            MACRO_F1,
            TRAINED_AT,
            GITHUB_COMMIT
        FROM {DATABASE}.{RESULTS_SCHEMA}.MODEL_METRICS
        WHERE MODEL_NAME = '{model_name}'
        ORDER BY TRAINED_AT DESC
        LIMIT 1
        """
    ).to_pandas()


def show_metrics(metrics):

    if metrics.empty:
        st.info(
            "No evaluation metrics are available for this model yet."
        )
        return

    metric = metrics.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{float(metric['ACCURACY']):.2%}"
    )

    col2.metric(
        "Weighted F1",
        f"{float(metric['WEIGHTED_F1']):.2%}"
    )

    col3.metric(
        "Macro F1",
        f"{float(metric['MACRO_F1']):.2%}"
    )

    st.caption(
        f"Registered version: {metric['MODEL_VERSION']} | "
        f"Commit: {metric['GITHUB_COMMIT']}"
    )


# --------------------------------------------------
# SAVE PREDICTION HISTORY
# --------------------------------------------------

def save_prediction(
    model_name,
    version_name,
    algorithm,
    input_data,
    predicted_drug,
    github_commit
):

    row = {
        "PREDICTION_ID": str(uuid.uuid4()),
        "MODEL_NAME": model_name,
        "MODEL_VERSION": version_name,
        "ALGORITHM": algorithm,
        "AGE": int(input_data.iloc[0]["Age"]),
        "SEX": str(input_data.iloc[0]["Sex"]),
        "BP": str(input_data.iloc[0]["BP"]),
        "CHOLESTEROL": str(input_data.iloc[0]["Cholesterol"]),
        "NA_TO_K": float(input_data.iloc[0]["Na_to_K"]),
        "PREDICTED_DRUG": str(predicted_drug),
        "GITHUB_COMMIT": str(github_commit)
    }

    history_df = pd.DataFrame([row])

    session.write_pandas(
        history_df,
        table_name="PREDICTION_HISTORY",
        database=DATABASE,
        schema=RESULTS_SCHEMA,
        auto_create_table=False,
        overwrite=False
    )


# --------------------------------------------------
# MODEL PREDICTION PAGE
# --------------------------------------------------

def show_model_page(display_name, config):

    model_name = config["name"]
    algorithm = config["algorithm"]

    st.header(f"{display_name} Model")

    try:
        model_version, version_name = (
            get_registered_model_version(model_name)
        )

        st.success(
            f"Model loaded successfully | Version: {version_name}"
        )

    except Exception as error:
        st.error(f"Unable to load model: {error}")
        return

    metrics = get_model_metrics(model_name)

    st.subheader("Model Evaluation")
    show_metrics(metrics)

    github_commit = "UNKNOWN"

    if not metrics.empty:
        github_commit = str(
            metrics.iloc[0]["GITHUB_COMMIT"]
        )

    st.divider()
    st.subheader("Enter Input Features")

    with st.form(f"prediction_form_{model_name}"):

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=15,
                max_value=100,
                value=30
            )

            sex = st.selectbox(
                "Sex",
                ["M", "F"],
                key=f"sex_{model_name}"
            )

            bp = st.selectbox(
                "Blood Pressure",
                ["LOW", "NORMAL", "HIGH"],
                key=f"bp_{model_name}"
            )

        with col2:

            cholesterol = st.selectbox(
                "Cholesterol",
                ["NORMAL", "HIGH"],
                key=f"cholesterol_{model_name}"
            )

            na_to_k = st.number_input(
                "Na_to_K",
                min_value=0.1,
                max_value=100.0,
                value=15.0,
                step=0.1
            )

        submitted = st.form_submit_button(
            "Predict Drug",
            type="primary",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "BP": bp,
            "Cholesterol": cholesterol,
            "Na_to_K": na_to_k
        }])

        try:

            with st.spinner(
                "Generating prediction in Snowflake..."
            ):

                prediction = model_version.run(
                    input_data,
                    function_name="predict"
                )

                if hasattr(prediction, "to_pandas"):
                    prediction = prediction.to_pandas()

                prediction = pd.DataFrame(prediction)

                if prediction.empty:
                    raise ValueError(
                        "The model returned an empty prediction."
                    )

                predicted_drug = prediction.iloc[0, 0]

                save_prediction(
                    model_name=model_name,
                    version_name=version_name,
                    algorithm=algorithm,
                    input_data=input_data,
                    predicted_drug=predicted_drug,
                    github_commit=github_commit
                )

            st.subheader("Prediction Result")

            st.success(
                f"Predicted Drug Category: {predicted_drug}"
            )

            st.write("Submitted Input")
            st.dataframe(
                input_data,
                use_container_width=True
            )

            st.write("Model Output")
            st.dataframe(
                prediction,
                use_container_width=True
            )

            st.caption(
                f"Saved to Snowflake prediction history | "
                f"{model_name} | {version_name}"
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# CHAMPION MODEL
# --------------------------------------------------

def show_champion():

    champion = session.sql(
        f"""
        SELECT
            RUN_COMMIT,
            MODEL_NAME,
            ALGORITHM,
            MODEL_VERSION,
            SELECTION_METRIC,
            METRIC_VALUE,
            SELECTED_AT
        FROM {DATABASE}.{RESULTS_SCHEMA}.MODEL_CHAMPION_HISTORY
        ORDER BY SELECTED_AT DESC
        LIMIT 1
        """
    ).to_pandas()

    if champion.empty:
        st.info(
            "Champion selection has not been recorded yet."
        )
        return

    result = champion.iloc[0]

    st.subheader("Current Champion Model")

    st.success(
        f"Champion: {result['ALGORITHM']} "
        f"({result['MODEL_NAME']})"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Champion Macro F1",
        f"{float(result['METRIC_VALUE']):.2%}"
    )

    col2.metric(
        "Model Version",
        str(result["MODEL_VERSION"])
    )

    col3.metric(
        "Selection Metric",
        str(result["SELECTION_METRIC"])
    )

    st.caption(
        f"Training commit: {result['RUN_COMMIT']} | "
        f"Selected at: {result['SELECTED_AT']}"
    )


# --------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------

def show_overview():

    show_champion()

    st.divider()
    st.header("Model Comparison")

    comparison = session.sql(
        f"""
        SELECT
            MODEL_NAME,
            ALGORITHM,
            MODEL_VERSION,
            ACCURACY,
            WEIGHTED_F1,
            MACRO_F1,
            TRAINED_AT
        FROM {DATABASE}.{RESULTS_SCHEMA}.MODEL_METRICS
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY MODEL_NAME
            ORDER BY TRAINED_AT DESC
        ) = 1
        ORDER BY ACCURACY DESC
        """
    ).to_pandas()

    if comparison.empty:

        st.info("No model evaluation results found.")

    else:

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Accuracy Comparison")

        chart_data = comparison[
            ["ALGORITHM", "ACCURACY"]
        ].set_index("ALGORITHM")

        st.bar_chart(chart_data)

    st.caption(
        "Each model is registered and versioned independently "
        "in the Snowflake Model Registry."
    )


# --------------------------------------------------
# PREDICTION HISTORY PAGE
# --------------------------------------------------

def show_prediction_history():

    st.header("Prediction History")

    try:

        history = session.sql(
            f"""
            SELECT
                PREDICTED_AT,
                MODEL_NAME,
                MODEL_VERSION,
                ALGORITHM,
                AGE,
                SEX,
                BP,
                CHOLESTEROL,
                NA_TO_K,
                PREDICTED_DRUG,
                GITHUB_COMMIT
            FROM {DATABASE}.{RESULTS_SCHEMA}.PREDICTION_HISTORY
            ORDER BY PREDICTED_AT DESC
            LIMIT 500
            """
        ).to_pandas()

        if history.empty:

            st.info(
                "No predictions have been recorded yet. "
                "Make a prediction from one of the model pages."
            )
            return

        model_options = ["All Models"] + sorted(
            history["MODEL_NAME"].dropna().unique().tolist()
        )

        selected_model = st.selectbox(
            "Filter by Model",
            model_options
        )

        if selected_model != "All Models":

            history = history[
                history["MODEL_NAME"] == selected_model
            ]

        st.metric(
            "Displayed Predictions",
            len(history)
        )

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

    except Exception as error:
        st.error(
            f"Unable to retrieve prediction history: {error}"
        )


# --------------------------------------------------
# VERSION COMPARISON PAGE
# --------------------------------------------------

def show_version_comparison():

    st.header("Model Version Comparison")

    history = session.sql(
        f"""
        SELECT
            MODEL_NAME,
            ALGORITHM,
            MODEL_VERSION,
            ACCURACY,
            WEIGHTED_F1,
            MACRO_F1,
            TRAINED_AT,
            GITHUB_COMMIT
        FROM {DATABASE}.{RESULTS_SCHEMA}.MODEL_METRICS
        ORDER BY TRAINED_AT DESC
        """
    ).to_pandas()

    if history.empty:

        st.info("No model version history found.")
        return

    model_options = ["All Models"] + sorted(
        history["MODEL_NAME"].dropna().unique().tolist()
    )

    selected_model = st.selectbox(
        "Choose a model",
        model_options,
        key="version_comparison_model"
    )

    if selected_model != "All Models":

        history = history[
            history["MODEL_NAME"] == selected_model
        ]

    st.subheader("Version History")

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Performance Across Versions")

    chart_metric = st.selectbox(
        "Choose evaluation metric",
        ["ACCURACY", "WEIGHTED_F1", "MACRO_F1"],
        index=2,
        key="version_comparison_metric"
    )

    chart_data = history[
        ["MODEL_NAME", "MODEL_VERSION", chart_metric]
    ].copy()

    chart_data["VERSION_LABEL"] = (
        chart_data["MODEL_NAME"]
        + " - "
        + chart_data["MODEL_VERSION"]
    )

    chart_data = chart_data.set_index("VERSION_LABEL")

    st.bar_chart(
        chart_data[[chart_metric]]
    )

    st.caption(
        "Each row represents one registered model version "
        "and its evaluation results."
    )


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page_options = [
    "Overview",
    "Random Forest",
    "Decision Tree",
    "Logistic Regression",
    "Version Comparison",
    "Prediction History"
]

selected_page = st.sidebar.radio(
    "Select a page",
    page_options
)

if st.sidebar.button("Refresh model versions"):

    get_registered_model_version.clear()
    st.rerun()


# --------------------------------------------------
# PAGE ROUTING
# --------------------------------------------------

if selected_page == "Overview":

    show_overview()

elif selected_page == "Random Forest":

    show_model_page(
        "Random Forest",
        MODEL_CONFIG["Random Forest"]
    )

elif selected_page == "Decision Tree":

    show_model_page(
        "Decision Tree",
        MODEL_CONFIG["Decision Tree"]
    )

elif selected_page == "Logistic Regression":

    show_model_page(
        "Logistic Regression",
        MODEL_CONFIG["Logistic Regression"]
    )

elif selected_page == "Version Comparison":

    show_version_comparison()

elif selected_page == "Prediction History":

    show_prediction_history()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Drug Classification MLOps | Python | Scikit-learn | "
    "GitHub Actions | Snowflake Model Registry"
)
