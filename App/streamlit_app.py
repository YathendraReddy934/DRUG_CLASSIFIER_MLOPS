
import streamlit as st
import pandas as pd

from snowflake.snowpark.context import get_active_session
from snowflake.ml.registry import Registry

DATABASE = "AGENT_QUICKSTART"
MODEL_SCHEMA = "DRUG_CLASSIFIER_ML"
MODEL_NAME = "DRUG_CLASSIFIER_RF"

st.set_page_config(
    page_title="Drug Classification",
    page_icon="🧪",
    layout="centered"
)

st.title("Drug Classification")
st.caption(
    "Educational Random Forest classification demonstration"
)

st.warning(
    "This application is for educational use only. "
    "It must not be used for medical decisions or treatment."
)

st.subheader("Enter input features")

age = st.slider("Age", 15, 80, 35)

sex = st.selectbox(
    "Sex",
    ["F", "M"]
)

bp = st.selectbox(
    "Blood Pressure",
    ["LOW", "NORMAL", "HIGH"]
)

cholesterol = st.selectbox(
    "Cholesterol",
    ["NORMAL", "HIGH"]
)

na_to_k = st.number_input(
    "Na_to_K ratio",
    min_value=0.1,
    max_value=100.0,
    value=15.0,
    step=0.1
)

if st.button("Predict drug class", type="primary"):
    try:
        session = get_active_session()

        registry = Registry(
            session=session,
            database_name=DATABASE,
            schema_name=MODEL_SCHEMA
        )

        model = registry.get_model(MODEL_NAME)
        versions = model.show_versions()

        if versions.empty:
            st.error("No registered model versions found.")
            st.stop()

        version_column = (
            "name" if "name" in versions.columns
            else "version_name"
        )

        if version_column not in versions.columns:
            st.error(
                f"Unexpected version columns: {list(versions.columns)}"
            )
            st.stop()

        versions["created_on"] = pd.to_datetime(
            versions["created_on"]
        )

        latest = versions.sort_values(
            "created_on",
            ascending=False
        ).iloc[0]

        version_name = latest[version_column]

        model_version = model.version(version_name)

        input_df = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "BP": bp,
            "Cholesterol": cholesterol,
            "Na_to_K": na_to_k
        }])

        prediction = model_version.run(
            input_df,
            function_name="predict"
        )

        predicted_class = prediction.iloc[0, -1]

        st.success(f"Predicted class: {predicted_class}")
        st.caption(f"Model version: {version_name}")

    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)




        

