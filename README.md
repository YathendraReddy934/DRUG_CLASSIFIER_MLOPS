
# Drug Classification MLOps

An educational machine learning project that predicts a drug
class using a Random Forest classification pipeline.

## Technology stack

- Python
- Pandas and Scikit-learn
- Random Forest
- Git and GitHub
- GitHub Actions
- CML
- Snowflake Model Registry
- Streamlit in Snowflake

## Dataset

Drug200 dataset with the following input features:

- Age
- Sex
- BP
- Cholesterol
- Na_to_K

Target: Drug

## Local setup

Create and activate a Python environment:

    py -3.11 -m venv .venv

    .\.venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

Train:

    python train.py

Evaluate:

    python evaluate.py

## Project workflow

1. Develop and test locally.
2. Push source code to GitHub.
3. GitHub Actions trains and evaluates the model.
4. CML publishes the evaluation report.
5. The main branch registers the model in Snowflake.
6. Streamlit in Snowflake uses the registered model.

## Disclaimer

This is an educational machine learning demonstration.
It is not intended for clinical or medical decision-making.