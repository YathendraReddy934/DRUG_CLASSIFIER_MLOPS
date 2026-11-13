
from pathlib import Path
import json
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT_DIR / "artifacts" / "models"
REPORT_DIR = ROOT_DIR / "artifacts" / "reports"


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
# EVALUATION
# --------------------------------------------------

def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    test_data_path = REPORT_DIR / "test_data.joblib"

    if not test_data_path.exists():
        raise FileNotFoundError(
            f"Test data not found: {test_data_path}. "
            "Run scripts/train.py first."
        )

    X_test, y_test = joblib.load(test_data_path)

    all_metrics = {}

    for model_name, config in MODELS.items():

        model_path = MODEL_DIR / config["filename"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Run scripts/train.py first."
            )

        model = joblib.load(model_path)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        weighted_f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        # Store metrics for Snowflake registration.
        all_metrics[model_name] = {
            "algorithm": config["algorithm"],
            "accuracy": float(accuracy),
            "weighted_f1": float(weighted_f1),
            "macro_f1": float(macro_f1)
        }

        print(f"\n===== {config['algorithm']} =====")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")

        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

        # Generate confusion matrix.
        matrix = confusion_matrix(
            y_test,
            predictions
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix
        )

        display.plot(
            cmap="Blues",
            xticks_rotation=45
        )

        plt.title(
            f"{config['algorithm']} - Confusion Matrix"
        )

        plt.tight_layout()

        model_suffix = model_name.split("_")[-1].lower()

        image_path = (
            REPORT_DIR /
            f"confusion_matrix_{model_suffix}.png"
        )

        plt.savefig(image_path)
        plt.close()

        print(f"Confusion matrix saved: {image_path}")

    # Save all three model metrics.
    metrics_path = REPORT_DIR / "metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_metrics,
            file,
            indent=4
        )

    print(f"\nMetrics saved to: {metrics_path}")
    print("\nAll three models evaluated successfully.")


if __name__ == "__main__":
    main()
