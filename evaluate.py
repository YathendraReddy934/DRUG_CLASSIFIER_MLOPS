
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "Model" / "drug_pipeline.joblib"
TEST_PATH = ROOT / "Results" / "test_data.joblib"
RESULTS_DIR = ROOT / "Results"


def main():
    if not MODEL_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Run train.py before evaluate.py."
        )

    model = joblib.load(MODEL_PATH)
    test_data = joblib.load(TEST_PATH)

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted"
    )
    macro_f1 = f1_score(
        y_test, predictions, average="macro"
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1)
    }

    with open(
        RESULTS_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    with open(
        RESULTS_DIR / "metrics.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write("DRUG CLASSIFICATION MODEL EVALUATION\n")
        file.write("=" * 45 + "\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Weighted F1: {weighted_f1:.4f}\n")
        file.write(f"Macro F1: {macro_f1:.4f}\n\n")
        file.write("Classification Report\n")
        file.write(report)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        xticks_rotation=45
    )
    display.figure_.tight_layout()
    display.figure_.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=150
    )
    plt.close(display.figure_)

    print("Evaluation completed.")
    print(json.dumps(metrics, indent=4))
    print("\nClassification report:\n")
    print(report)


if __name__ == "__main__":
    main()
