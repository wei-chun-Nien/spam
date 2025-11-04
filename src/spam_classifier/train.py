import os
import json
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import joblib

from .data import download_dataset, load_dataset
from .preprocess import prepare_features


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)[:, 1]
    except Exception:
        # fallback to decision function
        try:
            y_prob = model.decision_function(X)
        except Exception:
            y_prob = None

    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y, y_pred))
    metrics["precision"] = float(precision_score(y, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y, y_pred, zero_division=0))
    if y_prob is not None and len(set(y)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y, y_prob))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    metrics["confusion_matrix"] = confusion_matrix(y, y_pred).tolist()
    return metrics


def main():
    print("Downloading dataset...")
    csv = download_dataset()
    df = load_dataset(csv)
    print(f"Loaded {len(df)} rows")

    print("Preparing features...")
    splits, vectorizer = prepare_features(df)

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    print("Training LinearSVC (SVM)...")
    base = LinearSVC(class_weight="balanced", max_iter=10000)
    # calibrate to get probabilities
    clf = CalibratedClassifierCV(base, cv=3)
    clf.fit(X_train, y_train)

    print("Evaluating on validation set...")
    val_metrics = evaluate_model(clf, X_val, y_val)
    print("Validation metrics:", val_metrics)

    print("Evaluating on test set...")
    test_metrics = evaluate_model(clf, X_test, y_test)
    print("Test metrics:", test_metrics)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
    joblib.dump(clf, MODEL_DIR / "model.pkl")

    metrics = {"validation": val_metrics, "test": test_metrics}
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model artifacts to models/")


if __name__ == "__main__":
    main()
