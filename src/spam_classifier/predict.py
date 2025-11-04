import joblib
from pathlib import Path
from typing import Tuple, Union

MODEL_DIR = Path("models")


def load_artifacts(model_dir: Union[str, Path] = MODEL_DIR) -> Tuple[object, object]:
    model_dir = Path(model_dir)
    vectorizer = joblib.load(model_dir / "vectorizer.pkl")
    model = joblib.load(model_dir / "model.pkl")
    return vectorizer, model


def predict(text: str, threshold: float = 0.5, model_dir: Union[str, Path] = MODEL_DIR):
    vectorizer, model = load_artifacts(model_dir)
    X = vectorizer.transform([text])
    try:
        prob = float(model.predict_proba(X)[:, 1][0])
    except Exception:
        # fallback to decision_function normalized
        score = float(model.decision_function(X)[0])
        prob = 1.0 / (1.0 + pow(2.718281828, -score))
    label = "spam" if prob >= threshold else "ham"
    return {"probability": prob, "label": label}
