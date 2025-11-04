import sys
import pathlib

# Ensure src/ is on sys.path so Streamlit Cloud (or other hosts) can import the package
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from spam_classifier.data import download_dataset, load_dataset
from spam_classifier.predict import load_artifacts, predict
import joblib
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

MODEL_DIR = Path("models")


def main():
    st.title("Spam classifier demo (SVM)")

    st.sidebar.header("Model")
    if not MODEL_DIR.exists():
        st.sidebar.warning("No trained model found. Run training first (python -m spam_classifier.train)")

    # Data
    csv = download_dataset()
    df = load_dataset(csv)

    st.header("Data overview")
    st.write(df.head())

    st.subheader("Class distribution")
    counts = df["label"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_ylabel("count")
    st.pyplot(fig)

    st.header("Model & Metrics")
    metrics_path = MODEL_DIR / "metrics.json"
    # --- Debug: show model artifact presence and fitted state ---
    st.subheader("Model artifacts debug")
    try:
        vec_path = MODEL_DIR / "vectorizer.pkl"
        model_path = MODEL_DIR / "model.pkl"
        st.write("vectorizer exists:", vec_path.exists(), "size:", vec_path.stat().st_size if vec_path.exists() else None)
        st.write("model exists:", model_path.exists(), "size:", model_path.stat().st_size if model_path.exists() else None)
        if vec_path.exists():
            try:
                v = joblib.load(vec_path)
                try:
                    check_is_fitted(v)
                    st.success("Vectorizer is fitted")
                except Exception as e:
                    st.error(f"Vectorizer not fitted or invalid: {e}")
            except Exception as e:
                st.error(f"Failed to load vectorizer.pkl: {e}")
    except Exception as e:
        st.warning(f"Model debug check failed: {e}")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.json(metrics)
    else:
        st.info("Model metrics not found. Train a model to see metrics.")

    st.header("Predict")
    text = st.text_area("Enter message to classify", height=150)
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5)
    if st.button("Predict"):
        if not MODEL_DIR.exists():
            st.error("Model not found. Train first.")
        else:
            try:
                result = predict(text, threshold=threshold, model_dir=MODEL_DIR)
                st.write("Probability (spam):", result["probability"])
                st.write("Label:", result["label"])
            except NotFittedError as e:
                st.error("Model or vectorizer is not fitted. This usually means the uploaded artifacts are not the trained files. Please ensure models/vectorizer.pkl and models/model.pkl are the trained artefacts (not 404 HTML or empty files).\nCheck the 'Model artifacts debug' section above for details.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
