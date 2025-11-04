import sys
import pathlib
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

# 確保 src/ 在 sys.path 中
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from spam_classifier.data import download_dataset, load_dataset
from spam_classifier.predict import predict

# 使用絕對路徑
MODEL_DIR = ROOT / "models"

def load_vectorizer():
    """
    只檢查 vectorizer 是否存在和已 fit。
    model 不檢查 fit，避免 CalibratedClassifierCV 引發 AttributeError。
    """
    vec_path = MODEL_DIR / "vectorizer.pkl"
    model_path = MODEL_DIR / "model.pkl"

    if not vec_path.exists() or not model_path.exists():
        return None, "Model files not found. Make sure vectorizer.pkl and model.pkl exist in models/"

    try:
        vectorizer = joblib.load(vec_path)
    except Exception as e:
        return None, f"Failed to load vectorizer.pkl: {e}"

    try:
        check_is_fitted(vectorizer)
    except NotFittedError:
        return None, "Vectorizer is not fitted. Upload the correct trained file."

    return vectorizer, None


def main():
    st.title("Spam Classifier Demo (SVM)")

    # --- Model debug ---
    st.subheader("Model artifacts debug")
    vec_path = MODEL_DIR / "vectorizer.pkl"
    model_path = MODEL_DIR / "model.pkl"
    st.write("vectorizer exists:", vec_path.exists(), "size:", vec_path.stat().st_size if vec_path.exists() else None)
    st.write("model exists:", model_path.exists(), "size:", model_path.stat().st_size if model_path.exists() else None)

    vectorizer, error_msg = load_vectorizer()
    if error_msg:
        st.error(error_msg)
        st.stop()

    # --- Data ---
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

    # --- Metrics ---
    st.header("Model & Metrics")
    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.json(metrics)
    else:
        st.info("Model metrics not found. Train a model to see metrics.")

    # --- Prediction ---
    st.header("Predict")
    text = st.text_area("Enter message to classify", height=150)
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5)

    if st.button("Predict"):
        try:
            result = predict(text, threshold=threshold, model_dir=MODEL_DIR)
            st.write("Probability (spam):", result["probability"])
            st.write("Label:", result["label"])
        except NotFittedError:
            st.error("Model is not fitted. Make sure you uploaded the trained model.pkl")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
