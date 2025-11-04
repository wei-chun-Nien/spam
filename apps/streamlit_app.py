import sys
import pathlib
from pathlib import Path
import json
import subprocess

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- 確保 spam_classifier 可以被 import ---
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from spam_classifier.data import download_dataset, load_dataset
from spam_classifier.predict import predict

# 模型目錄
MODEL_DIR = Path("models")


def check_vectorizer_and_model(vec_path, model_path):
    """
    嘗試簡單測試 vectorizer 和 model 可用性
    """
    vectorizer_ok = False
    model_ok = False

    # vectorizer
    if vec_path.exists() and vec_path.stat().st_size > 0:
        try:
            vec = joblib.load(vec_path)
            vec.transform(["test"])
            vectorizer_ok = True
            st.success("Vectorizer is usable ✅")
        except Exception as e:
            st.warning(f"Vectorizer load/transform failed: {e}")
    else:
        st.warning("Vectorizer not found or empty")

    # model
    if model_path.exists() and model_path.stat().st_size > 0:
        try:
            model = joblib.load(model_path)
            if vectorizer_ok:
                dummy = vec.transform(["test"])
                model.predict(dummy)
                model_ok = True
                st.success("Model is usable ✅")
        except Exception as e:
            st.warning(f"Model load/predict failed: {e}")
    else:
        st.warning("Model not found or empty")

    return vectorizer_ok and model_ok


def train_model():
    """
    在 Streamlit 內呼叫 train.py 重新訓練模型
    """
    st.info("Training model... This may take a few minutes.")
    try:
        # 使用 subprocess 執行 train 模組
        subprocess.run([sys.executable, "-m", "spam_classifier.train"], check=True)
        st.success("Training finished! Model, vectorizer, and metrics.json updated ✅")
    except subprocess.CalledProcessError as e:
        st.error(f"Training failed: {e}")


def main():
    st.title("Spam Classifier Demo (SVM)")

    # --- 一鍵訓練按鈕 ---
    if st.sidebar.button("Train / Retrain Model"):
        train_model()

    # --- Data ---
    st.header("Data Overview")
    csv = download_dataset()
    df = load_dataset(csv)
    st.write(df.head())

    st.subheader("Class Distribution")
    counts = df["label"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_ylabel("Count")
    st.pyplot(fig, clear_figure=True)

    # --- Model & Metrics ---
    st.header("Model & Metrics")
    metrics_path = MODEL_DIR / "metrics.json"
    vec_path = MODEL_DIR / "vectorizer.pkl"
    model_path = MODEL_DIR / "model.pkl"

    st.subheader("Model Artifacts Debug")
    model_ready = check_vectorizer_and_model(vec_path, model_path)

    # 顯示 metrics
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.json(metrics)
        except Exception as e:
            st.error(f"Failed to load metrics.json: {e}")
    else:
        st.info("Model metrics not found. Train the model to see metrics.")

    # --- Prediction ---
    st.header("Predict a Message")
    text = st.text_area("Enter message to classify", height=150)
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5)

    if st.button("Predict"):
        if not model_ready:
            st.error("Model or vectorizer not usable. Please train first.")
        elif not text.strip():
            st.warning("Please enter some text to classify.")
        else:
            try:
                result = predict(text, threshold=threshold, model_dir=MODEL_DIR)
                st.write("**Probability (spam):**", result.get("probability"))
                st.write("**Predicted Label:**", result.get("label"))
            except Exception as e:
                st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
