import sys
import pathlib
from pathlib import Path
import json

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- 確保 spam_classifier 可以被 import ---
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  # 假設 spam_classifier 在專案根目錄下

from spam_classifier.data import download_dataset, load_dataset
from spam_classifier.predict import predict

# 模型目錄
MODEL_DIR = Path("models")


def check_vectorizer_and_model(vec_path, model_path):
    """
    安全檢查 vectorizer 和 model 是否可用
    - 不用 check_is_fitted，避免 AttributeError
    - 直接測試 predict 一個簡單訊息
    """
    vectorizer_ok = False
    model_ok = False

    # 檢查 vectorizer
    if not vec_path.exists() or vec_path.stat().st_size == 0:
        st.error(f"Vectorizer file not found or empty: {vec_path}")
    else:
        try:
            vec = joblib.load(vec_path)
            # 嘗試轉換簡單文字測試
            vec.transform(["test"])
            vectorizer_ok = True
            st.success("Vectorizer is usable ✅")
        except Exception as e:
            st.error(f"Vectorizer load/transform failed: {e}")

    # 檢查 model
    if not model_path.exists() or model_path.stat().st_size == 0:
        st.error(f"Model file not found or empty: {model_path}")
    else:
        try:
            model = joblib.load(model_path)
            # 嘗試對 dummy vector 做 predict
            if vectorizer_ok:
                dummy = vec.transform(["test"])
                model.predict(dummy)
                model_ok = True
                st.success("Model is usable ✅")
        except Exception as e:
            st.error(f"Model load/predict failed: {e}")

    return vectorizer_ok and model_ok


def main():
    st.title("Spam Classifier Demo (SVM)")

    st.sidebar.header("Model Status")
    if not MODEL_DIR.exists():
        st.sidebar.warning("No trained model found. Run training first:\npython -m spam_classifier.train")

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
        st.info("Model metrics not found. Train a model to see metrics.")

    # --- Prediction ---
    st.header("Predict a Message")
    text = st.text_area("Enter message to classify", height=150)
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5)

    if st.button("Predict"):
        if not model_ready:
            st.error("Model or vectorizer not usable. Cannot predict.")
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
