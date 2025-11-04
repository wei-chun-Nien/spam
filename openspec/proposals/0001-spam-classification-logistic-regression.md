<!--
OpenSpec proposal: Spam classification baseline (Logistic Regression + Streamlit)
Status: Draft
-->
# 0001: Spam classification — Phase 1 baseline (Logistic Regression + Streamlit)

## Summary

Build a Phase-1 baseline for spam classification using classical ML. The
pipeline trains a Logistic Regression binary classifier on a public SMS spam
dataset and exposes results in a Streamlit demo that displays data distribution,
training results, and an input box to predict whether a message is spam. The
app also exposes an adjustable decision threshold.

## Data

Primary dataset (baseline):
https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

Note: confirm licensing before broad redistribution.

## Phase 1 scope
- Data ingestion and preprocessing (download & cache CSV, parse label/text, basic cleaning)
- TF-IDF feature extraction (configurable max_features, ngram_range)
- Logistic Regression training and evaluation (accuracy, precision, recall, F1, ROC AUC)
- Streamlit UI: data overview, metrics, confusion matrix, ROC curve, prediction box, threshold slider
- Save artifacts: `vectorizer.pkl`, `model.pkl`, `metrics.json`

## Acceptance criteria / Tests
- Pipeline runs and produces model artifacts.
- Streamlit app loads and shows data distribution, metrics, confusion matrix, and ROC curve.
- Predict endpoint in app returns probability in [0,1] and label based on chosen threshold.
- Unit tests: preprocessing, vectorizer transform shapes, model predict_proba output shapes, threshold application.

## Implementation plan
1. Create `requirements.txt` (streamlit, scikit-learn, pandas, matplotlib, seaborn)
2. Implement `src/spam_classifier/` modules:
   - `data.py` (download & parsing)
   - `preprocess.py` (cleaning & vectorizer)
   - `train.py` (train & evaluate)
   - `predict.py` (load artifacts & predict with threshold)
3. Implement `apps/streamlit_app.py` for demo UI
4. Add `tests/test_pipeline.py`
5. Add README with run instructions

## Open questions
- Primary model: Logistic Regression (recommended) vs SVM? (Logistic allows probability output which helps thresholding.)
- Do you want a REST predict endpoint in addition to Streamlit?

## Timeline (estimate)
- Pipeline + tests: 1–2 days
- Streamlit demo: 1 day
- Optional containerization & CI: +1 day

