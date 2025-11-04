# Spam classification demo (SVM + Streamlit)

This workspace contains a small spam classification baseline and a Streamlit demo.

Quick start (Windows PowerShell):

1. Create and activate a virtual environment (optional):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train the model:

```powershell
python -m spam_classifier.train
```

4. Run Streamlit app:

```powershell
streamlit run apps/streamlit_app.py
```

Model artifacts will be written to `models/`.
