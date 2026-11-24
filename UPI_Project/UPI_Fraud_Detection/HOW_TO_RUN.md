# UPI Fraud Detection Platform - how to run

## 1. Where to put the files
Copy everything from this download INTO your existing `UPI_Fraud_Detection` folder (the one your notebook created in section 14), so it looks like:

```
UPI_Fraud_Detection/
    app.py                  <- new
    requirements.txt        <- new
    HOW_TO_RUN.md           <- new
    .streamlit/config.toml  <- new (dark theme)
    upi_app/                <- new (UI + integration code)
    models/                 <- YOUR files from the notebook (unchanged)
    data/                   <- YOUR files from the notebook (unchanged)
```
(`.streamlit` is a hidden folder - make sure it is copied too.) Your notebook is not touched.

## 2. Install (once)
Open Command Prompt in the `UPI_Fraud_Detection` folder:
```
pip install -r requirements.txt
```
Use the same Python/environment in which you ran the notebook, so the saved models load without version warnings.

## 3. Run
```
streamlit run app.py
```
Then open http://localhost:8501 (it usually opens automatically).

## 4. Using it
- Sidebar → **Everyday user**: Check Your UPI Transaction, Transaction History (+ compare), How the AI Decides.
- Sidebar → **Analytics**: Data Exploration, Preprocessing Pipeline, Model Performance, Explainable AI (SHAP).

## Troubleshooting
- "files not found" → run section 14 of the notebook so `models/` and `data/` exist next to `app.py`.
- "SHAP is not available" → `pip install shap`.
- Report downloads as .html instead of .pdf → `pip install reportlab`.
