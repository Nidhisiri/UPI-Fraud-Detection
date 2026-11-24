# UPI-Fraud-Detection
# UPI Fraud Detection

**CSL236 – Introduction to AI & ML | The NorthCap University**
Student: Nidhi Sharma (24csu140)

A machine learning project that detects fraudulent UPI transactions and explains each prediction with SHAP (Explainable AI). It includes a Jupyter Notebook with the full ML workflow and an interactive Streamlit dashboard.

## Features
- Data cleaning, EDA and preprocessing (missing values, encoding, scaling, binning demos, feature selection)
- Two final models: **Model A** (4 behaviour features, Random Forest) and **Model B** (best 5 features, Decision Tree)
- Evaluation with accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix and ROC curves
- Risk levels (Low / Medium / High) from the fraud probability
- SHAP explanations: global (beeswarm, bar) and individual (waterfall)
- Separate K-Means clustering analysis
- Streamlit dashboard with two modes:
  - **Everyday user:** Check Your UPI Transaction, transaction history, compare two transactions, downloadable report, How the AI Decides
  - **Analytics:** Data Exploration, Preprocessing Pipeline, Model Performance, Explainable AI (SHAP)

## Dataset
`fraud_dataset.csv`: 26,393 transactions, 65 columns, target `is_fraud` (17.2% fraud). The dataset is synthetic.

## Results (test set, Model B – Decision Tree)
| Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|
| 0.975 | 0.930 | 0.925 | 0.928 | 0.995 |

## Project structure
```
UPI_Fraud_Detection_Clean_Project_v3_with_outputs.ipynb   # full ML workflow
fraud_dataset.csv                                          # dataset
UPI_Fraud_Detection/
    app.py                 # Streamlit app entry point
    requirements.txt
    upi_app/               # UI and integration code
    models/                # trained models exported from the notebook
    data/                  # cleaned dataset and SHAP background data
    .streamlit/config.toml # dark theme
```

## How to run
1. Clone or download this repository.
2. Install the requirements:
   ```
   cd UPI_Fraud_Detection
   pip install -r requirements.txt
   ```
3. Start the app:
   ```
   streamlit run app.py
   ```
4. Open http://localhost:8501

To recreate the `models/` and `data/` folders, run the notebook from top to bottom (section 14 exports them).

## Disclaimer
This is an ML-based assessment for learning purposes. The prediction is an estimate, not a guarantee that a transaction is safe or fraudulent.
