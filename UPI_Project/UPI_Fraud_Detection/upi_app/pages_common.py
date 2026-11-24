"""Overview (landing) and About pages."""
import streamlit as st

from . import backend as B
from .theme import C, html, icon, kpi_row, note, page_header, qa_block


def _go(mode, page):
    st.session_state["mode"] = mode
    st.session_state["nav_page"] = page


def page_overview():
    ds, a = B.dataset(), B.artifacts()
    n, fr = len(ds), int(ds["is_fraud"].sum())
    best = a["evalr"]["B"]["results"][0]
    html(f'<div class="hero"><span class="pill v">Explainable AI</span><span class="pill c">UPI fraud detection</span>'
         '<h1>UPI Fraud Detection Platform</h1>'
         '<p>AI-powered transaction analysis with Explainable AI. Check a payment in seconds, or open the full analytics '
         'behind the model: data, preprocessing, evaluation and SHAP.</p></div>')
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        html(f'<div class="card link"><h4>{icon("shield", 22, C["cyan"])}&nbsp; Check Your UPI Transaction</h4>'
             '<p>For everyone. Enter a few details, get an assessment and a plain-English explanation of <i>why</i>. No technical knowledge needed.</p></div>')
        st.button("Open the checker", key="ov_user", type="primary", on_click=_go, args=("Everyday user", "Check Your UPI Transaction"))
    with c2:
        html(f'<div class="card link"><h4>{icon("chart", 22, C["violet"])}&nbsp; Advanced Analytics</h4>'
             '<p>For students, teachers and evaluators. Dataset analysis, the preprocessing pipeline, model performance and SHAP.</p></div>')
        st.button("Open analytics", key="ov_tech", on_click=_go, args=("Analytics", "Data Exploration"))
    st.write("")
    st.markdown("#### Project at a glance")
    kpi_row([("Total transactions", f"{n:,}", "cleaned dataset"), ("Fraudulent", f"{fr:,}", f"{fr/n*100:.1f}% of all"),
             ("Legitimate", f"{n-fr:,}", f"{(n-fr)/n*100:.1f}% of all"),
             ("Best model F1 (test)", f"{best['F1-Score']:.3f}", f"{best['Model']} · {len(a['info']['B']['features'])} features")])
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        html('<div class="card"><h4>Problem statement</h4><p>UPI makes payments instant, which also makes fraud fast. The goal is to classify a transaction as '
             'genuine or fraudulent from behavioural and account signals, and to <b>explain</b> each decision so that people can trust it.</p></div>')
    with c2:
        html(f'<div class="card"><h4>About the dataset</h4><p>{n:,} transactions, {ds.shape[1]-1} input columns after cleaning. It is a '
             'synthetic (computer-generated) research dataset, so results show how the method works, not real-world accuracy.</p></div>')


def page_about():
    page_header("About the Project", "UPI Fraud Detection using Machine Learning and Explainable AI (SHAP) - CSL236, Introduction to AI & ML.")
    html('<div class="card"><h4>Prepared by</h4><p>Nidhi Sharma · 24csu140 (as written in the project notebook)</p></div>')
    st.write("")
    st.markdown("#### The pipeline (from the notebook)")
    steps = ["Load data → basic commands → cleaning (constant columns, 97%-empty columns)", "EDA", "Remove ID/free-text columns and the leaking column; median/mode fill; one-hot encoding; 80/20 stratified split",
             "Feature selection: Random Forest importance + Mutual Information → 13 features", "Model A (4 behaviour features) and Model B (best 5 of 1,287 combinations)",
             "Logistic Regression, Decision Tree, Random Forest with GridSearchCV; evaluation on the unseen test set", "Risk score (Low / Medium / High)", "SHAP explanations of Model B", "K-Means clustering (separate analysis)"]
    for i, s in enumerate(steps, 1):
        st.markdown(f"{i}. {s}")
    st.markdown("#### How this app is organised")
    st.markdown("- `models/` and `data/` - files exported by your notebook (section 14). The app only **loads** them.\n"
                "- `app.py` and `upi_app/` - the new user interface and integration code.\n"
                "- The notebook's data cleaning, feature selection, training and SHAP are **not** re-implemented or changed.")
    st.markdown("#### Honest notes and limitations")
    ok = B.replay_check()
    r = B.replay()
    st.markdown(f"- **Replay check:** the app rebuilds the same test set as the notebook and its predictions "
                f"{'reproduce' if ok else 'do NOT reproduce'} the exported confusion matrix.")
    st.markdown(f"- **Missing values and the split:** the notebook fills missing values before the train/test split, but the modelling columns contain "
                f"**{r['missing_before_fill']} missing values** at that point, so the fill changes nothing and no information leaks. (Explain this in the viva; a cleaner order would be split → fit imputer on train.)")
    st.markdown("- **Synthetic data:** some columns almost decide fraud by themselves, which is why the notebook builds Model A/B on fewer features.")
    st.markdown("- **Model B inputs:** it uses behavioural signals (e.g. typing speed, receiver history), not the rupee amount. The checker therefore asks for those signals.")
    st.markdown("- **Outliers:** the notebook did not remove outliers; the Preprocessing page investigates them and explains why removal was not applied.")
    st.markdown("- **Not a bank-grade system.** Results are an ML estimate, never a guarantee.")
