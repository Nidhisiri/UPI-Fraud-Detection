"""Integration layer. Loads what the notebook exported and reuses it. NOTHING here trains the final models.

The notebook's own steps are only *replayed* (read-only) so the app can show the same X_test, risk table and
SHAP explanations the notebook produced. Each replayed block names the notebook cells it mirrors."""
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS, DATA = ROOT / "models", ROOT / "data"

REQUIRED = [MODELS / "model_B_version3.pkl", MODELS / "model_A_version2.pkl", MODELS / "model_info.pkl",
            MODELS / "evaluation_results.pkl", DATA / "dataset.csv"]

# Notebook cell 69 (columns removed before modelling) - copied unchanged.
COLUMNS_TO_REMOVE = ['transaction_id', 'user_id', 'merchant_id', 'device_id', 'ip_address', 'location',
                     'description', 'request_description', 'business_name_match', 'timestamp',
                     'handle_verification_status']

# Facts printed by the notebook on the RAW file (not available inside dataset.csv). Source = notebook output.
NB_FACTS = dict(raw_rows=26393, raw_cols=65, cleaned_cols=59,
                constant_cols=['request_description', 'relationship_to_requester', 'upi_handle_age',
                               'handle_contains_official_terms', 'social_media_presence'],
                url_referrer_missing_pct=97.13, request_description_missing_pct=97.23,
                leak_unverified_fraud=4545, leak_verified_genuine=21848)

# ------------------------------------------------------------------ friendly wording
FRIENDLY = {
    "receiver_transaction_history": dict(
        label="Receiver's past transactions", kind="int",
        question="How many past transactions does the receiver's account have?",
        help="The number of earlier transactions recorded for the receiver. Accounts with little history are harder to trust."),
    "unusual_transaction_amount_flag": dict(
        label="Unusual amount for you", kind="bool",
        question="Is this amount unusually high compared with what you normally send?",
        help="Answer Yes if this payment is much larger than your usual payments."),
    "handle_registration_pattern_recent": dict(
        label="Receiver's UPI ID is new", kind="bool",
        question="Was the receiver's UPI ID created recently?",
        help="Answer Yes if the receiver's UPI ID was registered only recently (for example, it is a new or unfamiliar ID)."),
    "keyboard_input_speed": dict(
        label="Typing speed", kind="float",
        question="Typing speed while entering the payment",
        help="A behaviour signal normally measured by the payment app (relative speed). If you do not know it, use the typical value."),
    "authentication_attempt_count": dict(
        label="PIN / OTP attempts", kind="int",
        question="How many PIN / OTP attempts were needed?",
        help="Number of times the PIN or OTP had to be entered for this payment."),
}


def pretty(feat):
    return FRIENDLY.get(feat, {}).get("label", feat.replace("_", " ").capitalize())


def value_text(feat, v):
    kind = FRIENDLY.get(feat, {}).get("kind", "float")
    if kind == "bool":
        return "Yes" if float(v) >= 0.5 else "No"
    if kind == "int":
        return f"{int(round(float(v)))}"
    return f"{float(v):.2f}"


def missing_files():
    return [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]


# ------------------------------------------------------------------ artifacts
@st.cache_resource(show_spinner="Loading trained models…")
def artifacts():
    a = dict(
        model_A=joblib.load(MODELS / "model_A_version2.pkl"),
        model_B=joblib.load(MODELS / "model_B_version3.pkl"),
        info=joblib.load(MODELS / "model_info.pkl"),
        evalr=joblib.load(MODELS / "evaluation_results.pkl"),
        kmeans=None,
    )
    if (MODELS / "kmeans_model.pkl").exists():
        a["kmeans"] = joblib.load(MODELS / "kmeans_model.pkl")
    return a


@st.cache_resource(show_spinner="Loading dataset…")
def dataset():
    return pd.read_csv(DATA / "dataset.csv")


@st.cache_resource(show_spinner="Preparing data…")
def replay():
    """Read-only replay of notebook cells 49-73 on the exported cleaned dataset.
    Same steps, same order, same settings (random_state=42, stratified 80/20) -> same X_train / X_test."""
    from sklearn.model_selection import train_test_split
    df = dataset()
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    Xm = X.drop(columns=[c for c in COLUMNS_TO_REMOVE if c in X.columns])
    missing_before = int(Xm.isnull().sum().sum())                      # evidence for the leakage audit
    for col in Xm.columns:                                             # cell 70 (unchanged)
        if pd.api.types.is_numeric_dtype(Xm[col]):
            Xm[col] = Xm[col].fillna(Xm[col].median())
        else:
            Xm[col] = Xm[col].fillna(Xm[col].mode()[0])
    text_cols = Xm.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    n_before_encoding = Xm.shape[1]
    Xm = pd.get_dummies(Xm, columns=text_cols, drop_first=True).astype(float)   # cell 72
    X_train, X_test, y_train, y_test = train_test_split(Xm, y, test_size=0.2, random_state=42, stratify=y)
    return dict(X_model=Xm, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, text_cols=text_cols,
                missing_before_fill=missing_before, n_before_encoding=n_before_encoding, n_removed=len(
                    [c for c in COLUMNS_TO_REMOVE if c in X.columns]))


def replay_check():
    """Does the replayed X_test reproduce the confusion matrix the notebook exported?"""
    a, r = artifacts(), replay()
    f = a["info"]["B"]["features"]
    best = a["evalr"]["B"]["best_model"]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(r["y_test"], a["model_B"].predict(r["X_test"][f])).tolist()
    return cm == a["evalr"]["B"]["models"][best]["confusion_matrix"]


# ------------------------------------------------------------------ prediction
def risk_level(p):
    lim = artifacts()["info"]["risk_limits"]          # notebook cell 103: low 0.50 / high 0.80
    return "Low" if p <= lim["low"] else ("Medium" if p <= lim["high"] else "High")


def reference_stats():
    """Per-feature median/range for genuine vs fraud, from the notebook's training split (real data)."""
    r, f = replay(), artifacts()["info"]["B"]["features"]
    out = {}
    for c in f:
        g, fr = r["X_train"].loc[r["y_train"] == 0, c], r["X_train"].loc[r["y_train"] == 1, c]
        out[c] = dict(g_med=float(g.median()), f_med=float(fr.median()), g_min=float(g.min()), g_max=float(g.max()),
                      f_min=float(fr.min()), f_max=float(fr.max()))
    return out


@st.cache_resource(show_spinner="Preparing explainer…")
def explainer():
    import shap
    inner = artifacts()["model_B"].named_steps["model"]
    if not hasattr(inner, "feature_importances_"):
        raise RuntimeError("SHAP TreeExplainer needs a tree model (Decision Tree / Random Forest).")
    return shap.TreeExplainer(inner)


def analyze(values, assumed):
    """values: {feature: number}. Uses the exported Model B pipeline + the notebook's SHAP lines."""
    a = artifacts()
    f = a["info"]["B"]["features"]
    row = pd.DataFrame([[float(values[c]) for c in f]], columns=f)
    model = a["model_B"]
    prob = float(model.predict_proba(row)[0, 1])
    pred = int(model.predict(row)[0])
    ex = explainer()(row)[:, :, 1]                      # notebook: fraud_explanation = explanation[:, :, 1]
    sv = ex.values[0]
    contrib = [dict(feature=c, label=pretty(c), value=float(row[c].iloc[0]), vtext=value_text(c, row[c].iloc[0]),
                    shap=float(s), assumed=c in assumed) for c, s in zip(f, sv)]
    contrib.sort(key=lambda d: abs(d["shap"]), reverse=True)
    return dict(prob=prob, pred=pred, risk=risk_level(prob), base=float(ex.base_values[0]), contrib=contrib,
                inputs={c: float(row[c].iloc[0]) for c in f}, assumed=list(assumed),
                time=datetime.now().strftime("%d %b %Y, %H:%M:%S"))


def sentence(d):
    pts = d["shap"] * 100
    if abs(pts) < 1:
        return f"{d['label']} ({d['vtext']}) had almost no effect on the result."
    way = "increased" if pts > 0 else "reduced"
    return f"{d['label']} ({d['vtext']}) {way} the fraud estimate by about {abs(pts):.0f} percentage points."


def explanation_text(res):
    """Plain-English summary built only from the real SHAP values of this transaction."""
    up = [d for d in res["contrib"] if d["shap"] * 100 >= 1]
    down = [d for d in res["contrib"] if d["shap"] * 100 <= -1]
    fraud = res["pred"] == 1
    lines = []
    if fraud:
        lines.append("The model flagged this transaction mainly because: " +
                     ("; ".join(d["label"].lower() for d in up[:3]) if up else "the combined pattern of your answers") + ".")
        if down:
            lines.append("Factors that pointed towards genuine: " + "; ".join(d["label"].lower() for d in down[:2]) + ".")
    else:
        lines.append("The model classified this transaction as legitimate mainly because: " +
                     ("; ".join(d["label"].lower() for d in down[:3]) if down else "no single factor stood out") + ".")
        if up:
            lines.append("Factors that still pointed towards fraud: " + "; ".join(d["label"].lower() for d in up[:2]) + ".")
    lines.append(f"The model starts from a neutral {res['base']*100:.0f}% estimate and each factor moves it up or down, "
                 f"ending at {res['prob']*100:.1f}%.")
    return lines


def safety_tips(res, answers):
    tips = ["Never share your UPI PIN or OTP with anyone - not even someone claiming to be from your bank.",
            "Verify the receiver's name and UPI ID on the confirmation screen before you pay.",
            "Be cautious of unexpected payment or 'collect' requests, especially ones that create urgency.",
            "A 'legitimate' result is not a guarantee of safety. Always double-check the details yourself."]
    if res["pred"] == 1:
        tips.insert(0, "Pause before paying. Contact the receiver through a number you already trust to confirm this payment.")
        tips.append("If you think you have been defrauded, call 1930 (National Cyber Crime Helpline, India) or report at cybercrime.gov.in, and inform your bank immediately.")
    if answers.get("handle_registration_pattern_recent", 0) >= 0.5:
        tips.append("The receiver's UPI ID is new - confirm it belongs to the person or business you intend to pay.")
    if answers.get("unusual_transaction_amount_flag", 0) >= 0.5:
        tips.append("This amount is unusual for you - consider sending a small test amount first.")
    return tips


# ------------------------------------------------------------------ SHAP (global) - notebook cell 109
@st.cache_resource(show_spinner="Computing SHAP values for 1,000 test transactions…")
def global_shap():
    a, r = artifacts(), replay()
    f = a["info"]["B"]["features"]
    X_shap = r["X_test"][f].sample(n=min(1000, len(r["X_test"])), random_state=42)
    expl = explainer()(X_shap)[:, :, 1]
    real = a["model_B"].predict_proba(X_shap)[:, 1]
    gap = float(np.abs(real - (expl.base_values + expl.values.sum(axis=1))).max())     # cell 112 check
    return dict(X=X_shap, expl=expl, gap=gap)


def explain_rows(X_rows):
    a = artifacts()
    f = a["info"]["B"]["features"]
    return explainer()(X_rows[f])[:, :, 1]


# ------------------------------------------------------------------ history helpers
def history():
    return st.session_state.setdefault("history", [])


def add_history(res, amount):
    h = history()
    h.append(dict(res, id=len(h) + 1, amount=amount))


def short_factors(res, n=2):
    top = [d for d in res["contrib"] if abs(d["shap"]) * 100 >= 1][:n]
    return ", ".join(("▲ " if d["shap"] > 0 else "▼ ") + d["label"] for d in top) or "-"
