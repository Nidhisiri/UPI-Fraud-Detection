"""Mode A - Advanced Analytics: EDA, preprocessing, model performance, SHAP."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import backend as B
from .pages_user import contribution_bar, factor_list
from .theme import C, html, kpi_row, note, page_header, qa_block, show_df, show_plot, style_fig

CLS = {0: "Genuine", 1: "Fraud"}
CMAP = {"Genuine": C["genuine"], "Fraud": C["fraud"]}


def _cls(df):
    d = df.copy()
    d["Class"] = d["is_fraud"].map(CLS)
    return d


# ================================================================== Data Exploration
def page_eda():
    page_header("Data Exploration", "Exploratory data analysis on the cleaned dataset. Use the filters to slice the data; every chart updates.")
    df = B.dataset()
    f1, f2, f3 = st.columns(3)
    types = f1.multiselect("Transaction type", sorted(df["transaction_type"].unique()), default=sorted(df["transaction_type"].unique()))
    lo, hi = float(df["amount"].min()), float(df["amount"].max())
    amt = f2.slider("Amount range (₹)", lo, hi, (lo, hi))
    hrs = f3.slider("Hour of day", 0, 23, (0, 23))
    d = df[df["transaction_type"].isin(types) & df["amount"].between(*amt) & df["transaction_time_of_day"].between(*hrs)]
    if d.empty:
        note("No transactions match these filters.", "warn")
        return
    kpi_row([("Transactions shown", f"{len(d):,}"), ("Fraudulent", f"{int(d['is_fraud'].sum()):,}"),
             ("Legitimate", f"{int((1-d['is_fraud']).sum()):,}"), ("Fraud rate", f"{d['is_fraud'].mean()*100:.1f}%")])
    st.write("")
    dc = _cls(d)
    tabs = st.tabs(["Fraud vs genuine", "Amount", "Type & merchant", "Time", "Security signals", "Correlation", "Feature explorer"])

    with tabs[0]:
        vc = dc["Class"].value_counts().reindex(["Genuine", "Fraud"]).fillna(0)
        fig = go.Figure(go.Pie(labels=vc.index, values=vc.values, hole=.62, marker=dict(colors=[CMAP[k] for k in vc.index], line=dict(color=C["bg"], width=3)),
                               textinfo="label+percent"))
        show_plot(style_fig(fig, 340, "Class balance", legend=False), key="eda_pie")
        note("Fraud is the minority class (about 17%), which is why the models use <b>class_weight='balanced'</b> and why Precision/Recall/F1 matter more than Accuracy.")

    with tabs[1]:
        c1, c2 = st.columns(2)
        logx = c1.checkbox("Log scale", value=False, key="eda_log")
        fig = px.histogram(dc, x="amount", color="Class", nbins=60, barmode="overlay", opacity=.75, color_discrete_map=CMAP, log_x=logx)
        show_plot(style_fig(fig, 340, "Transaction amount distribution"), key="eda_amt")
        fig = px.box(dc, x="Class", y="amount", color="Class", color_discrete_map=CMAP, points=False)
        show_plot(style_fig(fig, 320, "Amount by class", legend=False), key="eda_box")
        g = d.groupby("is_fraud")["amount"].agg(["mean", "median", "min", "max"]).round(0)
        g.index = [CLS[i] for i in g.index]
        show_df(g.reset_index(names="Class"))

    with tabs[2]:
        c1, c2 = st.columns(2)
        t = d.groupby("transaction_type")["is_fraud"].agg(["size", "mean"]).reset_index()
        fig = px.bar(t, x="transaction_type", y=t["mean"] * 100, text=(t["mean"] * 100).round(1), color_discrete_sequence=[C["violet"]], labels={"y": "Fraud rate (%)", "transaction_type": ""})
        with c1:
            show_plot(style_fig(fig, 320, "Fraud rate by transaction type", legend=False), key="eda_type")
        m = d.groupby("merchant_category_code")["is_fraud"].mean().sort_values() * 100
        fig = px.bar(x=m.values, y=m.index, orientation="h", text=m.round(1).values, color_discrete_sequence=[C["indigo"]], labels={"x": "Fraud rate (%)", "y": ""})
        with c2:
            show_plot(style_fig(fig, 320, "Fraud rate by merchant category", legend=False), key="eda_merch")
        note("All <b>collection_request</b> transactions in this dataset are fraud, and the <b>unknown</b> merchant category is 100% fraud - strong, dataset-specific patterns.")

    with tabs[3]:
        h = d.groupby("transaction_time_of_day")["is_fraud"].agg(["size", "mean"]).reset_index()
        fig = go.Figure(go.Scatter(x=h["transaction_time_of_day"], y=h["mean"] * 100, mode="lines+markers", line=dict(color=C["violet"], width=2.5),
                                   marker=dict(size=8, color=C["cyan"]), hovertemplate="Hour %{x}:00<br>%{y:.1f}% fraud<extra></extra>"))
        fig.update_xaxes(title="Hour of day", dtick=2)
        fig.update_yaxes(title="Fraud rate (%)")
        show_plot(style_fig(fig, 340, "Fraud rate by hour of day", legend=False), key="eda_hour")
        note("The <code>timestamp</code> column only holds minutes:seconds, so the real hour comes from <code>transaction_time_of_day</code> (as in the notebook).")

    with tabs[4]:
        rows = []
        for c in ["unusual_device_flag", "unusual_ip_flag", "unusual_location_flag", "unusual_transaction_amount_flag"]:
            for v, r in d.groupby(c)["is_fraud"].mean().items():
                rows.append(dict(Signal=c.replace("_", " "), Flag="Yes" if v == 1 else "No", FraudRate=r * 100))
        fig = px.bar(pd.DataFrame(rows), x="Signal", y="FraudRate", color="Flag", barmode="group", color_discrete_sequence=[C["muted"], C["fraud"]], labels={"FraudRate": "Fraud rate (%)"})
        show_plot(style_fig(fig, 340, "Fraud rate when a warning flag is raised"), key="eda_flags")
        fig = px.box(dc, x="Class", y="authentication_attempts", color="Class", color_discrete_map=CMAP)
        show_plot(style_fig(fig, 300, "Authentication attempts by class", legend=False), key="eda_auth")

    with tabs[5]:
        num = d.select_dtypes(include=np.number)
        num = num.loc[:, num.nunique() > 1]
        n_top = st.slider("Number of features", 8, min(40, num.shape[1]), min(18, num.shape[1]), key="eda_ntop")
        cor = num.corr()
        top = cor["is_fraud"].abs().sort_values(ascending=False).head(n_top).index
        fig = px.imshow(cor.loc[top, top], color_continuous_scale=[C["genuine"], C["surface2"], C["fraud"]], zmin=-1, zmax=1, aspect="auto")
        show_plot(style_fig(fig, 560, "Correlation heatmap (features most related to fraud)", legend=False), key="eda_corr")
        cw = cor["is_fraud"].drop("is_fraud").sort_values()
        pick = pd.concat([cw.head(6), cw.tail(6)])
        fig = go.Figure(go.Bar(x=pick.values, y=pick.index, orientation="h", marker_color=[C["fraud"] if v > 0 else C["genuine"] for v in pick.values]))
        show_plot(style_fig(fig, 380, "Strongest correlations with fraud", legend=False), key="eda_corr_bar")

    with tabs[6]:
        numc = [c for c in d.select_dtypes(include=np.number).columns if c != "is_fraud" and d[c].nunique() > 1]
        catc = [c for c in d.select_dtypes(exclude=[np.number]).columns if 1 < d[c].nunique() <= 12 and c not in ("transaction_id", "user_id", "merchant_id", "timestamp")]
        kind = st.radio("Feature type", ["Numerical", "Categorical"], horizontal=True, key="eda_kind")
        if kind == "Numerical":
            c = st.selectbox("Feature", numc, key="eda_num")
            fig = px.histogram(dc, x=c, color="Class", barmode="overlay", opacity=.75, nbins=50, color_discrete_map=CMAP)
            show_plot(style_fig(fig, 340, f"{c} by class"), key="eda_numhist")
        else:
            c = st.selectbox("Feature", catc, key="eda_cat")
            g = dc.groupby([c, "Class"]).size().reset_index(name="count")
            fig = px.bar(g, x=c, y="count", color="Class", barmode="group", color_discrete_map=CMAP)
            show_plot(style_fig(fig, 340, f"{c} by class"), key="eda_cathist")


# ================================================================== Preprocessing
@st.cache_resource(show_spinner="Re-running the notebook's feature-selection step…")
def _selection():
    """Replay of notebook cells 75-77 (same settings) on the training split only."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif
    r = B.replay()
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1).fit(r["X_train"], r["y_train"])
    rfi = pd.DataFrame({"Feature": r["X_train"].columns, "RF importance": rf.feature_importances_}).sort_values("RF importance", ascending=False)
    mi = pd.DataFrame({"Feature": r["X_train"].columns, "Mutual information": mutual_info_classif(r["X_train"], r["y_train"], random_state=42)}).sort_values("Mutual information", ascending=False)
    sel = list(dict.fromkeys(rfi.head(10)["Feature"].tolist() + mi.head(10)["Feature"].tolist()))
    return rfi, mi, sel


def page_preprocessing():
    page_header("Preprocessing Pipeline", "Every step the notebook performed - what was done, which columns, why, and the result. Numbers are computed from the real data.")
    df, r, a, F = B.dataset(), B.replay(), B.artifacts(), B.NB_FACTS
    kpi_row([("Raw columns", F["raw_cols"], "notebook"), ("After cleaning", F["cleaned_cols"], "constant / empty dropped"),
             ("Model matrix", r["X_model"].shape[1], "after encoding"), ("Selected", 13, "RF + mutual information"),
             ("Final (A / B)", f"{len(a['info']['A']['features'])} / {len(a['info']['B']['features'])}", "features")])
    st.write("")
    tabs = st.tabs(["Dataset & types", "Missing values", "Encoding", "Scaling & binning", "Outliers", "Feature selection", "Leakage audit"])

    with tabs[0]:
        st.markdown(f"**Shape in the exported file:** {df.shape[0]:,} rows × {df.shape[1]} columns (the leaking column was already removed).")
        dt = df.dtypes.astype(str).value_counts().rename_axis("Data type").reset_index(name="Columns")
        c1, c2 = st.columns([1, 2])
        with c1:
            show_df(dt)
        with c2:
            show_df(pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values, "Unique values": df.nunique().values}))
        qa_block("Removed constant columns and the 97%-empty <code>url_referrer</code>.", ", ".join(F["constant_cols"]) + ", url_referrer",
                 "A column with one value (or almost no values) cannot help a model tell fraud from genuine.", f"{F['raw_cols']} → {F['cleaned_cols']} columns.")
        st.write("")
        qa_block("Removed IDs, free text, location/IP/device identifiers and <code>timestamp</code> before modelling.", ", ".join(c for c in B.COLUMNS_TO_REMOVE),
                 "Identifiers only name a row; the timestamp holds minutes:seconds (not a real date), and one column leaks the answer.", f"{r['n_removed']} columns removed here (the leaking column was already dropped from the exported file) → {r['n_before_encoding']} remain.")

    with tabs[1]:
        miss = df.isnull().sum()
        miss = miss[miss > 0]
        show_df(pd.DataFrame({"Column": miss.index, "Missing": miss.values, "Percent": (miss.values / len(df) * 100).round(2)})) if len(miss) else st.write("No missing values in the exported file.")
        qa_block("Checked for missing values and duplicates; dropped columns that were ~97% empty; filled the rest.",
                 f"url_referrer ({F['url_referrer_missing_pct']}% empty), request_description ({F['request_description_missing_pct']}% empty) → dropped. Numeric → median, text → most common value.",
                 "Dropping is safer than inventing 97% of a column. Median is robust to extreme values; the mode is the natural fill for categories.",
                 f"Missing values in the modelling columns before filling: <b>{r['missing_before_fill']}</b>. Duplicate rows: <b>{int(df.duplicated().sum())}</b>.")
        note(f"<code>timestamp</code> has {int(df['timestamp'].isnull().sum()):,} unreadable values, but it is removed before modelling, so it never reaches the model.")

    with tabs[2]:
        tc = r["text_cols"]
        show_df(pd.DataFrame({"Text column": tc, "Unique values": [df[c].nunique() for c in tc], "Values": [", ".join(map(str, df[c].dropna().unique()[:4])) for c in tc]}))
        qa_block("One-Hot Encoding with <code>drop_first=True</code>; every text column becomes 0/1 columns.", ", ".join(tc),
                 "The text columns are nominal (no natural order), so numbers like 1,2,3 would invent an order. Dropping the first level avoids redundant columns.",
                 f"{r['n_before_encoding']} → {r['X_model'].shape[1]} columns.")
        st.caption("Label, Ordinal, Frequency, Count and Hash encoding were also demonstrated in the notebook (section 5.1) on copies of the data; they are not used by the models.")

    with tabs[3]:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(r["X_train"][["amount"]])
        z = pd.DataFrame({"Original amount": r["X_test"]["amount"].values, "Standardised (fit on train)": sc.transform(r["X_test"][["amount"]]).ravel()}).describe().round(2)
        show_df(z.reset_index(names="stat"))
        qa_block("StandardScaler sits <b>inside the Logistic Regression pipeline</b> only (fit on the training data). Decision Tree and Random Forest use no scaling.",
                 "All numeric inputs of Logistic Regression", "Logistic Regression is sensitive to feature scale; tree models split on thresholds, so scale does not matter to them.",
                 "Scaling is learned from training data and applied to test data automatically by the Pipeline.")
        st.write("")
        b = pd.cut(df["session_duration"], [-np.inf, 100, 300, np.inf], labels=["Low", "Medium", "High"])
        bt = pd.crosstab(b, df["is_fraud"].map(CLS)).reset_index().rename(columns={"session_duration": "Session level"})
        show_df(bt)
        qa_block("Binning demonstration: session duration → Low / Medium / High (<code>pd.cut</code>).", "session_duration",
                 "Shows how discretisation turns a number into ordered categories.", "<b>Demonstration only</b> - the binned column is not used by the final models (trees already find their own split points).")

    with tabs[4]:
        cols = [c for c in ["amount", "session_duration", "transaction_amount_vs_sender_history", "geographic_disparity", "receiver_account_age", "receiver_transaction_history"] if c in df]
        rows = []
        for c in cols:
            q1, q3 = df[c].quantile([.25, .75])
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            m = (df[c] < lo) | (df[c] > hi)
            rows.append(dict(Column=c, Outliers=int(m.sum()), Percent=round(m.mean() * 100, 2), FraudShareOfOutliers=round(df.loc[m, "is_fraud"].mean() * 100, 1) if m.any() else 0.0))
        show_df(pd.DataFrame(rows).rename(columns={"FraudShareOfOutliers": "Fraud % among outliers"}))
        fig = px.box(_cls(df), x="Class", y="amount", color="Class", color_discrete_map=CMAP, log_y=True, points="outliers")
        show_plot(style_fig(fig, 340, "Amount by class (log scale) - IQR outliers", legend=False), key="pp_out")
        qa_block("Outliers investigated with the IQR rule (1.5 × IQR) and box-plots.", ", ".join(cols),
                 "Extreme amounts here are mostly <b>fraud</b>, so they carry signal. Removing or capping them would erase fraud evidence; Decision Tree / Random Forest are also not sensitive to extreme values.",
                 "<b>No rows were removed or capped</b> (the notebook did not apply outlier treatment - this is a documented decision, not an omission).")

    with tabs[5]:
        rfi, mi, sel = _selection()
        c1, c2 = st.columns(2)
        with c1:
            show_df(rfi.head(10).round(4))
        with c2:
            show_df(mi.head(10).round(4))
        st.markdown(f"**Selected (top 10 of each, joined, {len(sel)} features):** " + ", ".join(f"`{s}`" for s in sel))
        st.markdown(f"**Model A (4 features):** {', '.join(a['info']['A']['features'])}  \n**Model B (5 features):** {', '.join(a['info']['B']['features'])}")
        qa_block("Two feature-selection methods on the <b>training data only</b>; then Model B = best of all 1,287 five-feature combinations (5-fold CV on training data); Model A removes 6 'giveaway' columns.",
                 "52 encoded columns → 13 → 5 (B) / 4 (A)", "Fewer, meaningful features are easier to explain and less likely to overfit. Using two methods reduces dependence on one viewpoint.",
                 "PCA is not used for the classifier (it would destroy feature names, which SHAP needs); it appears only to plot K-Means clusters.")

    with tabs[6]:
        ok = B.replay_check()
        audit = [("Leaking column removed", f"<code>handle_verification_status</code> matched the label 100% ({F['leak_unverified_fraud']:,} unverified = fraud, {F['leak_verified_genuine']:,} verified = genuine) → removed.", "ok"),
                 ("Split before fitting", "Feature selection, grid search and all scalers (inside Pipelines) use the <b>training split only</b>.", "ok"),
                 ("Median/mode fill happens before the split", f"In the notebook the fill runs before <code>train_test_split</code>. The modelling columns had <b>{r['missing_before_fill']}</b> missing values, so nothing was actually filled - no leakage in effect.", "ok" if r["missing_before_fill"] == 0 else "warn"),
                 ("One-Hot (get_dummies)", "A fixed rule, not fitted on data, so it cannot leak.", "ok"),
                 ("Scenario cross-validation (8.1)", "Uses the full dataset with Pipelines for <i>comparison</i>; the final scores come from the untouched 20% test set.", "ok"),
                 ("App ↔ notebook consistency", "The app's rebuilt test set reproduces the exported confusion matrix." if ok else "The rebuilt test set does NOT match the exported confusion matrix - library versions may differ.", "ok" if ok else "warn")]
        for t, body, k in audit:
            note(f"<b>{t}.</b> {body}", "note" if k == "ok" else "warn")
            st.write("")


# ================================================================== Model Performance
def page_model():
    page_header("Model Performance", "Real evaluation results on the 20% test set the models never saw.")
    a, r = B.artifacts(), B.replay()
    ver = st.radio("Version", ["Version 3 · Model B (5 features)", "Version 2 · Model A (4 features)"], horizontal=True, key="mp_ver")
    key = "B" if ver.startswith("Version 3") else "A"
    ev = a["evalr"][key]
    res = pd.DataFrame(ev["results"])
    best = ev["best_model"]
    b = res.iloc[0]
    kpi_row([("Accuracy", f"{b['Accuracy']:.3f}"), ("Precision", f"{b['Precision']:.3f}"), ("Recall", f"{b['Recall']:.3f}"),
             ("F1-score", f"{b['F1-Score']:.3f}"), ("ROC-AUC", f"{b['ROC-AUC']:.3f}")])
    st.caption(f"Best model: **{best}** · features: {', '.join(ev['features'])}")
    tabs = st.tabs(["Compare models", "Confusion matrix", "ROC curve", "Train / test split", "Risk levels", "Scenarios", "K-Means"])
    with tabs[0]:
        show_df(res.round(3))
        m = res.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(m, x="Metric", y="Score", color="Model", barmode="group", color_discrete_sequence=[C["violet"], C["cyan"], C["indigo"]])
        fig.update_yaxes(range=[0, 1.05])
        show_plot(style_fig(fig, 360, "Model comparison (test data)"), key="mp_cmp")
    with tabs[1]:
        mname = st.selectbox("Model", list(ev["models"]), index=list(ev["models"]).index(best), key="mp_cm")
        cm = np.array(ev["models"][mname]["confusion_matrix"])
        fig = px.imshow(cm, text_auto=True, x=["Genuine", "Fraud"], y=["Genuine", "Fraud"], color_continuous_scale=[C["surface"], C["indigo"], C["violet"]], labels=dict(x="Predicted", y="Actual", color="Count"))
        show_plot(style_fig(fig, 380, f"Confusion matrix - {mname}", legend=False), key="mp_cmx")
        rep = pd.DataFrame(ev["models"][mname]["report"]).T.round(3).reset_index(names="class")
        show_df(rep)
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"Of **{tp+fn}** real frauds the model caught **{tp}** and missed **{fn}**; it wrongly flagged **{fp}** of **{tn+fp}** genuine transactions.")
    with tabs[2]:
        fig = go.Figure()
        for (n, d), colr in zip(ev["models"].items(), [C["violet"], C["cyan"], C["indigo"]]):
            fig.add_scatter(x=d["fpr"], y=d["tpr"], name=f"{n} (AUC {d['roc_auc']:.3f})", mode="lines", line=dict(color=colr, width=2.5))
        fig.add_scatter(x=[0, 1], y=[0, 1], name="Random guess", line=dict(color=C["muted"], dash="dash"))
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate")
        show_plot(style_fig(fig, 400, "ROC curves"), key="mp_roc")
    with tabs[3]:
        yt, ys = r["y_train"], r["y_test"]
        d = pd.DataFrame({"Split": ["Training (80%)", "Testing (20%)"], "Rows": [len(yt), len(ys)], "Fraud": [int(yt.sum()), int(ys.sum())], "Fraud %": [round(yt.mean() * 100, 2), round(ys.mean() * 100, 2)]})
        show_df(d)
        note("Stratified split with <code>random_state=42</code>: both parts keep the same fraud percentage.")
    with tabs[4]:
        f = ev["features"]
        model = a["model_B"] if key == "B" else a["model_A"]
        p = model.predict_proba(r["X_test"][f])[:, 1]
        lim = a["info"]["risk_limits"]
        lvl = pd.cut(p, [-0.001, lim["low"], lim["high"], 1.0], labels=["Low", "Medium", "High"])
        t = pd.DataFrame({"Risk": lvl, "fraud": r["y_test"].values}).groupby("Risk", observed=False)["fraud"].agg(Transactions="size", Fraud_cases="sum", Fraud_rate=lambda s: round(s.mean() * 100, 1) if len(s) else 0).reset_index()
        show_df(t)
        st.caption("Low ≤ 50%, Medium 50-80%, High > 80% fraud probability (notebook section 10).")
    with tabs[5]:
        sc = a["evalr"].get("scenarios") or []
        if sc:
            show_df(pd.DataFrame(sc).round(3))
            note("Scenarios 1-2 score ≈100% because the synthetic dataset has near-giveaway columns; scenarios 3-4 are the realistic ones.")
        else:
            st.write("No scenario results were found in the exported file.")
    with tabs[6]:
        cs = a["evalr"].get("cluster_summary") or []
        st.markdown("K-Means is **unsupervised** and separate from classification; it never changes the scores above.")
        if cs:
            show_df(pd.DataFrame(cs))
        km = a["kmeans"]
        if km:
            from sklearn.decomposition import PCA
            cols = a["info"]["cluster_columns"]
            Xtr = km["scaler"].transform(r["X_train"][cols])
            lab = km["kmeans"].predict(Xtr)
            pts = PCA(n_components=2, random_state=42).fit_transform(Xtr)
            idx = np.random.RandomState(1).choice(len(pts), 3000, replace=False)
            dd = pd.DataFrame({"PCA 1": pts[idx, 0], "PCA 2": pts[idx, 1], "Cluster": lab[idx].astype(str), "Class": r["y_train"].values[idx]})
            dd["Class"] = dd["Class"].map(CLS)
            c1, c2 = st.columns(2)
            with c1:
                show_plot(style_fig(px.scatter(dd, x="PCA 1", y="PCA 2", color="Cluster", color_discrete_sequence=[C["violet"], C["cyan"], C["indigo"]], opacity=.7), 340, "K-Means clusters"), key="km1")
            with c2:
                show_plot(style_fig(px.scatter(dd, x="PCA 1", y="PCA 2", color="Class", color_discrete_map=CMAP, opacity=.7), 340, "Same points by true label"), key="km2")


# ================================================================== SHAP
def _native(fn):
    import matplotlib.pyplot as plt
    plt.close("all")
    fn()
    fig = plt.gcf()
    fig.patch.set_facecolor("#F3F5FC")
    st.pyplot(fig)
    plt.close(fig)


def page_shap():
    page_header("Explainable AI (SHAP)", "SHAP shows why the model made each prediction - globally and for a single transaction. All values come from the trained Model B.")
    try:
        import shap
        g = B.global_shap()
    except Exception as e:
        st.error(f"SHAP is not available: {e}\n\nInstall it with `pip install shap` and restart the app.")
        return
    a, r = B.artifacts(), B.replay()
    f = a["info"]["B"]["features"]
    tabs = st.tabs(["How SHAP works", "Global explanation", "One transaction", "What-if explorer"])

    with tabs[0]:
        html('<div class="card"><h4>The idea in one paragraph</h4><p>SHAP splits the model\'s fraud probability between the features. '
             '<b>Base value</b> = the model\'s average prediction. Every feature then <b>adds or removes</b> some probability. '
             '<b>Base value + all pushes = the final fraud probability.</b> Positive SHAP = towards fraud, negative = towards genuine (class 1).</p></div>')
        st.write("")
        kpi_row([("Base value", f"{float(g['expl'].base_values[0]):.3f}", "model average"), ("Transactions explained", f"{len(g['X']):,}", "from the test set"),
                 ("Max |model − SHAP| gap", f"{g['gap']:.1e}", "≈0 → SHAP matches the model")])
        st.markdown("```python\nexplainer = shap.TreeExplainer(model_B)      # 1. build from the trained model\nexplanation = explainer(X_shap)           # 2. push of every feature\nfraud_explanation = explanation[:, :, 1]  # 3. keep class 1 (fraud)\n```")
        note("The base value is 0.5 rather than 17% because the model was trained with <b>class_weight='balanced'</b>.")

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Summary (beeswarm) plot**")
            _native(lambda: shap.plots.beeswarm(g["expl"], show=False))
            st.caption("Each dot is a transaction. Right = towards fraud. Red = high feature value, blue = low.")
        with c2:
            st.markdown("**Mean absolute SHAP (feature importance)**")
            _native(lambda: shap.plots.bar(g["expl"], show=False))
        imp = pd.Series(np.abs(g["expl"].values).mean(axis=0), index=f).sort_values()
        fig = go.Figure(go.Bar(x=imp.values * 100, y=[B.pretty(c) for c in imp.index], orientation="h", marker_color=C["violet"]))
        fig.update_xaxes(title="Average push (percentage points of fraud probability)")
        show_plot(style_fig(fig, 300, "Interactive feature importance", legend=False), key="sh_imp")
        rows = []
        for i, c in enumerate(f):
            v, s = g["X"][c].values, g["expl"].values[:, i]
            corr = np.corrcoef(v, s)[0, 1] if np.std(v) > 0 and np.std(s) > 0 else 0
            rows.append(dict(Feature=B.pretty(c), MeanAbsPush=round(float(np.abs(s).mean()) * 100, 1),
                             Reading="Higher value → more fraud" if corr > .1 else ("Higher value → less fraud" if corr < -.1 else "Mixed / non-linear")))
        show_df(pd.DataFrame(rows).sort_values("MeanAbsPush", ascending=False))

    with tabs[2]:
        n = len(r["X_test"])
        def _pick(label):
            st.session_state["sh_pos_in"] = int(np.random.choice(np.where(r["y_test"].values == label)[0]))
        if "sh_pos_in" not in st.session_state:
            st.session_state["sh_pos_in"] = int(np.where(r["y_test"].values == 1)[0][0])
        c1, c2, c3 = st.columns(3)
        c1.button("Random real FRAUD example", key="sh_f", on_click=_pick, args=(1,))
        c2.button("Random real GENUINE example", key="sh_g", on_click=_pick, args=(0,))
        pos = int(c3.number_input("Test row number", 0, n - 1, key="sh_pos_in"))
        one = r["X_test"].iloc[[pos]]
        ex = B.explain_rows(one)
        prob = float(a["model_B"].predict_proba(one[f])[0, 1])
        actual = "Fraud" if r["y_test"].iloc[pos] == 1 else "Genuine"
        kpi_row([("Actual label", actual), ("Model fraud probability", f"{prob*100:.1f}%"), ("Base value", f"{float(ex.base_values[0]):.2f}")])
        st.write("")
        _native(lambda: shap.plots.waterfall(ex[0], show=False))
        res = B.analyze({c: one[c].iloc[0] for c in f}, [])
        show_plot(contribution_bar(res["contrib"]), key="sh_one")
        factor_list(res["contrib"], B.reference_stats())

    with tabs[3]:
        feat = st.selectbox("Vary this feature", f, format_func=B.pretty, key="wi_feat")
        info = a["info"]["B"]
        base = {c: info["defaults"][c] for c in f}
        lo, hi = info["min"][feat], info["max"][feat]
        xs = np.array([0.0, 1.0]) if info["is_binary"][feat] else np.linspace(lo, hi, 60)
        grid = pd.DataFrame([{**base, feat: float(x)} for x in xs])[f]
        pr = a["model_B"].predict_proba(grid)[:, 1] * 100
        fig = go.Figure(go.Scatter(x=xs, y=pr, mode="lines+markers", line=dict(color=C["violet"], width=3, shape="hv" if len(xs) > 2 else "linear"), marker=dict(color=C["cyan"], size=7)))
        fig.update_xaxes(title=B.pretty(feat))
        fig.update_yaxes(title="Model fraud probability (%)", range=[-2, 102])
        show_plot(style_fig(fig, 340, f"Real model output as “{B.pretty(feat)}” changes", legend=False), key="wi")
        st.caption("All other features are held at their typical training values (medians from model_info.pkl). Every point is a real model prediction.")
