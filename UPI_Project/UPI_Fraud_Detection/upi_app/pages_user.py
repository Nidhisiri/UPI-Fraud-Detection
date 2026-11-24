"""Mode B - everyday-user pages: checker, history/compare, and 'How the AI decides'."""
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import backend as B
from . import report
from .theme import C, html, icon, kpi, note, page_header, show_df, show_plot, style_fig

DISCLAIMER = ("This is an ML-based assessment from a model trained on a research dataset. It is not a guarantee that a "
              "transaction is safe or fraudulent. Please verify the details yourself.")


# ------------------------------------------------------------------ helpers
def parse_amount(text):
    """Returns (value, error). Accepts '4500', '4,500', '₹ 4,500.50'."""
    t = (text or "").strip()
    if not t:
        return None, "Please enter the transaction amount."
    t = re.sub(r"[₹,\s]|rs\.?|inr", "", t, flags=re.I)
    try:
        v = float(t)
    except ValueError:
        return None, "The amount must be a number, for example 4500 or 4,500.50."
    if not np.isfinite(v):
        return None, "The amount must be a valid number."
    if v <= 0:
        return None, "The amount must be greater than zero."
    return v, None


def gauge(prob):
    lim = B.artifacts()["info"]["risk_limits"]
    col = C["safe"] if prob <= lim["low"] else (C["amber"] if prob <= lim["high"] else C["danger"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number=dict(suffix="%", valueformat=".1f", font=dict(size=38)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=C["muted"]), bar=dict(color=col, thickness=.28),
                   bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[0, lim["low"] * 100], color="rgba(52,211,153,.18)"),
                          dict(range=[lim["low"] * 100, lim["high"] * 100], color="rgba(251,191,36,.18)"),
                          dict(range=[lim["high"] * 100, 100], color="rgba(248,113,113,.22)")])))
    return style_fig(fig, height=230, legend=False).update_layout(margin=dict(l=20, r=20, t=20, b=0))


def contribution_bar(contrib, height=None):
    d = sorted(contrib, key=lambda x: abs(x["shap"]))
    vals = [x["shap"] * 100 for x in d]
    fig = go.Figure(go.Bar(
        x=vals, y=[f"{x['label']} ({x['vtext']})" for x in d], orientation="h",
        marker=dict(color=[C["fraud"] if v > 0 else C["genuine"] for v in vals]),
        text=[f"{v:+.1f}" for v in vals], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:+.1f} percentage points<extra></extra>"))
    fig.add_vline(x=0, line_color=C["muted"], line_width=1)
    fig.update_xaxes(title="◀ Reduced fraud estimate  |  Increased fraud estimate ▶  (percentage points)")
    return style_fig(fig, height or 90 + 52 * len(d), legend=False)


def waterfall(res):
    c = sorted(res["contrib"], key=lambda x: -abs(x["shap"]))
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute"] + ["relative"] * len(c) + ["total"],
        x=["Starting point"] + [x["label"] for x in c] + ["Final estimate"],
        y=[res["base"] * 100] + [x["shap"] * 100 for x in c] + [res["prob"] * 100],
        increasing=dict(marker=dict(color=C["fraud"])), decreasing=dict(marker=dict(color=C["genuine"])),
        totals=dict(marker=dict(color=C["indigo"])), connector=dict(line=dict(color=C["border"])),
        text=[f"{res['base']*100:.0f}%"] + [f"{x['shap']*100:+.1f}" for x in c] + [f"{res['prob']*100:.1f}%"],
        textposition="outside"))
    fig.update_yaxes(title="Fraud estimate (%)", range=[0, 112])
    return style_fig(fig, 380, legend=False)


def factor_list(contrib, ref):
    rows = []
    for d in contrib:
        pts = d["shap"] * 100
        if abs(pts) < 1:
            colr, tag = C["muted"], "No real effect"
        elif pts > 0:
            colr, tag = C["fraud"], "Increased fraud prediction"
        else:
            colr, tag = C["genuine"], "Reduced fraud prediction"
        r = ref.get(d["feature"])
        ctx = ""
        if r:
            f = lambda v: B.value_text(d["feature"], v)
            ctx = f" · In the training data, typical genuine = {f(r['g_med'])}, typical fraud = {f(r['f_med'])}"
        assumed = " · <i>typical value assumed</i>" if d["assumed"] else ""
        rows.append(f'<div class="factor"><div class="dot" style="background:{colr}"></div><div>'
                    f'<div class="t">{B.sentence(d)}</div><div class="m">{tag}{ctx}{assumed}</div></div></div>')
    html("".join(rows))


# ------------------------------------------------------------------ Check Your UPI Transaction
def page_checker():
    page_header("Check Your UPI Transaction",
                "Understand your transaction risk in seconds. No technical knowledge required.")
    a = B.artifacts()
    info = a["info"]["B"]
    ref = B.reference_stats()
    ds = B.dataset()

    html('<div class="step"><div class="n">1</div><div><b>Enter transaction details</b><br>'
         '<span style="color:#8D99C4;font-size:.88rem">Answer what you can. If you do not know a detail, tick the '
         '"typical value" box and we will clearly mark it as assumed in the result.</span></div></div>')
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    assumed, answers, errors = [], {}, []

    with c1:
        amt_txt = st.text_input("Transaction amount (₹)", key="chk_amount", placeholder="e.g. 4,500",
                                help="Enter the amount you are planning to send through UPI.")
        amt, amt_err = parse_amount(amt_txt)
        if amt is not None:
            g = ds.loc[ds["is_fraud"] == 0, "amount"]
            fr = ds.loc[ds["is_fraud"] == 1, "amount"]
            st.caption(f"For context: ₹{amt:,.0f} is higher than {(g < amt).mean()*100:.0f}% of genuine and "
                       f"{(fr < amt).mean()*100:.0f}% of fraudulent transactions in the dataset "
                       f"(₹{ds['amount'].min():,.0f}–₹{ds['amount'].max():,.0f}).")
            if amt > ds["amount"].max() * 1.0 or amt < ds["amount"].min():
                st.warning("This amount is outside the range the model learned from, so treat the result with extra caution.")

        # receiver history ---------------------------------------------------------
        f = "receiver_transaction_history"
        meta = B.FRIENDLY[f]
        unk = st.checkbox("I don't know - use the typical value", key="unk_hist")
        hist = st.number_input(meta["question"], min_value=int(info["min"][f]), max_value=int(info["max"][f]),
                               value=None, step=1, key="chk_hist", disabled=unk, placeholder="e.g. 25", help=meta["help"])
        if unk:
            answers[f] = info["defaults"][f]; assumed.append(f)
        elif hist is None:
            errors.append("Enter how many past transactions the receiver has, or tick 'I don't know'.")
        else:
            answers[f] = float(hist)
        st.caption(f"In the training data: genuine receivers had {ref[f]['g_min']:.0f}–{ref[f]['g_max']:.0f} past "
                   f"transactions (typical {ref[f]['g_med']:.0f}); fraud cases had {ref[f]['f_min']:.0f}–{ref[f]['f_max']:.0f} "
                   f"(typical {ref[f]['f_med']:.0f}).")

    with c2:
        for f in ["unusual_transaction_amount_flag", "handle_registration_pattern_recent"]:
            meta = B.FRIENDLY[f]
            choice = st.selectbox(meta["question"], ["Select…", "No", "Yes", "I don't know"], key=f"chk_{f}", help=meta["help"])
            if choice == "Select…":
                errors.append(f"Please answer: {meta['question']}")
            elif choice == "I don't know":
                answers[f] = info["defaults"][f]; assumed.append(f)
            else:
                answers[f] = 1.0 if choice == "Yes" else 0.0

        f = "authentication_attempt_count"
        meta = B.FRIENDLY[f]
        answers[f] = float(st.number_input(meta["question"], min_value=int(info["min"][f]), max_value=int(info["max"][f]),
                                           value=int(info["min"][f]), step=1, key="chk_auth", help=meta["help"]))

        f = "keyboard_input_speed"
        meta = B.FRIENDLY[f]
        unk = st.checkbox(f"Use the typical typing speed ({info['defaults'][f]:.2f})", value=True, key="unk_speed")
        spd = st.slider(meta["question"], float(info["min"][f]), float(info["max"][f]), float(info["defaults"][f]),
                        0.01, key="chk_speed", disabled=unk, help=meta["help"])
        if unk:
            answers[f] = info["defaults"][f]; assumed.append(f)
        else:
            answers[f] = float(spd)

    st.write("")
    html('<div class="step"><div class="n">2</div><div><b>Analyze</b></div></div>')
    go_click = st.button("Analyze My Transaction", type="primary", key="btn_analyze")

    if go_click:
        errs = ([amt_err] if amt_err else []) + errors
        if errs:
            for e in errs:
                st.error(e)
            st.session_state.pop("last", None)
        else:
            try:
                with st.spinner("Analyzing…"):
                    res = B.analyze(answers, assumed)
            except Exception as e:                       # e.g. shap not installed
                st.error(f"Could not analyze the transaction: {e}")
            else:
                res["amount"] = amt
                st.session_state["last"] = res
                B.add_history(res, amt)

    res = st.session_state.get("last")
    if not res:
        return
    st.write("")
    html('<div class="step"><div class="n">3</div><div><b>Your result</b></div></div>')
    st.write("")
    bad = res["pred"] == 1
    if bad:
        html(f'<div class="verdict bad">{icon("alert", 30, C["danger"])}<h2>Potential Fraud Detected</h2>'
             '<p>Our model has identified suspicious characteristics in this transaction. Please review the following '
             'factors before proceeding.</p></div>')
    else:
        html(f'<div class="verdict ok">{icon("check", 30, C["safe"])}<h2>Transaction Appears Legitimate</h2>'
             '<p>Based on the information provided, our model has classified this transaction as legitimate.</p></div>')
    st.write("")
    k1, k2 = st.columns([1, 1.3], gap="large")
    with k1:
        show_plot(gauge(res["prob"]), key="gauge")
        st.caption("Model fraud probability - how strongly the model leans towards fraud. It is an estimate, not a certainty.")
    with k2:
        lvl_cls = {"Low": "ok", "Medium": "am", "High": "bad"}[res["risk"]]
        html(f'<span class="pill {lvl_cls}">Risk level: {res["risk"]}</span>'
             f'<span class="pill v">Model: {a["info"]["B"]["model_name"]}</span>')
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            html(kpi("Fraud probability", f"{res['prob']*100:.1f}%"))
        with c2:
            html(kpi("Amount entered", f"₹{res['amount']:,.0f}", "context only"))
        st.write("")
        note("The model does not read the rupee amount directly. It uses your answer to "
             "<b>“Is this amount unusual for you?”</b>. The amount is kept for your history and report.")

    st.write("")
    html('<div class="step"><div class="n">4</div><div><b>Why did the AI say this?</b><br>'
         '<span style="color:#8D99C4;font-size:.88rem">Our AI model evaluates multiple transaction characteristics '
         'together. These are the factors that influenced its decision most.</span></div></div>')
    st.write("")
    t1, t2 = st.tabs(["What mattered most", "Step-by-step breakdown"])
    with t1:
        show_plot(contribution_bar(res["contrib"]), key="contrib")
        html(f'<span class="pill" style="color:{C["fraud"]}">■ Increased fraud prediction</span>'
             f'<span class="pill" style="color:{C["genuine"]}">■ Reduced fraud prediction</span>')
    with t2:
        show_plot(waterfall(res), key="wf")
        st.caption("Starts from the model's neutral starting point, then every factor moves the estimate up or down.")
    st.write("")
    html(f'<div class="card"><h4>{"Why was this transaction flagged?" if bad else "Why was this transaction classified as legitimate?"}</h4></div>')
    factor_list(res["contrib"], ref)
    for line in B.explanation_text(res):
        st.markdown(f"- {line}")
    if res["assumed"]:
        note("Typical values were assumed for: <b>" + ", ".join(B.pretty(c) for c in res["assumed"]) +
             "</b>. Answer these yourself for a more reliable result.", "warn")

    st.write("")
    html('<div class="step"><div class="n">5</div><div><b>Stay safe</b></div></div>')
    for t in B.safety_tips(res, res["inputs"]):
        st.markdown(f"- {t}")
    note(DISCLAIMER, "warn")
    st.write("")
    try:
        data, mime, fname = report.build(res, res["amount"])
        st.download_button("Download analysis report", data, file_name=fname, mime=mime, key="dl_report")
    except Exception as e:
        st.warning(f"Report could not be created: {e}")


# ------------------------------------------------------------------ History + Compare
def page_history():
    page_header("Transaction History", "Transactions you analyzed in this session. Nothing is stored outside this browser session.")
    h = B.history()
    t1, t2 = st.tabs(["History", "Compare two transactions"])
    with t1:
        if not h:
            html('<div class="card"><h4>No transactions yet</h4><p>Analyze a transaction on the “Check Your UPI Transaction” page and it will appear here.</p></div>')
        else:
            df = pd.DataFrame([dict(No=x["id"], Time=x["time"], Amount=f"₹{x['amount']:,.0f}",
                                    Result="Potential fraud" if x["pred"] else "Legitimate",
                                    Probability=f"{x['prob']*100:.1f}%", Risk=x["risk"],
                                    Factors=B.short_factors(x)) for x in reversed(h)])
            show_df(df)
            c1, c2 = st.columns([1, 4])
            with c1:
                if st.button("Clear history", key="clear_hist"):
                    st.session_state["history"] = []
                    st.session_state.pop("last", None)
                    st.rerun()
            with c2:
                st.download_button("Download history (CSV)", df.to_csv(index=False).encode(), "upi_history.csv", "text/csv", key="dl_hist")
            st.caption("▲ increased and ▼ reduced the fraud prediction.")
    with t2:
        if len(h) < 2:
            html('<div class="card"><h4>Analyze at least two transactions to compare them</h4><p>Try one typical payment and one suspicious-looking one.</p></div>')
            return
        opts = {x["id"]: f"#{x['id']} · {x['time']} · {'Potential fraud' if x['pred'] else 'Legitimate'} · {x['prob']*100:.0f}%" for x in h}
        ids = list(opts)
        c1, c2 = st.columns(2)
        ia = c1.selectbox("Transaction A", ids, index=len(ids) - 2, format_func=opts.get, key="cmp_a")
        ib = c2.selectbox("Transaction B", ids, index=len(ids) - 1, format_func=opts.get, key="cmp_b")
        ra, rb = h[ia - 1], h[ib - 1]
        if ia == ib:
            note("Pick two different transactions.", "warn")
            return
        k = st.columns(4)
        for col, (lab, v) in zip(k, [("A: fraud probability", f"{ra['prob']*100:.1f}%"), ("B: fraud probability", f"{rb['prob']*100:.1f}%"),
                                     ("A: result", "Potential fraud" if ra["pred"] else "Legitimate"),
                                     ("B: result", "Potential fraud" if rb["pred"] else "Legitimate")]):
            with col:
                html(kpi(lab, v))
        feats = list(ra["inputs"])
        sa = {d["feature"]: d for d in ra["contrib"]}
        sb = {d["feature"]: d for d in rb["contrib"]}
        fig = go.Figure()
        for nm, s, colr in [("Transaction A", sa, C["violet"]), ("Transaction B", sb, C["cyan"])]:
            fig.add_bar(name=nm, y=[B.pretty(f) for f in feats], x=[s[f]["shap"] * 100 for f in feats], orientation="h", marker_color=colr)
        fig.update_layout(barmode="group")
        fig.update_xaxes(title="◀ Reduced fraud estimate  |  Increased fraud estimate ▶  (percentage points)")
        show_plot(style_fig(fig, 90 + 70 * len(feats), "How each factor pushed each prediction"), key="cmp_chart")
        diff = sorted(feats, key=lambda f: -abs(sa[f]["shap"] - sb[f]["shap"]))
        show_df(pd.DataFrame([dict(Factor=B.pretty(f), A_value=sa[f]["vtext"], B_value=sb[f]["vtext"],
                                   A_push=f"{sa[f]['shap']*100:+.1f}", B_push=f"{sb[f]['shap']*100:+.1f}") for f in diff]))
        st.markdown("**Why the predictions differ**")
        gap = (ra["prob"] - rb["prob"]) * 100
        st.markdown(f"- Transaction A is {abs(gap):.1f} points {'higher' if gap > 0 else 'lower'} in fraud probability than B.")
        for f in diff[:2]:
            d = (sa[f]["shap"] - sb[f]["shap"]) * 100
            if abs(d) >= 1:
                st.markdown(f"- **{B.pretty(f)}**: A = {sa[f]['vtext']}, B = {sb[f]['vtext']}. This moves A "
                            f"{abs(d):.0f} points {'towards fraud' if d > 0 else 'towards genuine'} relative to B.")
        if abs(gap) < 0.05:
            st.markdown("- The two transactions received (almost) the same estimate.")


# ------------------------------------------------------------------ How the AI decides (plain-English SHAP)
def page_how_ai():
    page_header("How the AI Decides", "A plain-English look at what the model pays attention to - using real results from the trained model.")
    html('<div class="card"><h4>What is SHAP?</h4><p>Imagine a team wins a match and you want to know how many runs each player contributed. '
         'SHAP does the same for the AI: it splits the final fraud estimate between the details of a transaction, so you can see which '
         'details pushed it <b>towards fraud</b> and which pushed it <b>towards genuine</b>. SHAP does not change the prediction - it only explains it.</p></div>')
    st.write("")
    try:
        g = B.global_shap()
    except Exception as e:
        st.error(f"SHAP could not be computed: {e}")
        return
    f = B.artifacts()["info"]["B"]["features"]
    imp = pd.Series(np.abs(g["expl"].values).mean(axis=0), index=f).sort_values()
    fig = go.Figure(go.Bar(x=imp.values * 100, y=[B.pretty(c) for c in imp.index], orientation="h",
                           marker_color=C["violet"], hovertemplate="%{y}: %{x:.1f} points<extra></extra>"))
    fig.update_xaxes(title="Average push on the fraud estimate (percentage points)")
    st.markdown("#### Which details matter most overall?")
    show_plot(style_fig(fig, 320, legend=False), key="howai_imp")
    top = imp.index[::-1][:2]
    st.markdown(f"Across 1,000 test transactions, the biggest influences were **{B.pretty(top[0])}** and **{B.pretty(top[1])}**.")
    st.markdown("#### See it on real examples")
    r = B.replay()
    f_idx = r["X_test"][r["y_test"] == 1].index[0]
    g_idx = r["X_test"][r["y_test"] == 0].index[0]
    ref = B.reference_stats()
    c1, c2 = st.columns(2, gap="large")
    for col, idx, title in [(c1, f_idx, "A transaction that really was fraud"), (c2, g_idx, "A transaction that really was genuine")]:
        with col:
            row = r["X_test"].loc[[idx]]
            res = B.analyze({c: row[c].iloc[0] for c in f}, [])
            st.markdown(f"**{title}** - model estimate: {res['prob']*100:.1f}%")
            show_plot(contribution_bar(res["contrib"]), key=f"ex_{idx}")
            factor_list(res["contrib"][:3], ref)
    note("These examples come from the held-out test data the model never trained on.")
