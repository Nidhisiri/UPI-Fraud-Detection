"""Design system: colours, CSS, small HTML components and Plotly styling."""
import re
import streamlit as st

C = dict(
    bg="#0A1128", surface="#101A3A", surface2="#16224A", border="#243064",
    text="#E6EAF7", muted="#8D99C4", indigo="#6366F1", violet="#8B5CF6", cyan="#22D3EE",
    safe="#34D399", danger="#F87171", amber="#FBBF24",
    # chart colours: blue/coral pair is colour-blind friendlier than green/red
    genuine="#22D3EE", fraud="#FB7185",
)

_ICONS = {
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
}


def icon(name, size=20, color="currentColor"):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-4px">'
            f'{_ICONS[name]}</svg>')


def html(s):
    """Render HTML safely through st.markdown (strips indentation/blank lines that Markdown would turn into code)."""
    s = "\n".join(line.strip() for line in s.strip().splitlines() if line.strip())
    st.markdown(s, unsafe_allow_html=True)


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
:root{--bg:#0A1128;--s1:#101A3A;--s2:#16224A;--bd:#243064;--tx:#E6EAF7;--mu:#8D99C4;--ind:#6366F1;--vio:#8B5CF6;--cy:#22D3EE;--ok:#34D399;--bad:#F87171;--am:#FBBF24;}
html,body,[class*="css"],.stApp{font-family:'Inter',system-ui,sans-serif;color:var(--tx);}
h1,h2,h3,h4{font-family:'Plus Jakarta Sans','Inter',sans-serif!important;letter-spacing:-.02em;color:var(--tx)!important;}
.stApp{background:radial-gradient(1200px 600px at 8% -10%,rgba(99,102,241,.22),transparent 60%),radial-gradient(900px 500px at 100% 0%,rgba(34,211,238,.08),transparent 60%),var(--bg);}
#MainMenu,footer,[data-testid="stDecoration"]{visibility:hidden;height:0;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2.2rem;padding-bottom:4rem;max-width:1200px;}
/* sidebar */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1637 0%,#0A1128 100%);border-right:1px solid var(--bd);}
[data-testid="stSidebar"] [role="radiogroup"]{gap:2px;}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:9px 12px;border-radius:10px;border:1px solid transparent;transition:background .15s,border-color .15s;cursor:pointer;width:100%;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:rgba(139,92,246,.10);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:linear-gradient(90deg,rgba(99,102,241,.28),rgba(139,92,246,.10));border-color:rgba(139,92,246,.55);}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child{display:none;}
[data-testid="stSidebar"] [role="radiogroup"] label p{font-size:.93rem;font-weight:500;}
.brand{display:flex;gap:12px;align-items:center;padding:6px 4px 14px 4px;}
.brand-mark{width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,#6366F1,#8B5CF6 60%,#22D3EE);display:flex;align-items:center;justify-content:center;color:#fff;box-shadow:0 6px 20px rgba(99,102,241,.45);}
.brand-name{font-family:'Plus Jakarta Sans';font-weight:800;font-size:1.02rem;line-height:1.1;}
.brand-sub{color:var(--mu);font-size:.74rem;}
.side-label{color:var(--mu);font-size:.76rem;margin:14px 0 6px 4px;}
/* cards */
.card{background:linear-gradient(180deg,rgba(22,34,74,.85),rgba(16,26,58,.85));border:1px solid var(--bd);border-radius:16px;padding:20px 22px;box-shadow:0 10px 30px rgba(3,7,25,.35);}
.card.tight{padding:14px 16px;}
.card h4{margin:0 0 6px 0;font-size:1.02rem;}
.card p{color:var(--mu);margin:0;font-size:.92rem;line-height:1.55;}
.card.link:hover{border-color:rgba(139,92,246,.7);}
.kpi{background:var(--s1);border:1px solid var(--bd);border-radius:14px;padding:16px 18px;height:100%;}
.kpi .v{font-family:'Plus Jakarta Sans';font-weight:700;font-size:1.65rem;letter-spacing:-.02em;}
.kpi .l{color:var(--mu);font-size:.8rem;margin-top:2px;}
.kpi .s{color:var(--cy);font-size:.78rem;margin-top:6px;}
.hero{padding:30px 32px;border-radius:20px;border:1px solid var(--bd);background:linear-gradient(120deg,rgba(99,102,241,.30),rgba(139,92,246,.14) 45%,rgba(16,26,58,.9));box-shadow:0 20px 50px rgba(3,7,25,.45);}
.hero h1{font-size:2.15rem;margin:0 0 8px 0;font-weight:800;}
.hero p{color:#B9C2E6;font-size:1.02rem;max-width:62ch;margin:0;line-height:1.6;}
.pagehead h2{margin:0;font-size:1.65rem;font-weight:800;}
.pagehead p{color:var(--mu);margin:4px 0 18px 0;max-width:75ch;}
.pill{display:inline-block;padding:3px 11px;border-radius:999px;font-size:.76rem;font-weight:600;border:1px solid var(--bd);color:var(--mu);margin-right:6px;}
.pill.v{color:#C4B5FD;border-color:rgba(139,92,246,.5);background:rgba(139,92,246,.12);}
.pill.c{color:#67E8F9;border-color:rgba(34,211,238,.45);background:rgba(34,211,238,.10);}
.pill.ok{color:#6EE7B7;border-color:rgba(52,211,153,.5);background:rgba(52,211,153,.10);}
.pill.bad{color:#FCA5A5;border-color:rgba(248,113,113,.5);background:rgba(248,113,113,.10);}
.pill.am{color:#FCD34D;border-color:rgba(251,191,36,.5);background:rgba(251,191,36,.10);}
/* verdict */
.verdict{border-radius:18px;padding:24px 26px;border:1px solid;}
.verdict.ok{background:linear-gradient(120deg,rgba(52,211,153,.18),rgba(16,26,58,.9));border-color:rgba(52,211,153,.55);}
.verdict.bad{background:linear-gradient(120deg,rgba(248,113,113,.20),rgba(16,26,58,.9));border-color:rgba(248,113,113,.6);}
.verdict h2{margin:6px 0 6px 0;font-size:1.6rem;font-weight:800;}
.verdict p{color:#C9D1EE;margin:0;line-height:1.55;}
.factor{display:flex;gap:12px;align-items:flex-start;padding:11px 0;border-bottom:1px solid rgba(36,48,100,.7);}
.factor:last-child{border-bottom:none;}
.factor .dot{width:10px;height:10px;border-radius:50%;margin-top:7px;flex:none;}
.factor .t{font-size:.95rem;line-height:1.5;}
.factor .m{color:var(--mu);font-size:.8rem;margin-top:2px;}
.step{display:flex;gap:14px;align-items:flex-start;}
.step .n{width:28px;height:28px;border-radius:50%;background:rgba(139,92,246,.2);border:1px solid rgba(139,92,246,.6);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.82rem;flex:none;}
.note{border-left:3px solid var(--cy);background:rgba(34,211,238,.07);padding:10px 14px;border-radius:0 10px 10px 0;color:#BFEAF4;font-size:.9rem;line-height:1.5;}
.warn{border-left:3px solid var(--am);background:rgba(251,191,36,.07);padding:10px 14px;border-radius:0 10px 10px 0;color:#F6E3A7;font-size:.9rem;line-height:1.5;}
.qa{display:grid;grid-template-columns:110px 1fr;gap:6px 14px;font-size:.9rem;}
.qa b{color:#A5B4FC;font-weight:600;}.qa span{color:#C9D1EE;line-height:1.5;}
/* controls */
.stButton>button,.stDownloadButton>button{border-radius:12px;border:1px solid var(--bd);background:var(--s2);color:var(--tx);font-weight:600;padding:.55rem 1.1rem;transition:transform .12s,border-color .15s,box-shadow .15s;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--vio);box-shadow:0 0 0 3px rgba(139,92,246,.18);}
.stButton>button:active{transform:scale(.98);}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#6366F1,#8B5CF6);border:none;color:#fff;box-shadow:0 8px 24px rgba(99,102,241,.4);width:100%;padding:.75rem 1.2rem;font-size:1rem;}
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--bd);}
.stTabs [data-baseweb="tab"]{border-radius:10px 10px 0 0;padding:8px 16px;color:var(--mu);}
.stTabs [aria-selected="true"]{color:var(--tx);}
[data-baseweb="input"],[data-baseweb="select"]>div,.stNumberInput input,.stTextInput input{background:var(--s1)!important;border-radius:10px!important;}
[data-testid="stMetric"]{background:var(--s1);border:1px solid var(--bd);border-radius:14px;padding:12px 16px;}
[data-testid="stDataFrame"]{border:1px solid var(--bd);border-radius:12px;overflow:hidden;}
details{border:1px solid var(--bd)!important;border-radius:12px!important;background:rgba(16,26,58,.6);}
@media (max-width:760px){.hero{padding:20px;}.hero h1{font-size:1.55rem;}.qa{grid-template-columns:1fr;}}
@media (prefers-reduced-motion:reduce){*{transition:none!important;}}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------- small components ----------
def page_header(title, subtitle=""):
    html(f'<div class="pagehead"><h2>{title}</h2><p>{subtitle}</p></div>')


def kpi(label, value, sub=""):
    s = f'<div class="s">{sub}</div>' if sub else ""
    return f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div>{s}</div>'


def kpi_row(items):
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        with col:
            html(kpi(*it))


def card(title, body, extra_class=""):
    html(f'<div class="card {extra_class}"><h4>{title}</h4><p>{body}</p></div>')


def note(text, kind="note"):
    html(f'<div class="{kind}">{text}</div>')


def qa_block(what, cols, why, result):
    html(f'<div class="qa"><b>What was done</b><span>{what}</span><b>Columns</b><span>{cols}</span>'
         f'<b>Why</b><span>{why}</span><b>Result</b><span>{result}</span></div>')


# ---------- version-tolerant Streamlit wrappers ----------
def show_plot(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo": False})


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def style_fig(fig, height=360, title=None, legend=True):
    fig.update_layout(
        template="plotly_dark", height=height, title=dict(text=title, x=0, font=dict(size=15)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=C["text"], size=12),
        margin=dict(l=10, r=10, t=44 if title else 16, b=10), showlegend=legend,
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=C["surface2"], bordercolor=C["border"], font_color=C["text"]),
    )
    fig.update_xaxes(gridcolor="rgba(141,153,196,.12)", zerolinecolor="rgba(141,153,196,.25)", linecolor=C["border"])
    fig.update_yaxes(gridcolor="rgba(141,153,196,.12)", zerolinecolor="rgba(141,153,196,.25)", linecolor=C["border"])
    return fig


def sidebar_brand():
    html(f'<div class="brand"><div class="brand-mark">{icon("shield", 22, "#fff")}</div>'
         '<div><div class="brand-name">UPI Fraud Detection</div>'
         '<div class="brand-sub">AI-powered, explainable</div></div></div>')
