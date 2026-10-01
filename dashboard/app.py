"""Streamlit dashboard (dark purple UI).

All numbers come from the trained model via the FastAPI service and from the saved
evaluation reports; nothing is hardcoded. Place this file at dashboard/app.py.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.feature_engineering import RAW_NUMERIC_FEATURES  # noqa: E402

API_URL = config.API_URL.rstrip("/")
SENSORS = RAW_NUMERIC_FEATURES
SENSOR_COLS = [*SENSORS, "machine_type"]
SENSOR_LABELS = {
    "air_temperature_k": "Air temperature [K]",
    "process_temperature_k": "Process temperature [K]",
    "rotational_speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
}
NAV = {
    "Overview": "🏠",
    "Machine Monitoring": "📡",
    "Sensor Analysis": "📊",
    "Prediction": "🧠",
    "Explainability": "💡",
    "Model Performance": "🎯",
}
SCOPE_TEST = "Held-out test split"
SCOPE_ALL = "All rows (incl. training)"
DISCLAIMER = ("Model-derived feature contributions (SHAP), not guaranteed causal explanations. "
              "Anomaly score is not a failure probability.")
TONE = {"LOW": "ok", "MEDIUM": "warn", "HIGH": "bad", "NORMAL": "ok", "ANOMALY": "bad"}
PURPLES = ["#8b6cff", "#6d4aff", "#38bdf8", "#22c55e", "#f59e0b", "#ef4444"]

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#0a0e1a;--panel:#0d1324;--card:#111830;--field:#0c1226;--line:#1f2a4d;
--text:#e8ebf7;--muted:#8d96b8;--p1:#6d4aff;--p2:#8b6cff;--ok:#22c55e;--warn:#f59e0b;--bad:#ef4444;}
.stApp{background:var(--bg);font-family:'Inter','Segoe UI',sans-serif;}
header[data-testid="stHeader"]{background:var(--bg);}
.block-container{padding-top:2.2rem;max-width:1250px;}
.stApp, .stApp label, .stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp [data-testid="stMarkdownContainer"]{color:var(--text);}
.stApp [data-testid="stCaptionContainer"], .stApp small{color:var(--muted);}
section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div{background:var(--panel);}
section[data-testid="stSidebar"]{border-right:1px solid var(--line);}

/* brand + sidebar labels */
.stApp .brand{display:flex;align-items:center;gap:.7rem;font-weight:700;font-size:1.05rem;margin:.2rem 0 1.4rem;color:#fff;}
.stApp .brand span.logo{font-size:1.5rem;}
.stApp .sb-label{letter-spacing:.14em;font-size:.72rem;font-weight:600;color:var(--muted);margin:1.3rem 0 .6rem;}

/* nav radio */
.st-key-nav div[role="radiogroup"]{gap:.25rem;}
.st-key-nav div[role="radiogroup"] > label{position:relative;padding:.7rem .9rem;border-radius:12px;width:100%;
 border:1px solid transparent;cursor:pointer;}
.st-key-nav div[role="radiogroup"] > label > div:first-child{display:none;}
.st-key-nav div[role="radiogroup"] > label:hover{background:#141c3a;}
.st-key-nav div[role="radiogroup"] > label:has(input:checked){background:linear-gradient(90deg,#27206f,#1a2150);border-color:#3b34a8;}
.stApp .st-key-nav label p{color:var(--muted);font-weight:500;font-size:.95rem;}
.stApp .st-key-nav label:has(input:checked) p{color:#fff;font-weight:600;}
.st-key-nav label:has(input:checked)::after{content:"";position:absolute;right:16px;top:50%;width:9px;height:9px;
 margin-top:-4px;border-radius:50%;background:var(--p2);}

/* model card */
.stApp .mcard{display:flex;gap:.8rem;align-items:center;background:#0f1630;border:1px solid var(--line);
 border-radius:14px;padding:.9rem;}
.stApp .mcard .mi{width:44px;height:44px;border-radius:12px;background:#171f42;display:flex;align-items:center;
 justify-content:center;font-size:1.4rem;}
.stApp .mcard .m1{font-size:.72rem;color:var(--muted);}
.stApp .mcard .m2{font-weight:700;color:#fff;}
.stApp .mcard .m3{font-size:.72rem;color:var(--muted);margin-bottom:.3rem;}
.stApp .ready{display:inline-block;background:rgba(34,197,94,.14);color:var(--ok);border-radius:8px;
 padding:.1rem .55rem;font-size:.75rem;font-weight:600;}

/* page header */
.stApp .ph{display:flex;gap:1.1rem;align-items:center;margin-bottom:1.4rem;}
.stApp .ph .pi{width:76px;height:76px;border-radius:20px;background:linear-gradient(145deg,#1b2350,#141b3d);
 border:1px solid var(--line);display:flex;align-items:center;justify-content:center;font-size:2.1rem;}
.stApp .ph h1{margin:0;font-size:2.1rem;font-weight:800;color:#fff;letter-spacing:-.02em;padding:0;}
.stApp .ph p{margin:.25rem 0 0;color:var(--muted);}

/* cards, forms, containers */
[data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"]{background:var(--card);
 border:1px solid var(--line)!important;border-radius:18px;}
[data-testid="stForm"]{padding:1.4rem;}
.stApp [data-testid="stExpander"]{background:var(--card);border:1px solid var(--line);border-radius:14px;}

/* inputs */
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"], .stApp [data-baseweb="select"] > div{
 background:var(--field)!important;border-color:var(--line)!important;border-radius:12px!important;}
.stApp input{color:var(--text)!important;}
.stApp [data-testid="stNumberInputContainer"] button{background:transparent!important;color:var(--muted)!important;border:none!important;}
[data-baseweb="popover"] ul, [data-baseweb="menu"]{background:#0e1424!important;}
.stApp [data-baseweb="select"] *{color:var(--text);}

/* primary button */
.stApp .stButton > button, .stApp [data-testid="stFormSubmitButton"] button{
 background:linear-gradient(180deg,#7c5cff,#5b3df5);color:#fff;border:none;border-radius:12px;
 padding:.7rem 1.5rem;font-weight:600;box-shadow:0 8px 24px rgba(91,61,245,.35);}
.stApp .stButton > button:hover, .stApp [data-testid="stFormSubmitButton"] button:hover{filter:brightness(1.1);color:#fff;}

/* KPI cards */
.stApp .kpi{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:1rem 1.1rem;height:100%;}
.stApp .kpi .kl{color:var(--muted);font-size:.8rem;font-weight:500;}
.stApp .kpi .kv{color:#fff;font-size:1.65rem;font-weight:800;margin:.25rem 0 .1rem;letter-spacing:-.02em;}
.stApp .kpi .ks{color:var(--muted);font-size:.75rem;}
.stApp .kpi.ok .kv{color:var(--ok);} .stApp .kpi.warn .kv{color:var(--warn);} .stApp .kpi.bad .kv{color:var(--bad);}
.stApp .meter{height:8px;border-radius:6px;background:#1a2345;margin-top:.5rem;overflow:hidden;}
.stApp .meter div{height:100%;background:linear-gradient(90deg,#6d4aff,#a78bfa);}

/* info box */
.stApp .infobox{display:flex;gap:1rem;background:var(--card);border:1px solid var(--line);
 border-left:3px solid var(--p2);border-radius:14px;padding:1.1rem 1.3rem;margin-top:1.2rem;}
.stApp .infobox .ii{width:40px;height:40px;border-radius:50%;background:#171f42;display:flex;
 align-items:center;justify-content:center;flex:none;}
.stApp .infobox b{color:#fff;} .stApp .infobox div div{color:var(--muted);font-size:.9rem;margin-top:.3rem;}

/* html tables + chips */
.stApp table.tbl{width:100%;border-collapse:collapse;font-size:.88rem;}
.stApp table.tbl th{color:var(--muted);text-align:left;font-weight:600;padding:.55rem .7rem;border-bottom:1px solid var(--line);}
.stApp table.tbl td{color:var(--text);padding:.55rem .7rem;border-bottom:1px solid #162043;}
.stApp .chip{display:inline-block;background:#171f42;border:1px solid var(--line);border-radius:999px;
 padding:.25rem .8rem;margin:.15rem .3rem .15rem 0;font-size:.82rem;color:var(--text);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---- UI helpers --------------------------------------------------------------
def page_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="ph"><div class="pi">{icon}</div><div><h1>{title}</h1><p>{subtitle}</p></div></div>',
        unsafe_allow_html=True,
    )


def kpi_html(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return (f'<div class="kpi {tone}"><div class="kl">{html.escape(label)}</div>'
            f'<div class="kv">{html.escape(str(value))}</div><div class="ks">{html.escape(sub)}</div></div>')


def kpi_row(items: list[tuple]) -> None:
    for col, item in zip(st.columns(len(items)), items):
        col.markdown(kpi_html(*item), unsafe_allow_html=True)


def info_box(title: str, body: str) -> None:
    st.markdown(f'<div class="infobox"><div class="ii">ℹ️</div><div><b>{title}</b><div>{body}</div></div></div>',
                unsafe_allow_html=True)


def html_table(df: pd.DataFrame, digits: int = 4) -> None:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = "".join(
        "<tr>" + "".join(
            f"<td>{v:.{digits}f}</td>" if isinstance(v, float) else f"<td>{html.escape(str(v))}</td>" for v in r
        ) + "</tr>" for r in df.itertuples(index=False)
    )
    st.markdown(f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>',
                unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#c9d0ea"), margin=dict(l=10, r=10, t=45, b=10), height=height,
                      colorway=PURPLES, legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#1c2545", zerolinecolor="#1c2545")
    fig.update_yaxes(gridcolor="#1c2545", zerolinecolor="#1c2545")
    return fig


def show(fig: go.Figure, height: int = 320) -> None:
    st.plotly_chart(style_fig(fig, height), use_container_width=True)


# ---- API / data --------------------------------------------------------------
def api_get(path: str) -> dict:
    r = requests.get(f"{API_URL}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict, params: dict | None = None) -> dict:
    r = requests.post(f"{API_URL}{path}", json=payload, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED_DATA_FILE)


@st.cache_data(show_spinner="Scoring machines through the API...", ttl=3600)
def score(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[["machine_id", *SENSOR_COLS]].to_dict("records")
    out: list[dict] = []
    for i in range(0, len(rows), 1000):
        out += api_post("/predict/batch", {"readings": rows[i:i + 1000]})["predictions"]
    scored_df = pd.DataFrame(out).drop(columns=["factor_details", "top_factors"], errors="ignore")
    return df.merge(scored_df.drop(columns=["model_version"]), on="machine_id", how="left")


def load_report(name: str) -> dict | None:
    path = config.REPORTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


# ---- startup checks ----------------------------------------------------------
try:
    health = api_get("/health")
except requests.RequestException as exc:
    st.error(f"Cannot reach the API at {API_URL}: {exc}")
    st.stop()
if not health["model_loaded"]:
    st.error(f"API is up but no model is loaded: {health.get('detail')}. Run `python -m src.train`.")
    st.stop()
if not config.PROCESSED_DATA_FILE.exists():
    st.error(f"{config.PROCESSED_DATA_FILE} not found. Run `python -m src.train` first.")
    st.stop()

data = load_data()
info = api_get("/model/info")

# ---- sidebar -----------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand"><span class="logo">👑</span>Predictive Maintenance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">PAGES</div>', unsafe_allow_html=True)
    page = st.radio("Page", list(NAV), format_func=lambda p: f"{NAV[p]}   {p}",
                    label_visibility="collapsed", key="nav")
    st.markdown('<div class="sb-label">MODEL</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="mcard"><div class="mi">🧊</div><div><div class="m1">Model</div>'
        f'<div class="m2">{html.escape(info["model_type"])}</div>'
        f'<div class="m3">{html.escape(info["model_version"])}</div><span class="ready">● Ready</span></div></div>',
        unsafe_allow_html=True)
    st.markdown('<div class="sb-label">DATA SCOPE</div>', unsafe_allow_html=True)
    scope = st.radio("Data scope", [SCOPE_TEST, SCOPE_ALL],
                     captions=["Recommended for evaluation", "Use entire dataset"],
                     label_visibility="collapsed", key="scope")
    if scope == SCOPE_ALL:
        st.warning("Training rows are in-sample; their scores look better than real-world behaviour.")

view = (data[data["split"] == "test"] if scope == SCOPE_TEST else data).reset_index(drop=True)


def scored_view() -> pd.DataFrame:
    return score(view)


# ---- pages -------------------------------------------------------------------
if page == "Overview":
    page_header("🏠", "Fleet overview", "Model-scored health of the machines in the selected data scope.")
    scored = scored_view()
    at_risk = scored["risk_level"].isin(["MEDIUM", "HIGH"])
    anomalous = scored["anomaly_status"] == "ANOMALY"
    total = scored["machine_id"].nunique()
    healthy = int(((~at_risk) & (~anomalous)).sum())
    pct = lambda n: f"{100 * n / max(total, 1):.1f}% of machines"  # noqa: E731
    kpi_row([
        ("Total machines", f"{total:,}", "in scope"),
        ("Healthy", f"{healthy:,}", pct(healthy), "ok"),
        ("At risk (medium/high)", f"{int(at_risk.sum()):,}", pct(int(at_risk.sum())), "warn"),
        ("Anomalous", f"{int(anomalous.sum()):,}", pct(int(anomalous.sum())), "warn"),
        ("Predicted failures", f"{int((scored['prediction'] == 1).sum()):,}",
         pct(int((scored['prediction'] == 1).sum())), "bad"),
    ])
    st.caption("Healthy = LOW risk and NORMAL anomaly status. Risk and anomaly status are separate signals.")
    left, right = st.columns(2)
    with left, st.container(border=True):
        show(px.histogram(scored, x="failure_probability", nbins=40, title="Failure probability"))
    with right, st.container(border=True):
        show(px.pie(scored, names="risk_level", hole=0.55, title="Risk level share", color="risk_level",
                    color_discrete_map={"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}))

elif page == "Machine Monitoring":
    page_header("📡", "Machine monitoring", "Inspect one machine's risk, anomaly status and sensor position.")
    scored = scored_view()
    top = scored.sort_values("failure_probability", ascending=False)["machine_id"].head(500).tolist()
    c1, c2 = st.columns([1, 2])
    typed = c1.text_input("Search machine ID (exact)")
    machine_id = typed if typed in set(scored["machine_id"]) else c2.selectbox("Top-risk machines", top)
    row = scored[scored["machine_id"] == machine_id].iloc[0]
    kpi_row([
        ("Machine ID", machine_id),
        ("Prediction", "FAILURE" if row["prediction"] == 1 else "NO FAILURE", "", "bad" if row["prediction"] == 1 else "ok"),
        ("Failure probability", f"{row['failure_probability']:.1%}"),
        ("Risk level", row["risk_level"], "", TONE[row["risk_level"]]),
        ("Anomaly status", row["anomaly_status"], "", TONE[row["anomaly_status"]]),
    ])
    st.caption(f"Recorded failure label in dataset: {int(row['machine_failure'])}. AI4I has one reading per "
               "machine, so charts show where this machine sits in the fleet distribution (red line).")
    for chunk in (SENSORS[:3], SENSORS[3:]):
        for col, sensor in zip(st.columns(3), chunk):
            fig = px.histogram(view, x=sensor, nbins=40, title=SENSOR_LABELS[sensor])
            fig.add_vline(x=row[sensor], line_color="#ef4444", line_width=2)
            with col, st.container(border=True):
                show(fig, 260)

elif page == "Sensor Analysis":
    page_header("📊", "Sensor analysis", "Explore sensor distributions and how they relate to failures.")
    st.caption("The dataset has no timestamps, so date/time filtering is not available.")
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns(4)
        types = f1.multiselect("Machine type", ["L", "M", "H"], default=["L", "M", "H"])
        sensor = f2.selectbox("Sensor", SENSORS, format_func=SENSOR_LABELS.get)
        outcome = f3.selectbox("Show", ["All", "Failures only", "Non-failures only"])
        other = f4.selectbox("Compare against", [s for s in SENSORS if s != sensor], format_func=SENSOR_LABELS.get)
    sub = view[view["machine_type"].isin(types)]
    if outcome == "Failures only":
        sub = sub[sub["machine_failure"] == 1]
    elif outcome == "Non-failures only":
        sub = sub[sub["machine_failure"] == 0]
    sub = sub.assign(failure=sub["machine_failure"].map({0: "no failure", 1: "failure"}))
    cmap = {"no failure": "#8b6cff", "failure": "#ef4444"}
    a, b = st.columns(2)
    with a, st.container(border=True):
        show(px.histogram(sub, x=sensor, color="failure", barmode="overlay", nbins=50, opacity=0.8,
                          color_discrete_map=cmap, title=f"Distribution: {SENSOR_LABELS[sensor]}"))
    with b, st.container(border=True):
        show(px.box(sub, x="failure", y=sensor, color="failure", points="outliers", color_discrete_map=cmap,
                    title=f"Spread: {SENSOR_LABELS[sensor]}"))
    with st.container(border=True):
        show(px.scatter(sub, x=sensor, y=other, color="failure", opacity=0.6, color_discrete_map=cmap,
                        title=f"{SENSOR_LABELS[sensor]} vs {SENSOR_LABELS[other]}"), 380)

elif page == "Prediction":
    page_header("🧠", "Predict failure risk",
                "Enter the current operating conditions of the machine to estimate the risk of failure.")
    with st.form("predict_form"):
        r1 = st.columns(3)
        mtype = r1[0].selectbox("Machine type", ["L", "M", "H"], help="Product quality variant (L, M or H).")
        air = r1[1].number_input("Air temperature [K]", 250.0, 400.0, 300.0, 0.1, format="%.2f")
        proc = r1[2].number_input("Process temperature [K]", 250.0, 400.0, 310.0, 0.1, format="%.2f")
        r2 = st.columns(3)
        speed = r2[0].number_input("Rotational speed [rpm]", 1.0, 10000.0, 1500.0, 10.0, format="%.2f")
        torque = r2[1].number_input("Torque [Nm]", 0.0, 500.0, 40.0, 0.5, format="%.2f")
        wear = r2[2].number_input("Tool wear [min]", 0.0, 2000.0, 100.0, 1.0, format="%.2f")
        submitted = st.form_submit_button("✦  Predict risk")
    if submitted:
        try:
            res = api_post("/predict", {"machine_type": mtype, "air_temperature_k": air,
                                        "process_temperature_k": proc, "rotational_speed_rpm": speed,
                                        "torque_nm": torque, "tool_wear_min": wear})
        except requests.RequestException as exc:
            st.error(f"API error: {exc}")
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            p = res["failure_probability"]
            cols = st.columns(4)
            cols[0].markdown(
                f'<div class="kpi {TONE[res["risk_level"]]}"><div class="kl">Failure probability</div>'
                f'<div class="kv">{p:.1%}</div><div class="meter"><div style="width:{p * 100:.1f}%"></div></div></div>',
                unsafe_allow_html=True)
            cols[1].markdown(kpi_html("Risk level", res["risk_level"], "", TONE[res["risk_level"]]), unsafe_allow_html=True)
            cols[2].markdown(kpi_html("Anomaly status", res["anomaly_status"], f"score {res['anomaly_score']:.3f}",
                                      TONE[res["anomaly_status"]]), unsafe_allow_html=True)
            cols[3].markdown(kpi_html("Prediction", "FAILURE" if res["prediction"] else "NO FAILURE", "",
                                      "bad" if res["prediction"] else "ok"), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("**Top factors increasing risk**")
                chips = "".join(f'<span class="chip">{html.escape(SENSOR_LABELS.get(f, f))}</span>'
                                for f in res["top_factors"])
                st.markdown(chips or '<span class="chip">none increasing risk</span>', unsafe_allow_html=True)
                st.caption(DISCLAIMER)
    info_box("How it works",
             "The model estimates the probability of failure from the operating conditions you enter "
             "(trained on the AI4I 2020 dataset). Anomaly status is a separate signal and is not a failure "
             "probability. A higher probability means a higher estimated chance of failure.")

elif page == "Explainability":
    page_header("💡", "Explainability", DISCLAIMER)
    report = load_report("shap_global_importance.json")
    with st.container(border=True):
        if report:
            imp = pd.DataFrame(report["importance"]).head(12).iloc[::-1]
            show(px.bar(imp, x="mean_abs_shap", y="feature", orientation="h",
                        title="Global importance (mean |SHAP|, log-odds)"), 380)
        else:
            st.warning("shap_global_importance.json not found; run training.")
    st.subheader("Local explanation")
    scored = scored_view()
    mid = st.selectbox("Machine", scored.sort_values("failure_probability", ascending=False)["machine_id"].head(500))
    r = scored[scored["machine_id"] == mid].iloc[0]
    res = api_post("/predict", {"machine_id": mid, **{c: (r[c].item() if hasattr(r[c], "item") else r[c])
                                                      for c in SENSOR_COLS}})
    kpi_row([("Failure probability", f"{res['failure_probability']:.1%}", "", TONE[res["risk_level"]]),
             ("Risk level", res["risk_level"], "", TONE[res["risk_level"]]),
             ("Anomaly status", res["anomaly_status"], "", TONE[res["anomaly_status"]])])
    det = pd.DataFrame(res["factor_details"])
    if not det.empty:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            show(px.bar(det, x="contribution", y="feature", orientation="h", color="direction",
                        color_discrete_map={"increases_failure_risk": "#ef4444", "decreases_failure_risk": "#22c55e"},
                        title="Top contributions to this prediction"), 300)
            html_table(det.assign(direction=det["direction"].str.replace("_", " ")), 3)

else:
    page_header("🎯", "Model performance", "Held-out test-set results from the trained models.")
    m = load_report("metrics.json")
    if not m:
        st.warning("metrics.json not found; run training.")
        st.stop()
    served = m["served_model"]
    res = m["models"][served]
    st.caption(f"Served model: {served}. Test rows: {m.get('test_rows')}, failures: {m.get('test_failures')}.")
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    kpi_row([(k.upper().replace("_", "-"), f"{res[k]:.3f}") for k in keys])
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Model comparison**")
        html_table(pd.DataFrame([{"model": n, **{k: v[k] for k in keys}} for n, v in m["models"].items()]))
    cm = res["confusion_matrix"]
    a, b = st.columns(2)
    with a, st.container(border=True):
        show(px.imshow([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], text_auto=True,
                       x=["Pred 0", "Pred 1"], y=["True 0", "True 1"], color_continuous_scale="Purples",
                       title="Confusion matrix"), 340)
    for col, (key, title, xl, yl) in zip(
        [b, st.container()], [("roc", "ROC curve", "FPR", "TPR"), ("pr", "Precision-recall curve", "Recall", "Precision")]
    ):
        fig = go.Figure()
        for name, cv in m["curves"].items():
            fig.add_trace(go.Scatter(x=cv[key]["x"], y=cv[key]["y"], mode="lines", name=name))
        fig.update_layout(title=title, xaxis_title=xl, yaxis_title=yl)
        with col, st.container(border=True):
            show(fig, 340)
    with st.expander("Why these metrics?"):
        for k, v in m["metric_explanations"].items():
            st.markdown(f"**{k}**: {v}")