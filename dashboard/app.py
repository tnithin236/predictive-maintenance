"""Streamlit dashboard. All numbers come from the trained model via the FastAPI
service and from the saved evaluation reports; nothing is hardcoded."""
from __future__ import annotations

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
DISCLAIMER = ("Model-derived feature contributions (SHAP), not guaranteed causal explanations. "
              "Anomaly score is not a failure probability.")

st.set_page_config(page_title="Predictive Maintenance", layout="wide")


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
    scored = pd.DataFrame(out).drop(columns=["factor_details", "top_factors"], errors="ignore")
    return df.merge(scored.drop(columns=["model_version"]), on="machine_id", how="left")


def load_report(name: str) -> dict | None:
    path = config.REPORTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


# ---- startup checks --------------------------------------------------------
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
page = st.sidebar.radio("Page", ["Overview", "Machine Monitoring", "Sensor Analysis",
                                 "Prediction", "Explainability", "Model Performance"])
info = api_get("/model/info")
st.sidebar.caption(f"Model {info['model_version']} ({info['model_type']})")
scope = st.sidebar.radio("Data scope", ["Held-out test split", "All rows (incl. training)"])
view = data[data["split"] == "test"] if scope.startswith("Held") else data
if scope.startswith("All"):
    st.sidebar.warning("Training rows are in-sample; their scores look better than real-world behaviour.")
scored = score(view.reset_index(drop=True))

# ---- pages -----------------------------------------------------------------
if page == "Overview":
    st.title("Fleet overview")
    c = st.columns(5)
    at_risk = scored["risk_level"].isin(["MEDIUM", "HIGH"])
    anomalous = scored["anomaly_status"] == "ANOMALY"
    c[0].metric("Total machines", f"{scored['machine_id'].nunique():,}")
    c[1].metric("Healthy", f"{int(((~at_risk) & (~anomalous)).sum()):,}")
    c[2].metric("At risk (medium/high)", f"{int(at_risk.sum()):,}")
    c[3].metric("Anomalous", f"{int(anomalous.sum()):,}")
    c[4].metric("Predicted failures", f"{int((scored['prediction'] == 1).sum()):,}")
    st.caption("Healthy = LOW risk and NORMAL anomaly status. Risk and anomaly status are separate signals.")
    left, right = st.columns(2)
    left.plotly_chart(px.histogram(scored, x="failure_probability", nbins=40, title="Failure probability"),
                      use_container_width=True)
    right.plotly_chart(px.pie(scored, names="risk_level", title="Risk level share"), use_container_width=True)

elif page == "Machine Monitoring":
    st.title("Machine monitoring")
    top = scored.sort_values("failure_probability", ascending=False)["machine_id"].head(500).tolist()
    typed = st.text_input("Search machine ID (exact)")
    machine_id = typed if typed in set(scored["machine_id"]) else st.selectbox("Top-risk machines", top)
    row = scored[scored["machine_id"] == machine_id].iloc[0]
    c = st.columns(5)
    c[0].metric("Machine ID", machine_id)
    c[1].metric("Prediction", "FAILURE" if row["prediction"] == 1 else "NO FAILURE")
    c[2].metric("Failure probability", f"{row['failure_probability']:.1%}")
    c[3].metric("Risk level", row["risk_level"])
    c[4].metric("Anomaly status", row["anomaly_status"])
    st.caption(f"Recorded failure label in dataset: {int(row['machine_failure'])}. "
               "AI4I has one reading per machine, so charts show where this machine sits in the fleet distribution.")
    cols = st.columns(len(SENSORS))
    for col, sensor in zip(cols, SENSORS):
        fig = px.histogram(view, x=sensor, nbins=40, title=sensor)
        fig.add_vline(x=row[sensor], line_color="red")
        col.plotly_chart(fig, use_container_width=True)

elif page == "Sensor Analysis":
    st.title("Sensor analysis")
    st.caption("The dataset has no timestamps, so date/time filtering is not available.")
    types = st.multiselect("Machine type", ["L", "M", "H"], default=["L", "M", "H"])
    sensor = st.selectbox("Sensor", SENSORS)
    outcome = st.radio("Show", ["All", "Failures only", "Non-failures only"], horizontal=True)
    sub = view[view["machine_type"].isin(types)]
    if outcome == "Failures only":
        sub = sub[sub["machine_failure"] == 1]
    elif outcome == "Non-failures only":
        sub = sub[sub["machine_failure"] == 0]
    sub = sub.assign(failure=sub["machine_failure"].map({0: "no failure", 1: "failure"}))
    st.plotly_chart(px.histogram(sub, x=sensor, color="failure", barmode="overlay", nbins=50), use_container_width=True)
    st.plotly_chart(px.box(sub, x="failure", y=sensor, color="failure", points="outliers"), use_container_width=True)
    other = st.selectbox("Compare against", [s for s in SENSORS if s != sensor])
    st.plotly_chart(px.scatter(sub, x=sensor, y=other, color="failure", opacity=0.6), use_container_width=True)

elif page == "Prediction":
    st.title("Predict failure risk")
    with st.form("predict"):
        c = st.columns(3)
        mtype = c[0].selectbox("Machine type", ["L", "M", "H"])
        air = c[1].number_input("Air temperature [K]", 250.0, 400.0, 300.0)
        proc = c[2].number_input("Process temperature [K]", 250.0, 400.0, 310.0)
        speed = c[0].number_input("Rotational speed [rpm]", 1.0, 10000.0, 1500.0)
        torque = c[1].number_input("Torque [Nm]", 0.0, 500.0, 40.0)
        wear = c[2].number_input("Tool wear [min]", 0.0, 2000.0, 100.0)
        go_btn = st.form_submit_button("Predict")
    if go_btn:
        try:
            res = api_post("/predict", {"machine_type": mtype, "air_temperature_k": air,
                                        "process_temperature_k": proc, "rotational_speed_rpm": speed,
                                        "torque_nm": torque, "tool_wear_min": wear})
        except requests.RequestException as exc:
            st.error(f"API error: {exc}")
        else:
            c = st.columns(4)
            c[0].metric("Failure probability", f"{res['failure_probability']:.1%}")
            c[1].metric("Risk level", res["risk_level"])
            c[2].metric("Anomaly status", res["anomaly_status"])
            c[3].metric("Prediction", "FAILURE" if res["prediction"] else "NO FAILURE")
            st.write("Top factors:", ", ".join(res["top_factors"]) or "none increasing risk")
            st.caption(DISCLAIMER)

elif page == "Explainability":
    st.title("Explainability")
    st.caption(DISCLAIMER)
    report = load_report("shap_global_importance.json")
    if report:
        imp = pd.DataFrame(report["importance"]).head(12).iloc[::-1]
        st.plotly_chart(px.bar(imp, x="mean_abs_shap", y="feature", orientation="h",
                               title="Global importance (mean |SHAP|, log-odds)"), use_container_width=True)
    else:
        st.warning("shap_global_importance.json not found; run training.")
    st.subheader("Local explanation")
    mid = st.selectbox("Machine", scored.sort_values("failure_probability", ascending=False)["machine_id"].head(500))
    r = scored[scored["machine_id"] == mid].iloc[0]
    res = api_post("/predict", {"machine_id": mid, **{c: (r[c].item() if hasattr(r[c], "item") else r[c])
                                                      for c in SENSOR_COLS}})
    det = pd.DataFrame(res["factor_details"])
    st.write(f"Failure probability {res['failure_probability']:.1%} | risk {res['risk_level']}")
    if not det.empty:
        st.plotly_chart(px.bar(det, x="contribution", y="feature", orientation="h", color="direction"),
                        use_container_width=True)
        st.dataframe(det, use_container_width=True)

else:
    st.title("Model performance (held-out test set)")
    m = load_report("metrics.json")
    if not m:
        st.warning("metrics.json not found; run training.")
        st.stop()
    served = m["served_model"]
    st.caption(f"Served model: {served}. Test rows: {m.get('test_rows')}, failures: {m.get('test_failures')}.")
    res = m["models"][served]
    c = st.columns(6)
    for col, key in zip(c, ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]):
        col.metric(key.upper().replace("_", "-"), f"{res[key]:.3f}")
    table = pd.DataFrame([{"model": n, **{k: v[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]}}
                          for n, v in m["models"].items()])
    st.dataframe(table.round(4), use_container_width=True)
    cm = res["confusion_matrix"]
    st.plotly_chart(px.imshow([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], text_auto=True,
                              x=["Pred 0", "Pred 1"], y=["True 0", "True 1"], title="Confusion matrix"),
                    use_container_width=True)
    for key, title, xl, yl in [("roc", "ROC curve", "FPR", "TPR"), ("pr", "Precision-recall curve", "Recall", "Precision")]:
        fig = go.Figure()
        for name, cv in m["curves"].items():
            fig.add_trace(go.Scatter(x=cv[key]["x"], y=cv[key]["y"], mode="lines", name=name))
        fig.update_layout(title=title, xaxis_title=xl, yaxis_title=yl)
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Why these metrics?"):
        for k, v in m["metric_explanations"].items():
            st.markdown(f"**{k}**: {v}")
