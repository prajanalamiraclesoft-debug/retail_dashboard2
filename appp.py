# app.py  —  run with:  streamlit run app.py
# RETAIL FRAUD DASHBOARD (BIGQUERY • SCHEMA-AWARE • INTERNAL THRESHOLD = 0.02)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, auc
)

# ───────────────────────────── PAGE / THEME ─────────────────────────────
st.set_page_config("RETAIL FRAUD DASHBOARD", layout="wide")
alt.renderers.set_embed_options(actions=False)

# INTERNAL OPERATING THRESHOLD (NOT SHOWN ON UI)
THRESH = 0.02  # <— decision boundary used for alerts & evaluation

# ───────────────────────────── SIDEBAR: BQ TABLE INFO ─────────────────────────────
with st.sidebar:
    st.header("BQ TABLE INFO")
    P = st.text_input("Project", "mss-data-engineer-sandbox")
    D = st.text_input("Dataset", "retail")
    PT = st.text_input("Predictions table", f"{P}.{D}.predictions_latest")
    FT = st.text_input("Features table",   f"{P}.{D}.features_signals_v4")
    MT = st.text_input("(Optional) Daily metrics", f"{P}.{D}.predictions_daily_metrics")
    S  = st.date_input("Start", date(2023, 1, 1))
    E  = st.date_input("End",   date(2024, 12, 31))
    st.caption("The app automatically adapts to each table's actual schema.")

# ───────────────────────────── BIGQUERY CLIENT ─────────────────────────────
sa = dict(st.secrets["gcp_service_account"])
sa["private_key"] = sa["private_key"].replace("\\n", "\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq = bigquery.Client(credentials=creds, project=creds.project_id)

# ───────────────────────────── SCHEMA-AWARE HELPERS ─────────────────────────────
def _split_table_id(table_id: str):
    parts = table_id.split(".")
    if len(parts) != 3:
        raise ValueError(f"Bad table id: {table_id}. Use project.dataset.table")
    return parts[0], parts[1], parts[2]

@st.cache_data(show_spinner=False)
def _get_columns(client, table_id: str) -> set:
    proj, dset, tbl = _split_table_id(table_id)
    sql = f"""
      SELECT column_name
      FROM `{proj}.{dset}.INFORMATION_SCHEMA.COLUMNS`
      WHERE table_name = @T
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("T", "STRING", tbl)]
        ),
    )
    return set(job.result().to_dataframe()["column_name"].str.lower().tolist())

@st.cache_data(show_spinner=True)
def load_df_safe(PRED_T, FEAT_T, S, E) -> pd.DataFrame:
    """Builds a SAFE SELECT that only references columns that exist in BQ."""
    p_cols = _get_columns(bq, PRED_T)
    f_cols = _get_columns(bq, FEAT_T)

    sel = []
    joins = ""
    where = ""

    # Always include identifiers
    sel.append("p.order_id")

    # Optional timestamp
    if "timestamp" in p_cols:
        sel.append("p.timestamp")
        where = "WHERE DATE(p.timestamp) BETWEEN @S AND @E"
    else:
        sel.append("TIMESTAMP(NULL) AS timestamp")

    # Fraud score
    if "fraud_score" in p_cols:
        sel.append("CAST(p.fraud_score AS FLOAT64) AS fraud_score")
    else:
        sel.append("CAST(0.0 AS FLOAT64) AS fraud_score")

    # Lightweight context if present
    for c in ["customer_id","store_id","sku_id","sku_category",
              "order_amount","quantity","payment_method",
              "shipping_country","ip_country"]:
        if c in p_cols:
            sel.append(f"p.{c}")

    # Join features table if available
    strongs = [
        "strong_tri_mismatch_high_value","strong_high_value_express_geo",
        "strong_burst_multi_device","strong_price_drop_bulk","strong_giftcard_geo",
        "strong_return_whiplash","strong_price_inventory_stress","strong_country_flip_express",
        "high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag",
        "fraud_flag"
    ]
    if f_cols:
        joins = f"LEFT JOIN `{FEAT_T}` s USING(order_id)"
        for c in strongs:
            if c in f_cols:
                sel.append(f"s.{c}")
            else:
                sel.append(("CAST(NULL AS INT64) AS fraud_flag") if c == "fraud_flag"
                           else f"CAST(0 AS INT64) AS {c}")
    else:
        for c in strongs:
            sel.append(("CAST(NULL AS INT64) AS fraud_flag") if c == "fraud_flag"
                       else f"CAST(0 AS INT64) AS {c}")

    sql = f"""
      SELECT {", ".join(sel)}
      FROM `{PRED_T}` p
      {joins}
      {where}
      ORDER BY timestamp
    """
    params = []
    if where:
        params = [
            bigquery.ScalarQueryParameter("S","DATE", str(S)),
            bigquery.ScalarQueryParameter("E","DATE", str(E)),
        ]

    job = bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
    d = job.result().to_dataframe()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce").dt.tz_localize(None)
    d["fraud_score"] = pd.to_numeric(d["fraud_score"], errors="coerce").fillna(0.0)
    return d

# ───────────────────────────── LOAD DATA ─────────────────────────────
df = load_df_safe(PT, FT, S, E)
if df.empty:
    st.warning("NO ROWS IN THIS WINDOW.")
    st.stop()

# Decision column (INTERNAL THRESHOLD)
df["is_alert"] = (df["fraud_score"] >= THRESH).astype(int)

# ───────────────────────────── HEADER / KPIs ─────────────────────────────
st.title("RETAIL FRAUD DASHBOARD")

k1, k2, k3, k4 = st.columns([1, 1, 1, 2])
TOT = len(df)
AL  = int(df["is_alert"].sum())
k1.metric("TOTAL ROWS SCORED", TOT)
k2.metric("TOTAL ALERTS", AL)
k3.metric("ALERT RATE", f"{(AL / TOT if TOT else 0):.2%}")
window_txt = ""
if "timestamp" in df.columns and df["timestamp"].notna().any():
    window_txt = f"{df['timestamp'].min()} → {df['timestamp'].max()}"
k4.caption(f"SCORING WINDOW: {window_txt}")

st.markdown("---")

# ───────────────────────────── SCORE DISTRIBUTION ─────────────────────────────
st.subheader("FRAUD SCORE DISTRIBUTION")
hist = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("fraud_score:Q", bin=alt.Bin(maxbins=50), title="Fraud score"),
        y=alt.Y("count():Q", title="Rows"),
        tooltip=[alt.Tooltip("count()", title="Rows")],
    )
    .properties(height=220)
)
st.altair_chart(hist, use_container_width=True)

# ───────────────────────────── SIGNAL PREVALENCE (STRONG FEATURES) ─────────────────────────────
st.subheader("STRONG SIGNAL PREVALENCE")
left, right = st.columns(2)

def prevalence(subcols, title):
    sub = [c for c in subcols if c in df.columns]
    if not sub:
        st.info(f"No available columns for: {title}")
        return
    zz = df[["is_alert"] + sub].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    a = int(zz["is_alert"].sum()) or 1
    na = int((1 - zz["is_alert"]).sum()) or 1
    rows = [
        {"signal": c,
         "% in alerts": zz.loc[zz.is_alert == 1, c].sum() / a,
         "% in non-alerts": zz.loc[zz.is_alert == 0, c].sum() / na}
        for c in sub
    ]
    dd = (
        pd.DataFrame(rows)
        .sort_values("% in alerts", ascending=False)
        .melt(id_vars="signal", var_name="group", value_name="value")
    )
    chart = (
        alt.Chart(dd)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", axis=alt.Axis(format="%"), title="Prevalence"),
            y=alt.Y("signal:N", sort="-x", title=None),
            color="group:N",
            tooltip=["signal:N", alt.Tooltip("value:Q", format=".1%"), "group:N"],
        )
        .properties(height=300, title=title)
    )
    st.altair_chart(chart, use_container_width=True)

with left:
    prevalence(
        [
            "strong_tri_mismatch_high_value","strong_high_value_express_geo",
            "strong_burst_multi_device","strong_price_drop_bulk","strong_giftcard_geo",
            "strong_return_whiplash","strong_price_inventory_stress","strong_country_flip_express",
        ],
        "STRONG RULES (FRAUD PATTERN FLAGS)",
    )

with right:
    prevalence(
        ["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"],
        "PRICING & INVENTORY CONTEXT",
    )

# ───────────────────────────── CORRELATIONS (OPTIONAL) ─────────────────────────────
st.subheader("CORRELATION OF FRAUD SCORE WITH PRICING/INVENTORY SIGNALS")
cand = [c for c in ["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"] if c in df.columns]
if cand:
    co = (
        df[["fraud_score"] + cand]
        .apply(pd.to_numeric, errors="coerce").fillna(0)
        .corr().loc[cand, "fraud_score"]
        .reset_index().rename(columns={"index": "signal", "fraud_score": "corr"})
    )
    corr_chart = (
        alt.Chart(co)
        .mark_bar()
        .encode(x="corr:Q", y=alt.Y("signal:N", sort="x", title=None),
                tooltip=[alt.Tooltip("corr:Q", format=".3f")])
        .properties(height=140)
    )
    st.altair_chart(corr_chart, use_container_width=True)
else:
    st.info("No pricing/inventory columns available to correlate.")

# ───────────────────────────── TOP ALERTS TABLE ─────────────────────────────
st.subheader("TOP ALERTS")
view_cols = [c for c in [
    "order_id","timestamp","customer_id","store_id","sku_id","sku_category",
    "order_amount","quantity","payment_method","shipping_country","ip_country","fraud_score"
] if c in df.columns]
st.dataframe(
    df.sort_values(["fraud_score", "timestamp"], ascending=[False, False]).loc[:, view_cols].head(50),
    use_container_width=True,
    height=340,
)

# ───────────────────────────── MODEL EVALUATION ─────────────────────────────
st.subheader("MODEL EVALUATION (IF LABEL PRESENT)")
label_candidates = [c for c in ["fraud_flag", "is_fraud", "label", "ground_truth", "gt", "y"] if c in df.columns]

if label_candidates:
    lab = st.selectbox("GROUND-TRUTH COLUMN (1=FRAUD, 0=LEGIT)", label_candidates, index=0)
    y_true = df[lab].fillna(0).astype(int).values
else:
    st.info("No ground-truth label found. Using the alert decision as a proxy for demo metrics.")
    y_true = df["is_alert"].values  # proxy

y_pred  = df["is_alert"].values
y_score = df["fraud_score"].values

m1, m2, m3, m4 = st.columns(4)
m1.metric("ACCURACY",  f"{accuracy_score(y_true, y_pred):.2%}")
m2.metric("PRECISION", f"{precision_score(y_true, y_pred, zero_division=0):.2%}")
m3.metric("RECALL",    f"{recall_score(y_true, y_pred, zero_division=0):.2%}")
m4.metric("F1-SCORE",  f"{f1_score(y_true, y_pred, zero_division=0):.2%}")

# Confusion matrix heatmap
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
cm_long = (
    pd.DataFrame(cm, index=["Actual: 0", "Actual: 1"], columns=["Pred: 0", "Pred: 1"])
    .reset_index().melt(id_vars="index", var_name="Predicted", value_name="Count")
    .rename(columns={"index": "Actual"})
)
st.altair_chart(
    alt.Chart(cm_long).mark_rect().encode(
        x="Predicted:N", y="Actual:N",
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["Actual", "Predicted", "Count"]
    ).properties(height=180),
    use_container_width=True
)

# ROC & PR curves (if there is positive class)
try:
    auc_roc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    st.altair_chart(
        alt.Chart(pd.DataFrame({"fpr": fpr, "tpr": tpr}))
        .mark_line().encode(x=alt.X("fpr:Q", title="False Positive Rate"),
                            y=alt.Y("tpr:Q", title="True Positive Rate"))
        .properties(height=220, title=f"ROC CURVE • AUC = {auc_roc:.3f}"),
        use_container_width=True,
    )
except Exception:
    pass

prec, rec, _ = precision_recall_curve(y_true, y_score)
ap = auc(rec, prec)
st.altair_chart(
    alt.Chart(pd.DataFrame({"recall": rec, "precision": prec}))
    .mark_line().encode(x="recall:Q", y="precision:Q")
    .properties(height=220, title=f"PRECISION–RECALL • AP ≈ {ap:.3f}"),
    use_container_width=True,
)

