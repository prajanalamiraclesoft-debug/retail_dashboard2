# app.py — run with: streamlit run app.py
# Retail Fraud Dashboard — threshold slider only (no date range)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, auc
)

# ───────────────── Page / Altair
st.set_page_config("Retail Fraud Dashboard", layout="wide")
alt.renderers.set_embed_options(actions=False)

# ───────────────── Sidebar (threshold only)
with st.sidebar:
    st.header("BQ TABLE INFO")
    P  = st.text_input("Project", "mss-data-engineer-sandbox")
    D  = st.text_input("Dataset", "retail")
    PT = st.text_input("Predictions table", f"{P}.{D}.predictions_latest")
    FT = st.text_input("Features table",   f"{P}.{D}.features_signals_v4")
    MT = st.text_input("(Optional) Metrics table", f"{P}.{D}.predictions_daily_metrics")
    TH = st.slider("ALERT THRESHOLD (≥)", 0.00, 1.00, 0.30, 0.01)
    st.caption("Threshold affects alerts, tables, and evaluation below.")

# ───────────────── BigQuery client
sa = dict(st.secrets["gcp_service_account"])
sa["private_key"] = sa["private_key"].replace("\\n", "\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq = bigquery.Client(credentials=creds, project=creds.project_id)

# ───────────────── Helpers (schema aware & cache-safe)
def _split_table_id(table_id: str):
    parts = table_id.split(".")
    if len(parts) != 3:
        raise ValueError(f"Bad table id: {table_id} (use project.dataset.table)")
    return parts[0], parts[1], parts[2]

@st.cache_data(show_spinner=False)
def get_columns(table_id: str) -> set:
    """Return lowercase set of columns for the table. No client passed (hash-safe)."""
    proj, dset, tbl = _split_table_id(table_id)
    sql = f"""
      SELECT column_name
      FROM `{proj}.{dset}.INFORMATION_SCHEMA.COLUMNS`
      WHERE table_name = @T
    """
    job = bq.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("T", "STRING", tbl)]
        ),
    )
    return set(job.result().to_dataframe()["column_name"].str.lower().tolist())

@st.cache_data(show_spinner=True)
def load_df(pred_table: str, feat_table: str) -> pd.DataFrame:
    """SAFE SELECT that only references columns that exist. No date filter."""
    pcols = get_columns(pred_table)
    fcols = get_columns(feat_table)

    sel = ["p.order_id"]

    if "timestamp" in pcols:
        sel.append("p.timestamp")
    else:
        sel.append("TIMESTAMP(NULL) AS timestamp")

    sel.append("CAST(p.fraud_score AS FLOAT64) AS fraud_score" if "fraud_score" in pcols
               else "CAST(0.0 AS FLOAT64) AS fraud_score")

    for c in ["customer_id","store_id","sku_id","sku_category",
              "order_amount","quantity","payment_method",
              "shipping_country","ip_country"]:
        sel.append(f"p.{c}" if c in pcols else f"CAST(NULL AS STRING) AS {c}")

    joins = ""
    strongs = [
        "strong_tri_mismatch_high_value","strong_high_value_express_geo",
        "strong_burst_multi_device","strong_price_drop_bulk","strong_giftcard_geo",
        "strong_return_whiplash","strong_price_inventory_stress","strong_country_flip_express",
        "high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag",
        "fraud_flag"
    ]
    if fcols:
        joins = f"LEFT JOIN `{feat_table}` s USING(order_id)"
        for c in strongs:
            if c in fcols:
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
      FROM `{pred_table}` p
      {joins}
      ORDER BY timestamp
    """
    d = bq.query(sql).result().to_dataframe()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce").dt.tz_localize(None)
    d["fraud_score"] = pd.to_numeric(d["fraud_score"], errors="coerce").fillna(0.0)
    return d

# ───────────────── Load & prepare
df = load_df(PT, FT)
if df.empty:
    st.warning("No rows available.")
    st.stop()

df["is_alert"] = (df["fraud_score"] >= TH).astype(int)

# ───────────────── Header KPIs
st.title("Retail Fraud Dashboard")

k1, k2, k3, k4 = st.columns([1,1,1,2])
TOT = len(df)
AL  = int(df["is_alert"].sum())
k1.metric("TOTAL ROWS", TOT)
k2.metric("ALERTS (≥ THRESHOLD)", AL)
k3.metric("ALERT RATE", f"{(AL/TOT if TOT else 0):.2%}")
win_text = ""
if "timestamp" in df.columns and df["timestamp"].notna().any():
    win_text = f"{df['timestamp'].min()} → {df['timestamp'].max()}"
k4.caption(f"TABLE WINDOW: {win_text}  |  THRESHOLD = {TH:.2f}")
st.markdown("---")

# ───────────────── Score distribution
st.subheader("Fraud Score Distribution")
hist = alt.Chart(df).mark_bar().encode(
    x=alt.X("fraud_score:Q", bin=alt.Bin(maxbins=50), title="Fraud score"),
    y=alt.Y("count():Q", title="Rows"),
    tooltip=[alt.Tooltip("count()", title="Rows")]
).properties(height=220)
rule = alt.Chart(pd.DataFrame({"x":[TH]})).mark_rule(color="crimson").encode(x="x")
st.altair_chart(hist + rule, use_container_width=True)

# ───────────────── Strong signal prevalence
st.subheader("Strong Signal Prevalence")

def prevalence(cols, title):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        st.info(f"No columns found for: {title}")
        return
    z = df[["is_alert"] + cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    a = int(z["is_alert"].sum()) or 1
    na = int((1 - z["is_alert"]).sum()) or 1
    rows = [{"signal": c,
             "% in alerts": z.loc[z.is_alert == 1, c].sum() / a,
             "% in non-alerts": z.loc[z.is_alert == 0, c].sum() / na}
            for c in cols]
    dd = (pd.DataFrame(rows)
          .sort_values("% in alerts", ascending=False)
          .melt(id_vars="signal", var_name="group", value_name="value"))
    chart = alt.Chart(dd).mark_bar().encode(
        x=alt.X("value:Q", axis=alt.Axis(format="%"), title="Prevalence"),
        y=alt.Y("signal:N", sort="-x", title=None),
        color="group:N",
        tooltip=["signal:N", alt.Tooltip("value:Q", format=".1%"), "group:N"]
    ).properties(height=300, title=title)
    st.altair_chart(chart, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    prevalence(
        [
            "strong_tri_mismatch_high_value","strong_high_value_express_geo",
            "strong_burst_multi_device","strong_price_drop_bulk","strong_giftcard_geo",
            "strong_return_whiplash","strong_price_inventory_stress","strong_country_flip_express",
        ],
        "Fraud Pattern Flags"
    )
with c2:
    prevalence(
        ["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"],
        "Pricing & Inventory Context"
    )

# ───────────────── Top alerts
st.subheader("Top Alerts (by score)")
cols = [c for c in [
    "order_id","timestamp","customer_id","store_id","sku_id","sku_category",
    "order_amount","quantity","payment_method","shipping_country","ip_country","fraud_score"
] if c in df.columns]
st.dataframe(
    df.sort_values(["fraud_score","timestamp"], ascending=[False, False]).loc[:, cols].head(50),
    use_container_width=True, height=340
)

# ───────────────── Evaluation
st.subheader("Model Evaluation")
label_candidates = [c for c in ["fraud_flag","is_fraud","label","ground_truth","gt","y"] if c in df.columns]
if label_candidates:
    lab = st.selectbox("Ground truth (1=fraud, 0=legit)", label_candidates, index=0)
    y_true = df[lab].fillna(0).astype(int).values
    st.caption(f"Using label column: `{lab}`")
else:
    st.warning("No label column found; using alert decision as a proxy for demo.")
    y_true = df["is_alert"].values

y_pred  = df["is_alert"].values
y_score = df["fraud_score"].values

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy",  f"{accuracy_score(y_true, y_pred):.2%}")
m2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.2%}")
m3.metric("Recall",    f"{recall_score(y_true, y_pred, zero_division=0):.2%}")
m4.metric("F1-score",  f"{f1_score(y_true, y_pred, zero_division=0):.2%}")

cm = confusion_matrix(y_true, y_pred, labels=[0,1])
cm_long = (pd.DataFrame(cm, index=["Actual: 0","Actual: 1"], columns=["Pred: 0","Pred: 1"])
           .reset_index().melt(id_vars="index", var_name="Predicted", value_name="Count")
           .rename(columns={"index":"Actual"}))
st.altair_chart(
    alt.Chart(cm_long).mark_rect().encode(
        x="Predicted:N", y="Actual:N",
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["Actual","Predicted","Count"]
    ).properties(height=180),
    use_container_width=True
)

try:
    auc_roc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    st.altair_chart(
        alt.Chart(pd.DataFrame({"fpr": fpr, "tpr": tpr}))
        .mark_line().encode(x=alt.X("fpr:Q", title="False Positive Rate"),
                            y=alt.Y("tpr:Q", title="True Positive Rate"))
        .properties(height=220, title=f"ROC (AUC = {auc_roc:.3f})"),
        use_container_width=True
    )
except Exception:
    pass

prec, rec, _ = precision_recall_curve(y_true, y_score)
ap = auc(rec, prec)
st.altair_chart(
    alt.Chart(pd.DataFrame({"recall": rec, "precision": prec}))
    .mark_line().encode(x="recall:Q", y="precision:Q")
    .properties(height=220, title=f"Precision–Recall (AP ≈ {ap:.3f})"),
    use_container_width=True
)

# ───────────────── Operating point helper (optional metrics table, no date filter)
@st.cache_data(show_spinner=False)
def load_ops(metrics_table: str) -> pd.DataFrame:
    proj, dset, tbl = _split_table_id(metrics_table)
    sql = f"""
      SELECT threshold, AVG(precision) AS precision, AVG(recall) AS recall
      FROM `{metrics_table}`
      GROUP BY threshold
      ORDER BY threshold
    """
    return bq.query(sql).result().to_dataframe()

st.subheader("Operating Point Helper")
use_bq = True
try:
    OP = load_ops(MT)
    use_bq = not OP.empty
except Exception:
    use_bq = False

if not use_bq:
    # Fallback sweep if no metrics table
    grid = np.round(np.linspace(0.01, 0.99, 33), 2)
    y_true_fallback = y_true
    OP = pd.DataFrame([{
        "threshold": t,
        "precision": (((df["fraud_score"] >= t) & (y_true_fallback == 1)).sum()
                       / max(1, (df["fraud_score"] >= t).sum())),
        "recall": (((df["fraud_score"] >= t) & (y_true_fallback == 1)).sum()
                    / max(1, (y_true_fallback == 1).sum()))
    } for t in grid])

st.altair_chart(
    alt.Chart(OP.melt(id_vars="threshold", value_vars=["precision","recall"],
                      var_name="metric", value_name="value"))
    .mark_line(point=True)
    .encode(x="threshold:Q", y=alt.Y("value:Q", axis=alt.Axis(format="%")),
            color="metric:N", tooltip=["threshold:Q", alt.Tooltip("value:Q", format=".1%"), "metric:N"])
    .properties(height=220, title=("BigQuery metrics" if use_bq else "Local sweep")),
    use_container_width=True
)

st.caption("This dashboard is schema-aware and uses the threshold slider for decisions and evaluation.")

