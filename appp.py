# app.py — Retail Fraud Dashboard (fixed internal threshold, never shown)
# Run: streamlit run app.py

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

# ───────────────────────────── Page / Altair
st.set_page_config("Retail Fraud Dashboard", layout="wide")
alt.renderers.set_embed_options(actions=False)

# Fixed decision threshold (kept internal only; never rendered)
TH = 0.50

# ───────────────────────────── Sidebar (no date, no threshold controls)
with st.sidebar:
    st.header("BQ TABLE INFO")
    P  = st.text_input("Project", "mss-data-engineer-sandbox")
    D  = st.text_input("Dataset", "retail")
    PT = st.text_input("Predictions table", f"{P}.{D}.predictions_latest")
    FT = st.text_input("Features table",   f"{P}.{D}.features_signals_v4")

# ───────────────────────────── BigQuery client
sa = dict(st.secrets["gcp_service_account"])
sa["private_key"] = sa["private_key"].replace("\\n", "\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq = bigquery.Client(credentials=creds, project=creds.project_id)

# ───────────────────────────── Helpers (schema-aware)
def _split_table_id(table_id: str):
    parts = table_id.split(".")
    if len(parts) != 3:
        raise ValueError(f"Bad table id: {table_id} (use project.dataset.table)")
    return parts[0], parts[1], parts[2]

@st.cache_data(show_spinner=False)
def get_columns(table_id: str) -> set:
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
    pcols = get_columns(pred_table)
    fcols = get_columns(feat_table)

    sel = ["p.order_id"]
    sel.append("p.timestamp" if "timestamp" in pcols else "TIMESTAMP(NULL) AS timestamp")
    sel.append("CAST(p.fraud_score AS FLOAT64) AS fraud_score" if "fraud_score" in pcols
               else "CAST(0.0 AS FLOAT64) AS fraud_score")

    info_cols = ["customer_id","store_id","sku_id","sku_category",
                 "order_amount","quantity","payment_method",
                 "shipping_country","ip_country"]
    for c in info_cols:
        sel.append(f"p.{c}" if c in pcols else f"CAST(NULL AS STRING) AS {c}")

    strongs = [
        "strong_tri_mismatch_high_value","strong_high_value_express_geo",
        "strong_burst_multi_device","strong_price_drop_bulk","strong_giftcard_geo",
        "strong_return_whiplash","strong_price_inventory_stress","strong_country_flip_express",
        "high_price_anomaly","low_price_anomaly",
        "oversell_flag","stockout_risk_flag","hoarding_flag",
        "fraud_flag"
    ]

    joins = ""
    if get_columns(feat_table):
        joins = f"LEFT JOIN `{feat_table}` s USING(order_id)"
        for c in strongs:
            sel.append(f"s.{c}" if c in get_columns(feat_table) else
                       ("CAST(NULL AS INT64) AS fraud_flag" if c=="fraud_flag" else f"CAST(0 AS INT64) AS {c}"))
    else:
        for c in strongs:
            sel.append(("CAST(NULL AS INT64) AS fraud_flag" if c=="fraud_flag" else f"CAST(0 AS INT64) AS {c}"))

    sql = f"""
      SELECT {", ".join(sel)}
      FROM `{pred_table}` p
      {joins}
      ORDER BY timestamp
    """
    d = bq.query(sql).result().to_dataframe()
    d["timestamp"]   = pd.to_datetime(d["timestamp"], errors="coerce").dt.tz_localize(None)
    d["fraud_score"] = pd.to_numeric(d["fraud_score"], errors="coerce").fillna(0.0)
    d["geo_mismatch"] = (
        (d["shipping_country"].astype(str) != d["ip_country"].astype(str))
        if {"shipping_country","ip_country"}.issubset(d.columns) else 0
    ).astype(int)
    return d

# ───────────────────────────── Load & prepare
df = load_df(PT, FT)
if df.empty:
    st.warning("No rows available.")
    st.stop()

df["is_alert"] = (df["fraud_score"] >= TH).astype(int)

# ───────────────────────────── Header KPIs
st.title("Retail Fraud Dashboard")

k1, k2, k3, k4 = st.columns([1,1,1,2])
TOT = len(df); AL = int(df["is_alert"].sum())
k1.metric("TOTAL ROWS", TOT)
k2.metric("ALERTS", AL)
k3.metric("ALERT RATE", f"{(AL/TOT if TOT else 0):.2%}")
win_text = f"{df['timestamp'].min()} → {df['timestamp'].max()}" if df["timestamp"].notna().any() else ""
k4.caption(f"TABLE WINDOW: {win_text}")
st.markdown("---")

# ───────────────────────────── Strong features (plain English, one sentence each)
st.subheader("STRONG FEATURES (what they mean)")
st.markdown("""
- **strong_tri_mismatch_high_value** — order amount is in the top spend band and the shipping country does not match the IP country.  
- **strong_high_value_express_geo** — top spend combined with an express/geo pattern that is frequently abused.  
- **strong_burst_multi_device** — a short-time burst of orders from multiple devices tied to the same identity.  
- **strong_price_drop_bulk** — many units purchased while the unit price is unusually low for the category.  
- **strong_giftcard_geo** — gift card usage combined with a location pattern linked to abuse.  
- **strong_return_whiplash** — abnormal swings in returns following purchases.  
- **strong_price_inventory_stress** — pricing behavior aligned with inventory stress signals.  
- **strong_country_flip_express** — recent country change paired with express behavior used in fraud runs.  
- **high_price_anomaly** — total amount is unusually high for the category baseline.  
- **low_price_anomaly** — total amount is unusually low for the category baseline.  
- **oversell_flag / stockout_risk_flag / hoarding_flag** — purchase pattern adds inventory risk context.  
- **geo_mismatch** — shipping country and IP country are different on the order.
""")

# ───────────────────────────── Score distribution (no threshold line)
st.subheader("Fraud Score Distribution")
st.altair_chart(
    alt.Chart(df).mark_bar().encode(
        x=alt.X("fraud_score:Q", bin=alt.Bin(maxbins=50), title="Fraud score"),
        y=alt.Y("count():Q", title="Rows"),
        tooltip=[alt.Tooltip("count()", title="Rows")]
    ).properties(height=220),
    use_container_width=True
)

# ───────────────────────────── Strong-signal prevalence
st.subheader("Strong Signal Prevalence in Alerts vs Non-alerts")

def prevalence(cols, title):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        st.info(f"No columns found for: {title}")
        return
    z  = df[["is_alert"] + cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    a  = int(z["is_alert"].sum()) or 1
    na = int((1 - z["is_alert"]).sum()) or 1
    rows = [{"signal": c,
             "% in alerts":     z.loc[z.is_alert == 1, c].sum()/a,
             "% in non-alerts": z.loc[z.is_alert == 0, c].sum()/na}
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

left, right = st.columns(2)
with left:
    prevalence(
        [
            "strong_tri_mismatch_high_value","strong_high_value_express_geo",
            "strong_burst_multi_device","strong_price_drop_bulk","strong_giftcard_geo",
            "strong_return_whiplash","strong_price_inventory_stress","strong_country_flip_express",
        ],
        "Fraud pattern flags"
    )
with right:
    prevalence(
        ["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag","geo_mismatch"],
        "Pricing / Inventory / Geo context"
    )

# ───────────────────────────── Alerts table (entire set)
st.subheader("All Alerts")
cols_show = [c for c in [
    "order_id","timestamp","customer_id","store_id","sku_id","sku_category",
    "order_amount","quantity","payment_method","shipping_country","ip_country","fraud_score",
    "strong_tri_mismatch_high_value","strong_high_value_express_geo","strong_burst_multi_device",
    "strong_price_drop_bulk","strong_giftcard_geo","strong_return_whiplash",
    "strong_price_inventory_stress","strong_country_flip_express",
    "high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag",
    "geo_mismatch"
] if c in df.columns]
alerts = df[df["is_alert"] == 1].sort_values(["fraud_score","timestamp"], ascending=[False, False])
st.caption(f"Showing **{len(alerts)}** alert(s). Download below if needed.")
st.dataframe(alerts.loc[:, cols_show], use_container_width=True, height=min(700, 28*max(8, len(alerts))))
st.download_button("Download alerts (.csv)", alerts.loc[:, cols_show].to_csv(index=False).encode("utf-8"),
                   file_name="alerts.csv")

# ───────────────────────────── Model evaluation (threshold not displayed)
st.subheader("Model Evaluation")
label_candidates = [c for c in ["fraud_flag","is_fraud","label","ground_truth","gt","y"] if c in df.columns]
if label_candidates:
    lab = st.selectbox("Ground truth (1=fraud, 0=legit)", label_candidates, index=0)
    y_true = df[lab].fillna(0).astype(int).values
    st
