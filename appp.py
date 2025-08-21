# streamlit run app.py
import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

# ===================== PAGE & LIBS =====================
st.set_page_config("Retail Dashboard: Fraud & Inventory", layout="wide")
alt.data_transformers.enable("default", max_rows=None)
alt.renderers.set_embed_options(actions=False)

# ===================== BIGQUERY CLIENT (BEFORE SIDEBAR) =====================
sa = dict(st.secrets["gcp_service_account"])
sa["private_key"] = sa["private_key"].replace("\\n", "\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq = bigquery.Client(credentials=creds, project=creds.project_id)

# ===================== CONSTANT: FIXED THRESHOLD =====================
# Per manager feedback: ALL metrics and flags are based ONLY on threshold = 0.30
THRESHOLD = 0.30

# ===================== HELPERS =====================
@st.cache_data(show_spinner=False)
def get_min_max_ts(table_fqdn: str):
    sql = f"SELECT MIN(DATE(timestamp)) AS min_d, MAX(DATE(timestamp)) AS max_d FROM `{table_fqdn}`"
    df = bq.query(sql).result().to_dataframe()
    if df.empty:
        return date(2023, 1, 1), date.today()
    row = df.iloc[0]
    min_d = row["min_d"] if pd.notna(row["min_d"]) else date(2023, 1, 1)
    max_d = row["max_d"] if pd.notna(row["max_d"]) else date.today()
    return min_d, max_d

@st.cache_data(show_spinner=True)
def load_df(PT, FT, S, E):
    # Only rows in selected date range are returned.
    sql = f"""
    SELECT
      p.order_id, p.timestamp, p.customer_id, p.store_id, p.sku_id, p.sku_category,
      p.order_amount, p.quantity, p.payment_method, p.shipping_country, p.ip_country,
      CAST(p.fraud_score AS FLOAT64) AS fraud_score,
      s.strong_tri_mismatch_high_value, s.strong_high_value_express_geo, s.strong_burst_multi_device,
      s.strong_price_drop_bulk, s.strong_giftcard_geo, s.strong_return_whiplash,
      s.strong_price_inventory_stress, s.strong_country_flip_express,
      s.high_price_anomaly, s.low_price_anomaly, s.oversell_flag, s.stockout_risk_flag, s.hoarding_flag,
      SAFE_CAST(s.fraud_flag AS INT64) AS fraud_flag
    FROM `{PT}` p
    LEFT JOIN `{FT}` s USING(order_id)
    WHERE DATE(p.timestamp) BETWEEN @S AND @E
    ORDER BY p.timestamp
    """
    j = bq.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("S", "DATE", str(S)),
                bigquery.ScalarQueryParameter("E", "DATE", str(E)),
            ]
        ),
    )
    d = j.result().to_dataframe()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce").dt.tz_localize(None)
    return d

# ===================== SIDEBAR (DATA RANGE IS DYNAMIC) =====================
with st.sidebar:
    st.header("**BigQuery Tables**")
    P = st.text_input("Project", "mss-data-engineer-sandbox")
    D = st.text_input("Dataset", "retail")
    PT = st.text_input("Predictions", f"{P}.{D}.predictions_latest")
    FT = st.text_input("Features", f"{P}.{D}.features_signals_v4")
    MT = st.text_input("Metrics (optional)", f"{P}.{D}.predictions_daily_metrics")

    # Data bounds from table
    try:
        data_min, data_max = get_min_max_ts(PT)
    except Exception:
        data_min, data_max = date(2023, 1, 1), date.today()

    CT = ZoneInfo("America/Chicago")
    today_ct = datetime.now(CT).date()

    range_choice = st.selectbox(
        "Quick range",
        ["Last 7 days", "Last 30 days", "This month", "Previous month", "YTD", "Today", "All data", "Custom"],
        index=1,
    )

    def month_bounds(d: date):
        first = d.replace(day=1)
        if d.month == 12:
            next_first = d.replace(year=d.year + 1, month=1, day=1)
        else:
            next_first = d.replace(month=d.month + 1, day=1)
        last = next_first - timedelta(days=1)
        return first, last

    if range_choice == "Today":
        S, E = today_ct, today_ct
    elif range_choice == "Last 7 days":
        S, E = today_ct - timedelta(days=6), today_ct
    elif range_choice == "Last 30 days":
        S, E = today_ct - timedelta(days=29), today_ct
    elif range_choice == "This month":
        m_first, m_last = month_bounds(today_ct)
        S, E = m_first, min(m_last, today_ct)
    elif range_choice == "Previous month":
        this_first, _ = month_bounds(today_ct)
        prev_last = this_first - timedelta(days=1)
        S, E = month_bounds(prev_last)
    elif range_choice == "YTD":
        S, E = date(today_ct.year, 1, 1), today_ct
    elif range_choice == "All data":
        S, E = data_min, data_max
    else:
        S = st.date_input("Start", value=max(data_min, today_ct - timedelta(days=29)), min_value=data_min, max_value=data_max)
        E = st.date_input("End", value=data_max, min_value=data_min, max_value=data_max)

# ===================== HEADER =====================
st.markdown("## **Retail Risk Dashboard — Fraud & Inventory**")
st.info("**All metrics, alerts, and decisions in this app use a FIXED threshold of 0.30 (score ≥ 0.30).**")

# ===================== LOAD DATA (STRICTLY FILTERED BY SELECTED DATES) =====================
df = load_df(PT, FT, S, E)
if df.empty:
    st.warning("**No rows in the selected date range. Adjust the dates and try again.**")
    st.stop()

# Derived columns
df["fraud_score"] = pd.to_numeric(df["fraud_score"], errors="coerce").fillna(0.0)
df["is_alert"] = (df["fraud_score"] >= THRESHOLD).astype(int)  # Fixed threshold
df["day"] = df["timestamp"].dt.floor("D")

# ===================== KPIs =====================
st.markdown("### **Key Performance Indicators (based on selected date range & threshold = 0.30)**")
c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
TOTAL = len(df)
ALERTS = int(df["is_alert"].sum())
c1.metric("**Total transactions**", TOTAL)
c2.metric("**Fraud alerts (score ≥ 0.30)**", ALERTS)
c3.metric("**Alert rate (≥ 0.30)**", f"{(ALERTS / TOTAL if TOTAL else 0):.2%}")
c4.caption(f"**Window:** {df['timestamp'].min()} → {df['timestamp'].max()}  |  **Threshold:** {THRESHOLD:.2f}")
st.markdown("---")

# ===================== DAILY TREND =====================
st.markdown("### **Daily Trend**")
st.caption("**What it shows:** Daily **total transactions** and **fraud alerts** within the **selected date range** only.")

tr = df.groupby("day").agg(total_txn=("order_id", "count"), fraud_alerts=("is_alert", "sum")).reset_index()
if len(tr):
    tl = tr.melt(id_vars="day", value_vars=["total_txn", "fraud_alerts"], var_name="series", value_name="value")
    tl["series"] = tl["series"].replace({"total_txn": "Total transactions", "fraud_alerts": "Fraud alerts (≥ 0.30)"})
    st.altair_chart(
        alt.Chart(tl)
        .mark_line(point=True)
        .encode(
            x=alt.X("day:T", title="Day"),
            y=alt.Y("value:Q", title="Count"),
            color=alt.Color("series:N", title="Series"),
            tooltip=[alt.Tooltip("day:T", title="Day"), "series:N", alt.Tooltip("value:Q", title="Count")],
        )
        .properties(height=260),
        use_container_width=True,
    )
else:
    st.info("**No activity in this range.**")

# ===================== SCORE DISTRIBUTION =====================
st.markdown("### **Fraud-Score Distribution**")
st.caption("**What it shows:** Histogram of the model’s fraud scores in the selected window. **Red line** marks the **fixed threshold (0.30)** used for alerts and metrics.")

hist = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("fraud_score:Q", bin=alt.Bin(maxbins=50), title="Fraud score"),
        y=alt.Y("count():Q", title="Rows"),
        tooltip=[alt.Tooltip("count()", title="Rows")],
    )
    .properties(height=200)
)
st.altair_chart(
    hist + alt.Chart(pd.DataFrame({"x": [THRESHOLD]})).mark_rule(color="crimson").encode(x="x"),
    use_container_width=True,
)

# ===================== CONTEXT SIGNALS =====================
st.markdown("### **Context Signals — Prevalence**")
st.caption("**What it shows:** For each signal, the share of **alerts** (score ≥ 0.30) and **non-alerts** that contain that signal. Helps explain *why* items are flagged.")

def prevalence_chart(cols, title):
    cols = [c for c in cols if c in df]
    if not cols:
        st.info(f"**No signals found for: {title}**")
        return
    Z = df[["is_alert"] + cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    a = int(Z["is_alert"].sum()) or 1
    na = int((1 - Z["is_alert"]).sum()) or 1
    rows = [{"signal": c,
             "% in alerts": Z.loc[Z.is_alert == 1, c].sum() / a,
             "% in non-alerts": Z.loc[Z.is_alert == 0, c].sum() / na} for c in cols]
    dd = (
        pd.DataFrame(rows)
        .sort_values("% in alerts", ascending=False)
        .melt(id_vars="signal", var_name="group", value_name="value")
    )
    st.altair_chart(
        alt.Chart(dd)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", axis=alt.Axis(format="%"), title="Prevalence"),
            y=alt.Y("signal:N", sort="-x", title=None),
            color=alt.Color("group:N", title=None),
            tooltip=["signal:N", alt.Tooltip("value:Q", format=".1%"), "group:N"],
        )
        .properties(title=title, height=320),
        use_container_width=True,
    )

left, right = st.columns(2)
with left:
    prevalence_chart(
        [
            "strong_tri_mismatch_high_value",      # High-value + billing/shipping/IP mismatch
            "strong_high_value_express_geo",       # High-value + express shipping + geo mismatch
            "strong_burst_multi_device",           # Many orders from multiple devices in short time
            "strong_price_drop_bulk",              # Big price drop + bulk quantity
            "strong_giftcard_geo",                 # Gift card usage with geo mismatch
            "strong_return_whiplash",              # Rapid return/re-purchase behavior
            "strong_price_inventory_stress",       # Price anomaly while inventory is stressed
            "strong_country_flip_express",         # Country flip with express shipping
        ],
        "**Strong-Signal Prevalence (Fraud risk indicators)**",
    )
with right:
    prevalence_chart(
        [
            "high_price_anomaly",                  # Price unusually high vs baseline
            "low_price_anomaly",                   # Price unusually low vs baseline
            "oversell_flag",                       # Demand surge vs stock → oversell risk
            "stockout_risk_flag",                  # Low inventory risk of stockout
            "hoarding_flag",                       # Abnormally large/bulk purchase
        ],
        "**Pricing & Inventory Prevalence (Operational risk)**",
    )

with st.expander("**What each signal means (quick reference)**", expanded=False):
    st.markdown("""
- **strong_tri_mismatch_high_value** — High order value plus **billing vs shipping vs IP** mismatch.  
- **strong_high_value_express_geo** — High value + **express** shipping + **geo** mismatch.  
- **strong_burst_multi_device** — Short-window activity bursts across **multiple devices**.  
- **strong_price_drop_bulk** — **Large price drop** combined with **bulk** quantity.  
- **strong_giftcard_geo** — **Gift card** usage with **geo** mismatch.  
- **strong_return_whiplash** — Fast **return/repurchase** cycles (abuse).  
- **strong_price_inventory_stress** — **Price anomaly** while **inventory is tight**.  
- **strong_country_flip_express** — Country flips plus **express** shipping.  
- **high_price_anomaly / low_price_anomaly** — Price far from normal (too high / too low).  
- **oversell_flag** — Risk of selling more units than available.  
- **stockout_risk_flag** — Inventory near depletion.  
- **hoarding_flag** — Abnormally large orders / quantities.
""")

# ===================== CORRELATIONS =====================
st.markdown("### **Correlation with Fraud Score**")
st.caption("**What it shows:** Pearson correlation between **fraud_score** and pricing/inventory signals (higher = more positively associated with risk).")
C = [c for c in ["high_price_anomaly", "low_price_anomaly", "oversell_flag", "stockout_risk_flag", "hoarding_flag"] if c in df]
if C:
    co = (
        df[["fraud_score"] + C]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .corr()
        .loc[C, "fraud_score"]
        .reset_index()
        .rename(columns={"index": "signal", "fraud_score": "corr"})
    )
    st.altair_chart(
        alt.Chart(co)
        .mark_bar()
        .encode(
            x=alt.X("corr:Q", title="Correlation"),
            y=alt.Y("signal:N", sort="x", title=None),
            tooltip=[alt.Tooltip("corr:Q", format=".3f")],
        )
        .properties(height=140),
        use_container_width=True,
    )
else:
    st.info("**No pricing/inventory columns available to correlate.**")

# ===================== TOP ALERTS (SCORE ≥ 0.30) =====================
st.markdown("### **Top Alerts (score ≥ 0.30)**")
st.caption("**What it shows:** Highest-risk transactions in the selected window for investigation.")
cols = [
    c for c in [
        "order_id","timestamp","customer_id","store_id","sku_id","sku_category",
        "order_amount","quantity","payment_method","shipping_country","ip_country","fraud_score"
    ] if c in df
]
st.dataframe(
    df[df["is_alert"] == 1]
      .sort_values(["fraud_score","timestamp"], ascending=[False, False])
      .loc[:, cols]
      .head(50),
    use_container_width=True, height=340
)

# ===================== MODEL EVALUATION (FIXED THRESHOLD 0.30) =====================
st.markdown("### **Model Evaluation (Threshold = 0.30 only)**")
st.caption("**What it shows:** Metrics computed at **exactly 0.30** threshold — **Accuracy, Precision, Recall, F1**, plus **ROC** and **PR** curves for overall ranking quality.")

# Choose ground-truth label if present
label_candidates = [c for c in ["fraud_flag","is_fraud","label","ground_truth","gt","y"] if c in df]
if label_candidates:
    lab = st.selectbox("**Ground-truth column (1=fraud, 0=legit)**", label_candidates, 0)
    st.caption(f"**Using label:** `{lab}`")
    y_true = df[lab].fillna(0).astype(int).values
else:
    st.warning("**No ground-truth label found; using decision at 0.30 as proxy (for demonstration).**")
    y_true = (df["fraud_score"] >= THRESHOLD).astype(int).values

y_pred = (df["fraud_score"] >= THRESHOLD).astype(int).values
y_score = df["fraud_score"].values

m1, m2, m3, m4 = st.columns(4)
m1.metric("**Accuracy (at 0.30)**", f"{accuracy_score(y_true, y_pred):.2%}")
m2.metric("**Precision (at 0.30)**", f"{precision_score(y_true, y_pred, zero_division=0):.2%}")
m3.metric("**Recall (at 0.30)**", f"{recall_score(y_true, y_pred, zero_division=0):.2%}")
m4.metric("**F1-score (at 0.30)**", f"{f1_score(y_true, y_pred, zero_division=0):.2%}")

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
cm_long = (
    pd.DataFrame(cm, index=["Actual: 0 (Legit)", "Actual: 1 (Fraud)"], columns=["Pred: 0", "Pred: 1"])
    .reset_index()
    .melt(id_vars="index", var_name="Predicted", value_name="Count")
    .rename(columns={"index": "Actual"})
)
st.altair_chart(
    alt.Chart(cm_long)
    .mark_rect()
    .encode(
        x=alt.X("Predicted:N", title=None),
        y=alt.Y("Actual:N", title=None),
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["Actual:N","Predicted:N","Count:Q"]
    )
    .properties(height=200, title="**Confusion Matrix (at 0.30)**"),
    use_container_width=True,
)

try:
    AUC = roc_auc_score(y_true, y_score)
except Exception:
    AUC = float("nan")

fpr, tpr, _ = roc_curve(y_true, y_score)
st.altair_chart(
    alt.Chart(pd.DataFrame({"fpr": fpr, "tpr": tpr}))
    .mark_line()
    .encode(x=alt.X("fpr:Q", title="False Positive Rate"), y=alt.Y("tpr:Q", title="True Positive Rate"))
    .properties(height=220, title=f"**ROC Curve (AUC = {AUC:.3f})**"),
    use_container_width=True,
)

prec, rec, _ = precision_recall_curve(y_true, y_score)
AP = auc(rec, prec)
st.altair_chart(
    alt.Chart(pd.DataFrame({"recall": rec, "precision": prec}))
    .mark_line()
    .encode(x=alt.X("recall:Q", title="Recall"), y=alt.Y("precision:Q", title="Precision"))
    .properties(height=220, title=f"**Precision–Recall Curve (AP ≈ {AP:.3f})**"),
    use_container_width=True,
)

# ===================== FOOTNOTE =====================
st.caption("**Note:** All sections above reflect **only** the transactions within the **selected date range** and apply a **fixed threshold of 0.30** to compute alerts and evaluation metrics.")
