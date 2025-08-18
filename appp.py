# app.py — Fraud detection dashboard (BigQuery + Streamlit)
# ----------------------------------------------------------
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import bigquery
from datetime import date
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ------------------------- PAGE SETUP -------------------------
st.set_page_config(page_title="Fraud detection dashboard", page_icon="🛡️", layout="wide")
st.markdown(
    """
<style>
.kpi {font-size: 1.8rem; font-weight: 700; margin-top: -10px}
.kpi-meta {color: #6c757d; font-size: .9rem; margin-top: -8px}
hr {margin: 0.8rem 0;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🛡️ Fraud detection dashboard")

with st.expander("What you're seeing (quick tour)", expanded=True):
    st.markdown(
        """
**Goal**: Monitor model outputs and the rule/feature context that tends to co-occur with alerts.

**How to read this page**
1. Pick a date range and threshold on the left.  
2. **KPI cards** summarize volume and alert rate for that window.  
3. **Daily trend** shows scored vs. alerts day-by-day.  
4. **Score distribution** helps pick a sensible threshold.  
5. **Strong-signal** and **Pricing/Inventory** charts show which engineered signals are most prevalent among alerts vs non-alerts.  
6. **Top alerts** lists the highest-risk orders for human review.  
7. **Model evaluation** computes Accuracy/Precision/Recall/F1, Confusion Matrix, ROC and PR.  
8. **Operating point helper** shows precision/recall vs threshold (uses BigQuery metrics if available, otherwise a local fallback).
"""
    )

# ------------------------- SIDEBAR ----------------------------
with st.sidebar:
    st.header("Settings")
    project = st.text_input("Project", "mss-data-engineer-sandbox")
    dataset = st.text_input("Dataset", "retail")

    pred_table = st.text_input(
        "Predictions table (project.dataset.table)",
        f"{project}.{dataset}.predictions_latest",
    )
    feat_table = st.text_input(
        "Features table (project.dataset.table)",
        f"{project}.{dataset}.features_signals_v4",
    )

    # Optional: precomputed daily metrics table (if you created it)
    metrics_table = st.text_input(
        "Precomputed metrics table (optional)",
        f"{project}.{dataset}.predictions_daily_metrics",
    )

    start = st.date_input("Start (YYYY/MM/DD)", date(2024, 12, 1))
    end = st.date_input("End (YYYY/MM/DD)", date(2024, 12, 31))

    threshold = st.slider("Alert threshold (score ≥ threshold ⇒ alert)", 0.00, 1.00, 0.30, 0.01)

# ------------------------- BIGQUERY ---------------------------
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

# Read secrets and convert escaped "\n" to real newlines
sa_info = dict(st.secrets["gcp_service_account"])
sa_info["private_key"] = sa_info["private_key"].replace("\\n", "\n")

credentials = service_account.Credentials.from_service_account_info(sa_info)
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

@st.cache_data(show_spinner=True)
def load_data(_client: bigquery.Client, _pred_table: str, _feat_table: str, _start: date, _end: date) -> pd.DataFrame:
    """
    Pull a single joined dataset once for the chosen date window.
    """
    sql = f"""
    SELECT
      p.order_id,
      p.timestamp,
      p.customer_id,
      p.store_id,
      p.sku_id,
      p.sku_category,
      p.order_amount,
      p.quantity,
      p.payment_method,
      p.shipping_country,
      p.ip_country,
      CAST(p.fraud_score AS FLOAT64) AS fraud_score,

      -- STRONG COMBOS
      s.strong_tri_mismatch_high_value,
      s.strong_high_value_express_geo,
      s.strong_burst_multi_device,
      s.strong_price_drop_bulk,
      s.strong_giftcard_geo,
      s.strong_return_whiplash,
      s.strong_price_inventory_stress,
      s.strong_country_flip_express,

      -- PRICING / INVENTORY
      s.high_price_anomaly,
      s.low_price_anomaly,
      s.oversell_flag,
      s.stockout_risk_flag,
      s.hoarding_flag,

      -- GROUND TRUTH (if present)
      SAFE_CAST(s.fraud_flag AS INT64) AS fraud_flag
    FROM `{_pred_table}` p
    LEFT JOIN `{_feat_table}` s
      USING(order_id)
    WHERE DATE(p.timestamp) BETWEEN @start AND @end
    ORDER BY timestamp
    """
    job = _client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start", "DATE", str(_start)),
                bigquery.ScalarQueryParameter("end", "DATE", str(_end)),
            ]
        ),
    )
    df = job.result().to_dataframe(create_bqstorage_client=False)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    return df

df = load_data(client, pred_table, feat_table, start, end)

if df.empty:
    st.warning("No rows in this date window. Try expanding the date range.")
    st.stop()

# normalize basics used later
df["fraud_score"] = pd.to_numeric(df["fraud_score"], errors="coerce")
df["is_alert"] = (df["fraud_score"] >= threshold).astype(int)
df["date"] = df["timestamp"].dt.date

# ------------------------- KPIs --------------------------------
total_scored = int(len(df))
total_alerts = int(df["is_alert"].sum())
alert_rate = (total_alerts / total_scored) if total_scored else 0.0

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
c1.markdown(f"<div class='kpi'>{total_scored}</div>", unsafe_allow_html=True)
c1.markdown("<div class='kpi-meta'>Scored</div>", unsafe_allow_html=True)

c2.markdown(f"<div class='kpi'>{total_alerts}</div>", unsafe_allow_html=True)
c2.markdown("<div class='kpi-meta'>Alerts</div>", unsafe_allow_html=True)

c3.markdown(f"<div class='kpi'>{alert_rate:.2%}</div>", unsafe_allow_html=True)
c3.markdown("<div class='kpi-meta'>Alert rate</div>", unsafe_allow_html=True)

c4.markdown(
    f"<div class='kpi-meta'>Window: <b>{df['timestamp'].min()}</b> → "
    f"<b>{df['timestamp'].max()}</b> &nbsp;|&nbsp; Threshold: <b>{threshold:.2f}</b></div>",
    unsafe_allow_html=True,
)

st.markdown("<hr/>", unsafe_allow_html=True)

# ------------------------- DAILY TREND -------------------------
st.subheader("Daily trend")
trend = (
    df.groupby("date")
    .agg(n=("order_id", "count"), alerts=("is_alert", "sum"))
    .reset_index()
)

if trend.empty:
    st.info("No activity on the selected dates.")
else:
    trend_long = trend.melt(id_vars="date", value_vars=["n", "alerts"], var_name="series", value_name="count")
    line = (
        alt.Chart(trend_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color("series:N", title=None),
            tooltip=["date:T", "series:N", "count:Q"],
        )
        .properties(height=280)
    )
    st.altair_chart(line, use_container_width=True)

# ------------------------- SCORE DISTRIBUTION ------------------
st.subheader("Fraud-score distribution")
hist = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("fraud_score:Q", bin=alt.Bin(maxbins=50), title="Fraud score"),
        y=alt.Y("count():Q", title="Count"),
        tooltip=[alt.Tooltip("count()", title="Count")],
    )
    .properties(height=220)
)
th_rule = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(color="crimson").encode(x="x")
st.altair_chart(hist + th_rule, use_container_width=True)

# ------------------------- CONTEXT SIGNALS --------------------
def prevalence_bar(_df: pd.DataFrame, cols: list, title: str):
    present = [c for c in cols if c in _df.columns]
    if not present:
        st.info(f"No matching signals found for: {title}")
        return
    rows = []
    for c in present:
        col = pd.to_numeric(_df[c], errors="coerce").fillna(0).astype(int)
        in_alerts = int(col[_df["is_alert"] == 1].sum())
        in_non_alerts = int(col[_df["is_alert"] == 0].sum())
        denom_alerts = max(int((_df["is_alert"] == 1).sum()), 1)
        denom_non_alerts = max(int((_df["is_alert"] == 0).sum()), 1)
        rows.append({"signal": c, "% in alerts": in_alerts / denom_alerts, "% in non-alerts": in_non_alerts / denom_non_alerts})
    dd = pd.DataFrame(rows).sort_values("% in alerts", ascending=False)
    dd_long = dd.melt(id_vars="signal", var_name="group", value_name="value")
    chart = (
        alt.Chart(dd_long)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", axis=alt.Axis(format="%"), title="Prevalence"),
            y=alt.Y("signal:N", sort="-x", title=None),
            color=alt.Color("group:N", title=None),
            tooltip=["signal:N", alt.Tooltip("value:Q", title="prevalence", format=".1%"), "group:N"],
        )
        .properties(height=320, title=title)
    )
    st.altair_chart(chart, use_container_width=True)

st.subheader("Context signals")
left, right = st.columns(2)

with left:
    prevalence_bar(
        df,
        [
            "strong_tri_mismatch_high_value",
            "strong_high_value_express_geo",
            "strong_burst_multi_device",
            "strong_price_drop_bulk",
            "strong_giftcard_geo",
            "strong_return_whiplash",
            "strong_price_inventory_stress",
            "strong_country_flip_express",
        ],
        "Strong-signal prevalence (alerts vs non-alerts)",
    )

with right:
    prevalence_bar(
        df,
        ["high_price_anomaly", "low_price_anomaly", "oversell_flag", "stockout_risk_flag", "hoarding_flag"],
        "Pricing & Inventory signal prevalence",
    )

# ------------------------- CORRELATION ------------------------
st.caption("Correlation of `fraud_score` with pricing/inventory signals (Pearson, numeric-only safe)")
corr_cols = [c for c in ["high_price_anomaly", "low_price_anomaly", "oversell_flag", "stockout_risk_flag", "hoarding_flag"] if c in df.columns]
if corr_cols:
    corr_df = df.copy()
    corr_df["fraud_score"] = pd.to_numeric(corr_df["fraud_score"], errors="coerce")
    for c in corr_cols:
        corr_df[c] = pd.to_numeric(corr_df[c], errors="coerce")
    corr_series = corr_df[["fraud_score"] + corr_cols].corr(method="pearson")["fraud_score"].drop("fraud_score")
    corr_plot_df = corr_series.rename("corr").reset_index().rename(columns={"index": "signal"})
    corr_plot = (
        alt.Chart(corr_plot_df)
        .mark_bar()
        .encode(
            x=alt.X("corr:Q", title="Correlation (score vs signal)"),
            y=alt.Y("signal:N", sort="x", title=None),
            tooltip=[alt.Tooltip("corr:Q", format=".3f"), "signal:N"],
            color=alt.condition(alt.datum.corr > 0, alt.value("#1f77b4"), alt.value("#d62728")),
        )
        .properties(height=140)
    )
    st.altair_chart(corr_plot, use_container_width=True)
else:
    st.info("No pricing/inventory columns available to compute correlation.")

# ------------------------- TOP ALERTS -------------------------
st.subheader("Top alerts (highest scores)")
top = (
    df.sort_values(["fraud_score", "timestamp"], ascending=[False, False])
    .head(50)
    .loc[
        :,
        [
            c
            for c in [
                "order_id",
                "timestamp",
                "customer_id",
                "store_id",
                "sku_id",
                "sku_category",
                "order_amount",
                "quantity",
                "payment_method",
                "shipping_country",
                "ip_country",
                "fraud_score",
            ]
            if c in df.columns
        ],
    ]
)
st.dataframe(top, use_container_width=True, height=360)

# ------------------------- MODEL EVALUATION -------------------
st.subheader("Model evaluation")

label_candidates = ["fraud_flag", "is_fraud", "label", "ground_truth", "gt", "y"]
present = [c for c in label_candidates if c in df.columns]

if present:
    label_col = st.selectbox("Ground-truth label column (1 = fraud, 0 = legit)", options=present, index=0)
    y_true = df[label_col].fillna(0).astype(int).values
    st.caption(f"Using ground-truth column: `{label_col}`.")
else:
    st.warning(
        "No ground-truth label column found (e.g., `fraud_flag`). "
        "Using the model’s current decision as a temporary label for evaluation."
    )
    label_col = None
    y_true = (df["fraud_score"] >= threshold).astype(int).values

y_pred = (df["fraud_score"] >= threshold).astype(int).values
y_score = df["fraud_score"].values

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{acc:.2%}")
m2.metric("Precision", f"{prec:.2%}")
m3.metric("Recall", f"{rec:.2%}")
m4.metric("F1-score", f"{f1:.2%}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
cm_df = pd.DataFrame(cm, index=pd.Index(["Actual: 0", "Actual: 1"], name="Actual"), columns=["Pred: 0", "Pred: 1"]).reset_index()
cm_long = cm_df.melt(id_vars=["Actual"], var_name="Predicted", value_name="Count")

cm_chart = (
    alt.Chart(cm_long)
    .mark_rect()
    .encode(
        x=alt.X("Predicted:N"),
        y=alt.Y("Actual:N"),
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["Actual:N", "Predicted:N", "Count:Q"],
    )
    .properties(height=200)
)
cm_text = alt.Chart(cm_long).mark_text(baseline="middle", fontSize=14).encode(x="Predicted:N", y="Actual:N", text="Count:Q")
st.altair_chart(cm_chart + cm_text, use_container_width=True)

# ROC curve + AUC
try:
    auc_roc = roc_auc_score(y_true, y_score)
except ValueError:
    auc_roc = float("nan")

fpr, tpr, _ = roc_curve(y_true, y_score)
roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
roc_chart = (
    alt.Chart(roc_df)
    .mark_line()
    .encode(
        x=alt.X("fpr:Q", title="False Positive Rate"),
        y=alt.Y("tpr:Q", title="True Positive Rate"),
        tooltip=[alt.Tooltip("fpr:Q", format=".3f"), alt.Tooltip("tpr:Q", format=".3f")],
    )
    .properties(height=260, title=f"ROC curve (AUC = {auc_roc:.3f})")
)
diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_rule(strokeDash=[4, 4]).encode(x="x", y="y")
st.altair_chart(roc_chart + diag, use_container_width=True)

# Precision–Recall curve + AP
prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_score)
ap = auc(rec_curve, prec_curve)
pr_df = pd.DataFrame({"recall": rec_curve, "precision": prec_curve})
pr_chart = (
    alt.Chart(pr_df)
    .mark_line()
    .encode(
        x=alt.X("recall:Q", title="Recall"),
        y=alt.Y("precision:Q", title="Precision"),
        tooltip=[alt.Tooltip("recall:Q", format=".3f"), alt.Tooltip("precision:Q", format=".3f")],
    )
    .properties(height=260, title=f"Precision–Recall curve (AP ≈ {ap:.3f})")
)
st.altair_chart(pr_chart, use_container_width=True)

# --------------------- OPERATING POINT HELPER -----------------
st.subheader("Operating point helper (precision/recall vs threshold)")

def load_bq_op_metrics(_client: bigquery.Client, _metrics_table: str, _start: date, _end: date) -> pd.DataFrame:
    sql = f"""
    SELECT threshold,
           AVG(precision) AS precision,
           AVG(recall)    AS recall
    FROM `{_metrics_table}`
    WHERE dt BETWEEN @start AND @end
    GROUP BY threshold
    ORDER BY threshold
    """
    job = _client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start", "DATE", str(_start)),
                bigquery.ScalarQueryParameter("end", "DATE", str(_end)),
            ]
        ),
    )
    return job.result().to_dataframe(create_bqstorage_client=False)

use_bq = False
op_df = pd.DataFrame()
try:
    # try BigQuery metrics (table must exist)
    op_df = load_bq_op_metrics(client, metrics_table, start, end)
    use_bq = not op_df.empty
except Exception:
    use_bq = False

if use_bq:
    chart_df = op_df.melt("threshold", ["precision", "recall"], "metric", "value")
else:
    # local fallback: sweep thresholds on the current df
    grid = np.round(np.linspace(0.05, 0.95, 19), 2)
    rows = []
    for th in grid:
        pred = (df["fraud_score"] >= th).astype(int)
        tp = int(((pred == 1) & (df["is_alert"] == 1)).sum())
        fp = int(((pred == 1) & (df["is_alert"] == 0)).sum())
        fn = int(((pred == 0) & (df["is_alert"] == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append({"threshold": th, "precision": precision, "recall": recall})
    op_df = pd.DataFrame(rows)
    chart_df = op_df.melt("threshold", ["precision", "recall"], "metric", "value")

op_chart = (
    alt.Chart(chart_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("threshold:Q"),
        y=alt.Y("value:Q", axis=alt.Axis(format="%")),
        color=alt.Color("metric:N", title=None),
        tooltip=["threshold:Q", alt.Tooltip("value:Q", format=".2%")],
    )
    .properties(height=260, title=("BigQuery metrics" if use_bq else "Local fallback"))
)
st.altair_chart(op_chart, use_container_width=True)

if not use_bq:
    st.caption(
        "No precomputed BigQuery metrics found for the selected window "
        f"(`{metrics_table}`). Showing local fallback."
    )


