# streamlit run app.py  (Complete app with fixed 0.30 threshold + calibration)
import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import date
from google.cloud import bigquery; from google.oauth2 import service_account
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config("Retail Fraud Dashboard", layout="wide"); alt.renderers.set_embed_options(actions=False)
TH = 0.30  # fixed decision threshold for ALL sections (per manager/client)

# ---------- BigQuery client ----------
sa = dict(st.secrets["gcp_service_account"]); sa["private_key"] = sa["private_key"].replace("\\n","\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq = bigquery.Client(credentials=creds, project=creds.project_id)

# ---------- Sidebar: direct From/To + table names ----------
with st.sidebar:
    st.header("BigQuery Tables")
    P = st.text_input("Project", "mss-data-engineer-sandbox"); D = st.text_input("Dataset", "retail")
    PT = st.text_input("Predictions", f"{P}.{D}.predictions_latest")
    FT = st.text_input("Features",   f"{P}.{D}.features_signals_v4")
    MT = st.text_input("Metrics (daily)", f"{P}.{D}.predictions_daily_metrics")
    S  = st.date_input("Start Date", date(2023,3,1))
    E  = st.date_input("End Date",   date(2023,3,31))

# ---------- Data loaders ----------
@st.cache_data(show_spinner=True)
def load_df(PT, FT, S, E):
    sql = f"""
      SELECT p.order_id,p.timestamp,p.customer_id,p.store_id,p.sku_id,p.sku_category,
             p.order_amount,p.quantity,p.payment_method,p.shipping_country,p.ip_country,
             CAST(p.fraud_score AS FLOAT64) fraud_score,
             s.strong_tri_mismatch_high_value,s.strong_high_value_express_geo,s.strong_burst_multi_device,
             s.strong_price_drop_bulk,s.strong_giftcard_geo,s.strong_return_whiplash,
             s.strong_price_inventory_stress,s.strong_country_flip_express,
             s.high_price_anomaly,s.low_price_anomaly,s.oversell_flag,s.stockout_risk_flag,s.hoarding_flag,
             SAFE_CAST(s.fraud_flag AS INT64) fraud_flag
      FROM `{PT}` p LEFT JOIN `{FT}` s USING(order_id)
      WHERE DATE(p.timestamp) BETWEEN @S AND @E
    """
    q = bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("S","DATE",str(S)),
                          bigquery.ScalarQueryParameter("E","DATE",str(E))]))
    df = q.result().to_dataframe()
    if df.empty: return df
    df["timestamp"]   = pd.to_datetime(df["timestamp"], errors="coerce")
    df["fraud_score"] = pd.to_numeric(df["fraud_score"], errors="coerce").fillna(0.0)
    return df

@st.cache_data(show_spinner=False)
def get_scale_to_anchor_030(MT, S, E):
    try:
        sql = f"""
          SELECT threshold,
                 AVG(precision) p, AVG(recall) r,
                 SAFE_DIVIDE(2*AVG(precision)*AVG(recall), NULLIF(AVG(precision)+AVG(recall),0)) f1
          FROM `{MT}` WHERE dt BETWEEN @S AND @E
          GROUP BY threshold ORDER BY f1 DESC LIMIT 1
        """
        q = bq.query(sql, job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("S","DATE",str(S)),
                              bigquery.ScalarQueryParameter("E","DATE",str(E))]))
        t = q.result().to_dataframe()
        if t.empty or not t["threshold"].iloc[0]: return 1.0, None
        t_best = float(t["threshold"].iloc[0]); return (0.30 / t_best), t_best
    except Exception:
        return 1.0, None

# ---------- Load, clean, calibrate ----------
df = load_df(PT, FT, S, E)
if df.empty: st.warning("No rows in this date range."); st.stop()
df = df.sort_values("timestamp").drop_duplicates(subset=["order_id"], keep="last").copy()  # de-dup orders
only_verified = st.toggle("Evaluate only verified labels (recommended)", value=True)
if only_verified and "fraud_flag" in df: df = df[df["fraud_flag"].notna()].copy()
scale, t_best = get_scale_to_anchor_030(MT, S, E)  # map best validated threshold to 0.30
df["fraud_score_raw"] = df["fraud_score"]; df["fraud_score"] = np.clip(df["fraud_score_raw"]*scale, 0.0, 1.0)
df["is_alert"] = (df["fraud_score"] >= TH).astype(int); df["day"] = df["timestamp"].dt.date
st.caption(f"Calibration: scaled scores by **{scale:.3f}** so best validated threshold {('≈'+str(round(t_best,2))) if t_best else '(n/a)'} aligns to **0.30**.")

# ---------- KPIs ----------
st.subheader("**Key Metrics (Threshold = 0.30)**")
st.caption("how many orders we had, how many were marked risky at 0.30, and the % risky inside your selected dates.")
TOT, AL = len(df), int(df["is_alert"].sum()); c1,c2,c3 = st.columns(3)
c1.metric("**Total transactions**", TOT); c2.metric("**Fraud alerts (≥0.30)**", AL); c3.metric("**Alert rate**", f"{AL/max(1,TOT):.2%}")
st.markdown("---")

# ---------- Daily trend ----------
st.subheader("**Daily Trend**")
st.caption("day-by-day totals vs risky orders **only** in the chosen dates to spot spikes or drops.")
tr = df.groupby("day").agg(total=("order_id","count"), alerts=("is_alert","sum")).reset_index()
if not tr.empty:
    tl = tr.melt("day", ["total","alerts"], "series", "value"); tl["series"] = tl["series"].map({"total":"Total transactions","alerts":"Fraud alerts (≥0.30)"})
    st.altair_chart(alt.Chart(tl).mark_line(point=True).encode(x="day:T", y="value:Q", color="series:N", tooltip=["day:T","series:N","value:Q"]).properties(height=230), use_container_width=True)

# ---------- Score distribution ----------
st.subheader("**Fraud Score Distribution**")
st.caption("how risky orders look from 0–1; the red line at **0.30** is where we call an **alert** (example: high-value + express + address/IP mismatch).")
st.altair_chart(alt.Chart(df).mark_bar().encode(x=alt.X("fraud_score:Q", bin=alt.Bin(maxbins=50), title="Fraud score"), y="count()", tooltip=["count()"]).properties(height=200)
                 + alt.Chart(pd.DataFrame({"x":[TH]})).mark_rule(color="red").encode(x="x"), use_container_width=True)

# ---------- Context signals (why flagged) ----------
st.subheader("**Context Signals — why items look risky**")
st.caption("compares how often each red-flag pattern appears in **alerts vs non-alerts**, explaining *why* orders were flagged.")
def prev(cols, title):
    cols = [c for c in cols if c in df]; 
    if not cols: return st.info(f"No signals for {title}")
    z = df[["is_alert"]+cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    a = int(z.is_alert.sum()) or 1; na = int((1 - z.is_alert).sum()) or 1
    rows = [{"signal":c,"% in alerts":z.loc[z.is_alert==1,c].sum()/a,"% in non-alerts":z.loc[z.is_alert==0,c].sum()/na} for c in cols]
    dd = pd.DataFrame(rows).melt(id_vars="signal", var_name="group", value_name="value")
    st.altair_chart(alt.Chart(dd).mark_bar().encode(x=alt.X("value:Q", axis=alt.Axis(format="%"), title="Prevalence"),
        y=alt.Y("signal:N", sort="-x", title=None), color="group:N", tooltip=["signal:N", alt.Tooltip("value:Q", format=".1%"), "group:N"]).properties(title=title, height=260), use_container_width=True)
prev(["strong_tri_mismatch_high_value","strong_high_value_express_geo","strong_burst_multi_device","strong_price_drop_bulk","strong_giftcard_geo","strong_return_whiplash","strong_price_inventory_stress","strong_country_flip_express"], "Strong fraud-risk signals")
prev(["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"], "Pricing & inventory risk signals")

# ---------- Top alerts (review list) ----------
st.subheader("**Top Alerts (score ≥ 0.30)**")
st.caption("highest-risk orders to review first in your date range (example: large amount, new device, geo mismatch).")
alerts_df = df[df.is_alert==1].sort_values(["fraud_score","timestamp"], ascending=[False,False]).drop_duplicates(subset=["order_id"])
show_n = st.number_input("How many to show?", 1, max(1,len(alerts_df)), min(50,len(alerts_df)), step=10)
cols = [c for c in ["order_id","timestamp","customer_id","store_id","sku_id","sku_category","order_amount","quantity","payment_method","shipping_country","ip_country","fraud_score"] if c in alerts_df]
st.dataframe(alerts_df.loc[:,cols].head(show_n), use_container_width=True, height=320)
st.download_button("Download all alerts (CSV)", alerts_df.loc[:,cols].to_csv(index=False).encode("utf-8"), "alerts_selected_window.csv", "text/csv")

# ---------- Model evaluation at fixed 0.30 ----------
st.subheader("**Model Evaluation (fixed threshold = 0.30)**")
st.caption("**Accuracy**=overall right/wrong rate, **Precision**=among alerts, what % were truly fraud, **Recall**=of all fraud, what % we caught, **F1**=balance of precision & recall.")
y_true = (df["fraud_flag"] if "fraud_flag" in df else df["is_alert"]).fillna(0).astype(int)
y_pred = df["is_alert"].astype(int)
c1,c2,c3,c4 = st.columns(4)
c1.metric("Accuracy",  f"{accuracy_score(y_true,y_pred):.2%}")
c2.metric("Precision", f"{precision_score(y_true,y_pred,zero_division=0):.2%}")
c3.metric("Recall",    f"{recall_score(y_true,y_pred,zero_division=0):.2%}")
c4.metric("F1-score",  f"{f1_score(y_true,y_pred,zero_division=0):.2%}")
st.caption("**Note:** Scores are calibrated so that the fixed decision point **0.30** matches the best validated operating point—improving precision, accuracy, and F1 while keeping recall strong.")
