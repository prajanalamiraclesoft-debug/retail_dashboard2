# streamlit run app.py  (≤100 lines)
import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import date
from google.cloud import bigquery; from google.oauth2 import service_account
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config("Retail Fraud Dashboard", layout="wide"); alt.renderers.set_embed_options(actions=False)
TH=0.30  # fixed threshold per manager

# ---- BigQuery client ----
sa=dict(st.secrets["gcp_service_account"]); sa["private_key"]=sa["private_key"].replace("\\n","\n")
creds=service_account.Credentials.from_service_account_info(sa)
bq=bigquery.Client(credentials=creds, project=creds.project_id)

# ---- Sidebar: direct From / To filter ----
with st.sidebar:
    st.header("BigQuery Tables")
    P=st.text_input("Project","mss-data-engineer-sandbox"); D=st.text_input("Dataset","retail")
    PT=st.text_input("Predictions",f"{P}.{D}.predictions_latest")
    FT=st.text_input("Features",f"{P}.{D}.features_signals_v4")
    S=st.date_input("Start Date",date(2023,3,1)); E=st.date_input("End Date",date(2023,3,31))

# ---- Load data filtered by S/E ----
@st.cache_data
def load_df(PT,FT,S,E):
    sql=f"""SELECT p.order_id,p.timestamp,p.customer_id,p.store_id,p.sku_id,p.sku_category,
    p.order_amount,p.quantity,p.payment_method,p.shipping_country,p.ip_country,
    CAST(p.fraud_score AS FLOAT64) fraud_score,
    s.strong_tri_mismatch_high_value,s.strong_high_value_express_geo,s.strong_burst_multi_device,
    s.strong_price_drop_bulk,s.strong_giftcard_geo,s.strong_return_whiplash,
    s.strong_price_inventory_stress,s.strong_country_flip_express,
    s.high_price_anomaly,s.low_price_anomaly,s.oversell_flag,s.stockout_risk_flag,s.hoarding_flag,
    SAFE_CAST(s.fraud_flag AS INT64) fraud_flag
    FROM `{PT}` p LEFT JOIN `{FT}` s USING(order_id)
    WHERE DATE(p.timestamp) BETWEEN @S AND @E"""
    j=bq.query(sql,job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("S","DATE",str(S)),
        bigquery.ScalarQueryParameter("E","DATE",str(E))]))
    d=j.result().to_dataframe()
    if d.empty: return d
    d["timestamp"]=pd.to_datetime(d["timestamp"],errors="coerce")
    d["fraud_score"]=pd.to_numeric(d["fraud_score"],errors="coerce").fillna(0.0)
    d["is_alert"]=(d["fraud_score"]>=TH).astype(int); d["day"]=d["timestamp"].dt.date; return d

df=load_df(PT,FT,S,E)
if df.empty: st.warning("No rows in this date range."); st.stop()

# ---- Key Metrics ----
st.subheader("**Key Metrics (Threshold = 0.30)**")
st.caption("how many orders we had, how many were marked risky, and what percent were risky in your selected dates.")
TOT=len(df); AL=int(df["is_alert"].sum()); c1,c2,c3=st.columns(3)
c1.metric("**Total transactions**",TOT); c2.metric("**Fraud alerts (score ≥ 0.30)**",AL); c3.metric("**Alert rate**",f"{AL/max(1,TOT):.2%}")
st.markdown("---")

# ---- Daily Trend ----
st.subheader("**Daily Trend**")
st.caption("day-by-day totals vs risky orders **only** inside the chosen dates to spot spikes or drops.")
tr=df.groupby("day").agg(total=("order_id","count"),alerts=("is_alert","sum")).reset_index()
if not tr.empty:
    tl=tr.melt("day",["total","alerts"],"type","value")
    tl["type"]=tl["type"].replace({"total":"Total transactions","alerts":"Fraud alerts (≥0.30)"})
    st.altair_chart(alt.Chart(tl).mark_line(point=True).encode(
        x="day:T",y="value:Q",color="type:N",tooltip=["day:T","type:N","value:Q"]
    ).properties(height=240),use_container_width=True)

# ---- Score Distribution ----
st.subheader("**Fraud Score Distribution**")
st.caption("how risky orders look from 0–1; the red line at 0.30 is where we call an **alert** (example: $300 express order with billing–shipping–IP mismatch).")
st.altair_chart(
    alt.Chart(df).mark_bar().encode(
        x=alt.X("fraud_score:Q",bin=alt.Bin(maxbins=50),title="Fraud score"),y="count()",tooltip=["count()"]
    ).properties(height=200) + alt.Chart(pd.DataFrame({"x":[TH]})).mark_rule(color="red").encode(x="x"),
    use_container_width=True
)

# ---- Context Signals (why flagged) ----
st.subheader("**Context Signals — what risky behavior we see**")
st.caption("compares how often each red-flag pattern appears in **alerts vs non-alerts** to explain **why** items were flagged.")
def prev(cols,title):
    cols=[c for c in cols if c in df]
    if not cols: return st.info(f"No signals for {title}")
    z=df[["is_alert"]+cols].apply(pd.to_numeric,errors="coerce").fillna(0).astype(int)
    a=int(z.is_alert.sum()) or 1; na=int((1-z.is_alert).sum()) or 1
    rows=[{"signal":c,"% in alerts":z.loc[z.is_alert==1,c].sum()/a,"% in non-alerts":z.loc[z.is_alert==0,c].sum()/na} for c in cols]
    dd=pd.DataFrame(rows).melt(id_vars="signal",var_name="group",value_name="value")
    st.altair_chart(alt.Chart(dd).mark_bar().encode(
        x=alt.X("value:Q",axis=alt.Axis(format="%"),title="Prevalence"),
        y=alt.Y("signal:N",sort="-x",title=None),color="group:N"
    ).properties(title=title,height=260),use_container_width=True)
prev(["strong_tri_mismatch_high_value","strong_high_value_express_geo","strong_burst_multi_device",
      "strong_price_drop_bulk","strong_giftcard_geo","strong_return_whiplash",
      "strong_price_inventory_stress","strong_country_flip_express"],"Strong fraud-risk signals")
prev(["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"],
     "Pricing & inventory risk signals")

# ---- Top Alerts (no artificial cap; optional user cap) ----
st.subheader("**Top Alerts (score ≥ 0.30)**")
st.caption("the highest-risk orders to review first within your date range.")

# Build, sort, and (optionally) de-duplicate by order_id if your join can create duplicates
alerts_df = (
    df[df["is_alert"] == 1]
      .sort_values(["fraud_score", "timestamp"], ascending=[False, False])
      .drop_duplicates(subset=["order_id"])  # remove dup orders if any
)

# Let the user decide how many to show (default = all alerts)
max_rows = int(len(alerts_df))
show_n = st.number_input("Showing alerts in below excel sheet", 1, max_rows, max_rows, step=10)

cols = [c for c in [
    "order_id","timestamp","customer_id","store_id","sku_id","sku_category",
    "order_amount","quantity","payment_method","shipping_country","ip_country","fraud_score"
] if c in alerts_df]

st.dataframe(alerts_df.loc[:, cols].head(show_n), use_container_width=True, height=340)

# Easy export (so row counts in Sheets = data rows only)
csv = alerts_df.loc[:, cols].to_csv(index=False).encode("utf-8")
st.download_button("Download all alerts (CSV)", csv, "alerts_selected_window.csv", "text/csv")


# ---- Model Evaluation ----
st.subheader("**Model Evaluation at 0.30**")
st.caption("**Precision**=among alerts, how many were truly fraud; **Recall**=among true fraud, how many we caught; **F1**=balance of precision & recall; **Accuracy**=overall right/wrong rate.")
y_true=(df.get("fraud_flag",df["is_alert"])).fillna(0).astype(int); y_pred=df["is_alert"]
c1,c2,c3,c4=st.columns(4)
c1.metric("Accuracy",f"{accuracy_score(y_true,y_pred):.2%}")
c2.metric("Precision",f"{precision_score(y_true,y_pred,zero_division=0):.2%}")
c3.metric("Recall",f"{recall_score(y_true,y_pred,zero_division=0):.2%}")
c4.metric("F1-score",f"{f1_score(y_true,y_pred,zero_division=0):.2%}")

