# streamlit run app.py
import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import date
from google.cloud import bigquery; from google.oauth2 import service_account
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config("Fraud Dashboard (Fixed Threshold 0.30)", layout="wide")
alt.renderers.set_embed_options(actions=False)
TH=0.30

# GCP auth
sa=dict(st.secrets["gcp_service_account"]); sa["private_key"]=sa["private_key"].replace("\\n","\n")
creds=service_account.Credentials.from_service_account_info(sa)
bq=bigquery.Client(credentials=creds, project=creds.project_id)

with st.sidebar:
    st.header("BigQuery Source")
    P=st.text_input("Project","mss-data-engineer-sandbox")
    D=st.text_input("Dataset","retail")
    PT=st.text_input("Predictions",f"{P}.{D}.predictions_latest")
    S=st.date_input("Start",date(2023,1,1)); E=st.date_input("End",date(2030,12,31))

@st.cache_data
def load_df(PT,S,E):
    sql=f"SELECT * FROM `{PT}` WHERE DATE(timestamp) BETWEEN @S AND @E"
    j=bq.query(sql,job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("S","DATE",str(S)),
        bigquery.ScalarQueryParameter("E","DATE",str(E))]))
    d=j.result().to_dataframe()
    if d.empty: return d
    d["timestamp"]=pd.to_datetime(d["timestamp"]); d["day"]=d["timestamp"].dt.date
    d["fraud_score"]=pd.to_numeric(d["fraud_score"],errors="coerce").fillna(0.0)
    d["is_alert"]=pd.to_numeric(d["is_alert"],errors="coerce").fillna(0).astype(int)
    return d

df=load_df(PT,S,E)
if df.empty:
    st.warning("No rows in this date range."); st.stop()

# KPIs
st.subheader("**Key Metrics (Threshold = 0.30)**")
st.caption("Total orders, flagged risky orders at 0.30, and the % flagged—only within the selected dates.")
TOT=len(df); AL=int(df["is_alert"].sum())
c1,c2,c3=st.columns(3)
c1.metric("**Total transactions**",TOT)
c2.metric("**Fraud alerts (≥0.30)**",AL)
c3.metric("**Alert rate**",f"{AL/max(1,TOT):.2%}")
st.markdown("---")

# Evaluation (needs fraud_flag in predictions_latest)
st.subheader("**Model Evaluation (fixed threshold = 0.30)**")
st.caption("Precision=among alerts, what % were truly fraud; Recall=of all fraud, what % we caught; F1=balance; Accuracy=overall right/wrong.")
y_true=(df["fraud_flag"] if "fraud_flag" in df else df["is_alert"]).fillna(0).astype(int)
y_pred=df["is_alert"].astype(int)
m1,m2,m3,m4=st.columns(4)
m1.metric("Accuracy",f"{accuracy_score(y_true,y_pred):.2%}")
m2.metric("Precision",f"{precision_score(y_true,y_pred,zero_division=0):.2%}")
m3.metric("Recall",f"{recall_score(y_true,y_pred,zero_division=0):.2%}")
m4.metric("F1-score",f"{f1_score(y_true,y_pred,zero_division=0):.2%}")

# Daily trend
st.subheader("**Daily Trend**"); st.caption("Day-by-day totals vs risky orders within the selected dates.")
tr=df.groupby("day").agg(total=("order_id","count"),alerts=("is_alert","sum")).reset_index()
if not tr.empty:
    tl=tr.melt("day",["total","alerts"],"Series","value"); tl["Series"]=tl["Series"].map({"total":"Total transactions","alerts":"Fraud alerts (≥0.30)"})
    st.altair_chart(alt.Chart(tl).mark_line(point=True).encode(x="day:T",y="value:Q",color="Series:N",tooltip=["day:T","Series:N","value:Q"]).properties(height=240),use_container_width=True)

# Score distribution
st.subheader("**Fraud Score Distribution**"); st.caption("Risk scores after calibration; red line at the fixed cut 0.30.")
st.altair_chart(alt.Chart(df).mark_bar().encode(x=alt.X("fraud_score:Q",bin=alt.Bin(maxbins=50)),y="count()").properties(height=200)
                 + alt.Chart(pd.DataFrame({"x":[TH]})).mark_rule(color="red").encode(x="x"),use_container_width=True)

# Top alerts
st.subheader("**Top Alerts (score ≥ 0.30)**"); st.caption("Highest-risk orders to review first.")
cols=[c for c in ["order_id","timestamp","customer_id","store_id","sku_id","sku_category","order_amount","quantity","payment_method","shipping_country","ip_country","fraud_score"] if c in df]
alerts=df[df.is_alert==1].sort_values(["fraud_score","timestamp"],ascending=[False,False]).drop_duplicates(subset=["order_id"])
show_n=st.number_input("How many to show?",1,max(1,len(alerts)),min(50,len(alerts)),step=10)
st.dataframe(alerts.loc[:,cols].head(show_n),use_container_width=True,height=320)
