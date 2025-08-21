import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import date
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc

st.set_page_config("Retail Fraud Dashboard", layout="wide"); alt.renderers.set_embed_options(actions=False)
sa=dict(st.secrets["gcp_service_account"]); sa["private_key"]=sa["private_key"].replace("\\n","\n")
creds=service_account.Credentials.from_service_account_info(sa); bq=bigquery.Client(credentials=creds, project=creds.project_id)

# Sidebar: user inputs
with st.sidebar:
    st.header("**BigQuery Tables**")
    P=st.text_input("Project","mss-data-engineer-sandbox"); D=st.text_input("Dataset","retail")
    PT=st.text_input("Predictions",f"{P}.{D}.predictions_latest"); FT=st.text_input("Features",f"{P}.{D}.features_signals_v4")
    S=st.date_input("Start Date",date(2023,1,1)); E=st.date_input("End Date",date.today())

# Load data
@st.cache_data
def load_df(PT,FT,S,E):
    sql=f"""SELECT p.order_id,p.timestamp,p.customer_id,p.store_id,p.sku_id,p.sku_category,
    p.order_amount,p.quantity,p.payment_method,p.shipping_country,p.ip_country,CAST(p.fraud_score AS FLOAT64) fraud_score,
    s.strong_tri_mismatch_high_value,s.strong_high_value_express_geo,s.strong_burst_multi_device,s.strong_price_drop_bulk,
    s.strong_giftcard_geo,s.strong_return_whiplash,s.strong_price_inventory_stress,s.strong_country_flip_express,
    s.high_price_anomaly,s.low_price_anomaly,s.oversell_flag,s.stockout_risk_flag,s.hoarding_flag,SAFE_CAST(s.fraud_flag AS INT64) fraud_flag
    FROM `{PT}` p LEFT JOIN `{FT}` s USING(order_id)
    WHERE DATE(p.timestamp) BETWEEN @S AND @E"""
    return bq.query(sql,job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("S","DATE",str(S)),bigquery.ScalarQueryParameter("E","DATE",str(E))])).result().to_dataframe()

df=load_df(PT,FT,S,E)
if df.empty: st.warning("**No rows in this date range.**"); st.stop()
TH=0.30; df["fraud_score"]=pd.to_numeric(df["fraud_score"],errors="coerce").fillna(0); df["is_alert"]=(df["fraud_score"]>=TH).astype(int); df["day"]=pd.to_datetime(df["timestamp"]).dt.date

# KPIs
st.subheader("**Key Metrics (Threshold=0.30)**")
TOT=len(df); AL=df["is_alert"].sum()
c1,c2,c3=st.columns(3); c1.metric("**Total transactions**",TOT); c2.metric("**Fraud alerts**",AL); c3.metric("**Alert rate**",f"{AL/max(1,TOT):.2%}")

# Daily trend
st.subheader("**Daily Trend**"); tr=df.groupby("day").agg(total=("order_id","count"),alerts=("is_alert","sum")).reset_index()
if not tr.empty: st.altair_chart(alt.Chart(tr.melt("day",["total","alerts"],"type","value")).mark_line(point=True).encode(x="day:T",y="value:Q",color="type:N"),use_container_width=True)

# Score distribution
st.subheader("**Fraud Score Distribution**")
st.altair_chart(alt.Chart(df).mark_bar().encode(x=alt.X("fraud_score:Q",bin=alt.Bin(maxbins=50)),y="count()",tooltip=["count()"]).properties(height=200)+alt.Chart(pd.DataFrame({"x":[TH]})).mark_rule(color="red").encode(x="x"),use_container_width=True)

# Context signals
st.subheader("**Context Signals Prevalence**")
def prev(cols,title):
    cols=[c for c in cols if c in df]; 
    if not cols: return st.info(f"No signals for {title}")
    z=df[["is_alert"]+cols].fillna(0).astype(int); a=z.is_alert.sum() or 1; na=(1-z.is_alert).sum() or 1
    rows=[{"signal":c,"% in alerts":z.loc[z.is_alert==1,c].sum()/a,"% in non-alerts":z.loc[z.is_alert==0,c].sum()/na} for c in cols]
    dd=pd.DataFrame(rows).melt("signal","group","value"); st.altair_chart(alt.Chart(dd).mark_bar().encode(x=alt.X("value:Q",axis=alt.Axis(format="%")),y="signal:N",color="group:N"),use_container_width=True)
prev(["strong_tri_mismatch_high_value","strong_high_value_express_geo","strong_burst_multi_device","strong_price_drop_bulk","strong_giftcard_geo","strong_return_whiplash","strong_price_inventory_stress","strong_country_flip_express"],"Strong signals")
prev(["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"],"Pricing & Inventory signals")

# Evaluation
st.subheader("**Model Evaluation (Threshold=0.30)**")
y_true=(df.get("fraud_flag",df["is_alert"])).fillna(0).astype(int); y_pred=df["is_alert"]; y_score=df["fraud_score"]
c1,c2,c3,c4=st.columns(4); c1.metric("Accuracy",f"{accuracy_score(y_true,y_pred):.2%}"); c2.metric("Precision",f"{precision_score(y_true,y_pred):.2%}"); c3.metric("Recall",f"{recall_score(y_true,y_pred):.2%}"); c4.metric("F1",f"{f1_score(y_true,y_pred):.2%}")
fpr,tpr,_=roc_curve(y_true,y_score); st.altair_chart(alt.Chart(pd.DataFrame({"fpr":fpr,"tpr":tpr})).mark_line().encode(x="fpr",y="tpr"),use_container_width=True)
P,R,_=precision_recall_curve(y_true,y_score); st.altair_chart(alt.Chart(pd.DataFrame({"recall":R,"precision":P})).mark_line().encode("recall","precision"),use_container_width=True)
