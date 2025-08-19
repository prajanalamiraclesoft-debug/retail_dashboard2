import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import date
from google.cloud import bigquery; from google.oauth2 import service_account
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score
st.set_page_config("Retail Dashboard: Fraud & Inventory", layout="wide"); alt.data_transformers.enable("default", max_rows=None); alt.renderers.set_embed_options(actions=False)

with st.sidebar:
  st.header("Settings"); P=st.text_input("Project","mss-data-engineer-sandbox"); D=st.text_input("Dataset","retail")
  PT=st.text_input("Predictions",f"{P}.{D}.predictions_latest"); FT=st.text_input("Features",f"{P}.{D}.features_signals_v4")
  MT=st.text_input("Metrics (optional)",f"{P}.{D}.predictions_daily_metrics"); S=st.date_input("Start",date(2024,12,1)); E=st.date_input("End",date(2024,12,31)); TH=st.slider("Alert threshold (≥)",0.00,1.00,0.30,0.01)

sa=dict(st.secrets["gcp_service_account"]); sa["private_key"]=sa["private_key"].replace("\\n","\n")
creds=service_account.Credentials.from_service_account_info(sa); bq=bigquery.Client(credentials=creds, project=creds.project_id)

@st.cache_data(show_spinner=True)
def load_df(PT,FT,S,E):
  sql=f"""SELECT p.order_id,p.timestamp,p.customer_id,p.store_id,p.sku_id,p.sku_category,p.order_amount,p.quantity,p.payment_method,p.shipping_country,p.ip_country,CAST(p.fraud_score AS FLOAT64) fraud_score,
  s.strong_tri_mismatch_high_value,s.strong_high_value_express_geo,s.strong_burst_multi_device,s.strong_price_drop_bulk,s.strong_giftcard_geo,s.strong_return_whiplash,s.strong_price_inventory_stress,s.strong_country_flip_express,
  s.high_price_anomaly,s.low_price_anomaly,s.oversell_flag,s.stockout_risk_flag,s.hoarding_flag,SAFE_CAST(s.fraud_flag AS INT64) fraud_flag
  FROM `{PT}` p LEFT JOIN `{FT}` s USING(order_id) WHERE DATE(p.timestamp) BETWEEN @S AND @E ORDER BY p.timestamp"""
  j=bq.query(sql,job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("S","DATE",str(S)),bigquery.ScalarQueryParameter("E","DATE",str(E))]))
  d=j.result().to_dataframe(); d["timestamp"]=pd.to_datetime(d["timestamp"],errors="coerce").dt.tz_localize(None); return d

df=load_df(PT,FT,S,E)
if df.empty: st.warning("No rows in this window."); st.stop()
df["fraud_score"]=pd.to_numeric(df["fraud_score"],errors="coerce").fillna(0.0); df["is_alert"]=(df["fraud_score"]>=TH).astype(int); df["day"]=df["timestamp"].dt.floor("D")

c1,c2,c3,c4=st.columns([1,1,1,2]); TOT=len(df); AL=int(df["is_alert"].sum())
c1.metric("Scored",TOT); c2.metric("Alerts",AL); c3.metric("Alert rate",f"{(AL/TOT if TOT else 0):.2%}")
c4.caption(f"Window: {df['timestamp'].min()} → {df['timestamp'].max()} | Threshold: {TH:.2f}"); st.markdown("---")

st.subheader("Daily trend")
tr=df.groupby("day").agg(scored=("order_id","count"),alerts=("is_alert","sum")).reset_index()
if len(tr):
  tl=tr.melt("day",["scored","alerts"],"series","value")
  st.altair_chart(alt.Chart(tl).mark_line(point=True).encode(x="day:T",y="value:Q",color="series:N",tooltip=[alt.Tooltip("day:T"),"series:N",alt.Tooltip("value:Q")]).properties(height=260),use_container_width=True)
else: st.info("No activity.")

st.subheader("Fraud-score distribution")
h=alt.Chart(df).mark_bar().encode(x=alt.X("fraud_score:Q",bin=alt.Bin(maxbins=50),title="Fraud score"),y=alt.Y("count():Q",title="Rows"),tooltip=[alt.Tooltip("count()",title="Rows")]).properties(height=200)
st.altair_chart(h+alt.Chart(pd.DataFrame({"x":[TH]})).mark_rule(color="crimson").encode(x="x"),use_container_width=True)

def prev(cols,title):
  cols=[c for c in cols if c in df]
  if not cols: st.info(f"No signals for: {title}"); return
  z=df[["is_alert"]+cols].apply(pd.to_numeric,errors="coerce").fillna(0).astype(int); a=z["is_alert"].sum() or 1; na=(1-z["is_alert"]).sum() or 1
  dd=pd.DataFrame([{"signal":c,"% in alerts":z.loc[z.is_alert==1,c].sum()/a,"% in non-alerts":z.loc[z.is_alert==0,c].sum()/na} for c in cols]).sort_values("% in alerts",False).melt("signal","group","value")
  st.altair_chart(alt.Chart(dd).mark_bar().encode(x=alt.X("value:Q",axis=alt.Axis(format="%"),title="Prevalence"),y=alt.Y("signal:N",sort="-x",title=None),color="group:N",tooltip=["signal:N",alt.Tooltip("value:Q",format=".1%"),"group:N"]).properties(title=title,height=300),use_container_width=True)

st.subheader("Context signals"); l,r=st.columns(2)
with l: prev(["strong_tri_mismatch_high_value","strong_high_value_express_geo","strong_burst_multi_device","strong_price_drop_bulk","strong_giftcard_geo","strong_return_whiplash","strong_price_inventory_stress","strong_country_flip_express"],"Strong-signal prevalence")
with r: prev(["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"],"Pricing & Inventory prevalence")

st.caption("Correlation of fraud_score with pricing/inventory signals (Pearson)")
C=[c for c in ["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"] if c in df]
if C: co=df[["fraud_score"]+C].apply(pd.to_numeric,errors="coerce").fillna(0).corr().loc[C,"fraud_score"].reset_index().rename(columns={"index":"signal","fraud_score":"corr"}); st.altair_chart(alt.Chart(co).mark_bar().encode(x="corr:Q",y=alt.Y("signal:N",sort="x",title=None),tooltip=[alt.Tooltip("corr:Q",format=".3f")]).properties(height=120),use_container_width=True)
else: st.info("No pricing/inventory columns to correlate.")

st.subheader("Top alerts"); cols=[c for c in ["order_id","timestamp","customer_id","store_id","sku_id","sku_category","order_amount","quantity","payment_method","shipping_country","ip_country","fraud_score"] if c in df]
st.dataframe(df.sort_values(["fraud_score","timestamp"],ascending=[False,False]).loc[:,cols].head(50),use_container_width=True,height=320)

st.subheader("Model evaluation")
L=[c for c in ["fraud_flag","is_fraud","label","ground_truth","gt","y"] if c in df]; 
if L: lab=st.selectbox("Ground-truth (1=fraud,0=legit)",L,0); y_true=df[lab].fillna(0).astype(int).values; st.caption(f"Using: `{lab}`")
else: st.warning("No label column; using decision as proxy."); y_true=(df["fraud_score"]>=TH).astype(int).values
y_pred=(df["fraud_score"]>=TH).astype(int).values; y_score=df["fraud_score"].values
m1,m2,m3,m4=st.columns(4); m1.metric("Accuracy",f"{accuracy_score(y_true,y_pred):.2%}"); m2.metric("Precision",f"{precision_score(y_true,y_pred,zero_division=0):.2%}"); m3.metric("Recall",f"{recall_score(y_true,y_pred,zero_division=0):.2%}"); m4.metric("F1-score",f"{f1_score(y_true,y_pred,zero_division=0):.2%}")
cm=confusion_matrix(y_true,y_pred,labels=[0,1]); cl=pd.DataFrame(cm,index=["Actual: 0","Actual: 1"],columns=["Pred: 0","Pred: 1"]).reset_index().melt(id_vars=["index"],var_name="Predicted",value_name="Count").rename(columns={"index":"Actual"})
st.altair_chart(alt.Chart(cl).mark_rect().encode(x="Predicted:N",y="Actual:N",color=alt.Color("Count:Q",scale=alt.Scale(scheme="blues")),tooltip=["Actual:N","Predicted:N","Count:Q"]).properties(height=185)+alt.Chart(cl).mark_text(baseline="middle").encode(x="Predicted:N",y="Actual:N",text="Count:Q"),use_container_width=True)

try: A=roc_auc_score(y_true,y_score)
except: A=float("nan")
fpr,tpr,_=roc_curve(y_true,y_score); st.altair_chart(alt.Chart(pd.DataFrame({"fpr":fpr,"tpr":tpr})).mark_line().encode(x=alt.X("fpr:Q",title="False Positive Rate"),y=alt.Y("tpr:Q",title="True Positive Rate")).properties(height=205,title=f"ROC (AUC={A:.3f})")+alt.Chart(pd.DataFrame({"x":[0,1],"y":[0,1]})).mark_rule(strokeDash=[4,4]).encode(x="x",y="y"),use_container_width=True)

P,R,_=precision_recall_curve(y_true,y_score); AP=auc(R,P); st.altair_chart(alt.Chart(pd.DataFrame({"recall":R,"precision":P})).mark_line().encode(x="recall:Q",y="precision:Q").properties(height=205,title=f"Precision–Recall (AP≈{AP:.3f})"),use_container_width=True)

@st.cache_data(show_spinner=False)
def load_ops(MT,S,E):
  sql=f"SELECT threshold,AVG(precision) precision,AVG(recall) recall FROM `{MT}` WHERE dt BETWEEN @S AND @E GROUP BY threshold ORDER BY threshold"
  return bq.query(sql,job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("S","DATE",str(S)),bigquery.ScalarQueryParameter("E","DATE",str(E))])).result().to_dataframe()
st.subheader("Operating point helper"); 
use_bq=True
try: OP=load_ops(MT,S,E); use_bq=not OP.empty
except: use_bq=False
if not use_bq: grid=np.round(np.linspace(0.05,0.95,19),2); OP=pd.DataFrame([{"threshold":t,"precision":(((df["fraud_score"]>=t)&(y_true==1)).sum()/max(1,(df["fraud_score"]>=t).sum())),"recall":(((df["fraud_score"]>=t)&(y_true==1)).sum()/max(1,(y_true==1).sum()))} for t in grid])
st.altair_chart(alt.Chart(OP.melt("threshold",["precision","recall"],"metric","value")).mark_line(point=True).encode(x="threshold:Q",y=alt.Y("value:Q",axis=alt.Axis(format="%")),color="metric:N").properties(height=205,title=("BigQuery metrics" if use_bq else "Local fallback")),use_container_width=True)
if not use_bq: st.caption(f"No precomputed metrics at `{MT}`; showing local sweep.")
