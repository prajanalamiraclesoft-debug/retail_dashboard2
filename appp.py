# app.py —  ~100 lines, full dashboard
import numpy as np, pandas as pd, altair as alt, streamlit as st
from datetime import date
from google.cloud import bigquery
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score,precision_recall_curve,auc,accuracy_score,precision_score,recall_score,f1_score

st.set_page_config("Retail Dashboard", layout="wide")
st.title("Retail Dashboard — Fraud, Pricing & Inventory")

# --------- Inputs
sb=st.sidebar; sb.header("Settings")
P=sb.text_input("Project","mss-data-engineer-sandbox")
D=sb.text_input("Dataset","retail")
PT=sb.text_input("Predictions table",f"{P}.{D}.predictions_latest")
FT=sb.text_input("Features table",f"{P}.{D}.features_signals_v4")
MT=sb.text_input("Metrics table (optional)",f"{P}.{D}.predictions_daily_metrics")
S=sb.date_input("Start",date(2024,12,1)); E=sb.date_input("End",date(2024,12,31))
TH=sb.slider("Alert threshold (score ≥ threshold ⇒ alert)",0.0,1.0,0.30,0.01)

# --------- Data
@st.cache_data(show_spinner=True)
def load(pt,ft,s,e,project):
    cli=bigquery.Client(project=project)
    sql=f"""SELECT p.order_id,p.timestamp,p.customer_id,p.store_id,p.sku_id,p.sku_category,
                   p.order_amount,p.quantity,p.payment_method,p.shipping_country,p.ip_country,
                   CAST(p.fraud_score AS FLOAT64) fraud_score,
                   s.strong_tri_mismatch_high_value,s.strong_high_value_express_geo,
                   s.strong_burst_multi_device,s.strong_price_drop_bulk,s.strong_giftcard_geo,
                   s.strong_return_whiplash,s.strong_price_inventory_stress,s.strong_country_flip_express,
                   s.high_price_anomaly,s.low_price_anomaly,s.oversell_flag,s.stockout_risk_flag,s.hoarding_flag,
                   SAFE_CAST(s.fraud_flag AS INT64) fraud_flag
            FROM `{pt}` p LEFT JOIN `{ft}` s USING(order_id)
            WHERE DATE(p.timestamp) BETWEEN @s AND @e ORDER BY timestamp"""
    job=cli.query(sql,job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("s","DATE",str(s)),
                          bigquery.ScalarQueryParameter("e","DATE",str(e))]))
    df=job.result().to_dataframe(create_bqstorage_client=False)
    if "timestamp" in df: df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
    return df
df=load(PT,FT,S,E,P)
if df.empty: st.warning("No rows in this window."); st.stop()
df["fraud_score"]=pd.to_numeric(df["fraud_score"],errors="coerce")
df["is_alert"]=(df["fraud_score"]>=TH).astype(int)
df["date"]=df["timestamp"].dt.date

# --------- KPIs
c1,c2,c3,c4=st.columns([1,1,1,2])
n=len(df); a=int(df["is_alert"].sum()); r=a/n if n else 0
c1.metric("Scored",n); c2.metric("Alerts",a); c3.metric("Alert rate",f"{r:.2%}")
c4.caption(f"Window: {df['timestamp'].min()} → {df['timestamp'].max()}  |  Threshold: {TH:.2f}")

# --------- Daily trend
trend=df.groupby("date").agg(n=("order_id","count"),alerts=("is_alert","sum")).reset_index()
st.subheader("Daily trend"); 
st.altair_chart(alt.Chart(trend.melt("date",["n","alerts"],"series","count"))
    .mark_line(point=True).encode(x="date:T",y="count:Q",color="series:N",tooltip=["date:T","series:N","count:Q"])
    .properties(height=240),use_container_width=True)

# --------- Score distribution
st.subheader("Fraud-score distribution")
hist=(alt.Chart(df).mark_bar().encode(x=alt.X("fraud_score:Q",bin=alt.Bin(maxbins=50)),
                                      y=alt.Y("count():Q"),tooltip=[alt.Tooltip("count()",title="Count")])
       .properties(height=220))
rule=alt.Chart(pd.DataFrame({"x":[TH]})).mark_rule(color="crimson").encode(x="x:Q")
st.altair_chart(hist+rule,use_container_width=True)

# --------- Signal prevalence (alerts vs non-alerts)
def prevalence(cols,title):
    present=[c for c in cols if c in df.columns]; 
    if not present: st.info(f"No signals for {title}."); return
    rows=[]
    for c in present:
        col=pd.to_numeric(df[c],errors="coerce").fillna(0).astype(int)
        da=max(int((df["is_alert"]==1).sum()),1); dn=max(int((df["is_alert"]==0).sum()),1)
        rows.append({"signal":c,"% in alerts":int(col[df["is_alert"]==1].sum())/da,"% in non-alerts":int(col[df["is_alert"]==0].sum())/dn})
    dd=pd.DataFrame(rows).sort_values("% in alerts",ascending=False)
    st.altair_chart(alt.Chart(dd.melt("signal",["% in alerts","% in non-alerts"],"group","value"))
        .mark_bar().encode(x=alt.X("value:Q",axis=alt.Axis(format="%")),y=alt.Y("signal:N",sort="-x"),
                           color="group:N",tooltip=["signal:N",alt.Tooltip("value:Q",format=".1%"),"group:N"])
        .properties(height=320,title=title),use_container_width=True)

st.subheader("Context signals")
lcol,rcol=st.columns(2)
with lcol: prevalence([
    "strong_tri_mismatch_high_value","strong_high_value_express_geo","strong_burst_multi_device",
    "strong_price_drop_bulk","strong_giftcard_geo","strong_return_whiplash",
    "strong_price_inventory_stress","strong_country_flip_express"],"Strong-signal prevalence")
with rcol: prevalence(["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"],
                      "Pricing & Inventory signal prevalence")

# --------- Correlation (score vs signals)
st.caption("Correlation of fraud_score with pricing/inventory signals (Pearson)")
cols=[c for c in ["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"] if c in df]
if cols:
    cdf=df[["fraud_score"]+cols].apply(pd.to_numeric,errors="coerce")
    cor=cdf.corr().loc[cols,"fraud_score"].rename("corr").reset_index().rename(columns={"index":"signal"})
    st.altair_chart(alt.Chart(cor).mark_bar().encode(x="corr:Q",y=alt.Y("signal:N",sort="x"),
                    tooltip=[alt.Tooltip("corr:Q",format=".3f"),"signal:N"]).properties(height=160),use_container_width=True)

# --------- Top alerts
st.subheader("Top alerts (highest scores)")
keep=[c for c in ["order_id","timestamp","customer_id","store_id","sku_id","sku_category","order_amount",
                  "quantity","payment_method","shipping_country","ip_country","fraud_score"] if c in df]
st.dataframe(df.sort_values(["fraud_score","timestamp"],ascending=[False,False]).loc[:,keep].head(50),
             use_container_width=True,height=360)

# --------- Model evaluation
st.subheader("Model evaluation")
lbls=[c for c in ["fraud_flag","is_fraud","label","ground_truth","gt","y"] if c in df]
if lbls:
    y=df[lbls[0]].fillna(0).astype(int).values; st.caption(f"Using ground-truth column: `{lbls[0]}`.")
else:
    y=(df["fraud_score"]>=TH).astype(int).values; st.warning("No ground-truth label found; using model decision as proxy.")
yhat=(df["fraud_score"]>=TH).astype(int).values; ysc=df["fraud_score"].values
m1,m2,m3,m4=st.columns(4)
m1.metric("Accuracy",f"{accuracy_score(y,yhat):.2%}"); m2.metric("Precision",f"{precision_score(y,yhat,zero_division=0):.2%}")
m3.metric("Recall",f"{recall_score(y,yhat,zero_division=0):.2%}"); m4.metric("F1-score",f"{f1_score(y,yhat,zero_division=0):.2%}")

cm=confusion_matrix(y,yhat,labels=[0,1]); cmd=pd.DataFrame(cm,index=["Actual: 0","Actual: 1"],columns=["Pred: 0","Pred: 1"]).reset_index()
cml=cmd.melt("Actual","Predicted","Count")
st.altair_chart(alt.Chart(cml).mark_rect().encode(x="Predicted:N",y="Actual:N",color=alt.Color("Count:Q",scale=alt.Scale(scheme="blues")),
                   tooltip=["Actual:N","Predicted:N","Count:Q"]).properties(height=200)
                +alt.Chart(cml).mark_text(baseline="middle").encode(x="Predicted:N",y="Actual:N",text="Count:Q"),
                use_container_width=True)

try: auc_roc=roc_auc_score(y,ysc)
except: auc_roc=float("nan")
fpr,tpr,_=roc_curve(y,ysc); pr,rc,_=precision_recall_curve(y,ysc); ap=auc(rc,pr)
st.altair_chart(alt.Chart(pd.DataFrame({"fpr":fpr,"tpr":tpr})).mark_line().encode(x="fpr:Q",y="tpr:Q",
    tooltip=[alt.Tooltip("fpr:Q",format=".3f"),alt.Tooltip("tpr:Q",format=".3f")]).properties(height=240,title=f"ROC (AUC={auc_roc:.3f})"),
    use_container_width=True)
st.altair_chart(alt.Chart(pd.DataFrame({"recall":rc,"precision":pr})).mark_line().encode(x="recall:Q",y="precision:Q",
    tooltip=[alt.Tooltip("recall:Q",format=".3f"),alt.Tooltip("precision:Q",format=".3f")]).properties(height=240,title=f"PR (AP≈{ap:.3f})"),
    use_container_width=True)

# --------- Operating point helper
st.subheader("Operating point helper (precision/recall vs threshold)")
@st.cache_data(show_spinner=False)
def bq_metrics(mt,s,e,project):
    try:
        cli=bigquery.Client(project=project)
        q=f"SELECT threshold,AVG(precision) precision,AVG(recall) recall FROM `{mt}` WHERE dt BETWEEN @s AND @e GROUP BY threshold"
        job=cli.query(q,job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("s","DATE",str(s)),
                              bigquery.ScalarQueryParameter("e","DATE",str(e))]))
        return job.result().to_dataframe(create_bqstorage_client=False)
    except: return pd.DataFrame()
op=bq_metrics(MT,S,E,P)
if op.empty:
    grid=np.round(np.linspace(0.05,0.95,19),2); rows=[]
    for th in grid:
        pred=(df["fraud_score"]>=th).astype(int); tp=int(((pred==1)&(df["is_alert"]==1)).sum())
        fp=int(((pred==1)&(df["is_alert"]==0)).sum()); fn=int(((pred==0)&(df["is_alert"]==1)).sum())
        pr=tp/(tp+fp) if (tp+fp)>0 else 0.0; rc=tp/(tp+fn) if (tp+fn)>0 else 0.0; rows.append({"threshold":th,"precision":pr,"recall":rc})
    op=pd.DataFrame(rows); st.caption("No precomputed BigQuery metrics; showing local fallback.")
st.altair_chart(alt.Chart(op.melt("threshold",["precision","recall"],"metric","value"))
    .mark_line(point=True).encode(x="threshold:Q",y=alt.Y("value:Q",axis=alt.Axis(format="%")),color="metric:N",
              tooltip=["threshold:Q",alt.Tooltip("value:Q",format=".2%")]).properties(height=260),use_container_width=True)


