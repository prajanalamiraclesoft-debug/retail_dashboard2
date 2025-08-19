# app.py — Compact Fraud/Inventory Dashboard (~100 lines)
import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import date
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score

st.set_page_config("Retail Dashboard: Fraud & Inventory", layout="wide")
alt.data_transformers.enable("default", max_rows=None); alt.renderers.set_embed_options(actions=False)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    proj = st.text_input("Project", "mss-data-engineer-sandbox")
    dset = st.text_input("Dataset", "retail")
    pred_tbl = st.text_input("Predictions", f"{proj}.{dset}.predictions_latest")
    feat_tbl = st.text_input("Features", f"{proj}.{dset}.features_signals_v4")
    met_tbl  = st.text_input("Metrics (optional)", f"{proj}.{dset}.predictions_daily_metrics")
    start = st.date_input("Start", date(2024,12,1)); end = st.date_input("End", date(2024,12,31))
    thr = st.slider("Alert threshold (≥)", 0.00, 1.00, 0.30, 0.01)

# ---------- BigQuery client ----------
sa = dict(st.secrets["gcp_service_account"]); sa["private_key"] = sa["private_key"].replace("\\n","\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq = bigquery.Client(credentials=creds, project=creds.project_id)

@st.cache_data(show_spinner=True)
def load_df(pred, feat, s, e):
    sql = f"""
    SELECT p.order_id,p.timestamp,p.customer_id,p.store_id,p.sku_id,p.sku_category,
           p.order_amount,p.quantity,p.payment_method,p.shipping_country,p.ip_country,
           CAST(p.fraud_score AS FLOAT64) fraud_score,
           s.strong_tri_mismatch_high_value,s.strong_high_value_express_geo,s.strong_burst_multi_device,
           s.strong_price_drop_bulk,s.strong_giftcard_geo,s.strong_return_whiplash,
           s.strong_price_inventory_stress,s.strong_country_flip_express,
           s.high_price_anomaly,s.low_price_anomaly,s.oversell_flag,s.stockout_risk_flag,s.hoarding_flag,
           SAFE_CAST(s.fraud_flag AS INT64) fraud_flag
    FROM `{pred}` p LEFT JOIN `{feat}` s USING(order_id)
    WHERE DATE(p.timestamp) BETWEEN @start AND @end ORDER BY p.timestamp
    """
    job = bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("start","DATE",str(s)),
                          bigquery.ScalarQueryParameter("end","DATE",str(e))]))
    df = job.result().to_dataframe()
    if "timestamp" in df: df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
    return df

df = load_df(pred_tbl, feat_tbl, start, end)
if df.empty: st.warning("No rows in this window."); st.stop()
df["fraud_score"]=pd.to_numeric(df["fraud_score"],errors="coerce").fillna(0.0)
df["is_alert"]=(df["fraud_score"]>=thr).astype(int); df["date"]=df["timestamp"].dt.date

# ---------- KPIs ----------
c1,c2,c3,c4 = st.columns([1,1,1,2])
tot=len(df); al=int(df["is_alert"].sum()); rate=(al/tot) if tot else 0
c1.metric("Scored", tot); c2.metric("Alerts", al); c3.metric("Alert rate", f"{rate:.2%}")
c4.caption(f"Window: {df['timestamp'].min()} → {df['timestamp'].max()}  |  Threshold: {thr:.2f}")
st.markdown("---")

# ---------- Daily trend ----------
st.subheader("Daily trend")
trend=(df.groupby("date").agg(n=("order_id","count"),alerts=("is_alert","sum")).reset_index())
if len(trend):
    tl=trend.melt("date",["n","alerts"],"series","count")
    st.altair_chart(alt.Chart(tl).mark_line(point=True).encode(
        x="date:T", y="count:Q", color="series:N", tooltip=["date:T","series:N","count:Q"]).properties(height=260),
        use_container_width=True)
else: st.info("No activity for selected dates.")

# ---------- Score distribution ----------
st.subheader("Fraud-score distribution")
hist=alt.Chart(df).mark_bar().encode(x=alt.X("fraud_score:Q",bin=alt.Bin(maxbins=50),title="Fraud score"),
                                     y=alt.Y("count():Q",title="Count"),
                                     tooltip=[alt.Tooltip("count()",title="Count")]).properties(height=220)
st.altair_chart(hist + alt.Chart(pd.DataFrame({"x":[thr]})).mark_rule(color="crimson").encode(x="x"),
                use_container_width=True)

# ---------- Signal prevalence ----------
def prevalence_block(cols, title):
    cols=[c for c in cols if c in df.columns]; 
    if not cols: st.info(f"No signals for: {title}"); return
    z=df[["is_alert"]+cols].copy()
    for c in cols: z[c]=pd.to_numeric(z[c],errors="coerce").fillna(0).astype(int)
    m=[]
    a_sum=z["is_alert"].sum() or 1; na_sum=(1-z["is_alert"]).sum() or 1
    for c in cols:
        m.append({"signal":c,"% in alerts":z.loc[z.is_alert==1,c].sum()/a_sum,
                  "% in non-alerts":z.loc[z.is_alert==0,c].sum()/na_sum})
    dd=pd.DataFrame(m).sort_values("% in alerts",ascending=False).melt("signal",var_name="group",value_name="value")
    st.altair_chart(alt.Chart(dd).mark_bar().encode(x=alt.X("value:Q",axis=alt.Axis(format="%"),title="Prevalence"),
        y=alt.Y("signal:N",sort="-x",title=None), color="group:N",
        tooltip=["signal:N",alt.Tooltip("value:Q",format=".1%"),"group:N"]).properties(title=title,height=320),
        use_container_width=True)

st.subheader("Context signals")
l,r=st.columns(2)
with l: prevalence_block(
    ["strong_tri_mismatch_high_value","strong_high_value_express_geo","strong_burst_multi_device",
     "strong_price_drop_bulk","strong_giftcard_geo","strong_return_whiplash",
     "strong_price_inventory_stress","strong_country_flip_express"],
    "Strong-signal prevalence (alerts vs non-alerts)")
with r: prevalence_block(
    ["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"],
    "Pricing & Inventory signal prevalence")

# ---------- Correlations ----------
st.caption("Correlation of fraud_score with pricing/inventory signals (Pearson)")
corr_cols=[c for c in ["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"] if c in df]
if corr_cols:
    cdf=df[["fraud_score"]+corr_cols].apply(pd.to_numeric,errors="coerce").fillna(0)
    co=cdf.corr().loc[corr_cols,"fraud_score"].reset_index().rename(columns={"index":"signal","fraud_score":"corr"})
    st.altair_chart(alt.Chart(co).mark_bar().encode(x=alt.X("corr:Q",title="Correlation"),
        y=alt.Y("signal:N",sort="x",title=None), tooltip=[alt.Tooltip("corr:Q",format=".3f")]).properties(height=140),
        use_container_width=True)
else: st.info("No pricing/inventory columns to correlate.")

# ---------- Top alerts ----------
st.subheader("Top alerts (highest scores)")
cols=[c for c in ["order_id","timestamp","customer_id","store_id","sku_id","sku_category",
                  "order_amount","quantity","payment_method","shipping_country","ip_country","fraud_score"] if c in df]
st.dataframe(df.sort_values(["fraud_score","timestamp"],ascending=[False,False]).loc[:,cols].head(50),
             use_container_width=True, height=360)

# ---------- Evaluation ----------
st.subheader("Model evaluation")
labs=[c for c in ["fraud_flag","is_fraud","label","ground_truth","gt","y"] if c in df]
if labs:
    lab=st.selectbox("Ground-truth column (1=fraud, 0=legit)", labs, index=0); y_true=df[lab].fillna(0).astype(int).values
    st.caption(f"Using: `{lab}`")
else:
    st.warning("No label column found; using current decision as proxy.")
    y_true=(df["fraud_score"]>=thr).astype(int).values
y_pred=(df["fraud_score"]>=thr).astype(int).values; y_score=df["fraud_score"].values
m1,m2,m3,m4=st.columns(4)
m1.metric("Accuracy",f"{accuracy_score(y_true,y_pred):.2%}")
m2.metric("Precision",f"{precision_score(y_true,y_pred,zero_division=0):.2%}")
m3.metric("Recall",f"{recall_score(y_true,y_pred,zero_division=0):.2%}")
m4.metric("F1-score",f"{f1_score(y_true,y_pred,zero_division=0):.2%}")

cm=confusion_matrix(y_true,y_pred,labels=[0,1]); cm_df=pd.DataFrame(cm, index=["Actual: 0","Actual: 1"], columns=["Pred: 0","Pred: 1"]).reset_index()
cm_long=cm_df.melt(id_vars=["index"], var_name="Predicted", value_name="Count").rename(columns={"index":"Actual"})
heat=alt.Chart(cm_long).mark_rect().encode(x="Predicted:N",y="Actual:N",color=alt.Color("Count:Q",scale=alt.Scale(scheme="blues")),
                                           tooltip=["Actual:N","Predicted:N","Count:Q"]).properties(height=200)
st.altair_chart(heat + alt.Chart(cm_long).mark_text(baseline="middle").encode(x="Predicted:N",y="Actual:N",text="Count:Q"),
                use_container_width=True)

try: auc_roc=roc_auc_score(y_true,y_score)
except: auc_roc=float("nan")
fpr,tpr,_=roc_curve(y_true,y_score); roc_df=pd.DataFrame({"fpr":fpr,"tpr":tpr})
st.altair_chart(alt.Chart(roc_df).mark_line().encode(x=alt.X("fpr:Q",title="False Positive Rate"),
    y=alt.Y("tpr:Q",title="True Positive Rate"), tooltip=[alt.Tooltip("fpr:Q",format=".3f"),alt.Tooltip("tpr:Q",format=".3f")])
    .properties(height=240,title=f"ROC (AUC={auc_roc:.3f})") +
    alt.Chart(pd.DataFrame({"x":[0,1],"y":[0,1]})).mark_rule(strokeDash=[4,4]).encode(x="x",y="y"),
    use_container_width=True)

prec,rec,_=precision_recall_curve(y_true,y_score); ap=auc(rec,prec)
pr_df=pd.DataFrame({"recall":rec,"precision":prec})
st.altair_chart(alt.Chart(pr_df).mark_line().encode(x="recall:Q",y="precision:Q",
    tooltip=[alt.Tooltip("recall:Q",format=".3f"),alt.Tooltip("precision:Q",format=".3f")])
    .properties(height=240,title=f"Precision–Recall (AP≈{ap:.3f})"), use_container_width=True)

# ---------- Operating point helper (BQ or local) ----------
@st.cache_data(show_spinner=False)
def load_bq_ops(tbl,s,e):
    sql=f"SELECT threshold,AVG(precision) precision,AVG(recall) recall FROM `{tbl}` WHERE dt BETWEEN @s AND @e GROUP BY threshold ORDER BY threshold"
    job=bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("s","DATE",str(s)),
                          bigquery.ScalarQueryParameter("e","DATE",str(e))]))
    return job.result().to_dataframe()
st.subheader("Operating point helper")
use_bq=True
try:
    op=load_bq_ops(met_tbl,start,end); use_bq=not op.empty
except: use_bq=False
if not use_bq:
    grid=np.round(np.linspace(0.05,0.95,19),2); op=[]
    y_true_local=y_true
    for th in grid:
        pred=(df["fraud_score"]>=th).astype(int)
        tp=((pred==1)&(y_true_local==1)).sum(); fp=((pred==1)&(y_true_local==0)).sum(); fn=((pred==0)&(y_true_local==1)).sum()
        op.append({"threshold":th,"precision": tp/(tp+fp) if (tp+fp) else 0.0,"recall": tp/(tp+fn) if (tp+fn) else 0.0})
    op=pd.DataFrame(op)
chart=op.melt("threshold",["precision","recall"],"metric","value")
st.altair_chart(alt.Chart(chart).mark_line(point=True).encode(x="threshold:Q",y=alt.Y("value:Q",axis=alt.Axis(format="%")),
    color="metric:N", tooltip=["threshold:Q",alt.Tooltip("value:Q",format=".2%")]).properties(height=240,
    title=("BigQuery metrics" if use_bq else "Local fallback")), use_container_width=True)
if not use_bq: st.caption(f"No precomputed metrics found at `{met_tbl}` for the window; showing local sweep.")
