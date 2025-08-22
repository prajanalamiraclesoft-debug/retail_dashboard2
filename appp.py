# streamlit run app.py
import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import date
from google.cloud import bigquery; from google.oauth2 import service_account
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config("Fraud Dashboard (Fixed TH=0.30)", layout="wide"); alt.renderers.set_embed_options(actions=False)
TH=0.30
with st.sidebar:
    st.header("BigQuery")
    PROJ=st.text_input("Project","mss-data-engineer-sandbox"); DATASET=st.text_input("Dataset","retail")
    RAW=st.text_input("Raw table",f"{PROJ}.{DATASET}.transaction_data")
    S=st.date_input("Start",date(2023,1,1)); E=st.date_input("End",date(2030,12,31))

sa=dict(st.secrets["gcp_service_account"]); sa["private_key"]=sa["private_key"].replace("\\n","\n")
creds=service_account.Credentials.from_service_account_info(sa); bq=bigquery.Client(credentials=creds, project=creds.project_id)

@st.cache_data(show_spinner=True)
def load_raw(RAW,S,E):
    sql=f"""
      SELECT order_id,TIMESTAMP(timestamp) ts,customer_id,store_id,sku_id,sku_category,
             SAFE_CAST(quantity AS FLOAT64) q,SAFE_CAST(unit_price AS FLOAT64) p,
             CAST(payment_method AS STRING) pay,CAST(shipping_country AS STRING) ship,CAST(ip_country AS STRING) ip,
             CAST(device_id AS STRING) dev,SAFE_CAST(account_created_at AS TIMESTAMP) acct,
             SAFE_CAST(coupon_discount AS FLOAT64) coup,SAFE_CAST(gift_card_amount AS FLOAT64) gift_amt,
             SAFE_CAST(gift_card_used AS BOOL) gift_used,SAFE_CAST(fraud_flag AS INT64) y
      FROM `{RAW}` WHERE DATE(timestamp) BETWEEN @S AND @E"""
    j=bq.query(sql,job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("S","DATE",str(S)), bigquery.ScalarQueryParameter("E","DATE",str(E))]))
    return j.result().to_dataframe()
df=load_raw(RAW,S,E)
if df.empty: st.warning("No rows in this range."); st.stop()

# -------- cleaning
df=df.sort_values("ts").drop_duplicates("order_id",keep="last"); df=df[(df.q>0)&(df.p>0)].copy()
df["amt"]=df.q*df.p; df["age"]=(pd.to_datetime(df["ts"]).dt.date-pd.to_datetime(df["acct"]).dt.date).dt.days.fillna(0)
for c in ["p","q"]:
    df[c]=df.groupby("sku_category")[c].transform(lambda s:s.clip(s.quantile(.01),s.quantile(.99)))
df["amt"]=df.q*df.p

# -------- features
cat_avg=df.groupby("sku_category")["p"].transform("mean").replace(0,np.nan)
df["price_ratio"]=(df["p"]/cat_avg).fillna(1.0)
den=df["amt"].replace(0,np.nan); df["coup_pct"]=(df["coup"]/den).fillna(0); df["gift_pct"]=(df["gift_amt"]/den).fillna(0)
df["geo"]=(df["ship"]!=df["ip"]).astype(int)

def roll_cnt_hours(g,h):
    t=(pd.to_datetime(g.ts).astype("int64")//1_000_000).to_numpy(); w=int(h*3600*1e6); a=np.sort(t); i=np.searchsorted(a,a-w,'left'); c=np.arange(1,len(a)+1)-i
    return pd.Series(c,index=g.sort_values("ts").index).reindex(g.index)
df=df.sort_values("ts")
df["cust_1h"]=df.groupby("customer_id",group_keys=False).apply(roll_cnt_hours,1)
df["cust_24h"]=df.groupby("customer_id",group_keys=False).apply(roll_cnt_hours,24)
df["dev_1h"]=df.groupby("dev",group_keys=False).apply(roll_cnt_hours,1)

def roll_stats(g):
    r=g["amt"].rolling(10,min_periods=1); return pd.DataFrame({"m":r.mean(),"s":r.std(ddof=0).replace(0,np.nan)})
rs=df.groupby("customer_id",group_keys=False).apply(roll_stats); df["c_m"]=rs["m"].values; df["c_s"]=rs["s"].values
df["z"]=((df["amt"]-df["c_m"])/df["c_s"]).replace([np.inf,-np.inf],0).fillna(0)
p90=np.nanpercentile(df["amt"],90) if len(df) else 0

# strong signals
df["s_price_bulk"]=((df["price_ratio"].sub(1).abs()>=.5)&(df["q"]>=3)).astype(int)
df["s_gc_geo"]=((df["gift_used"].fillna(False))&(df["geo"]==1)).astype(int)
df["s_burst"]=((df["cust_1h"]>=3)|(df["dev_1h"]>=3)).astype(int)
df["s_z"]= (df["z"].abs()>=2).astype(int)
df["s_geo_hi"]=((df["geo"]==1)&(df["amt"]>=p90)).astype(int)
df["s_any"]=(df[["s_price_bulk","s_gc_geo","s_burst","s_z","s_geo_hi"]].sum(axis=1)>0).astype(int)

# -------- train + calibrate to TH=0.30
lab=df[df.y.isin([0,1])].copy()
if lab.empty: st.error("No labeled rows (fraud_flag)."); st.stop()
cat_cols=["sku_category","pay","ship","ip","dev","store_id"]; num_cols=["amt","q","p","age","price_ratio","coup_pct","gift_pct","cust_1h","cust_24h","dev_1h","c_m","c_s","z","geo","s_price_bulk","s_gc_geo","s_burst","s_z","s_geo_hi","s_any"]
Xn=lab[num_cols].fillna(0); Xc=pd.get_dummies(lab[cat_cols].astype(str),dummy_na=False); X=pd.concat([Xn,Xc],axis=1); y=lab.y.astype(int).values
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42); pr=np.zeros(len(lab))
for tr,va in skf.split(X,y):
    m=HistGradientBoostingClassifier(max_iter=300,learning_rate=.1); m.fit(X.iloc[tr],y[tr]); pr[va]=m.predict_proba(X.iloc[va])[:,1]
ts=np.linspace(.01,.99,99); f1=[(lambda p,r:0 if p+r==0 else 2*p*r/(p+r))(*(precision_score(y,(pr>=t).astype(int),zero_division=0),recall_score(y,(pr>=t).astype(int),zero_division=0))) for t in ts]
t_star=ts[int(np.argmax(f1))]; scale=TH/t_star if t_star>0 else 1.0
M=HistGradientBoostingClassifier(max_iter=300,learning_rate=.1).fit(X,y)

def score_all(D):
    Xn=D[num_cols].fillna(0); Xc=pd.get_dummies(D[cat_cols].astype(str),dummy_na=False); Xc=Xc.reindex(columns=Xc.columns.union(X.columns[Xc.shape[1]-Xn.shape[1]:],sort=False),fill_value=0)
    Xc=Xc.reindex(columns=[c for c in X.columns if c not in Xn.columns],fill_value=0); X_full=pd.concat([Xn,Xc],axis=1); return np.clip(M.predict_proba(X_full)[:,1]*scale,0,1)

df["score"]=score_all(df)
safe=(df["amt"]<=100)&(df["q"]<=2)&(df["ship"]==df["ip"])&(df["gift_pct"]<.05)&(df["price_ratio"].sub(1).abs()<=.15)&(df["s_price_bulk"]==0)&(df["s_gc_geo"]==0)&(df["s_burst"]==0)&(df["s_z"]==0)&(df["s_geo_hi"]==0)
df["alert_raw"]=(df["score"]>=TH).astype(int); df["alert"]=((df["score"]>=TH)&(~safe)).astype(int)

# -------- KPIs + eval
st.subheader("**Key Metrics (Threshold = 0.30)**"); TOT=len(df); AL=int(df["alert"].sum())
c1,c2,c3=st.columns(3); c1.metric("**Total transactions**",TOT); c2.metric("**Fraud alerts (≥0.30)**",AL); c3.metric("**Alert rate**",f"{AL/max(1,TOT):.2%}")
st.caption(f"Calibration: best F1 t*≈{t_star:.2f} → scaled so decisions occur at 0.30.")

st.subheader("**Model Evaluation (fixed threshold = 0.30)**")
y_true=lab.y.astype(int).values; y_pred=df.loc[lab.index,"alert"].astype(int).values
m1,m2,m3,m4=st.columns(4); m1.metric("Accuracy",f"{accuracy_score(y_true,y_pred):.2%}")
m2.metric("Precision",f"{precision_score(y_true,y_pred,zero_division=0):.2%}")
m3.metric("Recall",f"{recall_score(y_true,y_pred,zero_division=0):.2%}")
m4.metric("F1-score",f"{f1_score(y_true,y_pred,zero_division=0):.2%}")

# -------- trend + distribution + top alerts
st.subheader("**Daily Trend**"); df["day"]=pd.to_datetime(df.ts).dt.date
tr=df.groupby("day").agg(total=("order_id","count"),alerts=("alert","sum")).reset_index()
if not tr.empty:
    tl=tr.melt("day",["total","alerts"],"Series","value").replace({"total":"Total transactions","alerts":"Fraud alerts (≥0.30)"})
    st.altair_chart(alt.Chart(tl).mark_line(point=True).encode(x="day:T",y="value:Q",color="Series:N"),use_container_width=True)
st.subheader("**Fraud Score Distribution**"); st.altair_chart(alt.Chart(df).mark_bar().encode(x=alt.X("score:Q",bin=alt.Bin(maxbins=50)),y="count()").properties(height=200)+alt.Chart(pd.DataFrame({"x":[TH]})).mark_rule(color="red").encode(x="x"),use_container_width=True)
st.subheader("**Top Alerts (score ≥ 0.30)**")
cols=[c for c in ["order_id","ts","customer_id","store_id","sku_id","sku_category","amt","q","pay","ship","ip","score"] if c in df]
al=df[df.alert==1].sort_values(["score","ts"],ascending=[False,False]).drop_duplicates("order_id")
st.dataframe(al.loc[:,cols].head(50),use_container_width=True,height=320)
