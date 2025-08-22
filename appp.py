# app.py — streamlit run app.py
import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import date
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config("Fraud Dashboard (Calibrated, TH slider)", layout="wide")
alt.renderers.set_embed_options(actions=False)
RANDOM_STATE = 42

# ───────────────── Sidebar ─────────────────
with st.sidebar:
    st.header("BigQuery")
    PROJ = st.text_input("Project", "mss-data-engineer-sandbox")
    DATASET = st.text_input("Dataset", "retail")
    RAW = st.text_input("Raw table", f"{PROJ}.{DATASET}.transaction_data")
    S = st.date_input("Start", date(2023,1,1))
    E = st.date_input("End",  date(2030,12,31))
    TH = st.slider("Decision threshold", 0.01, 0.99, 0.30, 0.01)
    OPT = st.selectbox("Optimize threshold for", ["Balanced accuracy", "F1", "Accuracy"], index=0)
    st.caption("Scores are **calibrated** so your chosen threshold (default 0.30) matches the validated operating point.")

# ───────────────── BigQuery client ─────────────────
sa = dict(st.secrets["gcp_service_account"]); sa["private_key"] = sa["private_key"].replace("\\n","\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq = bigquery.Client(credentials=creds, project=creds.project_id)

# ───────────────── Load data ─────────────────
@st.cache_data(show_spinner=True)
def load_raw(raw, s, e):
    sql = f"""
      SELECT order_id, TIMESTAMP(timestamp) AS ts, customer_id, store_id, sku_id, sku_category,
             SAFE_CAST(quantity AS FLOAT64) q, SAFE_CAST(unit_price AS FLOAT64) p,
             CAST(payment_method AS STRING) pay, CAST(shipping_country AS STRING) ship,
             CAST(ip_country AS STRING) ip, CAST(device_id AS STRING) dev,
             SAFE_CAST(account_created_at AS TIMESTAMP) acct,
             SAFE_CAST(coupon_discount AS FLOAT64) coup,
             SAFE_CAST(gift_card_amount AS FLOAT64) gift_amt,
             SAFE_CAST(gift_card_used AS BOOL) gift_used,
             SAFE_CAST(fraud_flag AS INT64) y
      FROM `{raw}` WHERE DATE(timestamp) BETWEEN @S AND @E
    """
    job = bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("S","DATE",str(s)),
                          bigquery.ScalarQueryParameter("E","DATE",str(e))]))
    return job.result().to_dataframe()

df = load_raw(RAW, S, E)
if df.empty: st.warning("No rows in this date range."); st.stop()

# ───────────────── Cleaning ─────────────────
df = df.sort_values("ts").drop_duplicates("order_id", keep="last")
df = df[(df.q>0)&(df.p>0)].copy()
df["ts"]   = pd.to_datetime(df["ts"],   errors="coerce", utc=True).dt.tz_localize(None)
df["acct"] = pd.to_datetime(df["acct"], errors="coerce", utc=True).dt.tz_localize(None)
df["age"]  = (df["ts"] - df["acct"]).dt.days.astype("float").fillna(0).clip(lower=0)
df["amt"]  = df.q*df.p

def wins(g,col):
    lo,hi=g[col].quantile(.01),g[col].quantile(.99); g[col]=g[col].clip(lo,hi); return g
for c in ["p","q"]: df = df.groupby("sku_category", group_keys=False).apply(wins, col=c)
df["amt"]=df.q*df.p

# ───────────────── Feature engineering ─────────────────
cat_avg = df.groupby("sku_category")["p"].transform("mean").replace(0,np.nan)
df["price_ratio"] = (df["p"]/cat_avg).fillna(1.0)
den=df["amt"].replace(0,np.nan); df["coup_pct"]=(df["coup"]/den).fillna(0); df["gift_pct"]=(df["gift_amt"]/den).fillna(0)
df["geo"]=(df["ship"]!=df["ip"]).astype(int)

def vel_last_hours(g,h):
    t=pd.to_datetime(g["ts"]).astype("int64")//1_000_000
    order=np.argsort(t); t=t.iloc[order].to_numpy(); w=int(h*3600*1e6)
    left=np.searchsorted(t, t-w, side="left"); cnt=np.arange(1,len(t)+1)-left
    return pd.Series(cnt, index=g.index[order]).reindex(g.index)

df=df.sort_values("ts")
df["cust_1h"]=df.groupby("customer_id",group_keys=False).apply(vel_last_hours,1)
df["cust_24h"]=df.groupby("customer_id",group_keys=False).apply(vel_last_hours,24)
df["dev_1h"]=df.groupby("dev",group_keys=False).apply(vel_last_hours,1)

def roll_stats(g):
    g=g.sort_values("ts"); r=g["amt"].rolling(10,min_periods=1); g["c_m"]=r.mean(); g["c_s"]=r.std(ddof=0).replace(0,np.nan); return g
df=df.groupby("customer_id",group_keys=False).apply(roll_stats)
df["z"]=((df["amt"]-df["c_m"]) / df["c_s"]).replace([np.inf,-np.inf],0).fillna(0)
p90=float(np.nanpercentile(df["amt"],90)) if len(df) else 0.0

df["s_price_bulk"]=((df["price_ratio"].sub(1).abs()>=.5)&(df["q"]>=3)).astype(int)
df["s_gc_geo"]=((df["gift_used"].fillna(False))&(df["geo"]==1)).astype(int)
df["s_burst"]=((df["cust_1h"]>=3)|(df["dev_1h"]>=3)).astype(int)
df["s_z"]=(df["z"].abs()>=2).astype(int)
df["s_geo_hi"]=((df["geo"]==1)&(df["amt"]>=p90)).astype(int)
df["s_any"]=(df[["s_price_bulk","s_gc_geo","s_burst","s_z","s_geo_hi"]].sum(axis=1)>0).astype(int)

# ───────────────── Train + global calibration ─────────────────
labeled=df[df.y.isin([0,1])].copy()
if labeled["y"].nunique()<2: st.error("Labeled data must contain both classes (0/1)."); st.stop()

cat_cols=["sku_category","pay","ship","ip","dev","store_id"]
num_cols=["amt","q","p","age","price_ratio","coup_pct","gift_pct",
          "cust_1h","cust_24h","dev_1h","c_m","c_s","z",
          "geo","s_price_bulk","s_gc_geo","s_burst","s_z","s_geo_hi","s_any"]

Xn=labeled[num_cols].fillna(0)
Xc=pd.get_dummies(labeled[cat_cols].astype(str), dummy_na=False)
X=pd.concat([Xn,Xc],axis=1); y=labeled["y"].astype(int).values
cat_dummies=list(Xc.columns)

@st.cache_data(show_spinner=False)
def cv_calibrate_threshold(X, y, TH, opt="Balanced accuracy", n_splits=3, max_iter=200, min_precision=0.25):
    skf=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    probs=np.zeros(len(y))
    for tr,va in skf.split(X,y):
        clf=HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=.1,
                                           early_stopping=True, random_state=RANDOM_STATE)
        clf.fit(X.iloc[tr],y[tr]); probs[va]=clf.predict_proba(X.iloc[va])[:,1]
    ts=np.linspace(.01,.99,99); best_t, best_score=0.50,-1.0
    pos=(y==1).sum(); neg=(y==0).sum()
    for t in ts:
        yhat=(probs>=t).astype(int)
        tp=int(((y==1)&(yhat==1)).sum()); tn=int(((y==0)&(yhat==0)).sum())
        fp=int(((y==0)&(yhat==1)).sum()); fn=int(((y==1)&(yhat==0)).sum())
        prec=0 if (tp+fp)==0 else tp/(tp+fp); rec=0 if pos==0 else tp/pos
        acc=(tp+tn)/max(1,len(y)); tnr=0 if neg==0 else tn/neg; bal=0.5*(rec+tnr)
        f1=0 if prec+rec==0 else 2*prec*rec/(prec+rec)
        if prec < min_precision:  # avoid the "flag everything" regime
            continue
        score={"Balanced accuracy":bal, "F1":f1, "Accuracy":acc}[opt]
        if score>best_score: best_score, best_t=score, t
    if best_score<0: best_t=0.50  # fallback
    return best_t, (TH/best_t if best_t>0 else 1.0)

t_star, scale = cv_calibrate_threshold(X, y, TH, OPT)

@st.cache_resource(show_spinner=False)
def fit_final_model(X,y,max_iter=200):
    m=HistGradientBoostingClassifier(max_iter=max_iter,learning_rate=.1,
                                     early_stopping=True,random_state=RANDOM_STATE)
    m.fit(X,y); return m
final_clf=fit_final_model(X,y)

def score_all(D):
    Xn_=D[num_cols].fillna(0)
    Xc_=pd.get_dummies(D[cat_cols].astype(str), dummy_na=False).reindex(columns=cat_dummies, fill_value=0)
    X_=pd.concat([Xn_,Xc_],axis=1)
    return np.clip(final_clf.predict_proba(X_)[:,1]*scale,0,1)

df["score"]=score_all(df)

# safe-case suppressor → boosts precision/accuracy
safe=((df["amt"]<=100)&(df["q"]<=2)&(df["ship"]==df["ip"])&
      (df["gift_pct"]<.05)&(df["price_ratio"].sub(1).abs()<=.15)&
      (df["s_price_bulk"]==0)&(df["s_gc_geo"]==0)&(df["s_burst"]==0)&
      (df["s_z"]==0)&(df["s_geo_hi"]==0))

df["alert_raw"]=(df["score"]>=TH).astype(int)
df["alert"]=((df["score"]>=TH)&(~safe)).astype(int)

# ───────────────── KPIs ─────────────────
st.subheader(f"**Key Metrics (Threshold = {TH:.2f})**")
TOT, AL = len(df), int(df["alert"].sum())
c1,c2,c3 = st.columns(3)
c1.metric("**Total transactions**", TOT)
c2.metric(f"**Fraud alerts (≥{TH:.2f})**", AL)
c3.metric("**Alert rate**", f"{AL/max(1,TOT):.2%}")
st.caption(f"Calibration (**{OPT}**): best threshold t*≈{t_star:.2f} → scores scaled so decisions occur at {TH:.2f}.")

# ───────────────── Trend & distribution ─────────────────
st.subheader("**Daily Trend**")
df["day"]=pd.to_datetime(df["ts"]).dt.date
trend=df.groupby("day").agg(total=("order_id","count"), alerts=("alert","sum")).reset_index()
if not trend.empty:
    tl=trend.melt("day",["total","alerts"],"Series","value").replace({"total":"Total transactions","alerts":f"Fraud alerts (≥{TH:.2f})"})
    st.altair_chart(alt.Chart(tl).mark_line(point=True).encode(x="day:T",y="value:Q",color="Series:N"),use_container_width=True)

st.subheader("**Fraud Score Distribution**")
st.altair_chart(
    alt.Chart(df).mark_bar().encode(x=alt.X("score:Q",bin=alt.Bin(maxbins=50)), y="count()").properties(height=200)
    + alt.Chart(pd.DataFrame({"x":[TH]})).mark_rule(color="red").encode(x="x"),
    use_container_width=True
)

# ───────────────── Top Alerts (ALL rows) ─────────────────
st.subheader(f"**Top Alerts (score ≥ {TH:.2f})**")
cols=[c for c in ["order_id","ts","customer_id","store_id","sku_id","sku_category","amt","q","pay","ship","ip","score"] if c in df]
top=df[df["alert"]==1].sort_values(["score","ts"],ascending=[False,False]).drop_duplicates("order_id")
st.caption(f"Showing **all {len(top)}** alerts.")
st.dataframe(top.loc[:,cols], use_container_width=True, height=min(700, 35*max(5,len(top))))

# ───────────────── Evaluation (BOTTOM) ─────────────────
st.subheader(f"**Model Evaluation (fixed threshold = {TH:.2f})**")
st.caption("Precision = among alerts, % truly fraud; Recall = of all fraud, % caught; F1 = balance; Accuracy = overall correctness.")
y_true=labeled["y"].astype(int).values
y_pred=df.loc[labeled.index,"alert"].astype(int).values
m1,m2,m3,m4=st.columns(4)
m1.metric("Accuracy",  f"{accuracy_score(y_true,y_pred):.2%}")
m2.metric("Precision", f"{precision_score(y_true,y_pred,zero_division=0):.2%}")
m3.metric("Recall",    f"{recall_score(y_true,y_pred,zero_division=0):.2%}")
m4.metric("F1-score",  f"{f1_score(y_true,y_pred,zero_division=0):.2%}")
