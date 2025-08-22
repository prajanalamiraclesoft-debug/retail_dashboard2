# app.py — streamlit run app.py
# Accuracy-first version: fixed 0.30, monthly top-k% mapping, strict 5-of-7 gate, force-alerts.

import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import date
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config("Fraud Dashboard — Accuracy First (0.30, top-k% monthly)", layout="wide")
alt.renderers.set_embed_options(actions=False)
RNG = 42
PROD_T = 0.30              # business decision threshold (fixed)
TARGET_RATE = 0.05         # top-k% monthly alerts (5% default). Lower -> higher accuracy.
RISK_GATE_MIN = 5          # require ≥5 of 7 risk signals

# ───────── Sidebar ─────────
with st.sidebar:
    st.header("BigQuery")
    PROJ = st.text_input("Project", "mss-data-engineer-sandbox")
    DATASET = st.text_input("Dataset", "retail")
    RAW = st.text_input("Raw table", f"{PROJ}.{DATASET}.transaction_data")
    S = st.date_input("Start", date(2023,1,1))
    E = st.date_input("End",  date(2030,12,31))
    TH = st.slider("Decision threshold (what-if view)", 0.01, 0.99, PROD_T, 0.01)
    RATE = st.slider("Target monthly alert rate (top-k%)", 0.01, 0.20, TARGET_RATE, 0.01)
    st.caption("Production stays at 0.30; we remap scores so 0.30 flags only the top-k% each month.")

# ───────── BigQuery ─────────
sa = dict(st.secrets["gcp_service_account"]); sa["private_key"]=sa["private_key"].replace("\\n","\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq = bigquery.Client(credentials=creds, project=creds.project_id)

@st.cache_data(show_spinner=True)
def load_raw(raw, s, e):
    sql = f"""
      SELECT order_id, TIMESTAMP(timestamp) ts, customer_id, store_id, sku_id, sku_category,
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

# ───────── Cleaning ─────────
df = df.sort_values("ts").drop_duplicates("order_id", keep="last")
df = df[(df.q>0) & (df.p>0)].copy()
df["ts"]   = pd.to_datetime(df["ts"],   errors="coerce", utc=True).dt.tz_localize(None)
df["acct"] = pd.to_datetime(df["acct"], errors="coerce", utc=True).dt.tz_localize(None)
df["age"]  = (df["ts"] - df["acct"]).dt.days.astype(float).fillna(0).clip(lower=0)
df["amt"]  = df.q * df.p
def wins(g,c): lo,hi=g[c].quantile(.01),g[c].quantile(.99); g[c]=g[c].clip(lo,hi); return g
for c in ["p","q"]: df = df.groupby("sku_category", group_keys=False).apply(wins, c)
df["amt"] = df.q * df.p

# ───────── Features ─────────
cat_avg = df.groupby("sku_category")["p"].transform("mean").replace(0,np.nan)
df["price_ratio"] = (df["p"]/cat_avg).fillna(1.0)
den = df["amt"].replace(0,np.nan)
df["coup_pct"] = (df["coup"]/den).fillna(0)
df["gift_pct"] = (df["gift_amt"]/den).fillna(0)
df["geo"] = (df["ship"] != df["ip"]).astype(int)
df["hour"] = pd.to_datetime(df["ts"]).dt.hour
df["dow"]  = pd.to_datetime(df["ts"]).dt.dayofweek
PAY_RISK = {"crypto":3,"paypal":2,"credit_card":2,"apple_pay":2,"google_pay":2,"debit_card":1,"bank_transfer":0,"cod":0}
df["pay_risk"] = df["pay"].map(PAY_RISK).fillna(1).astype(int)

def vel(g,h):
    t=pd.to_datetime(g["ts"]).astype("int64")//1_000_000
    o=np.argsort(t); t=t.iloc[o].to_numpy(); w=int(h*3600*1e6)
    L=np.searchsorted(t, t-w, side="left")
    return pd.Series(np.arange(1,len(t)+1)-L, index=g.index[o]).reindex(g.index)
df=df.sort_values("ts")
df["cust_1h"]=df.groupby("customer_id",group_keys=False).apply(vel,1)
df["cust_24h"]=df.groupby("customer_id",group_keys=False).apply(vel,24)
df["dev_1h"]=df.groupby("dev",group_keys=False).apply(vel,1)

def roll(g):
    g=g.sort_values("ts"); r=g["amt"].rolling(10,min_periods=1)
    g["c_m"]=r.mean(); g["c_s"]=r.std(ddof=0).replace(0,np.nan); return g
df=df.groupby("customer_id",group_keys=False).apply(roll)
df["z"]=((df["amt"]-df["c_m"])/df["c_s"]).replace([np.inf,-np.inf],0).fillna(0)
p90=float(np.nanpercentile(df["amt"],90)) if len(df) else 0

# strong shapes
df["s_price_bulk"]=((df["price_ratio"].sub(1).abs()>=.50)&(df["q"]>=3)).astype(int)
df["s_gc_geo"]=((df["gift_used"].fillna(False))&(df["geo"]==1)).astype(int)
df["s_burst"]=((df["cust_1h"]>=3)|(df["dev_1h"]>=3)).astype(int)
df["s_z"]=(df["z"].abs()>=2).astype(int)
df["s_geo_hi"]=((df["geo"]==1)&(df["amt"]>=p90)).astype(int)
df["s_any"]=(df[["s_price_bulk","s_gc_geo","s_burst","s_z","s_geo_hi"]].sum(axis=1)>0).astype(int)

# model data
labeled=df[df.y.isin([0,1])].copy()
if labeled["y"].nunique()<2: st.error("Labeled data must include both 0/1."); st.stop()
cat_cols=["sku_category","pay","ship","ip","dev","store_id","hour","dow"]
num_cols=["amt","q","p","age","price_ratio","coup_pct","gift_pct","pay_risk",
          "cust_1h","cust_24h","dev_1h","c_m","c_s","z",
          "geo","s_price_bulk","s_gc_geo","s_burst","s_z","s_geo_hi","s_any"]
Xn=labeled[num_cols].fillna(0)
Xc=pd.get_dummies(labeled[cat_cols].astype(str), dummy_na=False)
X=pd.concat([Xn,Xc],axis=1); y=labeled["y"].astype(int).values
cat_dummies=list(Xc.columns)

# train once (class-weighted a bit toward fraud)
pos=max(1,int((y==1).sum())); neg=max(1,len(y)-pos)
w_pos=min(20.0, neg/pos)
sw=np.where(y==1, w_pos, 1.0)

@st.cache_resource(show_spinner=False)
def fit_model(X, y, sw):
    m=HistGradientBoostingClassifier(max_iter=500, learning_rate=0.07, early_stopping=True, random_state=RNG)
    m.fit(X,y,sample_weight=sw); return m
clf = fit_model(X,y,sw)

# out-of-fold probs to set MONTHLY TOP-K% mapping
@st.cache_data(show_spinner=False)
def oof_probs_by_month(X, y):
    months = pd.to_datetime(labeled["ts"]).dt.to_period("M").astype(str).values
    skf=StratifiedKFold(n_splits=3, shuffle=True, random_state=RNG)
    p=np.zeros(len(y))
    for tr,va in skf.split(X,y):
        mm=HistGradientBoostingClassifier(max_iter=500, learning_rate=0.07, early_stopping=True, random_state=RNG)
        mm.fit(X.iloc[tr], y[tr], sample_weight=sw[tr])
        p[va]=mm.predict_proba(X.iloc[va])[:,1]
    return pd.DataFrame({"month":months,"prob":p,"y":y})
oof=oof_probs_by_month(X,y)

# compute per-month scale so that 0.30 == top-k% cut
scales={}
for m,grp in oof.groupby("month"):
    if len(grp)<50:
        scales[m]=PROD_T/np.clip(np.quantile(oof["prob"], 1-RATE), 1e-6, 1)  # global fallback
    else:
        t_q=np.quantile(grp["prob"], 1-RATE)  # top-k% cut
        scales[m]=PROD_T/np.clip(t_q, 1e-6, 1)

def score_all(D: pd.DataFrame) -> np.ndarray:
    Xn_=D[num_cols].fillna(0)
    Xc_=pd.get_dummies(D[cat_cols].astype(str), dummy_na=False).reindex(columns=cat_dummies, fill_value=0)
    base=clf.predict_proba(pd.concat([Xn_,Xc_],axis=1))[:,1]
    months=pd.to_datetime(D["ts"]).dt.to_period("M").astype(str)
    return np.clip(np.array([base[i]*scales.get(m,1.0) for i,m in enumerate(months)]),0,1)

df["score"]=score_all(df)

# decisions: strict gate + force-alerts
risk_bits=pd.DataFrame({
    "geo":(df["geo"]==1).astype(int),
    "vel":((df["cust_24h"]>=2)|(df["dev_1h"]>=2)).astype(int),
    "price":(df["price_ratio"].sub(1).abs()>=0.30).astype(int),
    "promo":(df["coup_pct"]>=0.25).astype(int),
    "gift":(df["gift_pct"]>=0.45).astype(int),
    "hiamt":(df["amt"]>=p90).astype(int),
    "pay":(df["pay_risk"]>=2).astype(int),
})
gate = (risk_bits.sum(axis=1) >= RISK_GATE_MIN)
force = ( (df["s_any"]==1) | ((df["geo"]==1)&((df["dev_1h"]>=2)|(df["cust_24h"]>=2))) | (df["gift_pct"]>=0.60) )
score_pass = (df["score"]>=TH)

df["alert"] = np.where(force, 1, (score_pass & gate).astype(int))

# ───────── KPIs ─────────
st.subheader(f"**Key Metrics (Threshold = {TH:.2f} | Prod = 0.30, top-k%={RATE:.0%})**")
TOT, AL = len(df), int(df["alert"].sum())
c1,c2,c3=st.columns(3)
c1.metric("**Total transactions**",TOT)
c2.metric(f"**Fraud alerts (≥{TH:.2f})**",AL)
c3.metric("**Alert rate**",f"{AL/max(1,TOT):.2%}")
st.caption("We remap scores so 0.30 flags only the top-k% each month, then require ≥5/7 risk signals; "
           "force-alerts preserve obvious fraud. This maximizes overall accuracy quickly.")

# trend
st.subheader("**Daily Trend**")
df["day"]=pd.to_datetime(df["ts"]).dt.date
trend=df.groupby("day").agg(total=("order_id","count"), alerts=("alert","sum")).reset_index()
if not trend.empty:
    tl=trend.melt("day",["total","alerts"],"Series","value").replace({"total":"Total transactions","alerts":f"Fraud alerts (≥{TH:.2f})"})
    st.altair_chart(alt.Chart(tl).mark_line(point=True).encode(x="day:T",y="value:Q",color="Series:N"), use_container_width=True)

# distribution
st.subheader("**Fraud Score Distribution**")
st.altair_chart(
    alt.Chart(df).mark_bar().encode(x=alt.X("score:Q",bin=alt.Bin(maxbins=50)), y="count()").properties(height=200)
    + alt.Chart(pd.DataFrame({"x":[TH]})).mark_rule(color="red").encode(x="x"),
    use_container_width=True
)

# top alerts (ALL)
st.subheader(f"**Top Alerts (score ≥ {TH:.2f})**")
cols=[c for c in ["order_id","ts","customer_id","store_id","sku_id","sku_category","amt","q","pay","ship","ip","score"] if c in df]
top=df[df["alert"]==1].sort_values(["score","ts"],ascending=[False,False]).drop_duplicates("order_id")
st.caption(f"Showing **all {len(top)}** alerts.")
st.dataframe(top.loc[:,cols], use_container_width=True, height=min(700, 35*max(5,len(top))))

# evaluation (on labeled rows)
st.subheader(f"**Model Evaluation (threshold = {TH:.2f})**")
st.caption("Precision = among alerts, % truly fraud • Recall = of all fraud, % caught • F1 = balance • Accuracy = overall correctness.")
y_true=labeled["y"].astype(int).values
y_pred=df.loc[labeled.index,"alert"].astype(int).values
m1,m2,m3,m4=st.columns(4)
m1.metric("Accuracy",  f"{accuracy_score(y_true,y_pred):.2%}")
m2.metric("Precision", f"{precision_score(y_true,y_pred,zero_division=0):.2%}")
m3.metric("Recall",    f"{recall_score(y_true,y_pred,zero_division=0):.2%}")
m4.metric("F1-score",  f"{f1_score(y_true,y_pred,zero_division=0):.2%}")
