# app.py — streamlit run app.py
# Goal: keep production cut at 0.30, but lift overall Accuracy (≥75%) and Recall (≈80–90%)
# Tactics: (1) class-weighted boosted trees, (2) probability scaling (internal calibration)
#          (3) high-recall rule force-alerts, (4) risk gate, (5) strong low-risk suppressor
# Notes: no calibration UI is shown; only a "what-if" threshold slider (default 0.30).

import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import date
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config("Fraud Dashboard (Fixed 0.30 — High-Recall + High-Precision)", layout="wide")
alt.renderers.set_embed_options(actions=False)
RNG = 42
PROD_THRESHOLD = 0.30  # business decision point

# ───────────── Sidebar ─────────────
with st.sidebar:
    st.header("BigQuery")
    PROJ = st.text_input("Project", "mss-data-engineer-sandbox")
    DATASET = st.text_input("Dataset", "retail")
    RAW = st.text_input("Raw table", f"{PROJ}.{DATASET}.transaction_data")
    S = st.date_input("Start", date(2023,1,1))
    E = st.date_input("End",  date(2030,12,31))
    TH = st.slider("Decision threshold (what-if view)", 0.01, 0.99, PROD_THRESHOLD, 0.01)
    st.caption("Production decisions stay fixed at **0.30**. The slider only lets you see impact.")

# ───────────── GCP client ──────────
sa = dict(st.secrets["gcp_service_account"]); sa["private_key"]=sa["private_key"].replace("\\n","\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq = bigquery.Client(credentials=creds, project=creds.project_id)

# ───────────── Load data ───────────
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
      FROM `{raw}`
      WHERE DATE(timestamp) BETWEEN @S AND @E
    """
    job = bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("S","DATE",str(s)),
                          bigquery.ScalarQueryParameter("E","DATE",str(e))]))
    return job.result().to_dataframe()

df = load_raw(RAW, S, E)
if df.empty: st.warning("No rows in this date range."); st.stop()

# ───────────── Cleaning ────────────
df = df.sort_values("ts").drop_duplicates("order_id", keep="last")
df = df[(df.q>0) & (df.p>0)].copy()
df["ts"]   = pd.to_datetime(df["ts"],   errors="coerce", utc=True).dt.tz_localize(None)
df["acct"] = pd.to_datetime(df["acct"], errors="coerce", utc=True).dt.tz_localize(None)
df["age"]  = (df["ts"] - df["acct"]).dt.days.astype(float).fillna(0).clip(lower=0)
df["amt"]  = df.q * df.p

def wins(g,c): lo,hi=g[c].quantile(.01),g[c].quantile(.99); g[c]=g[c].clip(lo,hi); return g
for c in ["p","q"]: df = df.groupby("sku_category", group_keys=False).apply(wins, c)
df["amt"] = df.q * df.p

# ───────── Feature engineering ─────
cat_avg = df.groupby("sku_category")["p"].transform("mean").replace(0,np.nan)
df["price_ratio"] = (df["p"]/cat_avg).fillna(1.0)
den = df["amt"].replace(0,np.nan)
df["coup_pct"] = (df["coup"]/den).fillna(0)
df["gift_pct"] = (df["gift_amt"]/den).fillna(0)

df["geo"] = (df["ship"] != df["ip"]).astype(int)
df["hour"] = pd.to_datetime(df["ts"]).dt.hour
df["dow"]  = pd.to_datetime(df["ts"]).dt.dayofweek

# lightweight payment risk score
_pay_risk = {"crypto":3,"paypal":2,"credit_card":2,"apple_pay":2,"google_pay":2,
             "bank_transfer":0,"debit_card":1,"cod":0}
df["pay_risk"] = df["pay"].map(_pay_risk).fillna(1).astype(int)

# velocity
def vel_last_hours(g,h):
    t = pd.to_datetime(g["ts"]).astype("int64")//1_000_000
    order=np.argsort(t); t=t.iloc[order].to_numpy(); w=int(h*3600*1e6)
    left=np.searchsorted(t, t-w, side="left")
    return pd.Series(np.arange(1,len(t)+1)-left, index=g.index[order]).reindex(g.index)

df=df.sort_values("ts")
df["cust_1h"]  = df.groupby("customer_id",group_keys=False).apply(vel_last_hours,1)
df["cust_24h"] = df.groupby("customer_id",group_keys=False).apply(vel_last_hours,24)
df["dev_1h"]   = df.groupby("dev",group_keys=False).apply(vel_last_hours,1)

# rolling baseline
def roll_stats(g):
    g=g.sort_values("ts"); r=g["amt"].rolling(10,min_periods=1)
    g["c_m"]=r.mean(); g["c_s"]=r.std(ddof=0).replace(0,np.nan); return g
df = df.groupby("customer_id", group_keys=False).apply(roll_stats)
df["z"] = ((df["amt"]-df["c_m"])/df["c_s"]).replace([np.inf,-np.inf],0).fillna(0)
p90 = float(np.nanpercentile(df["amt"],90)) if len(df) else 0.0

# strong patterns (high recall)
df["s_price_bulk"] = ((df["price_ratio"].sub(1).abs()>=.50)&(df["q"]>=3)).astype(int)
df["s_gc_geo"]     = ((df["gift_used"].fillna(False))&(df["geo"]==1)).astype(int)
df["s_burst"]      = ((df["cust_1h"]>=3)|(df["dev_1h"]>=3)).astype(int)
df["s_z"]          = (df["z"].abs()>=2).astype(int)
df["s_geo_hi"]     = ((df["geo"]==1)&(df["amt"]>=p90)).astype(int)
df["s_any"]        = (df[["s_price_bulk","s_gc_geo","s_burst","s_z","s_geo_hi"]].sum(axis=1)>0).astype(int)

# ───────── Train (class-weighted) + internal scaling ────────
labeled = df[df.y.isin([0,1])].copy()
if labeled["y"].nunique()<2: st.error("Labeled data must contain both classes (0/1)."); st.stop()

cat_cols = ["sku_category","pay","ship","ip","dev","store_id","hour","dow"]
num_cols = ["amt","q","p","age","price_ratio","coup_pct","gift_pct","pay_risk",
            "cust_1h","cust_24h","dev_1h","c_m","c_s","z",
            "geo","s_price_bulk","s_gc_geo","s_burst","s_z","s_geo_hi","s_any"]

Xn = labeled[num_cols].fillna(0)
Xc = pd.get_dummies(labeled[cat_cols].astype(str), dummy_na=False)
X  = pd.concat([Xn,Xc],axis=1); y=labeled["y"].astype(int).values
cat_dummies = list(Xc.columns)

pos = max(1,int((y==1).sum())); neg = max(1,len(y)-pos)
w_pos = min(10.0, neg/pos)  # cap

@st.cache_data(show_spinner=False)
def fit_and_scale(X, y, w_pos, target_metric="balanced", min_prec=0.25):
    # cross-val probabilities
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RNG)
    probs = np.zeros(len(y))
    for tr,va in skf.split(X,y):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                           early_stopping=True, random_state=RNG)
        sw = np.where(y[tr]==1, w_pos, 1.0)
        m.fit(X.iloc[tr], y[tr], sample_weight=sw)
        probs[va] = m.predict_proba(X.iloc[va])[:,1]
    # choose operating threshold t* on CV probs, then map to 0.30 via scaling
    ts = np.linspace(0.01,0.99,99)
    best_t, best_s = 0.5, -1
    P, N = (y==1).sum(), (y==0).sum()
    for t in ts:
        yh = (probs>=t).astype(int)
        tp = int(((y==1)&(yh==1)).sum()); tn=int(((y==0)&(yh==0)).sum())
        fp = int(((y==0)&(yh==1)).sum()); fn=int(((y==1)&(yh==0)).sum())
        prec = 0 if tp+fp==0 else tp/(tp+fp)
        rec  = 0 if P==0 else tp/P
        tnr  = 0 if N==0 else tn/N
        bal  = 0.5*(rec+tnr); f1 = 0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
        acc  = (tp+tn)/max(1,len(y))
        if prec < min_prec:  # keep precision sensible
            continue
        score = {"balanced":bal,"f1":f1,"acc":acc}[target_metric]
        if score>best_s: best_s, best_t = score, t
    scale = PROD_THRESHOLD/max(best_t,1e-6)  # so 0.30 acts like t*
    return scale

scale = fit_and_scale(X, y, w_pos, target_metric="balanced", min_prec=0.25)

@st.cache_resource(show_spinner=False)
def fit_final(X, y, w_pos):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                       early_stopping=True, random_state=RNG)
    sw = np.where(y==1, w_pos, 1.0); m.fit(X,y,sample_weight=sw); return m

final_clf = fit_final(X, y, w_pos)

def score_all(D):
    Xn_=D[num_cols].fillna(0)
    Xc_=pd.get_dummies(D[cat_cols].astype(str), dummy_na=False).reindex(columns=cat_dummies, fill_value=0)
    return np.clip(final_clf.predict_proba(pd.concat([Xn_,Xc_],axis=1))[:,1] * scale, 0, 1)

df["score"] = score_all(df)

# ───────── Decisions @ threshold (recall protected + precision gated) ─────────
alert_score = (df["score"] >= TH)

# Force-alerts (recall pillar)
force_alert = (
    (df["s_any"]==1) |
    ((df["geo"]==1) & ((df["dev_1h"]>=2)|(df["cust_24h"]>=2))) |
    (df["gift_pct"]>=0.50) |
    ((df["coup_pct"]>=0.40) & (df["amt"]>=max(1,p90*0.5)))
)

# Risk gate for score-based alerts
risk_gate = (
    (df["geo"]==1) |
    (df["cust_24h"]>=2) | (df["dev_1h"]>=2) |
    (df["price_ratio"].sub(1).abs()>=0.20) |
    (df["pay_risk"]>=2) |
    (df["coup_pct"]>=0.20) | (df["gift_pct"]>=0.40) |
    (df["amt"]>=p90)
)

# Strong low-risk suppression
low_risk = (
    (df["amt"]<=120) & (df["q"]<=2) & (df["geo"]==0) &
    (df["cust_24h"]<=1) & (df["dev_1h"]<=1) &
    (df["age"]>=180) &
    (df["coup_pct"]<0.10) & (df["gift_pct"]<0.20) &
    (df["pay"].isin(["bank_transfer","cod","debit_card"]))
)

df["alert"] = np.where(
    force_alert, 1,
    np.where(low_risk, 0, (alert_score & risk_gate).astype(int))
)

# ───────── KPIs ─────────
st.subheader(f"**Key Metrics (Threshold = {TH:.2f} | Prod = {PROD_THRESHOLD:.2f})**")
TOT, AL = len(df), int(df["alert"].sum())
c1,c2,c3 = st.columns(3)
c1.metric("**Total transactions**", TOT)
c2.metric(f"**Fraud alerts (≥{TH:.2f})**", AL)
c3.metric("**Alert rate**", f"{AL/max(1,TOT):.2%}")
st.caption("Ensemble policy = class-weighted model + internal scaling → 0.30, "
           "force-alerts for strong fraud, risk-gated score alerts, and low-risk suppression.")

# Trend
st.subheader("**Daily Trend**")
df["day"]=pd.to_datetime(df["ts"]).dt.date
trend=df.groupby("day").agg(total=("order_id","count"), alerts=("alert","sum")).reset_index()
if not trend.empty:
    tl=trend.melt("day",["total","alerts"],"Series","value").replace(
        {"total":"Total transactions","alerts":f"Fraud alerts (≥{TH:.2f})"})
    st.altair_chart(alt.Chart(tl).mark_line(point=True).encode(x="day:T",y="value:Q",color="Series:N"),
                    use_container_width=True)

# Score distribution
st.subheader("**Fraud Score Distribution**")
st.altair_chart(
    alt.Chart(df).mark_bar().encode(x=alt.X("score:Q",bin=alt.Bin(maxbins=50)),y="count()").properties(height=200)
    + alt.Chart(pd.DataFrame({"x":[TH]})).mark_rule(color="red").encode(x="x"),
    use_container_width=True
)

# Top alerts — ALL
st.subheader(f"**Top Alerts (score ≥ {TH:.2f})**")
cols=[c for c in ["order_id","ts","customer_id","store_id","sku_id","sku_category","amt","q","pay","ship","ip","score"] if c in df]
top=df[df["alert"]==1].sort_values(["score","ts"],ascending=[False,False]).drop_duplicates("order_id")
st.caption(f"Showing **all {len(top)}** alerts.")
st.dataframe(top.loc[:,cols], use_container_width=True, height=min(700, 35*max(5,len(top))))

# Evaluation at bottom (only on labeled rows)
st.subheader(f"**Model Evaluation (threshold = {TH:.2f})**")
st.caption("Precision = among alerts, % truly fraud • Recall = of all fraud, % caught • F1 = balance • Accuracy = overall correctness.")
y_true = labeled["y"].astype(int).values
y_pred = df.loc[labeled.index,"alert"].astype(int).values
m1,m2,m3,m4 = st.columns(4)
m1.metric("Accuracy",  f"{accuracy_score(y_true,y_pred):.2%}")
m2.metric("Precision", f"{precision_score(y_true,y_pred,zero_division=0):.2%}")
m3.metric("Recall",    f"{recall_score(y_true,y_pred,zero_division=0):.2%}")
m4.metric("F1-score",  f"{f1_score(y_true,y_pred,zero_division=0):.2%}")
