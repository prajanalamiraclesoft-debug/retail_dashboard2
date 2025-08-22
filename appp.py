# app.py — streamlit run app.py
import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import date
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config("Fraud Dashboard (Fixed threshold)", layout="wide")
alt.renderers.set_embed_options(actions=False)
RNG = 42

# ───────────── Sidebar ─────────────
with st.sidebar:
    st.header("BigQuery")
    PROJ = st.text_input("Project", "mss-data-engineer-sandbox")
    DATASET = st.text_input("Dataset", "retail")
    RAW = st.text_input("Raw table", f"{PROJ}.{DATASET}.transaction_data")
    S = st.date_input("Start", date(2023,1,1))
    E = st.date_input("End",  date(2030,12,31))
    TH = st.slider("Decision threshold", 0.01, 0.99, 0.30, 0.01)
    APPLY_SAFE = st.checkbox("Apply safe-case suppressor", True)
    st.caption("Model uses a fixed decision threshold (default 0.30). No calibration applied.")

# ───────────── BigQuery client ──────────
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
      FROM `{raw}` WHERE DATE(timestamp) BETWEEN @S AND @E
    """
    job = bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("S","DATE",str(s)),
                          bigquery.ScalarQueryParameter("E","DATE",str(e))]))
    return job.result().to_dataframe()

df = load_raw(RAW, S, E)
if df.empty:
    st.warning("No rows in this date range."); st.stop()

# ───────────── Cleaning ────────────
df = df.sort_values("ts").drop_duplicates("order_id", keep="last")
df = df[(df.q>0) & (df.p>0)].copy()
df["ts"]   = pd.to_datetime(df["ts"],   errors="coerce", utc=True).dt.tz_localize(None)
df["acct"] = pd.to_datetime(df["acct"], errors="coerce", utc=True).dt.tz_localize(None)
df["age"]  = (df["ts"] - df["acct"]).dt.days.astype("float").fillna(0).clip(lower=0)
df["amt"]  = df.q * df.p

def wins(g, col):
    lo, hi = g[col].quantile(.01), g[col].quantile(.99)
    g[col] = g[col].clip(lo, hi); return g
for c in ["p","q"]:
    df = df.groupby("sku_category", group_keys=False).apply(wins, col=c)
df["amt"] = df.q * df.p

# ───────── Feature engineering ─────
cat_avg = df.groupby("sku_category")["p"].transform("mean").replace(0, np.nan)
df["price_ratio"] = (df["p"]/cat_avg).fillna(1.0)
den = df["amt"].replace(0, np.nan)
df["coup_pct"] = (df["coup"]/den).fillna(0)
df["gift_pct"] = (df["gift_amt"]/den).fillna(0)
df["geo"] = (df["ship"] != df["ip"]).astype(int)

def vel_last_hours(g, hours):
    t = pd.to_datetime(g["ts"]).astype("int64") // 1_000_000
    order = np.argsort(t); t = t.iloc[order].to_numpy(); w = int(hours*3600*1e6)
    left = np.searchsorted(t, t - w, side="left")
    counts = np.arange(1, len(t)+1) - left
    return pd.Series(counts, index=g.index[order]).reindex(g.index)

df = df.sort_values("ts")
df["cust_1h"]  = df.groupby("customer_id", group_keys=False).apply(vel_last_hours, 1)
df["cust_24h"] = df.groupby("customer_id", group_keys=False).apply(vel_last_hours, 24)
df["dev_1h"]   = df.groupby("dev",         group_keys=False).apply(vel_last_hours, 1)

def roll_stats(g):
    g = g.sort_values("ts"); r = g["amt"].rolling(10, min_periods=1)
    g["c_m"] = r.mean(); g["c_s"] = r.std(ddof=0).replace(0, np.nan); return g
df = df.groupby("customer_id", group_keys=False).apply(roll_stats)
df["z"] = ((df["amt"] - df["c_m"]) / df["c_s"]).replace([np.inf, -np.inf], 0).fillna(0)
p90 = float(np.nanpercentile(df["amt"], 90)) if len(df) else 0.0

# strong signals (used for recall boost)
df["s_price_bulk"] = ((df["price_ratio"].sub(1).abs() >= .50) & (df["q"] >= 3)).astype(int)
df["s_gc_geo"]     = ((df["gift_used"].fillna(False)) & (df["geo"]==1)).astype(int)
df["s_burst"]      = ((df["cust_1h"]>=3) | (df["dev_1h"]>=3)).astype(int)
df["s_z"]          = (df["z"].abs() >= 2).astype(int)
df["s_geo_hi"]     = ((df["geo"]==1) & (df["amt"] >= p90)).astype(int)
df["s_any"]        = (df[["s_price_bulk","s_gc_geo","s_burst","s_z","s_geo_hi"]].sum(axis=1) > 0).astype(int)

# ───────── Train (weighted for recall) ───────
labeled = df[df.y.isin([0,1])].copy()
if labeled["y"].nunique() < 2:
    st.error("Labeled data must contain both classes (0/1)."); st.stop()

cat_cols = ["sku_category","pay","ship","ip","dev","store_id"]
num_cols = ["amt","q","p","age","price_ratio","coup_pct","gift_pct",
            "cust_1h","cust_24h","dev_1h","c_m","c_s","z",
            "geo","s_price_bulk","s_gc_geo","s_burst","s_z","s_geo_hi","s_any"]

Xn = labeled[num_cols].fillna(0)
Xc = pd.get_dummies(labeled[cat_cols].astype(str), dummy_na=False)
X  = pd.concat([Xn, Xc], axis=1)
y  = labeled["y"].astype(int).values
cat_dummies = list(Xc.columns)

# class imbalance weighting (helps recall at fixed 0.30)
pos = max(1, int((y==1).sum())); neg = max(1, len(y)-pos)
w_pos = min(10.0, neg/pos)  # cap to avoid overfitting

@st.cache_resource(show_spinner=False)
def fit_model(X, y, w_pos):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                       early_stopping=True, random_state=RNG)
    sw = np.where(y==1, w_pos, 1.0)
    m.fit(X, y, sample_weight=sw)
    return m

clf = fit_model(X, y, w_pos)

def score_all(D: pd.DataFrame) -> np.ndarray:
    Xn_ = D[num_cols].fillna(0)
    Xc_ = pd.get_dummies(D[cat_cols].astype(str), dummy_na=False).reindex(columns=cat_dummies, fill_value=0)
    X_  = pd.concat([Xn_, Xc_], axis=1)
    return clf.predict_proba(X_)[:,1]

df["score"] = score_all(df)

# decisions at fixed threshold with rule-based refinements
alert_score = (df["score"] >= TH).astype(int)

# 1) force-alert for strong patterns (improves recall)
force_alert = (df["s_any"]==1)

# 2) safe-case suppressor (optional; improves precision/accuracy)
safe = ((df["amt"]<=100)&(df["q"]<=2)&(df["ship"]==df["ip"])&
        (df["gift_pct"]<.05)&(df["price_ratio"].sub(1).abs()<=.15)&
        (df["s_price_bulk"]==0)&(df["s_gc_geo"]==0)&(df["s_burst"]==0)&
        (df["s_z"]==0)&(df["s_geo_hi"]==0))

if APPLY_SAFE:
    df["alert"] = np.where(force_alert, 1, np.where(safe & (alert_score==1), 0, alert_score))
else:
    df["alert"] = np.where(force_alert, 1, alert_score)

# ───────── KPIs ─────────
st.subheader(f"**Key Metrics (Threshold = {TH:.2f})**")
TOT, AL = len(df), int(df["alert"].sum())
c1,c2,c3 = st.columns(3)
c1.metric("**Total transactions**", TOT)
c2.metric(f"**Fraud alerts (≥{TH:.2f})**", AL)
c3.metric("**Alert rate**", f"{AL/max(1,TOT):.2%}")
st.caption("Training uses imbalance weighting and rule refinements (force-alert on strong fraud patterns; optional safe-case suppressor).")

# Trend
st.subheader("**Daily Trend**")
df["day"] = pd.to_datetime(df["ts"]).dt.date
trend = df.groupby("day").agg(total=("order_id","count"), alerts=("alert","sum")).reset_index()
if not trend.empty:
    tl = trend.melt("day", ["total","alerts"], "Series", "value").replace({"total":"Total transactions","alerts":f"Fraud alerts (≥{TH:.2f})"})
    st.altair_chart(alt.Chart(tl).mark_line(point=True).encode(x="day:T", y="value:Q", color="Series:N"), use_container_width=True)

# Score distribution
st.subheader("**Fraud Score Distribution**")
st.altair_chart(
    alt.Chart(df).mark_bar().encode(x=alt.X("score:Q", bin=alt.Bin(maxbins=50)), y="count()").properties(height=200)
    + alt.Chart(pd.DataFrame({"x":[TH]})).mark_rule(color="red").encode(x="x"),
    use_container_width=True
)

# Top alerts — show ALL
st.subheader(f"**Top Alerts (score ≥ {TH:.2f})**")
cols = [c for c in ["order_id","ts","customer_id","store_id","sku_id","sku_category","amt","q","pay","ship","ip","score"] if c in df]
top = df[df["alert"]==1].sort_values(["score","ts"], ascending=[False,False]).drop_duplicates("order_id")
st.caption(f"Showing **all {len(top)}** alerts.")
st.dataframe(top.loc[:, cols], use_container_width=True, height=min(700, 35*max(5,len(top))))

# ───────── Evaluation at bottom ────────
st.subheader(f"**Model Evaluation (fixed threshold = {TH:.2f})**")
st.caption("Precision = among alerts, % truly fraud; Recall = of all fraud, % caught; F1 = balance; Accuracy = overall correctness.")
y_true = labeled["y"].astype(int).values
y_pred = df.loc[labeled.index, "alert"].astype(int).values
m1,m2,m3,m4 = st.columns(4)
m1.metric("Accuracy",  f"{accuracy_score(y_true,y_pred):.2%}")
m2.metric("Precision", f"{precision_score(y_true,y_pred,zero_division=0):.2%}")
m3.metric("Recall",    f"{recall_score(y_true,y_pred,zero_division=0):.2%}")
m4.metric("F1-score",  f"{f1_score(y_true,y_pred,zero_division=0):.2%}")
