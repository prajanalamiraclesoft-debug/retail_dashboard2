# app.py — streamlit run app.py
import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import date
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config("Fraud Dashboard (Fixed TH=0.30, Fast)", layout="wide")
alt.renderers.set_embed_options(actions=False)
THRESH = 0.30
RANDOM_STATE = 42

# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.header("BigQuery")
    PROJ = st.text_input("Project", "mss-data-engineer-sandbox")
    DATASET = st.text_input("Dataset", "retail")
    RAW = st.text_input("Raw table", f"{PROJ}.{DATASET}.transaction_data")
    S = st.date_input("Start", date(2023,1,1))
    E = st.date_input("End",  date(2030,12,31))
    st.caption("Decision threshold is fixed at **0.30**; scores are calibrated so 0.30 equals the validated operating point.")

# ───────────────────────── GCP client ──────────────────────
sa = dict(st.secrets["gcp_service_account"])
sa["private_key"] = sa["private_key"].replace("\\n", "\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq = bigquery.Client(credentials=creds, project=creds.project_id)

# ───────────────────────── Data load ───────────────────────
@st.cache_data(show_spinner=True)
def load_raw(raw, s, e):
    sql = f"""
      SELECT
        order_id, TIMESTAMP(timestamp) AS ts, customer_id, store_id, sku_id, sku_category,
        SAFE_CAST(quantity AS FLOAT64) AS q, SAFE_CAST(unit_price AS FLOAT64) AS p,
        CAST(payment_method AS STRING) AS pay, CAST(shipping_country AS STRING) AS ship,
        CAST(ip_country AS STRING) AS ip, CAST(device_id AS STRING) AS dev,
        SAFE_CAST(account_created_at AS TIMESTAMP) AS acct,
        SAFE_CAST(coupon_discount AS FLOAT64) AS coup,
        SAFE_CAST(gift_card_amount AS FLOAT64) AS gift_amt,
        SAFE_CAST(gift_card_used AS BOOL) AS gift_used,
        SAFE_CAST(fraud_flag AS INT64) AS y
      FROM `{raw}`
      WHERE DATE(timestamp) BETWEEN @S AND @E
    """
    job = bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("S","DATE",str(s)),
            bigquery.ScalarQueryParameter("E","DATE",str(e))
        ]))
    return job.result().to_dataframe()

df = load_raw(RAW, S, E)
if df.empty:
    st.warning("No rows in this date range."); st.stop()

# ───────────────────────── Cleaning ────────────────────────
df = df.sort_values("ts").drop_duplicates("order_id", keep="last")
df = df[(df.q > 0) & (df.p > 0)].copy()

# fix datetimes & compute age safely
df["ts"]   = pd.to_datetime(df["ts"],   errors="coerce", utc=True).dt.tz_localize(None)
df["acct"] = pd.to_datetime(df["acct"], errors="coerce", utc=True).dt.tz_localize(None)
df["age"]  = (df["ts"] - df["acct"]).dt.days.astype("float").fillna(0).clip(lower=0)

df["amt"] = df.q * df.p

# winsorize per category (1–99%)
def wins(g, col):
    lo, hi = g[col].quantile(0.01), g[col].quantile(0.99)
    g[col] = g[col].clip(lo, hi); return g
for c in ["p","q"]:
    df = df.groupby("sku_category", group_keys=False).apply(wins, col=c)
df["amt"] = df.q * df.p

# ─────────────────── Feature engineering ───────────────────
# price baseline & ratios
cat_avg = df.groupby("sku_category")["p"].transform("mean").replace(0, np.nan)
df["price_ratio"] = (df["p"]/cat_avg).fillna(1.0)

# intensities
den = df["amt"].replace(0, np.nan)
df["coup_pct"] = (df["coup"]/den).fillna(0)
df["gift_pct"] = (df["gift_amt"]/den).fillna(0)

# geo mismatch
df["geo"] = (df["ship"] != df["ip"]).astype(int)

# velocity windows (counts in last 1h / 24h)
def vel_last_hours(g, hours):
    t = pd.to_datetime(g["ts"]).astype("int64") // 1_000_000
    order = np.argsort(t); t = t.iloc[order].to_numpy()
    w = int(hours*3600*1e6)
    left = np.searchsorted(t, t - w, side="left")
    counts = np.arange(1, len(t)+1) - left
    return pd.Series(counts, index=g.index[order]).reindex(g.index)

df = df.sort_values("ts")
df["cust_1h"]  = df.groupby("customer_id", group_keys=False).apply(vel_last_hours, 1)
df["cust_24h"] = df.groupby("customer_id", group_keys=False).apply(vel_last_hours, 24)
df["dev_1h"]   = df.groupby("dev",         group_keys=False).apply(vel_last_hours, 1)

# rolling customer baseline (last 10 orders)
def roll_stats(g):
    g = g.sort_values("ts"); r = g["amt"].rolling(10, min_periods=1)
    g["c_m"] = r.mean()
    g["c_s"] = r.std(ddof=0).replace(0,np.nan)
    return g
df = df.groupby("customer_id", group_keys=False).apply(roll_stats)
df["z"] = ((df["amt"] - df["c_m"]) / df["c_s"]).replace([np.inf,-np.inf], 0).fillna(0)

# high-value cut
p90 = float(np.nanpercentile(df["amt"], 90)) if len(df) else 0.0

# strong signals (0/1)
df["s_price_bulk"] = ((df["price_ratio"].sub(1).abs() >= 0.50) & (df["q"] >= 3)).astype(int)
df["s_gc_geo"]     = ((df["gift_used"].fillna(False)) & (df["geo"]==1)).astype(int)
df["s_burst"]      = ((df["cust_1h"]>=3) | (df["dev_1h"]>=3)).astype(int)
df["s_z"]          = (df["z"].abs() >= 2).astype(int)
df["s_geo_hi"]     = ((df["geo"]==1) & (df["amt"] >= p90)).astype(int)
df["s_any"]        = (df[["s_price_bulk","s_gc_geo","s_burst","s_z","s_geo_hi"]].sum(axis=1) > 0).astype(int)

# ───────────────── Train + calibrate (FAST) ────────────────
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

@st.cache_data(show_spinner=False)
def cv_calibrate_threshold(X, y, n_splits=3, max_iter=200):  # fast CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    probs = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1,
                                             early_stopping=True, random_state=RANDOM_STATE)
        clf.fit(X.iloc[tr], y[tr])
        probs[va] = clf.predict_proba(X.iloc[va])[:,1]
    ts = np.linspace(0.01, 0.99, 99)
    best_t, best_f1 = 0.5, -1
    for t in ts:
        yhat = (probs >= t).astype(int)
        p = precision_score(y, yhat, zero_division=0)
        r = recall_score(y, yhat, zero_division=0)
        f1 = 0 if p+r==0 else 2*p*r/(p+r)
        if f1 > best_f1: best_f1, best_t = f1, t
    scale = THRESH / best_t if best_t > 0 else 1.0
    return best_t, scale

t_star, scale = cv_calibrate_threshold(X, y)

@st.cache_resource(show_spinner=False)
def fit_final_model(X, y, max_iter=200):
    m = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1,
                                       early_stopping=True, random_state=RANDOM_STATE)
    m.fit(X, y); return m

final_clf = fit_final_model(X, y)

def score_all(D: pd.DataFrame) -> np.ndarray:
    Xn_ = D[num_cols].fillna(0)
    Xc_ = pd.get_dummies(D[cat_cols].astype(str), dummy_na=False).reindex(columns=cat_dummies, fill_value=0)
    X_  = pd.concat([Xn_, Xc_], axis=1)
    return np.clip(final_clf.predict_proba(X_)[:,1] * scale, 0, 1)

df["score"] = score_all(df)

# safe-case suppressor
safe = (
    (df["amt"] <= 100) & (df["q"] <= 2) & (df["ship"] == df["ip"]) &
    (df["gift_pct"] < 0.05) & (df["price_ratio"].sub(1).abs() <= 0.15) &
    (df["s_price_bulk"]==0) & (df["s_gc_geo"]==0) & (df["s_burst"]==0) &
    (df["s_z"]==0) & (df["s_geo_hi"]==0)
)
df["alert_raw"] = (df["score"] >= THRESH).astype(int)
df["alert"]     = ((df["score"] >= THRESH) & (~safe)).astype(int)

# ───────────────────── KPIs ────────────────────────────────
st.subheader("**Key Metrics (Threshold = 0.30)**")
TOT, AL = len(df), int(df["alert"].sum())
c1,c2,c3 = st.columns(3)
c1.metric("**Total transactions**", TOT)
c2.metric("**Fraud alerts (≥0.30)**", AL)
c3.metric("**Alert rate**", f"{AL/max(1,TOT):.2%}")
st.caption(f"Calibration: best-F1 threshold t* ≈ {t_star:.2f} → scaled so decisions occur at 0.30.")

# evaluation on labeled rows
st.subheader("**Model Evaluation (fixed threshold = 0.30)**")
st.caption("Precision = among alerts, % truly fraud; Recall = of all fraud, % caught; F1 = balance; Accuracy = overall correctness.")
y_true = labeled["y"].astype(int).values
y_pred = df.loc[labeled.index, "alert"].astype(int).values
m1,m2,m3,m4 = st.columns(4)
m1.metric("Accuracy",  f"{accuracy_score(y_true,y_pred):.2%}")
m2.metric("Precision", f"{precision_score(y_true,y_pred,zero_division=0):.2%}")
m3.metric("Recall",    f"{recall_score(y_true,y_pred,zero_division=0):.2%}")
m4.metric("F1-score",  f"{f1_score(y_true,y_pred,zero_division=0):.2%}")

# ───────────────────── Visuals ─────────────────────────────
st.subheader("**Daily Trend**")
df["day"] = pd.to_datetime(df["ts"]).dt.date
trend = df.groupby("day").agg(total=("order_id","count"), alerts=("alert","sum")).reset_index()
if not trend.empty:
    tl = trend.melt("day", ["total","alerts"], "Series", "value").replace({"total":"Total transactions","alerts":"Fraud alerts (≥0.30)"})
    st.altair_chart(alt.Chart(tl).mark_line(point=True).encode(x="day:T", y="value:Q", color="Series:N"), use_container_width=True)

st.subheader("**Fraud Score Distribution**")
st.altair_chart(
    alt.Chart(df).mark_bar().encode(x=alt.X("score:Q", bin=alt.Bin(maxbins=50)), y="count()").properties(height=200)
    + alt.Chart(pd.DataFrame({"x":[THRESH]})).mark_rule(color="red").encode(x="x"),
    use_container_width=True
)

st.subheader("**Top Alerts (score ≥ 0.30)**")
cols = [c for c in ["order_id","ts","customer_id","store_id","sku_id","sku_category","amt","q","pay","ship","ip","score"] if c in df]
top = df[df["alert"]==1].sort_values(["score","ts"], ascending=[False,False]).drop_duplicates("order_id")
st.dataframe(top.loc[:, cols].head(50), use_container_width=True, height=320)
