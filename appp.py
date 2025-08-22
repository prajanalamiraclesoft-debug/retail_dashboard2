# app.py  —  streamlit run app.py
import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import datetime
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config("Retail Fraud – Executive Dashboard", layout="wide")
alt.renderers.set_embed_options(actions=False)
RND = 42

# ───────────────────────── Sidebar (read-only DS + speed) ─────────────────────────
st.sidebar.header("Data source (read-only)")
PROJ   = st.sidebar.text_input("Project", "mss-data-engineer-sandbox", disabled=True)
DATASET= st.sidebar.text_input("Dataset", "retail", disabled=True)
RAW    = st.sidebar.text_input("Raw table", f"{PROJ}.{DATASET}.transaction_data", disabled=True)
FAST_MODE = st.sidebar.checkbox("Fast mode (skip heavy windows / rolling z-score)", True)

# ───────────────────────── BigQuery: pull a 4k snapshot ─────────────────────────
sa = dict(st.secrets["gcp_service_account"])
sa["private_key"] = sa["private_key"].replace("\\n", "\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq    = bigquery.Client(credentials=creds, project=creds.project_id)

@st.cache_data(ttl=900, show_spinner=True)
def load_snapshot(_secrets_guard: dict) -> pd.DataFrame:
    sql = f"""
    SELECT
      CAST(order_id AS STRING)          AS order_id,
      TIMESTAMP(timestamp)              AS ts,
      CAST(customer_id AS STRING)       AS customer_id,
      CAST(store_id AS STRING)          AS store_id,
      CAST(device_id AS STRING)         AS device_id,
      CAST(sku_id AS STRING)            AS sku_id,
      CAST(sku_category AS STRING)      AS sku_category,
      SAFE_CAST(quantity AS FLOAT64)    AS quantity,
      SAFE_CAST(unit_price AS FLOAT64)  AS unit_price,
      CAST(payment_method AS STRING)    AS payment_method,
      CAST(shipping_country AS STRING)  AS ship_country,
      CAST(ip_country AS STRING)        AS ip_country,
      SAFE_CAST(account_created_at AS TIMESTAMP) AS account_created_at,
      SAFE_CAST(coupon_discount AS FLOAT64)      AS coupon_discount,
      SAFE_CAST(gift_card_amount AS FLOAT64)     AS gift_balance_used,
      SAFE_CAST(fraud_flag AS INT64)    AS fraud_flag
    FROM `{RAW}`
    WHERE timestamp IS NOT NULL
    ORDER BY timestamp DESC
    LIMIT 4000
    """
    df = bq.query(sql).result().to_dataframe()
    return df

raw = load_snapshot(st.secrets)  # read-only guard for cache

# ───────────────────────── Feature engineering ─────────────────────────
def prepare_features(df0: pd.DataFrame, fast: bool) -> pd.DataFrame:
    df = df0.copy().dropna(subset=["order_id"]).drop_duplicates("order_id", keep="last")
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_localize(None)
    df["account_created_at"] = pd.to_datetime(df["account_created_at"], errors="coerce", utc=True).dt.tz_localize(None)
    df["account_age_days"] = (df["ts"] - df["account_created_at"]).dt.days.replace({np.nan: 0}).clip(lower=0)
    df["order_amount"] = df["quantity"] * df["unit_price"]

    # Winsorize per category
    def _wins(g, col):
        lo, hi = g[col].quantile(0.01), g[col].quantile(0.99)
        g[col] = g[col].clip(lo, hi); return g
    for c in ["unit_price", "quantity"]:
        df = df.groupby("sku_category", group_keys=False).apply(_wins, col=c)
    df["order_amount"] = df["quantity"] * df["unit_price"]

    # Strong features (business-intuitive)
    cat_avg = df.groupby("sku_category")["unit_price"].transform("mean").replace(0, np.nan)
    df["price_ratio"] = (df["unit_price"] / cat_avg).fillna(1.0)                    # price anomaly vs category
    df["s_price_bulk"] = ((df["price_ratio"].sub(1).abs() >= 0.50) &               # ±50% price & bulk qty
                          (df["quantity"] >= 3)).astype(int)

    df["geo_mismatch"] = (df["ip_country"] != df["ship_country"]).astype(int)      # network vs shipping
    p90 = float(np.nanpercentile(df["order_amount"], 90)) if len(df) else 0.0
    df["s_geo_highvalue"] = ((df["geo_mismatch"] == 1) &                           # geo mismatch on high value
                             (df["order_amount"] >= p90)).astype(int)

    den = df["order_amount"].replace(0, np.nan)
    df["coupon_pct"] = (df["coupon_discount"] / den).fillna(0)
    df["gift_pct"]   = (df["gift_balance_used"] / den).fillna(0)
    df["s_discount_gift_stack"] = ((df["coupon_pct"] >= 0.30) &                     # discount+gift stacking
                                   (df["gift_pct"]   >= 0.50)).astype(int)

    if not fast:
        # EXPENSIVE: burst in last hour (customer/device)
        df = df.sort_values("ts")
        def vel_last_hours(g, hours):
            t = pd.to_datetime(g["ts"]).astype("int64") // 1_000_000
            order = np.argsort(t); t = t.iloc[order].to_numpy(); w = int(hours*3600*1e6)
            left = np.searchsorted(t, t-w, side="left")
            counts = np.arange(1, len(t)+1) - left
            return pd.Series(counts, index=g.index[order]).reindex(g.index)
        df["cust_1h"] = df.groupby("customer_id", group_keys=False).apply(vel_last_hours, 1)
        df["dev_1h"]  = df.groupby("device_id",   group_keys=False).apply(vel_last_hours, 1)
        df["s_burst"] = ((df["cust_1h"] >= 3) | (df["dev_1h"] >= 3)).astype(int)

        # EXPENSIVE: customer spend deviation (rolling)
        def roll_stats(g):
            g = g.sort_values("ts")
            r = g["order_amount"].rolling(10, min_periods=1)
            g["c_amt_mean"] = r.mean(); g["c_amt_std"] = r.std(ddof=0).replace(0, np.nan)
            return g
        df = df.groupby("customer_id", group_keys=False).apply(roll_stats)
        z = ((df["order_amount"] - df["c_amt_mean"]) / df["c_amt_std"]).replace([np.inf,-np.inf],0).fillna(0)
        df["s_spend_deviation"] = (z.abs() >= 2).astype(int)
    else:
        # FAST proxies: burst via simple per-hour count & category-level z-score
        df["date_h"] = df["ts"].dt.floor("H")
        recent = (df.groupby(["customer_id","date_h"])["order_id"].transform("count")).fillna(0)
        df["s_burst"] = (recent >= 3).astype(int)
        cat_mean = df.groupby("sku_category")["order_amount"].transform("mean")
        cat_std  = df.groupby("sku_category")["order_amount"].transform("std").replace(0, np.nan)
        zc = ((df["order_amount"] - cat_mean) / cat_std).replace([np.inf,-np.inf],0).fillna(0)
        df["s_spend_deviation"] = (zc.abs() >= 2).astype(int)
        df["cust_1h"], df["dev_1h"] = 0, 0

    # Normalized channel + category
    df["payment_channel"] = df["payment_method"].replace(
        {"card":"Card","credit_card":"Card","debit_card":"Card",
         "digital_wallet":"DigitalWallet","wallet":"DigitalWallet",
         "gift_card":"GiftCard"}).fillna("Card")
    df["category"] = df["sku_category"].fillna("misc")

    df["y"] = df["fraud_flag"].fillna(0).astype(int)
    return df

df = prepare_features(raw, FAST_MODE)

# ───────────────────────── Train / test & auto threshold ─────────────────────────
df = df.sort_values("ts")
cut = int(len(df)*0.75)
train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

feat_num = ["order_amount","quantity","unit_price","account_age_days","price_ratio",
            "coupon_pct","gift_pct","cust_1h","dev_1h",
            "s_price_bulk","s_geo_highvalue","s_discount_gift_stack","s_burst","s_spend_deviation",
            "geo_mismatch"]
feat_cat = ["payment_channel","category","ship_country","ip_country","store_id"]

Xtr = pd.concat([train[feat_num].fillna(0),
                 pd.get_dummies(train[feat_cat].astype(str), dummy_na=False)], axis=1)
ytr = train["y"].values
Xte = pd.concat([test[feat_num].fillna(0),
                 pd.get_dummies(test[feat_cat].astype(str), dummy_na=False)], axis=1)
Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
yte = test["y"].values
cat_cols_model = Xtr.columns

# imbalance weight
pos = max(1, int((ytr==1).sum())); neg = max(1, int((ytr==0).sum()))
w_pos = neg/pos

clf = HistGradientBoostingClassifier(
    max_iter=160 if FAST_MODE else 300,
    learning_rate=0.08,
    min_samples_leaf=10,
    l2_regularization=0.01,
    early_stopping=True,
    random_state=RND
)
sw = np.where(ytr==1, w_pos, 1.0)
clf.fit(Xtr, ytr, sample_weight=sw)

probs = clf.predict_proba(Xte)[:,1]
grid  = np.linspace(0.05, 0.95, 31 if FAST_MODE else 91)
best_t, best_f1 = 0.5, -1
for t in grid:
    yp = (probs >= t).astype(int)
    p  = precision_score(yte, yp, zero_division=0)
    r  = recall_score(yte, yp, zero_division=0)
    f1 = 0 if (p+r)==0 else 2*p*r/(p+r)
    if p >= 0.25 and f1 > best_f1: best_f1, best_t = f1, t
yhat = (probs >= best_t).astype(int)

# Stats needed for new-order feature compute
p90_amount = float(np.nanpercentile(train["order_amount"], 90)) if len(train) else 0.0
cat_price_mean = train.groupby("sku_category")["unit_price"].mean().to_dict()
cat_amt_mean   = train.groupby("sku_category")["order_amount"].mean().to_dict()
cat_amt_std    = train.groupby("sku_category")["order_amount"].std().replace(0,np.nan).to_dict()

# ────────────────────────── L1. Snapshot (what business sees) ──────────────────────────
st.title("Retail Fraud – Executive Dashboard")
k1,k2,k3,k4 = st.columns(4)
k1.metric("Rows (snapshot)", f"{len(raw):,}")
k2.metric("Training window", f"{train['ts'].min().date()} → {train['ts'].max().date()}")
k3.metric("Testing window",  f"{test['ts'].min().date()} → {test['ts'].max().date()}")
k4.metric("Fast mode", "ON" if FAST_MODE else "OFF")

st.subheader("Raw 4,000 orders (snapshot view)")
show_cols = ["order_id","ts","customer_id","store_id","device_id","sku_id","sku_category",
             "quantity","unit_price","payment_method","ship_country","ip_country",
             "coupon_discount","gift_balance_used","fraud_flag"]
st.dataframe(raw.loc[:, show_cols], use_container_width=True, height=320)

# ────────────────────────── L2. Channel mix & volume (simple, readable) ──────────────────────────
st.subheader("Channel mix (share of orders)")
mix = (df.groupby("payment_channel")["order_id"].count().reset_index()
         .rename(columns={"order_id":"orders"}))
ch = alt.Chart(mix).mark_bar().encode(
    x=alt.X("payment_channel:N", title="Channel"),
    y=alt.Y("orders:Q", title="Orders"),
    tooltip=["payment_channel","orders"]
).properties(height=220)
st.altair_chart(ch, use_container_width=True)

# ────────────────────────── L3. Model evaluation (test set, auto-threshold) ──────────────────────────
st.subheader(f"Model evaluation (auto threshold = {best_t:.2f})")
m1,m2,m3,m4 = st.columns(4)
m1.metric("Accuracy",  f"{accuracy_score(yte,yhat):.2%}")
m2.metric("Precision", f"{precision_score(yte,yhat,zero_division=0):.2%}")
m3.metric("Recall",    f"{recall_score(yte,yhat,zero_division=0):.2%}")
m4.metric("F1-score",  f"{f1_score(yte,yhat,zero_division=0):.2%}")
st.caption("Auto-threshold picks the best F1 subject to a minimum precision guard (≥25%).")

# ────────────────────────── L4. New order – instant decision ──────────────────────────
st.subheader("New Order – Instant Decision")

# Helper: compute features for a single order using training stats
def single_features(rec: dict) -> pd.DataFrame:
    # normalize fields
    rec = rec.copy()
    rec["order_amount"] = rec["quantity"] * rec["unit_price"]
    prc_mean = cat_price_mean.get(rec["sku_category"], rec["unit_price"])
    rec["price_ratio"] = rec["unit_price"]/prc_mean if prc_mean else 1.0
    rec["s_price_bulk"] = int(abs(rec["price_ratio"]-1)>=0.50 and rec["quantity"]>=3)
    rec["geo_mismatch"] = int(rec["ip_country"] != rec["ship_country"])
    rec["s_geo_highvalue"] = int(rec["geo_mismatch"]==1 and rec["order_amount"]>=p90_amount)
    den = rec["order_amount"] or 1.0
    rec["coupon_pct"] = (rec["coupon_discount"]/den) if den else 0.0
    rec["gift_pct"]   = (rec["gift_balance_used"]/den) if den else 0.0
    rec["s_discount_gift_stack"] = int(rec["coupon_pct"]>=0.30 and rec["gift_pct"]>=0.50)

    if FAST_MODE:
        # burst proxy = 0 for a single order (no history); could be 1 if you wire recent counts
        rec["cust_1h"], rec["dev_1h"], rec["s_burst"] = 0,0,0
        m = cat_amt_mean.get(rec["sku_category"], rec["order_amount"])
        s = cat_amt_std.get(rec["sku_category"], np.nan)
        z = 0 if (s is np.nan or not s) else (rec["order_amount"]-m)/s
        rec["s_spend_deviation"] = int(abs(z)>=2.0)
    else:
        rec["cust_1h"], rec["dev_1h"] = 0,0
        rec["s_burst"], rec["s_spend_deviation"] = 0,0

    # one-row frame aligned to model features
    num = {k:rec[k] for k in ["order_amount","quantity","unit_price","account_age_days",
                              "price_ratio","coupon_pct","gift_pct","cust_1h","dev_1h",
                              "s_price_bulk","s_geo_highvalue","s_discount_gift_stack",
                              "s_burst","s_spend_deviation","geo_mismatch"]}
    cat = {"payment_channel":rec["payment_channel"],"category":rec["sku_category"],
           "ship_country":rec["ship_country"],"ip_country":rec["ip_country"],
           "store_id":rec["store_id"]}
    Xnum = pd.DataFrame([num]).fillna(0)
    Xcat = pd.get_dummies(pd.DataFrame([cat]).astype(str), dummy_na=False)
    X = pd.concat([Xnum, Xcat], axis=1).reindex(columns=cat_cols_model, fill_value=0)
    return X, rec

# Inputs minimal but complete for the strong features
c1,c2,c3 = st.columns(3)
sku_category   = c1.selectbox("Category", sorted(df["category"].unique()), index=0)
payment_channel= c2.selectbox("Payment channel", ["Card","DigitalWallet","GiftCard"], index=0)
ship_country   = c3.selectbox("Shipping country", sorted(df["ship_country"].dropna().unique())[:50], index=0)
c4,c5,c6 = st.columns(3)
ip_country     = c4.selectbox("Network country", sorted(df["ip_country"].dropna().unique())[:50], index=0)
quantity       = c5.number_input("Quantity", 1.0, 50.0, 1.0, 1.0)
unit_price     = c6.number_input("Unit price", 1.0, 5000.0, 120.0, 1.0)
c7,c8,c9 = st.columns(3)
coupon_discount= c7.number_input("Discount amount", 0.0, 5000.0, 0.0, 1.0)
gift_balance   = c8.number_input("Gift balance used", 0.0, 5000.0, 0.0, 1.0)
acct_age_days  = c9.number_input("Account age (days)", 0.0, 3650.0, 120.0, 1.0)

rec = {
    "sku_category": sku_category,
    "payment_channel": payment_channel,
    "ship_country": ship_country,
    "ip_country": ip_country,
    "quantity": float(quantity),
    "unit_price": float(unit_price),
    "coupon_discount": float(coupon_discount),
    "gift_balance_used": float(gift_balance),
    "account_age_days": float(acct_age_days),
    "store_id": "store_manual",
}

if st.button("Check"):
    Xin, rx = single_features(rec)
    pred = int((clf.predict_proba(Xin)[:,1] >= best_t)[0])
    label = "Fraud" if pred==1 else "Not Fraud"
    color = "#d62828" if pred==1 else "#2a9d8f"
    st.markdown(f"### Decision: <span style='color:{color}; font-weight:700'>{label}</span>", unsafe_allow_html=True)

    # Rule-based reasons (what business understands)
    reasons = []
    if rx["geo_mismatch"]: reasons.append("Network vs Shipping country mismatch")
    if rx["s_geo_highvalue"]: reasons.append("Mismatch on **high-value** order")
    if rx["s_price_bulk"]: reasons.append("Bulk quantity with **price anomaly**")
    if rx["s_discount_gift_stack"]: reasons.append("Heavy **discount + gift balance**")
    if rx["s_burst"]: reasons.append("Burst of orders in short time")
    if rx["s_spend_deviation"]: reasons.append("Order **deviates from category spend**")
    if not reasons: reasons = ["No strong risk rules fired"]

    st.write("**Why**:", " · ".join(reasons))
    st.caption("Rules are transparent signals; the model combines them with channel/geo/category context for the final decision.")
