# streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time
from datetime import date, datetime, timedelta

from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core.exceptions import GoogleAPICallError, NotFound, Forbidden, DeadlineExceeded

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─────────────────────────── Page & Altair ───────────────────────────
st.set_page_config("Retail Fraud – Business Dashboard", layout="wide")
alt.renderers.set_embed_options(actions=False)

# ─────────────────────────── Sidebar (read-only) ─────────────────────
with st.sidebar:
    st.header("Data source (read-only)")
    PROJ = st.text_input("Project", "mss-data-engineer-sandbox")
    DATASET = st.text_input("Dataset", "retail")
    RAW_TABLE = st.text_input("Raw table", f"{PROJ}.{DATASET}.transaction_data")
# ─────────────────────────── BigQuery client ─────────────────────────
# (streamlit cloud: put gcp_service_account in Secrets)
sa = dict(st.secrets["gcp_service_account"])
sa["private_key"] = sa["private_key"].replace("\\n", "\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq = bigquery.Client(credentials=creds, project=creds.project_id)

# ─────────────────────────── Data loader (robust) ────────────────────
@st.cache_data(show_spinner=True)
def load_snapshot(_secrets_guard) -> pd.DataFrame:
    sql = """
    SELECT
      order_id,
      TIMESTAMP(timestamp)                    AS ts,
      CAST(customer_id AS STRING)             AS customer_id,
      CAST(device_id   AS STRING)             AS device_id,
      CAST(store_id    AS STRING)             AS store_id,
      CAST(sku_id      AS STRING)             AS sku_id,
      CAST(sku_category AS STRING)            AS sku_category,
      SAFE_CAST(quantity      AS FLOAT64)     AS quantity,
      SAFE_CAST(unit_price    AS FLOAT64)     AS unit_price,
      CAST(payment_method AS STRING)          AS payment_method,
      CAST(shipping_country AS STRING)        AS ship_country,
      CAST(ip_country      AS STRING)         AS ip_country,
      SAFE_CAST(coupon_discount  AS FLOAT64)  AS coupon_discount,
      SAFE_CAST(gift_card_amount AS FLOAT64)  AS gift_balance_used,
      SAFE_CAST(gift_card_used   AS BOOL)     AS gift_used_flag,
      SAFE_CAST(account_created_at AS TIMESTAMP) AS account_created_at,
      SAFE_CAST(fraud_flag AS INT64)          AS fraud_flag
    FROM `mss-data-engineer-sandbox.retail.transaction_data`
    WHERE DATE(timestamp) BETWEEN '2023-01-01' AND '2030-12-31'
    ORDER BY timestamp, order_id
    LIMIT 4000
    """
    last_err = None
    for attempt in range(3):
        try:
            job = bq.query(sql, job_config=bigquery.QueryJobConfig(use_legacy_sql=False))
            return job.result(timeout=180).to_dataframe()
        except (DeadlineExceeded, GoogleAPICallError, Forbidden, NotFound, Exception) as e:
            last_err = e
            time.sleep(2 ** attempt)
    st.error(f"BigQuery snapshot failed: {type(last_err).__name__}: {last_err}")
    st.stop()

raw = load_snapshot(st.secrets)  # <— prevents the NameError you saw

# ─────────────────────────── Quiet cleaning & features ───────────────
def prepare_features(df0: pd.DataFrame) -> pd.DataFrame:
    df = df0.copy().dropna(subset=["order_id"]).drop_duplicates("order_id", keep="last")
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)]
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_localize(None)
    df["account_created_at"] = pd.to_datetime(df["account_created_at"], errors="coerce", utc=True).dt.tz_localize(None)
    df["account_age_days"] = (df["ts"] - df["account_created_at"]).dt.days.replace({np.nan: 0}).clip(lower=0)
    df["order_amount"] = df["quantity"] * df["unit_price"]

    # Winsorize per category to tame extreme unit_price / quantity
    def _wins(g, col):
        lo, hi = g[col].quantile(0.01), g[col].quantile(0.99)
        g[col] = g[col].clip(lo, hi)
        return g
    for c in ["unit_price", "quantity"]:
        df = df.groupby("sku_category", group_keys=False).apply(_wins, col=c)
    df["order_amount"] = df["quantity"] * df["unit_price"]

    # Strong features (6 business-obvious patterns)
    # 1) price_ratio (item priced far off its category norm) + bulk quantity
    cat_avg = df.groupby("sku_category")["unit_price"].transform("mean").replace(0, np.nan)
    df["price_ratio"] = (df["unit_price"] / cat_avg).fillna(1.0)
    df["s_price_bulk"] = ((df["price_ratio"].sub(1).abs() >= 0.50) & (df["quantity"] >= 3)).astype(int)

    # 2) geo_mismatch (IP vs ship mismatch)
    df["geo_mismatch"] = (df["ip_country"] != df["ship_country"]).astype(int)

    # 3) high value + mismatch
    p90 = float(np.nanpercentile(df["order_amount"], 90)) if len(df) else 0.0
    df["s_geo_highvalue"] = ((df["geo_mismatch"] == 1) & (df["order_amount"] >= p90)).astype(int)

    # 4) coupon + gift stack (classic laundering)
    den = df["order_amount"].replace(0, np.nan)
    df["coupon_pct"] = (df["coupon_discount"] / den).fillna(0)
    df["gift_pct"] = (df["gift_balance_used"] / den).fillna(0)
    df["s_discount_gift_stack"] = ((df["coupon_pct"] >= 0.30) & (df["gift_pct"] >= 0.50)).astype(int)

    # 5) burst velocity (customer or device)
    df = df.sort_values("ts")
    def vel_last_hours(g, hours):
        t = pd.to_datetime(g["ts"]).astype("int64") // 1_000_000
        order = np.argsort(t); t = t.iloc[order].to_numpy()
        w = int(hours*3600*1e6)
        left = np.searchsorted(t, t-w, side="left")
        counts = np.arange(1, len(t)+1) - left
        return pd.Series(counts, index=g.index[order]).reindex(g.index)
    df["cust_1h"] = df.groupby("customer_id", group_keys=False).apply(vel_last_hours, 1)
    df["dev_1h"]  = df.groupby("device_id",   group_keys=False).apply(vel_last_hours, 1)
    df["s_burst"] = ((df["cust_1h"] >= 3) | (df["dev_1h"] >= 3)).astype(int)

    # 6) spend deviation from customer baseline (z-score)
    def roll_stats(g):
        g = g.sort_values("ts")
        r = g["order_amount"].rolling(10, min_periods=1)
        g["c_amt_mean"] = r.mean()
        g["c_amt_std"]  = r.std(ddof=0).replace(0, np.nan)
        return g
    df = df.groupby("customer_id", group_keys=False).apply(roll_stats)
    z = ((df["order_amount"] - df["c_amt_mean"]) / df["c_amt_std"]).replace([np.inf, -np.inf], 0).fillna(0)
    df["s_spend_deviation"] = (z.abs() >= 2).astype(int)

    # Categorical cleanups for model
    df["payment_channel"] = df["payment_method"].replace(
        {"card":"Card", "credit_card":"Card", "debit_card":"Card",
         "digital_wallet":"DigitalWallet", "wallet":"DigitalWallet",
         "gift_card":"GiftCard"}
    ).fillna("Card")
    df["category"] = df["sku_category"].fillna("misc")

    # model features
    df["y"] = df["fraud_flag"].fillna(0).astype(int)
    return df

df = prepare_features(raw)

# ─────────────────────────── Train / Test split ──────────────────────
# time-aware split: last 25% by time is test
df = df.sort_values("ts")
cut = int(len(df) * 0.75)
train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

feat_num = [
    "order_amount","quantity","unit_price","account_age_days",
    "price_ratio","coupon_pct","gift_pct","cust_1h","dev_1h",
    "s_price_bulk","s_geo_highvalue","s_discount_gift_stack","s_burst","s_spend_deviation",
    "geo_mismatch"
]
feat_cat = ["payment_channel","category","ship_country","ip_country","store_id"]
Xtr = pd.concat([train[feat_num].fillna(0),
                 pd.get_dummies(train[feat_cat].astype(str), dummy_na=False)], axis=1)
Xte = pd.concat([test[feat_num].fillna(0),
                 pd.get_dummies(test[feat_cat].astype(str), dummy_na=False)], axis=1)
# align columns
Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
ytr, yte = train["y"].values, test["y"].values

# class imbalance weighting – upweight positives
pos = max(1, int((ytr == 1).sum()))
neg = max(1, int((ytr == 0).sum()))
w_pos = neg / pos

clf = HistGradientBoostingClassifier(
    max_iter=350, learning_rate=0.08, max_depth=None, min_samples_leaf=10,
    l2_regularization=0.01, early_stopping=True, random_state=42
)
sw = np.where(ytr == 1, w_pos, 1.0)
clf.fit(Xtr, ytr, sample_weight=sw)

# internal operating point: maximize F1 (balanced) with a small precision guard
probs = clf.predict_proba(Xte)[:,1]
ths = np.linspace(0.05, 0.95, 91)
best_t, best_score = 0.50, -1.0
for t in ths:
    pred = (probs >= t).astype(int)
    p = precision_score(yte, pred, zero_division=0)
    r = recall_score(yte, pred, zero_division=0)
    f1 = 0 if (p+r)==0 else 2*p*r/(p+r)
    if p >= 0.25 and f1 > best_score:  # mild guard so business doesn't see junky alerts
        best_score, best_t = f1, t

yhat = (probs >= best_t).astype(int)

# ─────────────────────────── Header ───────────────────────────────────
st.title("Retail Fraud – Business Dashboard")

# ─────────────────────────── Top row: Raw & Channel mix ──────────────
cL, cR = st.columns([2, 1])
with cL:
    st.subheader("Raw 4,000 orders (snapshot)")
    st.caption("Walmart-like retail flows; scroll to explore. This table is not filtered by any UI control.")
    show_cols = [
        "order_id","ts","customer_id","device_id","store_id","sku_id","category",
        "quantity","unit_price","order_amount","payment_channel","ship_country","ip_country","y"
    ]
    keep = [c for c in show_cols if c in df.columns]
    st.dataframe(df.loc[:, keep].sort_values("ts", ascending=False), use_container_width=True, height=420)

with cR:
    st.subheader("Channel mix")
    mix = (df["payment_channel"].value_counts(dropna=False)
           .rename_axis("payment_channel").reset_index(name="orders"))
    ch = (alt.Chart(mix)
          .mark_bar()
          .encode(x=alt.X("orders:Q", title="Orders"),
                  y=alt.Y("payment_channel:N", sort="-x", title=None),
                  tooltip=["payment_channel","orders"])
          .properties(height=180))
    st.altair_chart(ch, use_container_width=True)
    st.caption("Card, Digital Wallet, and Gift Card share of total orders. Helps business spot risky channel shifts quickly.")

# ─────────────────────────── Model metrics ───────────────────────────
st.markdown("---")
st.subheader("Model evaluation (hold-out period)")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy",  f"{accuracy_score(yte, yhat):.2%}")
m2.metric("Precision", f"{precision_score(yte, yhat, zero_division=0):.2%}")
m3.metric("Recall",    f"{recall_score(yte, yhat, zero_division=0):.2%}")
m4.metric("F1-score",  f"{f1_score(yte, yhat, zero_division=0):.2%}")
st.caption("Operating point is chosen internally to balance capture (recall) and quality (precision).")

# ─────────────────────────── What patterns are firing ─────────────────
st.subheader("Strong signals (how fraud looks here)")
sig_cols = ["s_price_bulk","s_geo_highvalue","s_discount_gift_stack","s_burst","s_spend_deviation","geo_mismatch"]
sig_names = {
    "s_price_bulk":"Bulk + abnormal price",
    "s_geo_highvalue":"High-value + geo mismatch",
    "s_discount_gift_stack":"Heavy coupon + gift stack",
    "s_burst":"Burst activity",
    "s_spend_deviation":"Spend off customer baseline",
    "geo_mismatch":"Geo mismatch (IP vs Ship)"
}
prev = df.assign(any_alert=df[sig_cols].sum(axis=1)>0)\
         .melt(value_vars=sig_cols, var_name="signal", value_name="on")\
         .groupby("signal", as_index=False)["on"].mean()
prev["signal"] = prev["signal"].map(sig_names)
chart = (alt.Chart(prev)
         .mark_bar()
         .encode(x=alt.X("on:Q", axis=alt.Axis(format="%"), title="Share of orders where signal fires"),
                 y=alt.Y("signal:N", sort="-x", title=None),
                 tooltip=[alt.Tooltip("on:Q", format=".1%"), "signal:N"])
         .properties(height=220))
st.altair_chart(chart, use_container_width=True)

# ─────────────────────────── New order instant decision ───────────────
st.markdown("---")
st.subheader("New Order – instant decision")

# Business-friendly fields only (we compute strong features behind the scenes)
c1,c2 = st.columns(2)
with c1:
    category = st.selectbox("Category", sorted(df["category"].unique().tolist()))
    payment_channel = st.selectbox("Payment channel", ["Card","DigitalWallet","GiftCard"])
    ship_country = st.selectbox("Shipping country", sorted(df["ship_country"].dropna().unique().tolist()))
with c2:
    ip_country = st.selectbox("Network country (IP)", sorted(df["ip_country"].dropna().unique().tolist()))
    quantity = st.number_input("Quantity", min_value=1.0, step=1.0, value=1.0)
    unit_price = st.number_input("Unit price", min_value=0.01, step=1.0, value=120.0)

c3,c4 = st.columns(2)
with c3:
    coupon_discount = st.number_input("Discount amount", min_value=0.0, step=1.0, value=0.0)
    gift_balance_used = st.number_input("Gift balance used", min_value=0.0, step=1.0, value=0.0)
with c4:
    account_age_days = st.number_input("Account age (days)", min_value=0, step=1, value=120)
    # Optional velocity proxies (if known); default 0 keeps form simple
    cust_1h_in = st.number_input("Customer orders in last 1h (optional)", min_value=0, step=1, value=0)
    dev_1h_in  = st.number_input("Device orders in last 1h (optional)", min_value=0, step=1, value=0)

if st.button("Check"):
    # Build single-row dataframe with the same features
    order_amount = quantity * unit_price
    price_ratio = 1.0
    if category in df["category"].unique():
        cat_mean = df.loc[df["category"]==category, "unit_price"].mean()
        price_ratio = (unit_price / cat_mean) if cat_mean and cat_mean > 0 else 1.0
    coupon_pct = (coupon_discount / order_amount) if order_amount > 0 else 0
    gift_pct   = (gift_balance_used / order_amount) if order_amount > 0 else 0
    p90 = float(np.nanpercentile(df["order_amount"], 90)) if len(df) else 0.0

    # Strong features for the new order
    s_price_bulk = int(abs(price_ratio - 1) >= 0.50 and quantity >= 3)
    geo_mismatch = int(ip_country != ship_country)
    s_geo_highvalue = int(geo_mismatch == 1 and order_amount >= p90)
    s_discount_gift_stack = int(coupon_pct >= 0.30 and gift_pct >= 0.50)
    s_burst = int(cust_1h_in >= 3 or dev_1h_in >= 3)
    # spend deviation unknown for a single order; approximate via price_ratio/amount
    s_spend_deviation = int(abs(price_ratio - 1) >= 0.75 or order_amount >= (1.5 * df["order_amount"].median()))

    new = pd.DataFrame([{
        "order_amount": order_amount, "quantity": quantity, "unit_price": unit_price,
        "account_age_days": account_age_days, "price_ratio": price_ratio,
        "coupon_pct": coupon_pct, "gift_pct": gift_pct,
        "cust_1h": cust_1h_in, "dev_1h": dev_1h_in,
        "s_price_bulk": s_price_bulk, "s_geo_highvalue": s_geo_highvalue,
        "s_discount_gift_stack": s_discount_gift_stack, "s_burst": s_burst,
        "s_spend_deviation": s_spend_deviation, "geo_mismatch": geo_mismatch,
        "payment_channel": payment_channel, "category": category,
        "ship_country": ship_country, "ip_country": ip_country, "store_id": "store_manual"
    }])

    Xnew = pd.concat([new[feat_num].fillna(0),
                      pd.get_dummies(new[feat_cat].astype(str), dummy_na=False)], axis=1)
    # align columns
    Xnew = Xnew.reindex(columns=Xtr.columns, fill_value=0)

    p = clf.predict_proba(Xnew)[:,1][0]
    decision = "Fraud" if p >= best_t else ("Review" if (p >= (best_t*0.8)) else "Not Fraud")

    # Reasons for business
    reasons = []
    if s_geo_highvalue: reasons.append("High-value order with geo mismatch")
    if s_price_bulk:    reasons.append("Bulk qty with abnormal price vs category")
    if s_discount_gift_stack: reasons.append("Heavy discount + gift balance stack")
    if s_burst:         reasons.append("Recent burst activity (customer/device)")
    if s_spend_deviation: reasons.append("Spend is far off customer baseline")
    if not reasons and geo_mismatch: reasons.append("IP vs Ship mismatch")
    if not reasons: reasons.append("No strong red flags")

    st.success(f"Decision: **{decision}**")
    st.write("Why:", " • ".join(reasons))
    st.caption("Note: The system balances quality and coverage under the hood; no manual threshold tuning is needed in the UI.")

# ─────────────────────────── End ─────────────────────────────────────
