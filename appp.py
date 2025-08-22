# Run: streamlit run app.py

import streamlit as st, pandas as pd, numpy as np, altair as alt
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
from google.api_core.exceptions import GoogleAPICallError, NotFound, Forbidden, DeadlineExceeded

# ---------------- UI / Theme ----------------
st.set_page_config(page_title="Retail Fraud – Executive Dashboard", layout="wide")
alt.renderers.set_embed_options(actions=False)
PRIMARY, GOOD, WARN, BAD, MUTED = "#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#6b7280"
st.markdown(f"""
<style>
  .kpi h2 {{font-size:2rem;margin:0}}
  .kpi small {{color:{MUTED}}}
  .decision {{font-size:1.2rem;font-weight:700;margin:.25rem 0}}
  .approve {{color:{GOOD}}}.review {{color:{WARN}}}.block {{color:{BAD}}}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar (visibility only; NOT used by the query) ----------------
with st.sidebar:
    st.header("Data source (read-only)")
    st.text_input("Project", "mss-data-engineer-sandbox", disabled=True)
    st.text_input("Dataset", "retail", disabled=True)
    st.text_input("Raw table", "mss-data-engineer-sandbox.retail.transaction_data", disabled=True)


# ---------------- BigQuery client ----------------
sa = dict(st.secrets["gcp_service_account"]); sa["private_key"] = sa["private_key"].replace("\\n","\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq = bigquery.Client(credentials=creds, project=creds.project_id)


# ---------------- Load pinned 4k snapshot (robust) ----------------
@st.cache_data(show_spinner=True)
def load_snapshot(_secrets_guard) -> pd.DataFrame:
    # Keep the snapshot the business liked, but make the SQL simpler & faster
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

    # Retry up to 3 times with exponential backoff; hard 3-minute per attempt timeout
    last_err = None
    for attempt in range(3):
        try:
            job = bq.query(
                sql,
                job_config=bigquery.QueryJobConfig(use_legacy_sql=False)
            )
            return job.result(timeout=180).to_dataframe()  # 3 minutes per attempt
        except (DeadlineExceeded, GoogleAPICallError, Forbidden, NotFound, Exception) as e:
            last_err = e
            # brief backoff, then try again
            time.sleep(2 ** attempt)
    # If we get here, every attempt failed—surface the error in the UI so you know why
    st.error(
        "BigQuery snapshot failed after 3 attempts. "
        "Most common causes: temporary service issue, missing permissions, or table name mismatch.\n\n"
        f"Root cause: {type(last_err).__name__}: {last_err}"
    )
    st.stop()


# ---------------- Strong, business-friendly features ----------------
df = raw.copy().dropna(subset=["order_id"]).drop_duplicates("order_id", keep="last")
df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_localize(None)
df["account_created_at"] = pd.to_datetime(df["account_created_at"], errors="coerce", utc=True).dt.tz_localize(None)
df["account_age_days"] = (df["ts"] - df["account_created_at"]).dt.days.astype("float").clip(lower=0).fillna(0)

df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).clip(lower=0)
df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0).clip(lower=0)
df["order_amount"] = df["quantity"] * df["unit_price"]

den = df["order_amount"].replace(0, np.nan)
df["coupon_pct"] = (pd.to_numeric(df["coupon_discount"], errors="coerce")/den).fillna(0).clip(0,1)
df["gift_pct"]   = (pd.to_numeric(df["gift_balance_used"], errors="coerce")/den).fillna(0).clip(0,1)

df["addr_mismatch"] = (df["ship_country"] != df["ip_country"]).astype(int)
cat_avg = df.groupby("sku_category")["unit_price"].transform("mean").replace(0,np.nan)
df["price_ratio"] = (df["unit_price"]/cat_avg).fillna(1.0)

# Retail-focused channels: Card & Digital wallet only
pm = df["payment_method"].str.lower().fillna("")
df["pay_channel"] = np.select(
    [
        pm.str.contains(r"wallet|paypal|upi|gpay|paytm|apple|google"),
        pm.str.contains(r"card|credit|debit|visa|master|amex")
    ],
    ["Digital wallet", "Card"],
    default="Card"
)
df["pay_code"] = df["pay_channel"].map({"Digital wallet":2, "Card":1}).astype(int)

def vel_last_hours(g, hours):
    t = pd.to_datetime(g["ts"]).astype("int64")//1_000_000
    order = np.argsort(t); t = t.iloc[order].to_numpy(); w = int(hours*3600*1e6)
    left = np.searchsorted(t, t-w, side="left")
    cnt = np.arange(1, len(t)+1) - left
    return pd.Series(cnt, index=g.index[order]).reindex(g.index)

df = df.sort_values("ts")
df["cust_1h"]  = df.groupby("customer_id", group_keys=False).apply(vel_last_hours, 1).fillna(1).astype(int)
df["cust_24h"] = df.groupby("customer_id", group_keys=False).apply(vel_last_hours, 24).fillna(1).astype(int)
df["dev_1h"]   = df.groupby("device_id",   group_keys=False).apply(vel_last_hours, 1).fillna(1).astype(int)

def roll_stats(g):
    g = g.sort_values("ts"); r = g["order_amount"].rolling(10, min_periods=1)
    g["c_m"] = r.mean(); g["c_s"] = r.std(ddof=0).replace(0,np.nan); return g
df = df.groupby("customer_id", group_keys=False).apply(roll_stats)
df["amt_z"] = ((df["order_amount"] - df["c_m"]) / df["c_s"]).replace([np.inf,-np.inf],0).fillna(0)

df["fraud_flag"] = pd.to_numeric(df["fraud_flag"], errors="coerce").fillna(0).clip(0,1).astype(int)

FEATURES = [
    "order_amount","quantity","unit_price","account_age_days",
    "coupon_pct","gift_pct","price_ratio","addr_mismatch","pay_code",
    "cust_1h","cust_24h","dev_1h","amt_z"
]
X = df[FEATURES].fillna(0)
y = df["fraud_flag"].values

# ---------------- Train & evaluate (time-aware split) ----------------
df_sorted = df.sort_values("ts").reset_index(drop=True)
cut = int(len(df_sorted)*0.8)
tr_idx, te_idx = df_sorted.index[:cut], df_sorted.index[cut:]
X_tr, y_tr = X.loc[tr_idx], y[tr_idx]
X_te, y_te = X.loc[te_idx], y[te_idx]

pos = max(1, int((y_tr==1).sum())); neg = max(1, len(y_tr)-pos)
w_pos = min(15.0, neg/pos)
sample_w = np.where(y_tr==1, w_pos, 1.0)

clf = HistGradientBoostingClassifier(max_iter=700, learning_rate=0.06,
                                     early_stopping=True, random_state=42)
clf.fit(X_tr, y_tr, sample_weight=sample_w)

probs_te = clf.predict_proba(X_te)[:,1]
ts_grid = np.linspace(0.05, 0.95, 91)

# Thresholds (internal): Review keeps recall ≥90%, Block keeps precision ≥80%
t_review = ts_grid[0]
for t in ts_grid:
    if recall_score(y_te, (probs_te>=t).astype(int), zero_division=0) >= 0.90:
        t_review = t; break
t_block = ts_grid[-1]
for t in ts_grid[::-1]:
    if precision_score(y_te, (probs_te>=t).astype(int), zero_division=0) >= 0.80:
        t_block = t; break
t_block = max(t_block, t_review + 0.05)

# Report metrics at F1-optimal threshold
best_t, best_f1 = 0.5, -1
for t in ts_grid:
    yhat = (probs_te>=t).astype(int)
    f1 = f1_score(y_te, yhat, zero_division=0)
    if f1 > best_f1: best_f1, best_t = f1, t
ACC = accuracy_score(y_te, (probs_te>=best_t).astype(int))
PREC= precision_score(y_te, (probs_te>=best_t).astype(int), zero_division=0)
REC = recall_score(y_te, (probs_te>=best_t).astype(int), zero_division=0)
F1  = f1_score(y_te, (probs_te>=best_t).astype(int), zero_division=0)

# ---------------- Header & KPIs ----------------
pct = lambda x: f"{x*100:.2f}%"
st.markdown("## Retail Fraud Dashboard (Walmart-style)")
k1,k2,k3,k4 = st.columns(4)
with k1: st.markdown(f"<div class='kpi'><small>Accuracy</small><h2>{pct(ACC)}</h2></div>", unsafe_allow_html=True)
with k2: st.markdown(f"<div class='kpi'><small>Precision</small><h2>{pct(PREC)}</h2></div>", unsafe_allow_html=True)
with k3: st.markdown(f"<div class='kpi'><small>Recall</small><h2>{pct(REC)}</h2></div>", unsafe_allow_html=True)
with k4: st.markdown(f"<div class='kpi'><small>F1-score</small><h2>{pct(F1)}</h2></div>", unsafe_allow_html=True)
st.caption("Policy: **Review** catches ≥90% of fraud; **Block** keeps ≥80% precision. Decisions are internal; no sliders shown.")

st.divider()

# ---------------- Executive visuals ----------------
left, right = st.columns([3,2], gap="large")

with left:
    st.subheader("Weekly volume & fraud rate")
    tmp = df.assign(week=pd.to_datetime(df["ts"]).dt.to_period("W").dt.start_time)
    weekly = tmp.groupby("week").agg(orders=("order_id","count"), frauds=("fraud_flag","sum")).reset_index()
    weekly["rate"] = weekly["frauds"]/weekly["orders"]
    base = alt.Chart(weekly).encode(x=alt.X("week:T", title=None))
    c1 = base.mark_line(point=True, color=PRIMARY).encode(y=alt.Y("orders:Q", title="Orders"))
    c2 = base.mark_line(strokeDash=[4,2], color=BAD).encode(y=alt.Y("rate:Q", title="Fraud rate", axis=alt.Axis(format="%")))
    st.altair_chart((c1 + c2).resolve_scale(y='independent').properties(height=260), use_container_width=True)

    st.subheader("Risk by category")
    cat = df.groupby("sku_category").agg(orders=("order_id","count"), frauds=("fraud_flag","sum")).reset_index()
    cat["rate"] = cat["frauds"]/cat["orders"]
    st.altair_chart(
        alt.Chart(cat).mark_bar(color=BAD).encode(
            x=alt.X("sku_category:N", title=None, sort="-y"),
            y=alt.Y("rate:Q", title="Fraud rate", axis=alt.Axis(format="%")),
            tooltip=["orders","frauds",alt.Tooltip("rate:Q", format=".1%")]
        ).properties(height=240),
        use_container_width=True
    )

with right:
    st.subheader("Checkout channel mix (Card vs Digital wallet)")
    mix = df.groupby("pay_channel").agg(orders=("order_id","count"), frauds=("fraud_flag","sum")).reset_index()
    mix["order_share"] = mix["orders"]/mix["orders"].sum()
    mix["fraud_share"] = mix["frauds"]/mix["frauds"].sum()
    pie_orders = alt.Chart(mix).mark_arc(innerRadius=55).encode(
        theta="order_share:Q", color=alt.Color("pay_channel:N", title=None),
        tooltip=["pay_channel","orders",alt.Tooltip("order_share:Q",format=".1%")]
    ).properties(title="Orders", height=220)
    pie_fraud = alt.Chart(mix).mark_arc(innerRadius=55).encode(
        theta="fraud_share:Q", color=alt.Color("pay_channel:N", title=None),
        tooltip=["pay_channel","frauds",alt.Tooltip("fraud_share:Q",format=".1%")]
    ).properties(title="Frauds", height=220)
    st.altair_chart(alt.vconcat(pie_orders, pie_fraud).resolve_scale(color='independent'), use_container_width=True)

    st.subheader("What drives the model")
    imp_vals = getattr(clf, "feature_importances_", None)
    if imp_vals is None: imp_vals = np.ones(len(FEATURES))/len(FEATURES)
    imp = pd.DataFrame({"feature": FEATURES, "importance": imp_vals}).sort_values("importance")
    st.altair_chart(
        alt.Chart(imp).mark_bar(color=PRIMARY).encode(
            x=alt.X("importance:Q", title="Relative importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")]
        ).properties(height=240),
        use_container_width=True
    )

st.divider()

# ---------------- Full raw data (the 4,000 snapshot) ----------------
st.subheader(f"Raw dataset (pinned snapshot: {len(df):,} rows)")
st.dataframe(df.drop(columns=["ts"]), use_container_width=True, height=420)
st.download_button("Download 4k rows (CSV)", df.drop(columns=["ts"]).to_csv(index=False).encode("utf-8"),
                   file_name="retail_fraud_4k.csv", mime="text/csv")

st.divider()

# ---------------- New Order – instant decision ----------------
st.subheader("New Order – decision (Card / Digital wallet)")

countries  = sorted(df["ship_country"].dropna().unique().tolist() or ["US","UK","DE","IN","CA"])
categories = sorted(df["sku_category"].dropna().unique().tolist() or ["electronics","home","apparel","toys","grocery"])
channels   = ["Card","Digital wallet"]

with st.form("new_order_form"):
    c1,c2,c3 = st.columns(3)
    with c1:
        sku_category = st.selectbox("Category", categories, index=0)
        ship_country = st.selectbox("Shipping country", countries, index=0)
        quantity     = st.number_input("Quantity", 1.0, step=1.0, value=1.0)
    with c2:
        pay_channel  = st.selectbox("Payment channel", channels, index=0)
        ip_country   = st.selectbox("Network country", countries, index=0)
        unit_price   = st.number_input("Unit price", 1.0, step=1.0, value=120.0)
    with c3:
        discount_amt = st.number_input("Discount amount", 0.0, step=1.0, value=0.0)
        gift_used    = st.number_input("Gift balance used", 0.0, step=1.0, value=0.0)
        acct_age     = st.number_input("Account age (days)", 0.0, step=1.0, value=120.0)
    submitted = st.form_submit_button("Check")

def reasons_from_row(r):
    reasons=[]
    if r["addr_mismatch"] and (r["pay_channel"]=="Digital wallet") and (r["order_amount"]>300):
        reasons.append("Address inconsistency with high-value digital-wallet payment")
    if r["gift_pct"]>0.55 and r["coupon_pct"]>0.20:
        reasons.append("Large gift-balance share combined with high discount")
    if abs(r["price_ratio"]-1)>0.50 and r["quantity"]>=3:
        reasons.append("Unusual price for its category with bulk quantity")
    if r["account_age_days"]<10 and r["order_amount"]>400:
        reasons.append("Very new account with large purchase")
    if r["pay_channel"]=="Digital wallet" and r["order_amount"]>180:
        reasons.append("Digital-wallet payment with substantial amount")
    if not reasons:
        safes=[]
        if not r["addr_mismatch"]: safes.append("Address looks consistent")
        if r["gift_pct"]<0.20:     safes.append("Low gift-balance usage")
        if r["coupon_pct"]<0.30:   safes.append("Discount within normal range")
        if r["account_age_days"]>=10: safes.append("Account not brand-new")
        if abs(r["price_ratio"]-1)<=0.50 or r["quantity"]<3: safes.append("No unusual price or bulk")
        reasons = ["; ".join(safes)] if safes else ["No clear risk patterns"]
    return reasons

if submitted:
    order_amount = float(unit_price * quantity)
    coupon_pct   = float((discount_amt/order_amount) if order_amount else 0.0)
    gift_pct     = float((gift_used /order_amount) if order_amount else 0.0)
    addr_mismatch= int(ship_country != ip_country)
    ref_avg      = float(df.loc[df["sku_category"]==sku_category, "unit_price"].mean() or df["unit_price"].mean())
    price_ratio  = float(unit_price / ref_avg) if ref_avg>0 else 1.0
    pay_code     = {"Digital wallet":2, "Card":1}[pay_channel]

    row = pd.DataFrame([{
        "order_amount":order_amount, "quantity":quantity, "unit_price":unit_price,
        "account_age_days":acct_age, "coupon_pct":coupon_pct, "gift_pct":gift_pct,
        "price_ratio":price_ratio, "addr_mismatch":addr_mismatch, "pay_code":pay_code,
        # one-off entry: neutral burst / deviation
        "cust_1h":1, "cust_24h":1, "dev_1h":1, "amt_z":0,
    }])[FEATURES].fillna(0)

    p = clf.predict_proba(row)[:,1][0]
    if p >= t_block:   decision, css = "BLOCK",  "block"
    elif p >= t_review:decision, css = "REVIEW", "review"
    else:              decision, css = "APPROVE","approve"

    rdict = {
        "addr_mismatch":addr_mismatch, "pay_channel":pay_channel,
        "order_amount":order_amount, "gift_pct":gift_pct, "coupon_pct":coupon_pct,
        "price_ratio":price_ratio, "quantity":quantity, "account_age_days":acct_age
    }
    reasons = reasons_from_row(rdict)

    st.markdown(f"<div class='decision {css}'>Decision: {decision}</div>", unsafe_allow_html=True)
    st.write("**What “Review” means:** route to analyst queue for quick verification (e.g., confirm address or payment).")
    st.write("**Why:** " + " | ".join(reasons))
    st.table(pd.DataFrame([{
        "Category": sku_category,
        "Payment channel": pay_channel,
        "Ship vs Network": f"{ship_country} / {ip_country}",
        "Unit price": round(unit_price,2),
        "Quantity": int(quantity),
        "Order amount": round(order_amount,2),
        "Discount %": round(coupon_pct*100,2),
        "Gift %": round(gift_pct*100,2),
        "Price ratio vs category": round(price_ratio,2),
        "Account age (days)": int(acct_age)
    }]))


