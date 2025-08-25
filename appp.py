# app.py — Unified ML-Driven Fraud Detection for Retail POS (signals-trained, tuned)
# Run: streamlit run app.py

import numpy as np, pandas as pd, streamlit as st, altair as alt, warnings
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve

warnings.filterwarnings("ignore")
np.random.seed(42)
st.set_page_config("Unified ML-Driven Fraud Detection for Retail POS", layout="wide")
alt.renderers.set_embed_options(actions=False)

# ───────────────────────── 1) Build 4,000 retail transactions (USA) ─────────────────────────
N = 4000
US = np.array(["CA","TX","NY","FL","IL","WA","GA","PA","OH","MI"])
stores = np.array(["US01","US02","US03","US04"])
store_state_pool = np.random.choice(US, size=len(stores), replace=False)
idx = np.random.choice(len(stores), N)

df = pd.DataFrame({
    "order_id"      : np.arange(1, N+1),
    "channel"       : np.random.choice(["Online","In-Store"], N, p=[0.70,0.30]),
    "store_id"      : stores[idx],
    "store_state"   : store_state_pool[idx],
    "sku_category"  : np.random.choice(["Electronics","Apparel","Grocery","Beauty","Home"], N),
    "order_amount"  : np.round(np.random.lognormal(5.45, 0.8, N), 2),
    "quantity"      : np.random.randint(1, 6, N),
    "returns_30d"   : np.random.poisson(0.2, N),
    "on_hand"       : np.random.randint(10, 250, N)
})
df["payment_method"] = np.where(
    df["channel"].eq("Online"),
    np.random.choice(["credit_card","debit_card","apple_pay","paypal"], N, p=[0.45,0.22,0.13,0.20]),
    np.random.choice(["credit_card","debit_card","apple_pay","gift_card"], N, p=[0.52,0.25,0.13,0.10]),
)
df["shipping_state"]   = np.where(df["channel"].eq("In-Store"), df["store_state"], np.random.choice(US, N))
df["billing_state"]    = np.random.choice(US, N)
df["express_shipping"] = np.where(df["channel"].eq("Online"), np.random.choice([0,1], N, p=[0.7,0.3]), 0)

# ───────────────────────── 2) Light cleaning (caps) + helper ─────────────────────────
df["order_amount"] = df["order_amount"].clip(lower=1, upper=np.percentile(df["order_amount"], 99))
df["quantity"]     = df["quantity"].clip(1, 50)
df["returns_30d"]  = df["returns_30d"].clip(0, 10)
df["on_hand"]      = df["on_hand"].clip(1, 500)
df["log_amount"]   = np.log1p(df["order_amount"])

# ───────────────────────── 3) Risk signals (these will be used to TRAIN) ─────────────────────────
# Address + payment
df["addr_mismatch"]   = (df["shipping_state"]!=df["billing_state"]).astype(int)               # billing ≠ shipping
df["payment_risky"]   = df["payment_method"].isin(["paypal","gift_card"]).astype(int)         # PayPal / gift card

# Category-relative amount + price anomalies
grp = df.groupby("sku_category")["order_amount"]
p90 = grp.transform(lambda s: s.quantile(0.90))
mu  = grp.transform("mean")
sd  = grp.transform("std").replace(0, 1.0)
z   = ((df["order_amount"]-mu)/sd).clip(-5,5)

df["high_amount_cat"] = (df["order_amount"]>=p90).astype(int)                                 # unusual vs category
df["price_high_anom"] = (z>=2).astype(int)                                                    # z≥2
df["price_low_anom"]  = (z<=-2).astype(int)                                                   # z≤−2

# Inventory / behavior
df["oversell_flag"]   = (df["quantity"]>0.60*df["on_hand"]).astype(int)                       # qty > 60% of stock
df["hoarding_flag"]   = (df["quantity"]>=4).astype(int)                                       # bulk buying
df["return_whiplash"] = (df["returns_30d"]>=2).astype(int)                                    # frequent returns
df["express_high_amt"]= ((df["express_shipping"]==1)&(df["high_amount_cat"]==1)).astype(int)  # rush & high value

signal_features = [
    "payment_risky","addr_mismatch","high_amount_cat","price_high_anom","price_low_anom",
    "oversell_flag","hoarding_flag","return_whiplash","express_high_amt","log_amount"
]

# ───────────────────────── 4) Ground truth: realistic (no 100% possible) ─────────────────────────
def sigmoid(x): return 1.0/(1.0+np.exp(-x))

# Unobserved driver + noise → keeps metrics realistic while rewarding signals
hidden = np.random.normal(0, 0.6, N)
logit = (
    -1.0
    + 1.2*df["payment_risky"]
    + 1.1*df["addr_mismatch"]
    + 1.0*df["high_amount_cat"]
    + 0.8*df["express_high_amt"]
    + 0.7*df["price_high_anom"]
    + 0.5*df["return_whiplash"]
    + 0.4*df["oversell_flag"]
    + 0.3*df["hoarding_flag"]
    - 0.2*df["price_low_anom"]
    + 0.6*hidden
    + np.random.normal(0, 0.45, N)
)
p_fraud = sigmoid(logit)
y_true = (np.random.rand(N) < p_fraud).astype(int)

# Label noise (3%) to prevent any chance of perfection
flip = np.random.rand(N) < 0.03
y_true = np.where(flip, 1 - y_true, y_true)
df["fraud_flag"] = y_true

# ───────────────────────── 5) Train classifier on signals (tuned RandomForest) ─────────────────────────
X, y = df[signal_features], df["fraud_flag"]
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(Xtr, ytr)

proba_all = rf.predict_proba(X)[:,1]
proba_val = rf.predict_proba(Xva)[:,1]

# Threshold chosen by max F1 (business can move the slider)
prec, rec, thr = precision_recall_curve(yva, proba_val)
f1_vals = (2*prec*rec)/(prec+rec+1e-12)
default_thr = float(thr[np.argmax(f1_vals)]) if len(thr) else 0.50

st.sidebar.header("Business control")
TH = st.sidebar.slider("Decision threshold", 0.00, 1.00, float(round(default_thr,2)), 0.01)
st.sidebar.caption("Default is tuned by ML (best F1). Slide to tighten/loosen control.")

flags  = (proba_all>=TH).astype(int)
alerts = int(flags.sum()); rate = alerts/len(df)

# ───────────────────────── 6) Header + KPIs ─────────────────────────
st.title("Unified ML-Driven Fraud Detection for Retail POS")
st.caption("Scores each order for fraud risk using learned patterns, flags high-risk orders, and shows which signals triggered the alert.")

k1,k2,k3 = st.columns(3)
k1.metric("Transactions", f"{len(df):,}")
k2.metric("Suspicious orders", f"{alerts:,}")
k3.metric("Alert rate", f"{rate:.0%}")

# ───────────────────────── 7) Retail transactions (first) ─────────────────────────
st.markdown("### Retail transactions")
tx_cols = ["order_id","channel","store_id","store_state","sku_category",
           "order_amount","quantity","payment_method","shipping_state","billing_state",
           "express_shipping","returns_30d"]
st.dataframe(df[tx_cols], use_container_width=True, height=420)

# ───────────────────────── 8) Fraud score distribution (second) ─────────────────────────
st.markdown("### Fraud score distribution")
st.caption("Distribution of ML-assigned fraud risk scores across all transactions.")
st.altair_chart(
    alt.Chart(pd.DataFrame({"score":proba_all})).mark_bar().encode(
        x=alt.X("score:Q", bin=alt.Bin(maxbins=40), title="Fraud score"),
        y=alt.Y("count():Q", title="Orders")
    ).properties(height=220),
    use_container_width=True
)

# ───────────────────────── 9) Model evaluation (third) ─────────────────────────
st.markdown("### Model evaluation")
yhat_val = (proba_val>=TH).astype(int)
c1,c2,c3,c4 = st.columns(4)
c1.metric("Accuracy",  f"{accuracy_score(yva,yhat_val):.0%}")
c2.metric("Precision", f"{precision_score(yva,yhat_val,zero_division=0):.0%}")
c3.metric("Recall",    f"{recall_score(yva,yhat_val,zero_division=0):.0%}")
c4.metric("F1-score",  f"{f1_score(yva,yhat_val,zero_division=0):.0%}")

# ───────────────────────── 10) Strong signals driving alerts (fourth) ─────────────────────────
st.markdown("### Strong signals driving alerts")
rename = {
    "addr_mismatch"  : "Address mismatch (billing ≠ shipping)",
    "payment_risky"  : "Payment risky (PayPal / gift card)",
    "high_amount_cat": "High order amount (unusual vs category)",
    "price_high_anom": "High price anomaly (z≥2)",
    "price_low_anom" : "Low price anomaly (z≤−2)",
    "oversell_flag"  : "Oversell flag (qty > 60% of stock)",
    "hoarding_flag"  : "Hoarding flag (bulk buying)",
    "return_whiplash": "Return whiplash (frequent returns)",
    "express_high_amt":"Express + high value (rush & big order)"
}
sig_cols = list(rename.keys())

rows=[]
for c in sig_cols:
    rows.append([
        rename[c],
        int(df.loc[flags==1, c].sum()),   # suspicious (flagged)
        int(df.loc[flags==0, c].sum())    # normal (cleared)
    ])
sig_tbl = (pd.DataFrame(rows, columns=[
    "Signal (what it checks)",
    "Suspicious orders (flagged) with this signal",
    "Normal orders (cleared) with this signal"
]).sort_values("Suspicious orders (flagged) with this signal", ascending=False))
st.dataframe(sig_tbl, use_container_width=True, height=320)
st.caption("Counts • Items at the top are the strongest drivers of alerts.")

# ───────────────────────── 11) ML-detected suspicious orders ─────────────────────────
st.markdown("### ML-detected suspicious orders")
susp = df.loc[flags==1, tx_cols].copy()
susp["fraud_score"] = np.round(proba_all[flags==1], 3)
for c in sig_cols:
    susp[rename[c]] = df.loc[flags==1, c].values
show_cols = tx_cols + ["fraud_score"] + [rename[c] for c in sig_cols]
st.dataframe(
    susp.sort_values("fraud_score", ascending=False)[show_cols].head(300),
    use_container_width=True, height=420
)

# ───────────────────────── 12) Instant decision (score a new order) ─────────────────────────
st.markdown("### Instant decision")
c1,c2,c3 = st.columns(3)
chan = c1.selectbox("Channel", ["Online","In-Store"])
catg = c2.selectbox("SKU category", ["Electronics","Apparel","Grocery","Beauty","Home"])
paym = c3.selectbox("Payment method", ["credit_card","debit_card","apple_pay","gift_card","paypal"])
c4,c5,c6 = st.columns(3)
amt  = c4.number_input("Order amount ($)", 1, 20000, 250)
qty  = c5.number_input("Quantity", 1, 50, 1)
expr = c6.selectbox("Express shipping", [0,1], index=0)
c7,c8 = st.columns(2)
ship = c7.selectbox("Shipping state", sorted(US))
bill = c8.selectbox("Billing state", sorted(US))

def score_one():
    r = pd.DataFrame([{
        "order_amount": amt, "quantity": int(qty), "express_shipping": int(expr),
        "shipping_state": ship, "billing_state": bill, "payment_method": paym,
        "sku_category": catg
    }])
    # compute signals for this one order using category stats
    cp = df[df["sku_category"]==catg]
    if cp.empty:
        mu, sd, p90 = 500.0, 120.0, 1000.0
    else:
        mu  = float(cp["order_amount"].mean())
        sd  = max(float(cp["order_amount"].std()), 1.0)
        p90 = float(cp["order_amount"].quantile(0.90))
    z = (amt-mu)/sd

    r["addr_mismatch"]   = int(ship!=bill)
    r["payment_risky"]   = int(paym in ["paypal","gift_card"])
    r["high_amount_cat"] = int(amt>=p90)
    r["price_high_anom"] = int(z>=2); r["price_low_anom"] = int(z<=-2)
    r["oversell_flag"]   = 0                       # not known in ad-hoc check
    r["hoarding_flag"]   = int(qty>=4)
    r["return_whiplash"] = 0
    r["express_high_amt"]= int((expr==1) and r["high_amount_cat"].iloc[0]==1)
    r["log_amount"]      = np.log1p(amt)

    score = float(rf.predict_proba(r[signal_features])[:,1])
    fired = [rename[c] for c in sig_cols if r[c].iloc[0]==1]
    return score, fired

if st.button("Score order"):
    score, fired = score_one()
    st.success(f"Decision: {'fraud' if score>=TH else 'not fraud'} • Fraud score ≈ {score:.2f} • Threshold {TH:.2f}")
    st.write("Signals: " + (", ".join(fired) if fired else "none"))
