# app.py — Unified ML-Driven Fraud Detection for Retail POS (Signals-Trained)
# Run: streamlit run app.py

import numpy as np, pandas as pd, streamlit as st, altair as alt, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve

warnings.filterwarnings("ignore")
np.random.seed(42)
st.set_page_config("Unified ML-Driven Fraud Detection for Retail POS", layout="wide")
alt.renderers.set_embed_options(actions=False)

# ───────────────────────── 1) 4,000 retail transactions (USA only) ─────────────────────────
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
    "on_hand"       : np.random.randint(10, 200, N)
})
df["payment_method"] = np.where(
    df["channel"].eq("Online"),
    np.random.choice(["credit_card","debit_card","apple_pay","paypal"], N, p=[0.45,0.22,0.13,0.20]),
    np.random.choice(["credit_card","debit_card","apple_pay","gift_card"], N, p=[0.52,0.25,0.13,0.10]),
)
df["shipping_state"]   = np.where(df["channel"].eq("In-Store"), df["store_state"], np.random.choice(US, N))
df["billing_state"]    = np.random.choice(US, N)
df["express_shipping"] = np.where(df["channel"].eq("Online"), np.random.choice([0,1], N, p=[0.7,0.3]), 0)
df["log_amount"]       = np.log1p(df["order_amount"])

# ───────────────────────── 2) Signals (features we will TRAIN on) ─────────────────────────
# Business-friendly wording is used later in the dashboard.
df["addr_mismatch"]   = (df["shipping_state"]!=df["billing_state"]).astype(int)            # Address mismatch (billing ≠ shipping)
df["payment_risky"]   = df["payment_method"].isin(["paypal","gift_card"]).astype(int)       # Payment risky (PayPal / gift card)
grp = df.groupby("sku_category")["order_amount"]
p90 = grp.transform(lambda s: s.quantile(0.90))
mu  = grp.transform("mean")
sd  = grp.transform("std").replace(0,1.0)
z   = (df["order_amount"]-mu)/sd
df["high_amount_cat"] = (df["order_amount"]>=p90).astype(int)                               # High order amount (unusual vs category)
df["price_high_anom"] = (z>=2).astype(int)                                                  # High price anomaly (z≥2 vs category)
df["price_low_anom"]  = (z<=-2).astype(int)                                                 # Low price anomaly (z≤−2 vs category)
df["oversell_flag"]   = (df["quantity"]>0.6*df["on_hand"]).astype(int)                      # Oversell flag (qty > 60% of stock)
df["hoarding_flag"]   = (df["quantity"]>=4).astype(int)                                     # Hoarding flag (bulk buying)
df["return_whiplash"] = (df["returns_30d"]>=2).astype(int)                                  # Return whiplash (frequent returns)
df["express_high_amt"]= ((df["express_shipping"]==1)&(df["high_amount_cat"]==1)).astype(int)# Express + high value (rush & big order)

signal_features = [
    "payment_risky","addr_mismatch","high_amount_cat","price_high_anom","price_low_anom",
    "oversell_flag","hoarding_flag","return_whiplash","express_high_amt","log_amount"
]

# ───────────────────────── 3) Ground truth (fraud_flag) — realistic, not perfectly separable ─────────────────────────
# We build the probability of fraud as a logistic function of signals plus:
#  - a hidden unobserved factor (not available to the model),
#  - and noise. This prevents 99% accuracy while still letting signals be predictive.
def sigmoid(x): return 1/(1+np.exp(-x))
hidden = np.random.normal(0, 1.0, N)  # unobserved factor (customer riskiness / device risk etc.)
logit = (
    -1.0
    + 1.3*df["payment_risky"]
    + 1.1*df["addr_mismatch"]
    + 1.0*df["high_amount_cat"]
    + 0.8*df["express_high_amt"]
    + 0.7*df["price_high_anom"]
    + 0.5*df["return_whiplash"]
    + 0.4*df["oversell_flag"]
    + 0.2*df["hoarding_flag"]
    - 0.2*df["price_low_anom"]
    + 0.5*hidden                         # unobserved driver
    + np.random.normal(0, 0.7, N)        # extra noise
)
p_fraud = sigmoid(logit)
df["fraud_flag"] = (np.random.rand(N) < p_fraud).astype(int)

# ───────────────────────── 4) Train on signals (Logistic Regression to keep it honest) ─────────────────────────
X, y = df[signal_features], df["fraud_flag"]
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc",  StandardScaler()),
    ("lr",  LogisticRegression(max_iter=200, class_weight="balanced"))
])
pipe.fit(Xtr, ytr)

proba_all = pipe.predict_proba(X)[:,1]
proba_val = pipe.predict_proba(Xva)[:,1]

# Best business operating point: threshold that maximizes F1 on validation
prec, rec, thr = precision_recall_curve(yva, proba_val)
f1 = (2*prec*rec)/(prec+rec+1e-12)
best_thr = float(thr[np.argmax(f1)]) if len(thr) else 0.50
TH = st.sidebar.slider("Decision threshold", 0.00, 1.00, float(round(best_thr,2)), 0.01)

flags  = (proba_all >= TH).astype(int)
alerts = int(flags.sum()); rate = alerts/len(df)

# ───────────────────────── 5) Header & KPIs ─────────────────────────
st.title("Unified ML-Driven Fraud Detection for Retail POS")
st.caption("Scores each order for fraud risk using learned patterns, flags high-risk orders, and shows which signals triggered the alert.")

c1,c2,c3 = st.columns(3)
c1.metric("Transactions", f"{len(df):,}")
c2.metric("Suspicious orders", f"{alerts:,}")
c3.metric("Alert rate", f"{rate:.0%}")

# ───────────────────────── 6) Retail transactions (clean) ─────────────────────────
st.markdown("### Retail transactions")
tx_cols = ["order_id","channel","store_id","store_state","sku_category",
           "order_amount","quantity","payment_method","shipping_state","billing_state",
           "express_shipping","returns_30d"]
st.dataframe(df[tx_cols], use_container_width=True, height=420)

# ───────────────────────── 7) Model evaluation (validation set, chosen threshold) ─────────────────────────
st.markdown("### Model evaluation")
yhat = (proba_val>=TH).astype(int)
m1,m2,m3,m4 = st.columns(4)
m1.metric("Accuracy",  f"{accuracy_score(yva,yhat):.0%}")
m2.metric("Precision", f"{precision_score(yva,yhat,zero_division=0):.0%}")
m3.metric("Recall",    f"{recall_score(yva,yhat,zero_division=0):.0%}")
m4.metric("F1-score",  f"{f1_score(yva,yhat,zero_division=0):.0%}")

# ───────────────────────── 8) Strong signals driving alerts (counts only, simple wording) ─────────────────────────
st.markdown("### Strong signals driving alerts")
rename = {
    "addr_mismatch"  : "Address mismatch (billing ≠ shipping)",
    "payment_risky"  : "Payment risky (PayPal / gift card)",
    "high_amount_cat": "High order amount (unusual vs category)",
    "price_high_anom": "High price anomaly (z≥2 vs category)",
    "price_low_anom" : "Low price anomaly (z≤−2 vs category)",
    "oversell_flag"  : "Oversell flag (qty > 60% of stock)",
    "hoarding_flag"  : "Hoarding flag (bulk buying)",
    "return_whiplash": "Return whiplash (frequent returns)",
    "express_high_amt":"Express + high value (rush & big order)"
}
rows=[]
for c in ["addr_mismatch","payment_risky","high_amount_cat","price_high_anom",
          "price_low_anom","oversell_flag","hoarding_flag","return_whiplash","express_high_amt"]:
    rows.append([rename[c], int(df.loc[flags==1, c].sum()), int(df.loc[flags==0, c].sum())])
sig_tbl = pd.DataFrame(rows, columns=[
    "Signal (what it checks)",
    "Flagged — orders with this signal",
    "Cleared — orders with this signal"
]).sort_values("Flagged — orders with this signal", ascending=False)
st.dataframe(sig_tbl, use_container_width=True, height=320)

# ───────────────────────── 9) ML-detected suspicious orders (scores only here) ─────────────────────────
st.markdown("### ML-detected suspicious orders")
susp = df.loc[flags==1, tx_cols].copy()
susp["fraud_score"] = np.round(proba_all[flags==1], 3)
for c in rename:
    susp[rename[c]] = df.loc[flags==1, c].values
show_cols = tx_cols + ["fraud_score"] + list(rename.values())
st.dataframe(susp.sort_values("fraud_score", ascending=False)[show_cols].head(300),
             use_container_width=True, height=420)

# ───────────────────────── 10) Instant decision (score a new order using the signals-trained model) ─────────────────────────
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
    # recompute signals for the single order
    cp = df[df["sku_category"]==catg]
    p90 = float(cp["order_amount"].quantile(0.90)) if not cp.empty else 1000.0
    mu  = float(cp["order_amount"].mean()) if not cp.empty else 500.0
    sd  = max(float(cp["order_amount"].std()), 1.0) if not cp.empty else 100.0
    z   = (amt-mu)/sd
    r["addr_mismatch"]   = int(ship!=bill)
    r["payment_risky"]   = int(paym in ["paypal","gift_card"])
    r["high_amount_cat"] = int(amt>=p90)
    r["price_high_anom"] = int(z>=2); r["price_low_anom"]=int(z<=-2)
    r["oversell_flag"]   = 0 if expr==1 else int(qty>60)  # not applicable here; keep zero
    r["hoarding_flag"]   = int(qty>=4)
    r["return_whiplash"] = 0
    r["express_high_amt"]= int((expr==1) and r["high_amount_cat"].iloc[0]==1)
    r["log_amount"]      = np.log1p(amt)
    score = float(pipe.predict_proba(r[signal_features])[:,1])
    on = [rename[k] for k in rename if r.get(k, pd.Series([0])).iloc[0]==1]
    return score, on

if st.button("Score order"):
    score, on = score_one()
    st.success(f"Decision: {'FLAGGED' if score>=TH else 'CLEARED'}  •  Fraud score ≈ {score:.2f}  •  Threshold {TH:.2f}")
    st.write("Signals: " + (", ".join(on) if on else "none"))

# ───────────────────────── 11) Fraud score distribution ─────────────────────────
st.markdown("### Fraud score distribution")
st.altair_chart(
    alt.Chart(pd.DataFrame({"score":proba_all})).mark_bar().encode(
        x=alt.X("score:Q", bin=alt.Bin(maxbins=40), title="Fraud score"),
        y=alt.Y("count():Q", title="Orders")
    ).properties(height=220),
    use_container_width=True
)
