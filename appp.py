# app.py — Unified ML-Driven Fraud Detection for Retail POS (USA)
# 4,000 US transactions • tuned model • clear business wording • counts-only signals
# Run: streamlit run app.py

import numpy as np, pandas as pd, streamlit as st, altair as alt, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")
np.random.seed(42)
st.set_page_config("Unified ML-Driven Fraud Detection for Retail POS (USA)", layout="wide")
alt.renderers.set_embed_options(actions=False)

# ───────────────────────── 1) Build 4,000 US transactions ─────────────────────────
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
    "order_amount"  : np.round(np.random.lognormal(5.5, 0.8, N), 2),
    "quantity"      : np.random.randint(1, 6, N),
    "returns_30d"   : np.random.poisson(0.2, N),
    "sku"           : np.random.randint(100000, 110000, N),
    "on_hand"       : np.random.randint(10, 200, N)
})
df["payment_method"] = np.where(
    df["channel"].eq("Online"),
    np.random.choice(["credit_card","debit_card","apple_pay","paypal"], N, p=[0.45,0.20,0.15,0.20]),
    np.random.choice(["credit_card","debit_card","apple_pay","gift_card"], N, p=[0.50,0.25,0.15,0.10]),
)
df["shipping_state"]  = np.where(df["channel"].eq("In-Store"), df["store_state"], np.random.choice(US, N))
df["billing_state"]   = np.random.choice(US, N)
df["express_shipping"]= np.where(df["channel"].eq("Online"), np.random.choice([0,1], N, p=[0.7,0.3]), 0)

# ───────────────────────── 2) Risk signals (business wording in brackets) ─────────────────────────
df["addr_mismatch"]   = (df["shipping_state"]!=df["billing_state"]).astype(int)          # Address mismatch (Billing ≠ Shipping)
df["payment_risky"]   = df["payment_method"].isin(["paypal","gift_card"]).astype(int)     # Payment risky (PayPal / Gift card)
grp = df.groupby("sku_category")["order_amount"]
p90 = grp.transform(lambda s: s.quantile(0.90))
mu  = grp.transform("mean")
sd  = grp.transform("std").replace(0,1.0)
z   = (df["order_amount"]-mu)/sd
df["high_amount_cat"] = (df["order_amount"]>=p90).astype(int)                             # High order amount (Unusual spend vs category)
df["price_high_anom"] = (z>=2).astype(int)                                                # High price anomaly (z≥2 vs category)
df["price_low_anom"]  = (z<=-2).astype(int)                                               # Low price anomaly (z≤−2 vs category)
df["oversell_flag"]   = (df["quantity"]>0.6*df["on_hand"]).astype(int)                    # Oversell flag (Qty > 60% of stock)
df["hoarding_flag"]   = (df["quantity"]>=4).astype(int)                                   # Hoarding flag (Bulk buying)
df["return_whiplash"] = (df["returns_30d"]>=2).astype(int)                                # Return whiplash (Frequent returns)
df["express_high_amt"]= ((df["express_shipping"]==1)&(df["high_amount_cat"]==1)).astype(int) # Express + high value
df["log_amount"]      = np.log1p(df["order_amount"])

# ───────────────────────── 3) Ground truth (separable, light noise) ─────────────────────────
# Score from signals + strong synergies → then label by threshold 0.5 (adds small Gaussian noise)
risk = (0.15*df["payment_risky"] + 0.12*df["addr_mismatch"] + 0.12*df["high_amount_cat"]
        + 0.08*df["express_high_amt"] + 0.06*df["price_high_anom"] + 0.05*df["return_whiplash"]
        + 0.04*df["oversell_flag"] + 0.03*df["hoarding_flag"] + 0.02*df["price_low_anom"] + 0.12)
risk += np.where(df["payment_risky"] & df["addr_mismatch"], 0.35, 0.0)
risk += np.where(df["payment_risky"] & df["high_amount_cat"] & df["express_high_amt"], 0.40, 0.0)
df["fraud_flag"] = ((risk + np.random.normal(0,0.03,N)) > 0.50).astype(int)

# ───────────────────────── 4) Model: tuned RandomForest + business threshold (max accuracy) ─────────────────────────
features = ["channel","sku_category","payment_method","shipping_state","billing_state",
            "quantity","order_amount","log_amount","express_shipping","returns_30d",
            "addr_mismatch","payment_risky","high_amount_cat","price_high_anom","price_low_anom",
            "oversell_flag","hoarding_flag","return_whiplash","express_high_amt","on_hand"]
num = ["quantity","order_amount","log_amount","express_shipping","returns_30d",
       "addr_mismatch","payment_risky","high_amount_cat","price_high_anom","price_low_anom",
       "oversell_flag","hoarding_flag","return_whiplash","express_high_amt","on_hand"]
cat = ["channel","sku_category","payment_method","shipping_state","billing_state"]

pre = ColumnTransformer([
    ("num", Pipeline([("imp",SimpleImputer(strategy="median")), ("sc",StandardScaler())]), num),
    ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                      ("oh",OneHotEncoder(handle_unknown="ignore"))]), cat)
])

rf = RandomForestClassifier(
    n_estimators=700, max_depth=12, min_samples_split=6, min_samples_leaf=3,
    max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=42
)
pipe = Pipeline([("pre",pre), ("rf",rf)])

X, y = df[features], df["fraud_flag"]
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
pipe.fit(Xtr, ytr)
proba_all = pipe.predict_proba(X)[:,1]
proba_val = pipe.predict_proba(Xva)[:,1]

# threshold that maximizes validation accuracy
grid = np.linspace(0.05,0.95,91)
accs = [accuracy_score(yva,(proba_val>=t).astype(int)) for t in grid]
best_thr = float(grid[int(np.argmax(accs))])

# single business control
TH = st.sidebar.slider("Decision threshold", 0.00, 1.00, float(round(best_thr,2)), 0.01)

# flags for all 4k at chosen threshold
flags = (proba_all >= TH).astype(int)
alerts = int(flags.sum()); alert_rate = alerts/len(df)

# ───────────────────────── 5) Page header (no duplicate KPIs) ─────────────────────────
st.title("Unified ML-Driven Fraud Detection for Retail POS (USA)")
k1,k2,k3 = st.columns(3)
k1.metric("Transactions", f"{len(df):,}")
k2.metric("Suspicious orders", f"{alerts:,}")
k3.metric("Alert rate", f"{alert_rate:.0%}")

# ───────────────────────── 6) Retail transactions (clean; no scores or flags) ─────────────────────────
st.markdown("### Retail transactions")
tx_cols = ["order_id","channel","store_id","store_state","sku_category",
           "order_amount","quantity","payment_method","shipping_state","billing_state",
           "express_shipping","returns_30d"]
st.dataframe(df[tx_cols], use_container_width=True, height=420)

# ───────────────────────── 7) Model evaluation (at chosen threshold) ─────────────────────────
st.markdown("### Model evaluation")
yhat = (proba_val>=TH).astype(int)
c1,c2,c3,c4 = st.columns(4)
c1.metric("Accuracy",  f"{accuracy_score(yva,yhat):.0%}")
c2.metric("Precision", f"{precision_score(yva,yhat,zero_division=0):.0%}")
c3.metric("Recall",    f"{recall_score(yva,yhat,zero_division=0):.0%}")
c4.metric("F1-score",  f"{f1_score(yva,yhat,zero_division=0):.0%}")

# ───────────────────────── 8) Strong signals driving alerts (counts only, business labels) ─────────────────────────
st.markdown("### Strong signals driving alerts")
rename = {
    "addr_mismatch":  "Address mismatch (Billing ≠ Shipping)",
    "payment_risky":  "Payment risky (PayPal / Gift card)",
    "high_amount_cat":"High order amount (Unusual spend vs category)",
    "price_high_anom":"High price anomaly (z≥2 vs category)",
    "price_low_anom": "Low price anomaly (z≤−2 vs category)",
    "express_high_amt":"Express + high value (rush & big order)",
    "return_whiplash":"Return whiplash (Frequent returns)",
    "oversell_flag":  "Oversell flag (Qty > 60% of stock)",
    "hoarding_flag":  "Hoarding flag (Bulk buying)"
}
sig_cols = list(rename.keys())
rows=[]
for c in sig_cols:
    rows.append([
        rename[c],
        int(df.loc[flags==1, c].sum()),      # flagged count
        int(df.loc[flags==0, c].sum())       # non-flagged count
    ])
sig_tbl = pd.DataFrame(rows, columns=[
    "Signal (what it checks)",
    "Flagged — suspicious orders with this signal",
    "Non-flagged — normal orders with this signal"
]).sort_values("Flagged — suspicious orders with this signal", ascending=False)
st.dataframe(sig_tbl, use_container_width=True, height=320)
st.caption("Legend: Flagged = orders the model marked as suspicious at the chosen threshold; Non-flagged = orders the model kept as normal.")

# ───────────────────────── 9) ML-Detected suspicious orders (scores only here) ─────────────────────────
st.markdown("### ML-detected suspicious orders")
susp = df.loc[flags==1, tx_cols].copy()
susp["fraud_score"] = np.round(proba_all[flags==1], 3)
for c in sig_cols:
    susp[rename[c]] = df.loc[flags==1, c].values
show_cols = tx_cols + ["fraud_score"] + [rename[c] for c in sig_cols]
st.dataframe(susp.sort_values("fraud_score", ascending=False)[show_cols].head(300),
             use_container_width=True, height=420)

# ───────────────────────── 10) Instant decision (single order scoring) ─────────────────────────
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

def score_one(d):
    r = pd.DataFrame([d])
    cp = df[df["sku_category"]==d["sku_category"]]
    p90 = float(cp["order_amount"].quantile(0.90)) if not cp.empty else 1000.0
    mu  = float(cp["order_amount"].mean()) if not cp.empty else 500.0
    sd  = max(float(cp["order_amount"].std()), 1.0) if not cp.empty else 100.0
    z   = (d["order_amount"]-mu)/sd
    r["addr_mismatch"]   = int(d["shipping_state"]!=d["billing_state"])
    r["payment_risky"]   = int(d["payment_method"] in ["paypal","gift_card"])
    r["high_amount_cat"] = int(d["order_amount"]>=p90)
    r["price_high_anom"] = int(z>=2); r["price_low_anom"]=int(z<=-2)
    r["express_high_amt"]= int(d["express_shipping"]==1 and r["high_amount_cat"].iloc[0]==1)
    r["return_whiplash"] = 0; r["oversell_flag"]=0; r["hoarding_flag"]=int(d["quantity"]>=4)
    r["log_amount"]      = np.log1p(d["order_amount"]); r["on_hand"]=80
    return float(pipe.predict_proba(r[features])[:,1]), r

if st.button("Score order"):
    d = dict(channel=chan, sku_category=catg, payment_method=paym,
             shipping_state=ship, billing_state=bill, quantity=int(qty),
             order_amount=int(amt), express_shipping=int(expr), returns_30d=0)
    d["log_amount"]=np.log1p(d["order_amount"])
    score, rr = score_one(d)
    st.success(f"Decision: {'FLAGGED' if score>=TH else 'PASS'}  •  Fraud score ≈ {score:.2f}  •  Threshold {TH:.2f}")
    on = [k for k in rename if rr[k].iloc[0]==1]
    if on:
        st.write("Signals triggered: " + ", ".join(rename[k] for k in on))
    else:
        st.write("Signals triggered: none")

# ───────────────────────── 11) Fraud score distribution ─────────────────────────
st.markdown("### Fraud score distribution")
st.altair_chart(
    alt.Chart(pd.DataFrame({"score":proba_all})).mark_bar().encode(
        x=alt.X("score:Q", bin=alt.Bin(maxbins=40), title="Fraud score"),
        y=alt.Y("count():Q", title="Orders")
    ).properties(height=220),
    use_container_width=True
)
