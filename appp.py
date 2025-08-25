# app.py — Unified ML-Driven Fraud Detection for Retail POS (USA)
# Business-ready: 4,000 US transactions, tuned model, clear labels, counts-only signals.

import numpy as np, pandas as pd, streamlit as st, altair as alt, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve

warnings.filterwarnings("ignore")
np.random.seed(42)
st.set_page_config("Unified ML-Driven Fraud Detection for Retail POS", layout="wide")
alt.renderers.set_embed_options(actions=False)

# ---------------------- 1) Create 4,000 USA transactions ----------------------
N = 4000
US = np.array(["CA","TX","NY","FL","IL","WA","GA","PA","OH","MI"])
stores = np.array(["US01","US02","US03","US04"])
store_state_pool = np.random.choice(US, size=len(stores), replace=False)
idx = np.random.choice(len(stores), N)
df = pd.DataFrame({
    "order_id"       : np.arange(1, N+1),
    "channel"        : np.random.choice(["Online","In-Store"], N, p=[0.70,0.30]),
    "store_id"       : stores[idx],
    "store_state"    : store_state_pool[idx],
    "sku_category"   : np.random.choice(["Electronics","Apparel","Grocery","Beauty","Home"], N),
    "order_amount"   : np.round(np.random.lognormal(5.5,0.8,N),2),
    "quantity"       : np.random.randint(1,6,N),
    "payment_method" : np.where(np.random.rand(N)<0.7,
                        np.random.choice(["credit_card","debit_card","apple_pay"], N),
                        np.random.choice(["gift_card","paypal"], N)),
    "returns_30d"    : np.random.poisson(0.2,N),
    "sku"            : np.random.randint(100000,110000,N),
    "on_hand"        : np.random.randint(10,200,N)
})
df["shipping_state"] = np.where(df["channel"].eq("In-Store"), df["store_state"], np.random.choice(US,N))
df["billing_state"]  = np.random.choice(US,N)
df["express_shipping"]= np.where(df["channel"].eq("Online"), np.random.choice([0,1],N,p=[0.7,0.3]), 0)

# ---------------------- 2) Business risk signals (binary) ----------------------
df["addr_mismatch"]   = (df["shipping_state"]!=df["billing_state"]).astype(int)  # Address mismatch (Billing ≠ Shipping)
df["payment_risky"]   = df["payment_method"].isin(["paypal","gift_card"]).astype(int)  # Payment risky (PayPal / Gift card)
grp = df.groupby("sku_category")["order_amount"]
p90 = grp.transform(lambda s: s.quantile(0.90))
mu  = grp.transform("mean")
sd  = grp.transform("std").replace(0,1.0)
z   = (df["order_amount"]-mu)/sd
df["high_amount_cat"] = (df["order_amount"]>=p90).astype(int)  # High order amount (unusual spend)
df["price_high_anom"] = (z>=2).astype(int)                     # Price anomaly high
df["price_low_anom"]  = (z<=-2).astype(int)                    # Price anomaly low
df["oversell_flag"]   = (df["quantity"]>0.6*df["on_hand"]).astype(int)  # Oversell flag (qty > stock)
df["hoarding_flag"]   = (df["quantity"]>=4).astype(int)               # Hoarding flag (bulk buying)
df["return_whiplash"] = (df["returns_30d"]>=2).astype(int)           # Return whiplash (frequent returns)
df["express_high_amt"]= ((df["express_shipping"]==1)&(df["high_amount_cat"]==1)).astype(int) # Express + high value
df["log_amount"]      = np.log1p(df["order_amount"])

# ---------------------- 3) Synthetic ground truth (business pattern) ----------------------
p = (0.02 + 0.42*df["payment_risky"] + 0.36*df["addr_mismatch"] + 0.32*df["high_amount_cat"]
     + 0.18*df["express_high_amt"] + 0.12*df["price_high_anom"] + 0.10*df["return_whiplash"]
     + 0.08*df["oversell_flag"] + 0.06*df["hoarding_flag"])
p = np.maximum(p, np.where(df["payment_risky"] & df["addr_mismatch"], 0.9, p))
p = np.maximum(p, np.where(df["payment_risky"] & df["high_amount_cat"] & df["express_high_amt"], 0.94, p))
df["fraud_flag"] = (np.random.rand(N) < np.clip(p,0,0.98)).astype(int)

# ---------------------- 4) Model: tuned RandomForest + best threshold (F1) ----------------------
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
    n_estimators=600, max_depth=10, min_samples_split=6, min_samples_leaf=3,
    max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=42
)
pipe = Pipeline([("pre",pre), ("rf",rf)])

X, y = df[features], df["fraud_flag"]
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
pipe.fit(Xtr, ytr)
proba_all = pipe.predict_proba(X)[:,1]
proba_val = pipe.predict_proba(Xva)[:,1]
prec, rec, thr = precision_recall_curve(yva, proba_val)
f1 = (2*prec*rec)/(prec+rec+1e-12); best_thr = float(thr[np.argmax(f1)])
TH = st.sidebar.slider("Decision Threshold", 0.00, 1.00, float(round(best_thr,2)), 0.01)

# ---------------------- 5) KPIs ----------------------
st.title("Unified ML-Driven Fraud Detection for Retail POS")
flags = (proba_all >= TH).astype(int)
alerts = int(flags.sum()); alert_rate = alerts/len(df)
k1, k2, k3 = st.columns(3)
k1.metric("Accuracy",  f"{accuracy_score(yva, (proba_val>=TH).astype(int)):.0%}")
k2.metric("Precision", f"{precision_score(yva, (proba_val>=TH).astype(int), zero_division=0):.0%}")
k3.metric("Recall",    f"{rec[np.argmax(thr>=TH)] if np.any(thr>=TH) else rec.max():.0%}")
k1b, k2b, k3b = st.columns(3)
k1b.metric("Transactions", f"{len(df):,}")
k2b.metric("Suspicious orders", f"{alerts:,}")
k3b.metric("Alert rate", f"{alert_rate:.0%}")

# ---------------------- 6) Retail transactions (no scores, no flags) ----------------------
st.markdown("### Retail transactions")
tx_cols = ["order_id","channel","store_id","store_state","sku_category",
           "order_amount","quantity","payment_method","shipping_state","billing_state",
           "express_shipping","returns_30d"]
st.dataframe(df[tx_cols], use_container_width=True, height=420)

# ---------------------- 7) Model evaluation (at chosen threshold) ----------------------
st.markdown("### Model evaluation")
yhat = (proba_val>=TH).astype(int)
c1,c2,c3,c4 = st.columns(4)
c1.metric("Accuracy",  f"{accuracy_score(yva,yhat):.0%}")
c2.metric("Precision", f"{precision_score(yva,yhat,zero_division=0):.0%}")
c3.metric("Recall",    f"{recall_score(yva,yhat,zero_division=0):.0%}")
c4.metric("F1-score",  f"{f1_score(yva,yhat,zero_division=0):.0%}")

# ---------------------- 8) Strong signals driving alerts (counts only) ----------------------
st.markdown("### Strong signals driving alerts")
rename = {
    "addr_mismatch":"Address mismatch (Billing ≠ Shipping)",
    "payment_risky":"Payment risky (PayPal / Gift card)",
    "high_amount_cat":"High order amount (Unusual spend)",
    "price_high_anom":"Price anomaly high",
    "price_low_anom":"Price anomaly low",
    "express_high_amt":"Express + high value",
    "return_whiplash":"Return whiplash (Frequent returns)",
    "oversell_flag":"Oversell flag (Qty > stock)",
    "hoarding_flag":"Hoarding flag (Bulk buying)"
}
sig_cols = list(rename.keys())
flag_series = pd.Series(flags, index=df.index)
rows=[]
for c in sig_cols:
    rows.append([rename[c],
                 int(df.loc[flag_series==1, c].sum()),
                 int(df.loc[flag_series==0, c].sum())])
sig_tbl = pd.DataFrame(rows, columns=[
    "Signal (what it checks)",
    "Flagged (suspicious orders with this signal)",
    "Non-flagged (normal orders with this signal)"
]).sort_values("Flagged (suspicious orders with this signal)", ascending=False)
st.dataframe(sig_tbl, use_container_width=True, height=320)

# ---------------------- 9) ML-Detected suspicious orders (scores only here) ----------------------
st.markdown("### ML-Detected suspicious orders")
susp = df.loc[flags==1, tx_cols].copy()
susp["fraud_score"] = np.round(proba_all[flags==1], 3)
for c in sig_cols: susp[rename[c]] = df.loc[flags==1, c].values
show_cols = tx_cols + ["fraud_score"] + [rename[c] for c in sig_cols]
st.dataframe(susp.sort_values("fraud_score", ascending=False)[show_cols].head(300),
             use_container_width=True, height=420)

# ---------------------- 10) Fraud score distribution ----------------------
st.markdown("### Fraud score distribution")
st.altair_chart(
    alt.Chart(pd.DataFrame({"score":proba_all})).mark_bar().encode(
        x=alt.X("score:Q", bin=alt.Bin(maxbins=40), title="Fraud score"),
        y=alt.Y("count():Q", title="Orders")
    ).properties(height=220),
    use_container_width=True
)
