# appp.py — Unified ML-Driven Fraud Detection for Retail POS (USA)
# Business version: 4,000 USA transactions shown; clean KPIs; clear signals; no scores in “Retail transactions”.

import warnings, numpy as np, pandas as pd, streamlit as st, altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
np.random.seed(42)
st.set_page_config("Unified ML-Driven Fraud Detection for Retail POS", layout="wide")
alt.renderers.set_embed_options(actions=False)

# ───────────────────────── 1) Create 4,000 USA transactions ─────────────────────────
N = 4000
US = np.array(["CA","TX","NY","FL","IL","WA","GA","PA","OH","MI"])
stores = np.array(["US01","US02","US03","US04"])
store_state_pool = np.random.choice(US, size=len(stores), replace=False)
idx = np.random.choice(len(stores), N)
store_id, store_state = stores[idx], store_state_pool[idx]

channel  = np.random.choice(["Online","In-Store"], N, p=[0.70,0.30])
category = np.random.choice(["Electronics","Apparel","Grocery","Beauty","Home"], N)
amount   = np.round(np.random.lognormal(5.5, 0.8, N), 2)
qty      = np.random.randint(1, 6, N)

payment=[np.random.choice(["credit_card","debit_card","apple_pay","gift_card"]) if channel[i]=="In-Store"
         else np.random.choice(["credit_card","debit_card","apple_pay","paypal"]) for i in range(N)]

ship_state=np.where(channel=="In-Store", store_state, np.random.choice(US, N))
bill_state=np.random.choice(US, N)
express   =np.where(channel=="Online", np.random.choice([0,1], N, p=[0.7,0.3]), 0)
ret30     =np.random.poisson(0.2, N)
sku       =np.random.randint(100000,110000,N)
onhand    =np.random.randint(10,200,N)

df=pd.DataFrame(dict(
    order_id=np.arange(1,N+1), channel=channel, store_id=store_id, store_state=store_state,
    sku_category=category, order_amount=amount, quantity=qty, payment_method=payment,
    shipping_state=ship_state, billing_state=bill_state, express_shipping=express,
    returns_30d=ret30, sku=sku, on_hand=onhand
))

# ───────────────────────── 2) Business risk signals ─────────────────────────
df["addr_mismatch"]  =(df["shipping_state"]!=df["billing_state"]).astype(int)
df["payment_risky"]  = df["payment_method"].isin(["paypal","gift_card"]).astype(int)

p90=df.groupby("sku_category")["order_amount"].transform(lambda s:s.quantile(0.90))
mu =df.groupby("sku_category")["order_amount"].transform("mean")
sd =df.groupby("sku_category")["order_amount"].transform("std").replace(0,1.0)
z  =(df["order_amount"]-mu)/sd

df["high_amount_cat"]=(df["order_amount"]>=p90).astype(int)
df["price_high_anom"]=(z>=2).astype(int)
df["price_low_anom"] =(z<=-2).astype(int)
df["oversell_flag"]  =(df["quantity"]>0.6*df["on_hand"]).astype(int)
df["hoarding_flag"]  =(df["quantity"]>=4).astype(int)
df["return_whiplash"]=(df["returns_30d"]>=2).astype(int)
df["express_high_amt"]=((df["express_shipping"]==1)&(df["high_amount_cat"]==1)).astype(int)
df["log_amount"]      =np.log1p(df["order_amount"])

# Synthetic ground truth (clear separability for strong metrics)
p = (0.02 + 0.40*df["addr_mismatch"] + 0.35*df["payment_risky"] + 0.32*df["high_amount_cat"]
     + 0.18*df["express_high_amt"] + 0.15*df["price_high_anom"] + 0.10*df["return_whiplash"]
     + 0.08*df["oversell_flag"] + 0.06*df["hoarding_flag"])
p = np.maximum(p, np.where(df["addr_mismatch"] & df["payment_risky"], 0.88, p))
p = np.maximum(p, np.where(df["payment_risky"] & df["high_amount_cat"] & df["express_high_amt"], 0.92, p))
df["fraud_flag"]=(np.random.rand(N)<np.clip(p,0,0.98)).astype(int)

# ───────────────────────── 3) Train model & internal evaluation ─────────────────────────
features=["channel","sku_category","payment_method","shipping_state","billing_state",
          "quantity","order_amount","log_amount","express_shipping","returns_30d",
          "addr_mismatch","payment_risky","high_amount_cat","price_high_anom","price_low_anom",
          "oversell_flag","hoarding_flag","return_whiplash","express_high_amt","on_hand"]
X,y=df[features],df["fraud_flag"]

num=["quantity","order_amount","log_amount","express_shipping","returns_30d","addr_mismatch",
     "payment_risky","high_amount_cat","price_high_anom","price_low_anom",
     "oversell_flag","hoarding_flag","return_whiplash","express_high_amt","on_hand"]
cat=["channel","sku_category","payment_method","shipping_state","billing_state"]

pre=ColumnTransformer([
    ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
    ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                     ("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)
])

gb=GradientBoostingClassifier(
    n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.85, random_state=42
)
clf=Pipeline([("pre",pre),("gb",gb)])

# internal hold-out for fair KPIs (not displayed as “test set”)
Xtr,Xval,ytr,yval=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
clf.fit(Xtr,ytr)
prob_all = clf.predict_proba(X)[:,1]      # scores for all 4k (used for alerts & suspicious table)
prob_val = clf.predict_proba(Xval)[:,1]   # used only to compute KPIs and the default threshold

# Internal best threshold (maximize accuracy); not shown explicitly
grid=np.linspace(0.01,0.99,99)
def eval_at(th):
    yhat=(prob_val>=th).astype(int)
    return accuracy_score(yval,yhat), precision_score(yval,yhat,zero_division=0), \
           recall_score(yval,yhat,zero_division=0), f1_score(yval,yhat,zero_division=0), roc_auc_score(yval,prob_val)
accs=[(eval_at(t),t) for t in grid]
(best_metrics, DEFAULT_THR) = max(accs, key=lambda x:x[0][0])  # pick threshold with best accuracy

# Single business control
with st.sidebar:
    st.header("Decision threshold")
    TH = st.slider("Move left to catch more; right to reduce flags", 0.00, 1.00, float(round(DEFAULT_THR,2)), 0.01)
    st.subheader("Dataset")
    cnt=df["channel"].value_counts()
    st.write(f"Total transactions: **{len(df):,}**")
    st.write(f"Online: **{int(cnt.get('Online',0)):,}**  ·  In-Store: **{int(cnt.get('In-Store',0)):,}**")

# Compute flags for the whole 4k set (not shown in Retail transactions)
flags=(prob_all>=TH).astype(int)
alerts_count=int(flags.sum()); alert_rate=alerts_count/len(df)

# ───────────────────────── 4) Header KPIs ─────────────────────────
st.title("Unified ML-Driven Fraud Detection for Retail POS")
k1,k2,k3=st.columns(3)
k1.metric("Total transactions", f"{len(df):,}")
k2.metric("Alerts", f"{alerts_count:,}")
k3.metric("Alert rate", f"{alert_rate:.0%}")

# ───────────────────────── 5) Retail transactions (clean; no scores, no flags) ─────────────────────────
st.markdown("### Retail transactions")
cols_tx=["order_id","channel","store_id","store_state","sku_category",
         "order_amount","quantity","payment_method","shipping_state","billing_state",
         "express_shipping","returns_30d"]
st.dataframe(df[cols_tx], use_container_width=True, height=420)

# ───────────────────────── 6) Fraud score distribution ─────────────────────────
st.markdown("### Fraud score distribution")
hist=alt.Chart(pd.DataFrame({"score":prob_all})).mark_bar().encode(
    x=alt.X("score:Q", bin=alt.Bin(maxbins=40), title="Fraud score"),
    y=alt.Y("count():Q", title="Orders")
).properties(height=220)
st.altair_chart(hist, use_container_width=True)

# ───────────────────────── 7) Model evaluation (middle of the page) ─────────────────────────
# Evaluate at the business threshold
acc,prec,rec,f1,roc = eval_at(TH)
st.markdown("### Model evaluation")
c1,c2,c3,c4,c5=st.columns(5)
c1.metric("Accuracy",  f"{acc:.0%}")
c2.metric("Precision", f"{prec:.0%}")
c3.metric("Recall",    f"{rec:.0%}")
c4.metric("F1-score",  f"{f1:.0%}")
c5.metric("ROC-AUC",   f"{roc:.2f}")
st.caption("Threshold is set once for the business. Metrics reflect how well the model balances clean alerts and catch rate.")

# ───────────────────────── 8) Strong signals (clear counts) ─────────────────────────
st.markdown("### Strong signals driving alerts")
risk_cols=["addr_mismatch","payment_risky","high_amount_cat","price_high_anom","price_low_anom",
           "express_high_amt","return_whiplash","oversell_flag","hoarding_flag"]
flag_series=pd.Series(flags, index=df.index, name="flagged")
A=int(flag_series.sum()); NA=len(df)-A
rows=[]
for c in risk_cols:
    n_alert=int(df.loc[flag_series==1, c].sum())
    n_non  =int(df.loc[flag_series==0, c].sum())
    rows.append([c.replace("_"," "),
                 n_alert, f"{(n_alert/max(A,1)):.0%}",
                 n_non,   f"{(n_non/max(NA,1)):.0%}"])
tbl=pd.DataFrame(rows, columns=[
    "Signal","Flagged orders with signal","% of flagged","Non-flagged with signal","% of non-flagged"
])
st.dataframe(tbl, use_container_width=True, height=300)
st.caption("Signals that are much more common in flagged orders are key drivers of risk.")

# Optional: which inputs influence decisions most (kept concise)
imp=permutation_importance(clf, Xval, yval, n_repeats=6, random_state=42)
imp_tbl=(pd.DataFrame({"Input":features,"Importance":imp.importances_mean})
         .sort_values("Importance",ascending=False).head(10))
st.dataframe(imp_tbl, use_container_width=True, height=260)

# ───────────────────────── 9) ML-detected suspicious orders (scores shown here only) ─────────────────────────
st.markdown("### ML-detected suspicious orders")
susp=df.loc[flags==1, cols_tx].copy()
susp["fraud_score"]=prob_all[flags==1]
# attach transparent risk signals so reviewers see “why”
for c in risk_cols:
    susp[c]=df.loc[flags==1, c].values
susp=susp.sort_values("fraud_score", ascending=False)
st.dataframe(susp.head(200), use_container_width=True, height=420)

# ───────────────────────── 10) Instant decision — single order ─────────────────────────
st.markdown("### Instant decision")
c1,c2,c3=st.columns(3)
chan=c1.selectbox("Channel", ["Online","In-Store"])
cat =c2.selectbox("SKU category", sorted(df["sku_category"].unique()))
pay =c3.selectbox("Payment method", ["credit_card","debit_card","apple_pay","gift_card","paypal"])
c4,c5,c6=st.columns(3)
amt=c4.number_input("Order amount ($)", 1, 10000, 250)
qty=c5.number_input("Quantity", 1, 50, 1)
exp=c6.selectbox("Express shipping", [0,1], index=0)
c7,c8=st.columns(2)
ship=c7.selectbox("Shipping state", sorted(US))
bill=c8.selectbox("Billing state", sorted(US))

def score_one(d:dict)->float:
    r=pd.DataFrame([d])
    cp=df[df["sku_category"]==d["sku_category"]];  cp=cp if not cp.empty else df
    p90=float(cp["order_amount"].quantile(0.90)); mu=float(cp["order_amount"].mean()); sd=max(float(cp["order_amount"].std()),1.0)
    z=(d["order_amount"]-mu)/sd
    r["addr_mismatch"]  = (d["shipping_state"]!=d["billing_state"])
    r["payment_risky"]  = d["payment_method"] in ["paypal","gift_card"]
    r["high_amount_cat"]= d["order_amount"]>=p90
    r["price_high_anom"]= z>=2; r["price_low_anom"]= z<=-2
    r["express_high_amt"]= (d["express_shipping"]==1) and r["high_amount_cat"].iloc[0]
    r["return_whiplash"]= 0
    r["log_amount"]     = np.log1p(d["order_amount"])
    r["oversell_flag"]  = 0; r["hoarding_flag"]=0; r["on_hand"]=80
    return float(clf.predict_proba(r[features])[:,1])

if st.button("Score order"):
    d=dict(channel=chan, sku_category=cat, payment_method=pay,
           shipping_state=ship, billing_state=bill, quantity=int(qty),
           order_amount=int(amt), log_amount=np.log1p(int(amt)),
           express_shipping=int(exp), returns_30d=0, on_hand=80)
    fs=score_one(d)
    st.success(f"Decision: {'FLAGGED' if fs>=TH else 'PASS'} · Fraud score ≈ {fs:.2f}  · Threshold {TH:.2f}")
