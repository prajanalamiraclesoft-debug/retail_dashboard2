# appp.py — Unified ML-Driven Fraud Detection for Retail POS (Business-Ready, USA-only)

import time, warnings, numpy as np, pandas as pd, altair as alt, streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             precision_recall_curve, roc_auc_score)

warnings.filterwarnings("ignore"); np.random.seed(42)
st.set_page_config(page_title="Unified ML-Driven Fraud Detection for Retail POS (USA)", layout="wide")
alt.renderers.set_embed_options(actions=False)

# ─────────────── 1) Create 4,000 USA transactions ───────────────
N=4000
US_STATES=np.array(["CA","TX","NY","FL","IL","WA","GA","PA","OH","MI"])
store_ids=np.array(["US01","US02","US03","US04"])
store_state_pool=np.random.choice(US_STATES, size=len(store_ids), replace=False)
idx=np.random.choice(len(store_ids), N)
store_id=store_ids[idx]; store_state=store_state_pool[idx]

channel=np.random.choice(["Online","In-Store"], N, p=[0.7,0.3])
category=np.random.choice(["Electronics","Apparel","Grocery","Beauty","Home"], N)
amount=np.round(np.random.lognormal(5.5,0.8,N),2)
quantity=np.random.randint(1,6,N)

payment=[np.random.choice(["credit_card","debit_card","apple_pay","gift_card"]) if channel[i]=="In-Store"
         else np.random.choice(["credit_card","debit_card","apple_pay","paypal"]) for i in range(N)]

shipping_state=np.where(channel=="In-Store", store_state, np.random.choice(US_STATES, N))
billing_state =np.random.choice(US_STATES, N)
express=np.where(channel=="Online", np.random.choice([0,1],N,p=[0.7,0.3]), 0)
returns30=np.random.poisson(0.2, N)
sku=np.random.randint(100000,110000,N)
onhand=np.random.randint(10,200,N)

df=pd.DataFrame(dict(
    order_id=np.arange(1,N+1), channel=channel, store_id=store_id, store_state=store_state,
    sku_category=category, order_amount=amount, quantity=quantity, payment_method=payment,
    shipping_state=shipping_state, billing_state=billing_state, express_shipping=express,
    returns_30d=returns30, sku=sku, on_hand=onhand
))

# ─────────────── 2) Strong Features ───────────────
df["addr_mismatch"]  =(df["shipping_state"]!=df["billing_state"]).astype(int)
df["payment_risky"]  = df["payment_method"].isin(["paypal","gift_card"]).astype(int)
p90=df.groupby("sku_category")["order_amount"].transform(lambda s:s.quantile(0.90))
mu=df.groupby("sku_category")["order_amount"].transform("mean")
sd=df.groupby("sku_category")["order_amount"].transform("std").replace(0,1.0)
z=(df["order_amount"]-mu)/sd
df["high_amount_cat"]=(df["order_amount"]>=p90).astype(int)
df["price_high_anom"]=(z>=2).astype(int)
df["price_low_anom"] =(z<=-2).astype(int)
df["oversell_flag"]  =(df["quantity"]>0.6*df["on_hand"]).astype(int)
df["hoarding_flag"]  =(df["quantity"]>=4).astype(int)
df["return_whiplash"]=(df["returns_30d"]>=2).astype(int)
df["express_high_amt"]=((df["express_shipping"]==1)&(df["high_amount_cat"]==1)).astype(int)
df["log_amount"]=np.log1p(df["order_amount"])

# Labels
p=(0.05 + 0.25*df["addr_mismatch"] + 0.20*df["payment_risky"] + 0.22*df["high_amount_cat"]
    + 0.10*df["return_whiplash"] + 0.12*df["express_high_amt"] + 0.07*df["oversell_flag"]
    + 0.06*df["hoarding_flag"])
df["fraud_flag"]=(np.random.rand(N)<np.clip(p,0,0.95)).astype(int)

# ─────────────── 3) Train Model (RandomForest → Higher Accuracy) ───────────────
features=["channel","sku_category","payment_method","shipping_state","billing_state",
          "quantity","order_amount","log_amount","express_shipping","returns_30d",
          "addr_mismatch","payment_risky","high_amount_cat","price_high_anom","price_low_anom",
          "oversell_flag","hoarding_flag","return_whiplash","express_high_amt","on_hand"]

X,y=df[features],df["fraud_flag"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

num=["quantity","order_amount","log_amount","express_shipping","returns_30d","addr_mismatch","payment_risky",
     "high_amount_cat","price_high_anom","price_low_anom","oversell_flag","hoarding_flag",
     "return_whiplash","express_high_amt","on_hand"]
cat=["channel","sku_category","payment_method","shipping_state","billing_state"]

pre=ColumnTransformer([
  ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
  ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                   ("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)
])

clf=Pipeline([("pre",pre),
              ("rf",RandomForestClassifier(n_estimators=200,max_depth=10,class_weight="balanced",random_state=42))]).fit(Xtr,ytr)
yprob=clf.predict_proba(Xte)[:,1]

# ─────────────── 4) Thresholds ───────────────
prec,rec,thr=precision_recall_curve(yte,yprob)
f1=(2*prec*rec)/(prec+rec+1e-9)
best_idx=int(np.argmax(f1))
BEST_THR=float(thr[best_idx]) if best_idx<len(thr) else 0.5

def evaluate_at(thr:float):
    yhat=(yprob>=thr).astype(int)
    return dict(
        acc=accuracy_score(yte,yhat),
        prec=precision_score(yte,yhat,zero_division=0),
        rec=recall_score(yte,yhat,zero_division=0),
        f1=f1_score(yte,yhat,zero_division=0),
        roc=roc_auc_score(yte,yprob),
        alert_rate=float(yhat.mean())
    )

# ─────────────── 5) Dashboard ───────────────
st.title("Unified ML-Driven Fraud Detection for Retail POS (USA)")
st.subheader("Bridging Behavioral Risk, Pricing Anomalies, and Inventory Stress")

with st.sidebar:
    st.header("Business Controls")
    st.metric("Best Threshold (Max F1)", f"{BEST_THR:.2f}")
    TH=st.slider("Business Threshold", 0.00, 1.00, float(round(BEST_THR,2)), 0.01)
    st.caption("Best threshold is fixed by ML evaluation. Slider lets you override for stricter/looser control.")

M=evaluate_at(TH)
k1,k2,k3,k4=st.columns(4)
k1.metric("Accuracy", f"{M['acc']:.0%}")
k2.metric("Precision", f"{M['prec']:.0%}")
k3.metric("Recall", f"{M['rec']:.0%}")
k4.metric("F1-Score", f"{M['f1']:.0%}")

st.info("**Details:** These 4 KPIs show how reliable the model is. Accuracy is overall correctness. Precision is the cleanliness of alerts. Recall is the share of frauds we caught. F1 balances both.")

# Transactions
st.markdown("### Retail Transaction Data (USA)")
cols_tx=["order_id","channel","store_id","store_state","sku_category","order_amount","quantity",
         "payment_method","shipping_state","billing_state","express_shipping","returns_30d"]
st.dataframe(df[cols_tx].head(100), use_container_width=True, height=320)
st.caption("**Details:** This is the raw transaction data (sample of 100 shown). All are USA orders.")

# Fraud score distribution
st.markdown("### Fraud Score Distribution")
hist=alt.Chart(pd.DataFrame({"fraud_score":yprob})).mark_bar().encode(
    x=alt.X("fraud_score:Q", bin=alt.Bin(maxbins=40)),
    y=alt.Y("count():Q")
).properties(height=200)
st.altair_chart(hist, use_container_width=True)
st.caption("**Details:** Fraud scores are probabilities (0–1). Higher score = more suspicious.")

# Suspicious orders
st.markdown("### ML-Detected Suspicious Orders")
scores=pd.Series(yprob,index=Xte.index,name="fraud_score")
alerts=(df.loc[Xte.index,cols_tx].assign(fraud_score=scores)
        .loc[lambda d:d["fraud_score"]>=TH]
        .sort_values("fraud_score",ascending=False))
st.dataframe(alerts.head(50), use_container_width=True, height=300)
st.caption("**Details:** These are the top suspicious orders flagged by the model. Fraud_score shows risk level.")

# Strong features
st.markdown("### Strong Features Driving Suspicion")
strong_cols=["addr_mismatch","payment_risky","high_amount_cat","price_high_anom","price_low_anom",
             "express_high_amt","return_whiplash","oversell_flag","hoarding_flag"]
rows=[]
for c in strong_cols:
    flagged=alerts[c].sum() if c in alerts else 0
    nonflagged=(df[c].sum()-flagged)
    rows.append([c, flagged, nonflagged])
tbl=pd.DataFrame(rows,columns=["Feature","Flagged Orders With Feature","Non-Flagged Orders With Feature"])
st.dataframe(tbl, use_container_width=True, height=300)
st.caption("**Details:** Shows how many flagged vs non-flagged orders had each risk signal.")

# Instant Decision
st.markdown("### Instant Decision — Test a New Order")
c1,c2,c3=st.columns(3)
chan_sel=c1.selectbox("Channel", ["Online","In-Store"])
cat_sel =c2.selectbox("SKU Category", sorted(df["sku_category"].unique()))
pay_sel =c3.selectbox("Payment Method", ["credit_card","debit_card","apple_pay","gift_card","paypal"])
c4,c5,c6=st.columns(3)
amt_sel=c4.number_input("Order Amount ($)", min_value=1,max_value=10000,value=250)
qty_sel=c5.number_input("Quantity",min_value=1,max_value=50,value=1)
exp_sel=c6.selectbox("Express Shipping", [0,1], index=0)
c7,c8=st.columns(2)
ship_sel=c7.selectbox("Shipping State", sorted(US_STATES))
bill_sel=c8.selectbox("Billing State", sorted(US_STATES))

def score_new_order(order:dict)->float:
    r=pd.DataFrame([order])
    p90=df.groupby("sku_category")["order_amount"].quantile(0.90).get(order["sku_category"],250)
    mu=df.groupby("sku_category")["order_amount"].mean().get(order["sku_category"],250)
    sd=df.groupby("sku_category")["order_amount"].std().get(order["sku_category"],50)
    z=(order["order_amount"]-mu)/max(sd,1)
    r["addr_mismatch"]=(order["shipping_state"]!=order["billing_state"])
    r["payment_risky"]=order["payment_method"] in ["paypal","gift_card"]
    r["high_amount_cat"]=order["order_amount"]>=p90
    r["price_high_anom"]=z>=2; r["price_low_anom"]=z<=-2
    r["express_high_amt"]=(order["express_shipping"]==1 and order["high_amount_cat"])
    r["return_whiplash"]=0; r["log_amount"]=np.log1p(order["order_amount"])
    r["oversell_flag"]=0; r["hoarding_flag"]=0; r["on_hand"]=80
    return float(clf.predict_proba(r[features])[:,1])

if st.button("Score Order"):
    fs=score_new_order(dict(channel=chan_sel, sku_category=cat_sel, payment_method=pay_sel,
                            shipping_state=ship_sel, billing_state=bill_sel,
                            quantity=int(qty_sel), order_amount=int(amt_sel), log_amount=np.log1p(amt_sel),
                            express_shipping=int(exp_sel), returns_30d=0, on_hand=80))
    st.success(f"Decision: {'FLAGGED' if fs>=TH else 'PASS'} · Fraud Score ≈ {fs:.2f}")
