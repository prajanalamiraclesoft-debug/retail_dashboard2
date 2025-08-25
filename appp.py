# appp.py — Unified ML-Driven Fraud Detection for Retail POS (USA-only, business-ready)

import time, warnings, numpy as np, pandas as pd, altair as alt, streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             precision_recall_curve, auc, roc_auc_score)

warnings.filterwarnings("ignore"); np.random.seed(42)
st.set_page_config(page_title="Unified ML-Driven Fraud Detection for Retail POS (USA)", layout="wide")
alt.renderers.set_embed_options(actions=False)

# ─────────────── 1) Create 4,000 USA transactions (Online + In-Store) ───────────────
N=4000
us_states=np.array(["CA","TX","NY","FL","IL","WA","GA","PA","OH","MI"])

store_ids=np.array(["US01","US02","US03","US04"])
store_state=np.random.choice(us_states, size=len(store_ids), replace=False)
idx=np.random.choice(len(store_ids), N)
store_id=store_ids[idx]; store_us_state=store_state[idx]          # store_country is implicitly US

channel=np.random.choice(["Online","In-Store"], N, p=[0.7,0.3])
category=np.random.choice(["Electronics","Apparel","Grocery","Beauty","Home"], N)

amount=np.round(np.random.lognormal(5.5,0.8,N),2)
quantity=np.random.randint(1,6,N)

# payment options depend on channel
payment=[np.random.choice(["credit_card","debit_card","apple_pay","gift_card"]) if channel[i]=="In-Store"
         else np.random.choice(["credit_card","debit_card","apple_pay","paypal"]) for i in range(N)]

# shipping/billing **states** (USA only). Allow realistic mismatches across states.
shipping_state=np.where(
    channel=="In-Store",
    store_us_state,
    np.random.choice(us_states, N, p=np.full(len(us_states), 1/len(us_states)))
)
billing_state=np.random.choice(us_states, N, p=np.full(len(us_states), 1/len(us_states)))

express=np.where(channel=="Online", np.random.choice([0,1],N,p=[0.7,0.3]), 0)
returns30=np.random.poisson(0.2, N)
sku=np.random.randint(100000,110000,N)
onhand=np.random.randint(10,200,N)

df=pd.DataFrame(dict(
    order_id=np.arange(1,N+1), channel=channel, store_id=store_id, store_state=store_us_state,
    sku_category=category, order_amount=amount, quantity=quantity, payment_method=payment,
    shipping_state=shipping_state, billing_state=billing_state, express_shipping=express,
    returns_30d=returns30, sku=sku, on_hand=onhand
))

# ─────────────── 2) Strong business features (no IP, USA-only) ───────────────
df["addr_mismatch"]  =(df["shipping_state"]!=df["billing_state"]).astype(int)
df["payment_risky"]  = df["payment_method"].isin(["paypal","gift_card"]).astype(int)

p90=df.groupby("sku_category")["order_amount"].transform(lambda s:s.quantile(0.90))
mean=df.groupby("sku_category")["order_amount"].transform("mean")
std =df.groupby("sku_category")["order_amount"].transform("std").replace(0,1.0)
z=(df["order_amount"]-mean)/std

df["high_amount_cat"]=(df["order_amount"]>=p90).astype(int)
df["price_high_anom"]=(z>=2).astype(int)     # unusually high (possible fraud)
df["price_low_anom"] =(z<=-2).astype(int)    # unusually low (discount abuse)
df["oversell_flag"]  =(df["quantity"]>0.6*df["on_hand"]).astype(int)
df["hoarding_flag"]  =(df["quantity"]>=4).astype(int)
df["return_whiplash"]=(df["returns_30d"]>=2).astype(int)
df["express_high_amt"]=((df["express_shipping"]==1)&(df["high_amount_cat"]==1)).astype(int)
df["log_amount"]      =np.log1p(df["order_amount"])

# ─────────────── 3) Outcome for training (probability → sampled label) ───────────────
# This creates a labeled outcome using the same business drivers the model will learn.
p = (0.04 + 0.22*df["addr_mismatch"] + 0.18*df["payment_risky"] + 0.20*df["high_amount_cat"]
     + 0.08*df["return_whiplash"] + 0.10*df["express_high_amt"] + 0.06*df["oversell_flag"]
     + 0.05*df["hoarding_flag"])
df["fraud_flag"]=(np.random.rand(N)<np.clip(p,0,0.95)).astype(int)

# ─────────────── 4) Train model → fraud_score (probability) ───────────────
# Model: Logistic Regression (explainable). fraud_score = P(fraud=1)
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
  ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)
])

clf=Pipeline([("pre",pre),("lr",LogisticRegression(max_iter=300,class_weight="balanced"))]).fit(Xtr,ytr)
yprob=clf.predict_proba(Xte)[:,1]  # fraud_score (probability 0..1)

# ─────────────── 5) Best Threshold (fixed) + Business Threshold (slider) ───────────────
prec,rec,thr=precision_recall_curve(yte,yprob)
f1=(2*prec*rec)/(prec+rec+1e-9)
best_idx=int(np.argmax(f1))
BEST_THR=float(thr[best_idx] if best_idx<len(thr) else 0.50)  # <-- FIXED best threshold (max F1)

def evaluate_at(thr: float):
    yhat=(yprob>=thr).astype(int)
    return dict(
        acc=accuracy_score(yte,yhat),
        prec=precision_score(yte,yhat,zero_division=0),
        rec=recall_score(yte,yhat,zero_division=0),
        f1=f1_score(yte,yhat,zero_division=0),
        roc=roc_auc_score(yte,yprob),
        pr_auc=auc(rec,prec),
        alert_rate=float(yhat.mean()),
    )

# ─────────────── 6) Dashboard UI ───────────────
st.title("Unified ML-Driven Fraud Detection for Retail POS (USA)")
st.subheader("Bridging Behavioral Risk, Pricing Anomalies, and Inventory Stress")
st.caption(f"Build: {time.strftime('%Y-%m-%d %H:%M:%S')} · Model: Logistic Regression · fraud_score = ML probability")

with st.sidebar:
    st.header("Business Control")
    st.metric("Best Threshold (Max F1)", f"{BEST_THR:.2f}")
    TH=st.slider("Business Threshold (you can override)", 0.00, 1.00, float(round(BEST_THR,2)), 0.01,
                 help="Lower = stricter (more alerts). Higher = looser (fewer alerts).")
    st.caption("Best Threshold is fixed from model evaluation; this slider is a policy choice.")

m=evaluate_at(TH)
k1,k2,k3,k4=st.columns(4)
k1.metric("Total Transactions (USA)", f"{len(df):,}")
k2.metric("ML-Detected Orders", f"{int(m['alert_rate']*len(yprob)):,}")
k3.metric("Alert Rate", f"{m['alert_rate']:.1%}")
k4.metric("ROC-AUC", f"{m['roc']:.2f}")

# --- ALL 4,000 transactions (USA) ---
st.markdown("### Retail Transaction Data (USA)")
cols_tx=["order_id","channel","store_id","store_state","sku_category",
         "order_amount","quantity","payment_method","shipping_state",
         "billing_state","express_shipping","returns_30d"]
st.dataframe(df[cols_tx], use_container_width=True, height=520)
st.download_button("Download all transactions (CSV)",
                   data=df[cols_tx].to_csv(index=False).encode("utf-8"),
                   file_name="retail_transactions_usa_4000.csv", mime="text/csv",
                   use_container_width=True)

# --- Fraud score distribution ---
st.markdown("### Fraud Score Distribution (ML Output)")
hist=alt.Chart(pd.DataFrame({"fraud_score":yprob})).mark_bar().encode(
    x=alt.X("fraud_score:Q", bin=alt.Bin(maxbins=40), title="Fraud Score (0–1)"),
    y=alt.Y("count():Q", title="Transactions")
).properties(height=220)
st.altair_chart(hist, use_container_width=True)

# --- ML-Detected suspicious orders (joined correctly) ---
st.markdown("### ML-Detected Suspicious Orders")
scores=pd.Series(yprob, index=Xte.index, name="fraud_score")
alerts=(df.loc[Xte.index, cols_tx]
          .assign(fraud_score=scores)
          .loc[lambda d: d["fraud_score"]>=TH]
          .sort_values("fraud_score", ascending=False))
st.dataframe(alerts[
    ["order_id","channel","store_id","store_state","sku_category",
     "order_amount","quantity","payment_method","shipping_state","billing_state","fraud_score"]
], use_container_width=True, height=420)
st.download_button("Download suspicious orders (CSV)",
                   data=alerts.to_csv(index=False).encode("utf-8"),
                   file_name="ml_detected_suspicious_orders_usa.csv", mime="text/csv",
                   use_container_width=True)

# --- Model evaluation (at Business Threshold) ---
st.markdown("### Model Evaluation (at Business Threshold)")
e1,e2,e3,e4=st.columns(4)
e1.metric("Accuracy", f"{m['acc']:.0%}")
e2.metric("Precision", f"{m['prec']:.0%}")
e3.metric("Recall", f"{m['rec']:.0%}")
e4.metric("F1-Score", f"{m['f1']:.0%}")
st.caption("Accuracy = overall correctness • Precision = alert cleanliness • Recall = % of fraud captured • F1 = balance.")

# --- Strong features (business framing) ---
st.markdown("### Strong Features Driving Risk (business-explainable)")
st.write(
"""
- **Address mismatch** — shipping state differs from billing state  
- **Payment channel risk** — PayPal (online) & Gift Cards (in-store) are riskier; credit/debit stable  
- **High amount for category** — spend above the 90th percentile of that category  
- **Price anomalies** — unusually high (possible fraud) or unusually low (discount abuse)  
- **Express + high value** — fast delivery requests on expensive orders  
- **Return whiplash** — multiple recent returns  
- **Inventory pressure** — oversell risk (qty > 60% of on-hand) or **hoarding** (bulk buys)
"""
)

# ─────────────── 7) Instant Decision (what-if scoring) ───────────────
def payment_options(chan:str):
    return ["credit_card","debit_card","apple_pay","gift_card"] if chan=="In-Store" \
           else ["credit_card","debit_card","apple_pay","paypal"]

def score_new_order(order:dict)->float:
    """Return fraud_score ∈ [0,1] for a single order dict (same model features)."""
    r=pd.DataFrame([order])
    cp=df[df["sku_category"]==r["sku_category"].iloc[0]];  cp = cp if not cp.empty else df
    p90=cp["order_amount"].quantile(0.90); mu=cp["order_amount"].mean(); sd=max(cp["order_amount"].std(),1.0)
    z=(r["order_amount"]-mu)/sd
    r["addr_mismatch"]=(r["shipping_state"]!=r["billing_state"]).astype(int)
    r["payment_risky"]=r["payment_method"].isin(["paypal","gift_card"]).astype(int)
    r["high_amount_cat"]=(r["order_amount"]>=p90).astype(int)
    r["price_high_anom"]=(z>=2).astype(int); r["price_low_anom"]=(z<=-2).astype(int)
    r["express_high_amt"]=((r["express_shipping"]==1)&(r["high_amount_cat"]==1)).astype(int)
    r["return_whiplash"]=(r["returns_30d"]>=2).astype(int)
    r["log_amount"]=np.log1p(r["order_amount"])
    for c in ["oversell_flag","hoarding_flag","on_hand"]:
        if c not in r.columns: r[c]=0
    return float(clf.predict_proba(r[features])[:,1])

st.markdown("### Instant Decision — Test a New Order")
cc1,cc2,cc3=st.columns(3)
chan_sel=cc1.selectbox("Channel", ["Online","In-Store"])
cat_sel =cc2.selectbox("SKU Category", sorted(df["sku_category"].unique()))
pay_sel =cc3.selectbox("Payment Method", payment_options(chan_sel))
c4,c5,c6=st.columns(3)
amt_sel=c4.number_input("Order Amount ($)", min_value=1, max_value=10000, value=250, step=1)
qty_sel=c5.slider("Quantity", 1, 10, 1)
exp_sel=c6.checkbox("Express Shipping", value=(chan_sel=="Online"))
c7,c8=st.columns(2)
ship_sel=c7.selectbox("Shipping State", sorted(us_states))
bill_sel=c8.selectbox("Billing State", sorted(us_states))

example=dict(channel=chan_sel, sku_category=cat_sel, payment_method=pay_sel,
             shipping_state=ship_sel, billing_state=bill_sel, quantity=qty_sel,
             order_amount=amt_sel, log_amount=np.log1p(amt_sel),
             express_shipping=int(exp_sel), returns_30d=0, on_hand=80)

if st.button("Score Order"):
    fs=score_new_order(example)
    st.success(f"Decision: {'ALERT' if fs>=TH else 'PASS'} · Fraud Score ≈ {fs:.2f} "
               f"(Best {BEST_THR:.2f} · Business {TH:.2f})")
