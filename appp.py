# appp.py — Unified ML-Driven Fraud Detection for Retail POS (business-ready, single file)

import numpy as np, pandas as pd, warnings, time, altair as alt, streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             precision_recall_curve, auc, roc_auc_score)

warnings.filterwarnings("ignore"); np.random.seed(42)
st.set_page_config(page_title="Unified ML-Driven Fraud Detection for Retail POS", layout="wide")
alt.renderers.set_embed_options(actions=False)

# ─────────────── 1) Create 4,000 retail transactions (Online + In-Store) ───────────────
N=4000
store_ids=np.array(["TX01","CA03","NY01","UK11"]); store_ctys=np.array(["US","CA","US","UK"])
idx=np.random.choice(len(store_ids),N); store_id,store_cty=store_ids[idx],store_ctys[idx]
channel=np.random.choice(["Online","In-Store"],N,p=[0.7,0.3])
category=np.random.choice(["Electronics","Apparel","Grocery","Beauty","Home"],N)
amount=np.round(np.random.lognormal(5.5,0.8,N),2); quantity=np.random.randint(1,6,N)
payment=[np.random.choice(["credit_card","debit_card","apple_pay","gift_card"]) if channel[i]=="In-Store"
         else np.random.choice(["credit_card","debit_card","apple_pay","paypal"]) for i in range(N)]
shipping=np.where(channel=="In-Store",store_cty,np.random.choice(["US","CA","UK"],N,p=[0.8,0.12,0.08]))
billing =np.random.choice(["US","CA","UK"],N,p=[0.82,0.10,0.08])
express =np.where(channel=="Online",np.random.choice([0,1],N,p=[0.7,0.3]),0)
returns30=np.random.poisson(0.2,N); sku=np.random.randint(100000,110000,N); onhand=np.random.randint(10,200,N)

df=pd.DataFrame(dict(order_id=np.arange(1,N+1),channel=channel,store_id=store_id,store_country=store_cty,
                     sku_category=category,order_amount=amount,quantity=quantity,payment_method=payment,
                     shipping_country=shipping,billing_country=billing,express_shipping=express,
                     returns_30d=returns30,sku=sku,on_hand=onhand))

# ─────────────── 2) Strong business features (no IP) ───────────────
df["addr_mismatch"]  =(df["shipping_country"]!=df["billing_country"]).astype(int)
df["payment_risky"]  = df["payment_method"].isin(["paypal","gift_card"]).astype(int)
p90=df.groupby("sku_category")["order_amount"].transform(lambda s:s.quantile(0.90))
mean=df.groupby("sku_category")["order_amount"].transform("mean")
std =df.groupby("sku_category")["order_amount"].transform("std").replace(0,1.0)
z=(df["order_amount"]-mean)/std
df["high_amount_cat"]=(df["order_amount"]>=p90).astype(int)
df["price_high_anom"]=(z>=2).astype(int); df["price_low_anom"]=(z<=-2).astype(int)
df["oversell_flag"]  =(df["quantity"]>0.6*df["on_hand"]).astype(int)
df["hoarding_flag"]  =(df["quantity"]>=4).astype(int)
df["return_whiplash"]=(df["returns_30d"]>=2).astype(int)
df["express_high_amt"]=((df["express_shipping"]==1)&(df["high_amount_cat"]==1)).astype(int)
df["log_amount"]      =np.log1p(df["order_amount"])

# ─────────────── 3) Outcome for training (probability → sampled label) ───────────────
p = (0.04 + 0.22*df["addr_mismatch"] + 0.18*df["payment_risky"] + 0.20*df["high_amount_cat"]
     + 0.08*df["return_whiplash"] + 0.10*df["express_high_amt"] + 0.06*df["oversell_flag"]
     + 0.05*df["hoarding_flag"])
df["fraud_flag"]=(np.random.rand(N)<np.clip(p,0,0.95)).astype(int)

# ─────────────── 4) Train model → fraud_score (probability) ───────────────
features=["channel","sku_category","payment_method","shipping_country","billing_country",
          "quantity","order_amount","log_amount","express_shipping","returns_30d",
          "addr_mismatch","payment_risky","high_amount_cat","price_high_anom","price_low_anom",
          "oversell_flag","hoarding_flag","return_whiplash","express_high_amt","on_hand"]
X,y=df[features],df["fraud_flag"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

num=["quantity","order_amount","log_amount","express_shipping","returns_30d","addr_mismatch","payment_risky",
     "high_amount_cat","price_high_anom","price_low_anom","oversell_flag","hoarding_flag",
     "return_whiplash","express_high_amt","on_hand"]
cat=["channel","sku_category","payment_method","shipping_country","billing_country"]

pre=ColumnTransformer([
  ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
  ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)
])
clf=Pipeline([("pre",pre),("lr",LogisticRegression(max_iter=300,class_weight="balanced"))]).fit(Xtr,ytr)
yprob=clf.predict_proba(Xte)[:,1]  # fraud_score

# Threshold by best F1
prec,rec,thr=precision_recall_curve(yte,yprob); f1=(2*prec*rec)/(prec+rec+1e-9)
best_idx=int(np.argmax(f1)); BEST_THR=float(thr[best_idx] if best_idx<len(thr) else 0.50)

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

# ─────────────── 5) Streamlit dashboard ───────────────
st.title("Unified ML-Driven Fraud Detection for Retail POS")
st.subheader("Bridging Behavioral Risk, Pricing Anomalies, and Inventory Stress")
st.caption(f"Build: {time.strftime('%Y-%m-%d %H:%M:%S')} · File: appp.py")

with st.sidebar:
    st.header("Business Control")
    TH=st.slider("Decision Threshold", 0.00, 1.00, float(round(BEST_THR,2)), 0.01,
                 help="Lower = stricter (more alerts). Higher = looser (fewer alerts).")
    st.caption("Threshold sets the line where an order is flagged as **ML-detected suspicious**.")

metrics=evaluate_at(TH)
c1,c2,c3,c4=st.columns(4)
c1.metric("Total Transactions", f"{len(df):,}")
c2.metric("ML-Detected Orders", f"{int(metrics['alert_rate']*len(yprob)):,}")
c3.metric("Alert Rate", f"{metrics['alert_rate']:.1%}")
c4.metric("Best-F1 Threshold", f"{BEST_THR:.2f}")

st.markdown("### Retail Transactions (Online + In-Store)")
cols_tx=["order_id","channel","store_id","store_country","sku_category","order_amount",
         "quantity","payment_method","shipping_country","billing_country","express_shipping","returns_30d"]
st.dataframe(df[cols_tx].head(200), use_container_width=True, height=260)

st.markdown("### Fraud Score Distribution (AI/ML Output)")
hist=alt.Chart(pd.DataFrame({"fraud_score":yprob})).mark_bar().encode(
    x=alt.X("fraud_score:Q", bin=alt.Bin(maxbins=40), title="Fraud Score (0–1)"),
    y=alt.Y("count():Q", title="Transactions")
).properties(height=220)
st.altair_chart(hist, use_container_width=True)

st.markdown("### ML-Detected Suspicious Orders")
alerts=(pd.DataFrame(Xte).assign(fraud_score=yprob)
        .loc[lambda d:(d["fraud_score"]>=TH)]
        .sort_values("fraud_score", ascending=False))
st.dataframe(alerts[cols_tx[1:5]+["order_amount","quantity","payment_method","shipping_country","billing_country","fraud_score"]]
             .head(200), use_container_width=True, height=300)

st.markdown("### Model Evaluation")
m=metrics
e1,e2,e3,e4=st.columns(4)
e1.metric("Accuracy", f"{m['acc']:.0%}")
e2.metric("Precision", f"{m['prec']:.0%}")
e3.metric("Recall", f"{m['rec']:.0%}")
e4.metric("F1-Score", f"{m['f1']:.0%}")
st.caption("Accuracy = overall correctness · Precision = quality of alerts · Recall = % of true fraud captured · F1 = balance.")

st.markdown("### Strong Features (used in real-time decisions)")
st.write(
"""
- **Address mismatch** — shipping and billing differ  
- **Risky payment channels** — PayPal (online) or gift cards (in-store)  
- **High amount for category** — spend above the 90th percentile  
- **Price anomalies** — unusually high (possible fraud) or low (discount abuse)  
- **Express + high value** — fast shipping on expensive orders  
- **Returns behavior** — multiple recent returns (return whiplash)  
- **Inventory pressure** — oversell risk or hoarding (bulk quantity)
"""
)

# ─────────────── Instant Decision (what-if) ───────────────
def score_new_order(order:dict)->float:
    r=pd.DataFrame([order])
    cp=df[df["sku_category"]==r["sku_category"].iloc[0]]
    if cp.empty: cp=df
    p90=cp["order_amount"].quantile(0.90); mu=cp["order_amount"].mean(); sd=max(cp["order_amount"].std(),1.0)
    z=(r["order_amount"]-mu)/sd
    r["addr_mismatch"]=(r["shipping_country"]!=r["billing_country"]).astype(int)
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
c1,c2,c3=st.columns(3)
cat_sel=c1.selectbox("SKU Category", sorted(df["sku_category"].unique()))
pay_sel=c2.selectbox("Payment Method",
                     ["credit_card","debit_card","apple_pay","gift_card","paypal"])
amt_sel=c3.number_input("Order Amount", min_value=1, max_value=10000, value=250, step=1)
c4,c5=st.columns(2)
ship_sel=c4.selectbox("Shipping Country", ["US","CA","UK"])
bill_sel=c5.selectbox("Billing Country", ["US","CA","UK"])
exp_sel=st.checkbox("Express Shipping", value=False)
qty_sel=st.slider("Quantity", 1, 10, 1)

example=dict(channel="Online" if pay_sel in ("paypal","credit_card","debit_card","apple_pay") else "In-Store",
             sku_category=cat_sel, payment_method=pay_sel, shipping_country=ship_sel, billing_country=bill_sel,
             quantity=qty_sel, order_amount=amt_sel, log_amount=np.log1p(amt_sel),
             express_shipping=int(exp_sel), returns_30d=0, on_hand=80)

if st.button("Score Order"):
    fs=score_new_order(example)
    st.success(f"Decision: {'ALERT' if fs>=TH else 'PASS'} · Fraud Score ≈ {fs:.2f} (Threshold {TH:.2f})")

