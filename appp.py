# appp.py — Unified ML-Driven Fraud Detection for Retail POS (USA-only, tuned for Accuracy)
import time, warnings, numpy as np, pandas as pd, altair as alt, streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             precision_recall_curve, roc_auc_score, auc)

warnings.filterwarnings("ignore"); np.random.seed(42)
st.set_page_config(page_title="Unified ML-Driven Fraud Detection for Retail POS (USA)", layout="wide")
alt.renderers.set_embed_options(actions=False)

# ───────── 1) Create 4,000 USA transactions (Online + In-Store) ─────────
N=4000
US_STATES=np.array(["CA","TX","NY","FL","IL","WA","GA","PA","OH","MI"])
store_ids=np.array(["US01","US02","US03","US04"])
store_state_pool=np.random.choice(US_STATES, size=len(store_ids), replace=False)
idx=np.random.choice(len(store_ids), N)
store_id=store_ids[idx]; store_state=store_state_pool[idx]

channel = np.random.choice(["Online","In-Store"], N, p=[0.7,0.3])
category= np.random.choice(["Electronics","Apparel","Grocery","Beauty","Home"], N)
amount  = np.round(np.random.lognormal(5.5,0.8,N),2)
quantity= np.random.randint(1,6,N)

payment=[]
for i in range(N):
    if channel[i]=="In-Store": payment.append(np.random.choice(["credit_card","debit_card","apple_pay","gift_card"]))
    else:                      payment.append(np.random.choice(["credit_card","debit_card","apple_pay","paypal"]))

shipping_state=np.where(channel=="In-Store", store_state,
                        np.random.choice(US_STATES, N))
billing_state =np.random.choice(US_STATES, N)
express = np.where(channel=="Online", np.random.choice([0,1],N,p=[0.7,0.3]), 0)
returns30 = np.random.poisson(0.2, N)
sku=np.random.randint(100000,110000,N)
onhand=np.random.randint(10,200,N)

df=pd.DataFrame(dict(
    order_id=np.arange(1,N+1), channel=channel, store_id=store_id, store_state=store_state,
    sku_category=category, order_amount=amount, quantity=quantity, payment_method=payment,
    shipping_state=shipping_state, billing_state=billing_state, express_shipping=express,
    returns_30d=returns30, sku=sku, on_hand=onhand
))

# ───────── 2) Strong features (clear, measurable) ─────────
df["addr_mismatch"]  =(df["shipping_state"]!=df["billing_state"]).astype(int)
df["payment_risky"]  = df["payment_method"].isin(["paypal","gift_card"]).astype(int)
p90 = df.groupby("sku_category")["order_amount"].transform(lambda s:s.quantile(0.90))
mu  = df.groupby("sku_category")["order_amount"].transform("mean")
sd  = df.groupby("sku_category")["order_amount"].transform("std").replace(0,1.0)
z   = (df["order_amount"]-mu)/sd
df["high_amount_cat"]=(df["order_amount"]>=p90).astype(int)
df["price_high_anom"]=(z>=2).astype(int)
df["price_low_anom"] =(z<=-2).astype(int)
df["oversell_flag"]  =(df["quantity"]>0.6*df["on_hand"]).astype(int)
df["hoarding_flag"]  =(df["quantity"]>=4).astype(int)
df["return_whiplash"]=(df["returns_30d"]>=2).astype(int)
df["express_high_amt"]=((df["express_shipping"]==1)&(df["high_amount_cat"]==1)).astype(int)
df["log_amount"]     = np.log1p(df["order_amount"])

# ───────── 3) Outcome label (generated from business drivers) ─────────
p = (0.04 + 0.22*df["addr_mismatch"] + 0.18*df["payment_risky"] + 0.20*df["high_amount_cat"]
     + 0.08*df["return_whiplash"] + 0.10*df["express_high_amt"] + 0.06*df["oversell_flag"]
     + 0.05*df["hoarding_flag"])
df["fraud_flag"] = (np.random.rand(N) < np.clip(p,0,0.95)).astype(int)

# ───────── 4) Train model → fraud_score (probability) ─────────
features=["channel","sku_category","payment_method","shipping_state","billing_state",
          "quantity","order_amount","log_amount","express_shipping","returns_30d",
          "addr_mismatch","payment_risky","high_amount_cat","price_high_anom","price_low_anom",
          "oversell_flag","hoarding_flag","return_whiplash","express_high_amt","on_hand"]
X,y=df[features], df["fraud_flag"]
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

num=["quantity","order_amount","log_amount","express_shipping","returns_30d","addr_mismatch","payment_risky",
     "high_amount_cat","price_high_anom","price_low_anom","oversell_flag","hoarding_flag",
     "return_whiplash","express_high_amt","on_hand"]
cat=["channel","sku_category","payment_method","shipping_state","billing_state"]

pre=ColumnTransformer([
    ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]), num),
    ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                     ("oh",OneHotEncoder(handle_unknown="ignore"))]), cat)
])
clf=Pipeline([("pre",pre),("lr",LogisticRegression(max_iter=300,class_weight="balanced"))]).fit(Xtr,ytr)
yprob = clf.predict_proba(Xte)[:,1]   # fraud_score (0..1)

# ───────── 5) Threshold search (Best-Accuracy & Best-F1) ─────────
def evaluate_at(thr: float):
    yhat=(yprob>=thr).astype(int)
    return dict(
        thr=thr,
        acc=accuracy_score(yte,yhat),
        prec=precision_score(yte,yhat,zero_division=0),
        rec=recall_score(yte,yhat,zero_division=0),
        f1=f1_score(yte,yhat,zero_division=0),
        alert_rate=float(yhat.mean())
    )

# grid of thresholds 0.00..0.99
grid=np.linspace(0,0.99,100)
evals=[evaluate_at(t) for t in grid]
best_acc = max(evals, key=lambda d:d["acc"])
# best F1 using PR curve
prec,rec,thr = precision_recall_curve(yte,yprob)
f1 = (2*prec*rec)/(prec+rec+1e-9)
best_f1_thr = float(thr[int(np.argmax(f1))]) if len(thr)>0 else 0.5
best_f1 = evaluate_at(best_f1_thr)

# ───────── 6) Dashboard UI ─────────
st.title("Unified ML-Driven Fraud Detection for Retail POS (USA)")
st.subheader("Bridging Behavioral Risk, Pricing Anomalies, and Inventory Stress")

with st.sidebar:
    st.header("Thresholds")
    st.metric("Best Accuracy Threshold", f"{best_acc['thr']:.2f}")
    st.metric("Best F1 Threshold", f"{best_f1['thr']:.2f}")
    TH = st.slider("Business Threshold (override)", 0.00, 1.00, float(round(best_acc["thr"],2)), 0.01,
                   help="Lower → more alerts (higher recall). Higher → fewer alerts (often higher accuracy).")

# KPIs at Business Threshold
M = evaluate_at(TH)
k1,k2,k3,k4 = st.columns(4)
k1.metric("Total Transactions (USA)", f"{len(df):,}")
k2.metric("ML-Detected Orders", f"{int(M['alert_rate']*len(yprob)):,}")
k3.metric("Alert Rate", f"{M['alert_rate']:.1%}")
k4.metric("Accuracy @ Threshold", f"{M['acc']:.0%}")

# ───────── 7) All 4,000 transactions (USA) ─────────
st.markdown("### Retail Transaction Data (USA)")
cols_tx=["order_id","channel","store_id","store_state","sku_category",
         "order_amount","quantity","payment_method","shipping_state",
         "billing_state","express_shipping","returns_30d"]
st.dataframe(df[cols_tx], use_container_width=True, height=420)
st.download_button("Download all transactions (CSV)",
                   data=df[cols_tx].to_csv(index=False).encode("utf-8"),
                   file_name="retail_transactions_usa_4000.csv", mime="text/csv",
                   use_container_width=True)

# ───────── 8) Fraud Score Distribution ─────────
st.markdown("### Fraud Score Distribution (ML Output)")
hist=alt.Chart(pd.DataFrame({"fraud_score":yprob})).mark_bar().encode(
    x=alt.X("fraud_score:Q", bin=alt.Bin(maxbins=40), title="Fraud Score (0–1)"),
    y=alt.Y("count():Q", title="Transactions")
).properties(height=200)
st.altair_chart(hist, use_container_width=True)

# ───────── 9) ML-Detected Suspicious Orders (joined correctly) ─────────
st.markdown("### ML-Detected Suspicious Orders")
scores=pd.Series(yprob, index=Xte.index, name="fraud_score")
alerts=(df.loc[Xte.index, cols_tx]
          .assign(fraud_score=scores)
          .loc[lambda d: d["fraud_score"]>=TH]
          .sort_values("fraud_score", ascending=False))
st.dataframe(alerts[["order_id","channel","store_id","store_state","sku_category",
                     "order_amount","quantity","payment_method","shipping_state",
                     "billing_state","fraud_score"]],
             use_container_width=True, height=320)
st.download_button("Download suspicious orders (CSV)",
                   data=alerts.to_csv(index=False).encode("utf-8"),
                   file_name="ml_detected_suspicious_orders_usa.csv", mime="text/csv",
                   use_container_width=True)

# ───────── 10) Strong Feature Counts (how many have each) ─────────
st.markdown("### Strong Feature Counts (Alerts vs Non-Alerts)")
def feature_counts(df_alert_flag: pd.Series, cols: list):
    A = df_alert_flag.sum()
    NA = len(df_alert_flag)-A
    # build table with counts in alerts & non-alerts
    rows=[]
    for c in cols:
        if df[c].dtype!=int and df[c].dtype!=bool: continue
        in_alerts = int(df.loc[df_alert_flag==1, c].sum())
        in_normal = int(df.loc[df_alert_flag==0, c].sum())
        rows.append([c, in_alerts, in_normal,
                     f"{(in_alerts/max(A,1)):.0%}", f"{(in_normal/max(NA,1)):.0%}"])
    return pd.DataFrame(rows, columns=["feature","count_in_alerts","count_in_normal",
                                       "%alerts_with_feature","%normal_with_feature"])

ALERT_FLAG = (scores>=TH).reindex(df.index, fill_value=False)  # mark only test rows realistically
ALERT_FLAG.loc[~df.index.isin(Xte.index)] = False
strong_cols=["addr_mismatch","payment_risky","high_amount_cat","price_high_anom",
             "price_low_anom","express_high_amt","return_whiplash","oversell_flag","hoarding_flag"]
tbl = feature_counts(ALERT_FLAG.astype(int), strong_cols)
st.dataframe(tbl, use_container_width=True, height=280)

# ───────── 11) Model Evaluation (at Business Threshold) ─────────
st.markdown("### Model Evaluation (at Business Threshold)")
e1,e2,e3,e4 = st.columns(4)
e1.metric("Accuracy", f"{M['acc']:.0%}")
e2.metric("Precision", f"{M['prec']:.0%}")
e3.metric("Recall", f"{M['rec']:.0%}")
e4.metric("F1-Score", f"{M['f1']:.0%}")
st.caption(f"Best Accuracy Threshold = {best_acc['thr']:.2f} · Best F1 Threshold = {best_f1['thr']:.2f} · ROC-AUC = {roc_auc_score(yte,yprob):.2f}")

# ───────── 12) Instant Decision — Test a New Order (numeric inputs) ─────────
st.markdown("### Instant Decision — Test a New Order")
def payment_options(chan:str):
    return ["credit_card","debit_card","apple_pay","gift_card"] if chan=="In-Store" \
           else ["credit_card","debit_card","apple_pay","paypal"]

c1,c2,c3 = st.columns(3)
chan_sel=c1.selectbox("Channel", ["Online","In-Store"])
cat_sel =c2.selectbox("SKU Category", sorted(df["sku_category"].unique()))
pay_sel =c3.selectbox("Payment Method", payment_options(chan_sel))
c4,c5,c6 = st.columns(3)
amt_sel =c4.number_input("Order Amount ($)", min_value=1, max_value=10000, value=250, step=1)
qty_sel =c5.number_input("Quantity", min_value=1, max_value=50, value=1, step=1)
exp_sel =c6.selectbox("Express Shipping", [0,1], index=0)
c7,c8 = st.columns(2)
ship_sel=c7.selectbox("Shipping State", sorted(US_STATES))
bill_sel=c8.selectbox("Billing State", sorted(US_STATES))

def score_new_order(order:dict)->float:
    r=pd.DataFrame([order])
    cp=df[df["sku_category"]==r["sku_category"].iloc[0]]
    if cp.empty: cp=df
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

example=dict(channel=chan_sel, sku_category=cat_sel, payment_method=pay_sel,
             shipping_state=ship_sel, billing_state=bill_sel, quantity=int(qty_sel),
             order_amount=int(amt_sel), log_amount=np.log1p(int(amt_sel)),
             express_shipping=int(exp_sel), returns_30d=0, on_hand=80)
if st.button("Score Order"):
    fs=score_new_order(example)
    st.success(f"Decision: {'ALERT' if fs>=TH else 'PASS'} · Fraud Score ≈ {fs:.2f} "
               f"(Best-Acc {best_acc['thr']:.2f} · Best-F1 {best_f1['thr']:.2f} · Business {TH:.2f})")
