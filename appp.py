# retail_fraud_end2end.py  — end-to-end, business-ready, no SQL/BQML
import numpy as np, pandas as pd, warnings, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, roc_auc_score
warnings.filterwarnings("ignore")

np.random.seed(7)

# ---------------- 1) Build 4,000 transactions (Online + In-Store) ----------------
N=4000
stores=[("TX01","US"),("CA03","CA"),("NY01","US"),("UK11","UK")]
store_id,store_cty=zip(*np.random.choice(stores,N))
ch=np.random.choice(["Online","In-Store"],N,p=[0.7,0.3])
cat=np.random.choice(["Electronics","Apparel","Grocery","Beauty","Home"],N)
amt=np.round(np.random.lognormal(5.5,0.8,N),2)
qty=np.random.randint(1,6,N)
# payments by channel: In-Store {cc,dc,apple,gift}, Online {cc,dc,apple,paypal}
pay=[]
for i in range(N):
    pay.append(np.random.choice(["credit_card","debit_card","apple_pay","gift_card"] if ch[i]=="In-Store"
                                else ["credit_card","debit_card","apple_pay","paypal"]))
pay=np.array(pay)
ship=np.where(ch=="In-Store",np.array(store_cty),np.random.choice(["US","CA","UK"],N,p=[0.8,0.12,0.08]))
bill=np.random.choice(["US","CA","UK"],N,p=[0.82,0.1,0.08])
express=np.where(ch=="Online",np.random.choice([0,1],N,p=[0.7,0.3]),0)
ret30=np.random.poisson(0.2,N)
sku=np.random.randint(100000,110000,N)
onhand=np.random.randint(10,200,N)

df=pd.DataFrame(dict(order_id=np.arange(1,N+1),channel=ch,store_id=store_id,store_country=store_cty,
                     sku_category=cat,order_amount=amt,quantity=qty,payment_method=pay,
                     shipping_country=ship,billing_country=bill,express_shipping=express,
                     returns_30d=ret30,sku=sku,on_hand=onhand))

# ---------------- 2) Feature engineering (BUSINESS signals) ----------------
# Address mismatch (no IP used)
df["addr_mismatch"]=(df["shipping_country"]!=df["billing_country"]).astype(int)
# Risky payment channels (PayPal online; Gift cards in-store)
df["payment_risky"]=df["payment_method"].isin(["paypal","gift_card"]).astype(int)
# Category-relative high amount (p90) + price anomalies (z-score)
p90=df.groupby("sku_category")["order_amount"].transform(lambda s:s.quantile(0.90))
mean=df.groupby("sku_category")["order_amount"].transform("mean")
std=df.groupby("sku_category")["order_amount"].transform("std").replace(0,1.0)
z=(df["order_amount"]-mean)/std
df["high_amount_cat"]=(df["order_amount"]>=p90).astype(int)
df["price_high_anom"]=(z>=2).astype(int)
df["price_low_anom"]=(z<=-2).astype(int)
# Inventory stress
df["oversell_flag"]=(df["quantity"]>0.6*df["on_hand"]).astype(int)
df["hoarding_flag"]=(df["quantity"]>=4).astype(int)
# Returns and express+amount synergy
df["return_whiplash"]=(df["returns_30d"]>=2).astype(int)
df["express_high_amt"]=((df["express_shipping"]==1)&(df["high_amount_cat"]==1)).astype(int)
df["log_amount"]=np.log1p(df["order_amount"])

# ---------------- 3) Create a training label (for demo) ----------------
# Fraud probability generator (business-plausible)
p = (0.04 + 0.22*df["addr_mismatch"] + 0.18*df["payment_risky"] +
     0.20*df["high_amount_cat"] + 0.08*df["return_whiplash"] +
     0.10*df["express_high_amt"] + 0.06*df["oversell_flag"] + 0.05*df["hoarding_flag"])
df["fraud_flag"]=(np.random.rand(N)<np.clip(p,0,0.95)).astype(int)

# ---------------- 4) Train model and compute fraud_score ----------------
features=["channel","sku_category","payment_method","shipping_country","billing_country",
          "quantity","order_amount","log_amount","express_shipping","returns_30d",
          "addr_mismatch","payment_risky","high_amount_cat","price_high_anom","price_low_anom",
          "oversell_flag","hoarding_flag","return_whiplash","express_high_amt","on_hand"]
X=df[features]; y=df["fraud_flag"].astype(int)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=7,stratify=y)

num=["quantity","order_amount","log_amount","express_shipping","returns_30d",
     "addr_mismatch","payment_risky","high_amount_cat","price_high_anom","price_low_anom",
     "oversell_flag","hoarding_flag","return_whiplash","express_high_amt","on_hand"]
cat=["channel","sku_category","payment_method","shipping_country","billing_country"]
pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
                       ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                                        ("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])
# Using LogisticRegression: fast, stable, explainable; outputs calibrated probabilities
clf=Pipeline([("pre",pre),("lr",LogisticRegression(max_iter=300,class_weight="balanced"))])
clf.fit(Xtr,ytr)
yprob=clf.predict_proba(Xte)[:,1]  # fraud_score = P(fraud=1)

# ---------------- 5) Threshold selection & evaluation ----------------
# Best threshold by F1 (balanced capture vs noise)
prec,rec,thr=precision_recall_curve(yte,yprob); f1=(2*prec*rec)/np.clip(prec+rec,1e-9,None)
best_idx=int(np.nanargmax(f1)); best_thr=(thr[best_idx] if best_idx<len(thr) else 0.5)
ypred=(yprob>=best_thr).astype(int)

acc=accuracy_score(yte,ypred); pr=precision_score(yte,ypred,zero_division=0)
rc=recall_score(yte,ypred,zero_division=0); f1s=f1_score(yte,ypred,zero_division=0)
roc=roc_auc_score(yte,yprob); pr_auc=auc(rec,prec)

# ---------------- 6) Business summary (prints you can read in the meeting) ----------------
print("\nUNIFIED ML-DRIVEN FRAUD FOR RETAIL POS — BUSINESS SUMMARY")
print(f"- Data: 4,000 transactions (Online + In-Store). Payments respected by channel.")
print("- Model: Logistic Regression in an sklearn Pipeline (numeric scaling + one-hot for categories).")
print("- fraud_score: model probability that a transaction is fraudulent (predict_proba for class=1).")
print(f"- Best threshold (max F1): {best_thr:.2f}  → at this setting:")
print(f"  Accuracy={acc:.2%}  Precision={pr:.2%}  Recall={rc:.2%}  F1={f1s:.2%}  ROC-AUC={roc:.2%}  PR-AUC={pr_auc:.2%}")
print("- How we choose threshold:")
print("  • Business can keep a fixed default (e.g., 0.50), or use the best-F1 threshold above")
print("    to balance catching fraud (recall) and keeping alerts clean (precision).")
print("- Strong features (business perspective, no IP used):")
feat_desc={
 "addr_mismatch":"Shipping vs Billing address differ.",
 "payment_risky":"PayPal (online) or Gift Cards (in-store).",
 "high_amount_cat":"Order above 90th percentile for its category.",
 "price_high_anom":"Price unusually high versus peers (z-score ≥ 2).",
 "price_low_anom":"Price unusually low (possible discount abuse).",
 "express_high_amt":"Express shipping combined with high amount.",
 "return_whiplash":"Multiple returns in last 30 days.",
 "oversell_flag":"Quantity too high relative to inventory on-hand.",
 "hoarding_flag":"Customer buying unusually large quantities."
}
for k,v in feat_desc.items(): print(f"  • {k}: {v}")
print("- What we did end-to-end: generated transaction data, engineered business signals,")
print("  trained the model, produced fraud_score for each order, selected the best threshold,")
print("  and evaluated results in business terms (accuracy, precision, recall, F1).")

# ---------------- 7) Example: score one new order (what-if) ----------------
def score_new_order(order:dict)->float:
    """Return fraud_score ∈ [0,1] for a single order dict (must contain model features)."""
    r=pd.DataFrame([order])
    # derive the same signals
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
    for c in ["oversell_flag","hoarding_flag","on_hand"]: r.setdefault(c,0)
    r=r[features]  # order columns for the pipeline
    return float(clf.predict_proba(r)[:,1])

example=dict(channel="Online",sku_category="Electronics",payment_method="paypal",
             shipping_country="US",billing_country="US",quantity=2,order_amount=1800,
             log_amount=np.log1p(1800),express_shipping=1,returns_30d=0,
             addr_mismatch=0,payment_risky=1,high_amount_cat=1,price_high_anom=0,
             price_low_anom=0,oversell_flag=0,hoarding_flag=0,return_whiplash=0,
             express_high_amt=1,on_hand=80)
fs=score_new_order(example)
print(f"\nInstant decision example → fraud_score={fs:.2f}  decision@{best_thr:.2f}="
      f"{'ALERT' if fs>=best_thr else 'PASS'}")

# (Optional) export predictions for a dashboard
pred=pd.DataFrame({"fraud_score":yprob,"is_alert":(yprob>=best_thr).astype(int),"actual":yte.values})
pred.to_csv("predictions_latest.csv",index=False)
print("\nSaved predictions_latest.csv for your dashboard.")
