# retail_fraud_end2end.py
# End-to-end Retail Fraud Detection demo (4k transactions, Online + In-store)
import numpy as np, pandas as pd, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             precision_recall_curve, auc, roc_auc_score)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------- 1) Generate 4k Transactions ----------------
N = 4000
store_ids = np.array(["TX01","CA03","NY01","UK11"])
store_ctys = np.array(["US","CA","US","UK"])
idx = np.random.choice(len(store_ids), N)
store_id, store_cty = store_ids[idx], store_ctys[idx]

channel = np.random.choice(["Online","In-Store"], N, p=[0.7,0.3])
category = np.random.choice(["Electronics","Apparel","Grocery","Beauty","Home"], N)
amount = np.round(np.random.lognormal(5.5,0.8,N),2)
quantity = np.random.randint(1,6,N)

payment = []
for i in range(N):
    if channel[i]=="In-Store":
        payment.append(np.random.choice(["credit_card","debit_card","apple_pay","gift_card"]))
    else:
        payment.append(np.random.choice(["credit_card","debit_card","apple_pay","paypal"]))

shipping = np.where(channel=="In-Store", store_cty,
            np.random.choice(["US","CA","UK"],N,p=[0.8,0.12,0.08]))
billing = np.random.choice(["US","CA","UK"], N, p=[0.82,0.1,0.08])
express = np.where(channel=="Online", np.random.choice([0,1],N,p=[0.7,0.3]),0)
returns30 = np.random.poisson(0.2,N)
sku = np.random.randint(100000,110000,N)
onhand = np.random.randint(10,200,N)

df = pd.DataFrame(dict(order_id=np.arange(1,N+1), channel=channel, store_id=store_id,
                       store_country=store_cty, sku_category=category, order_amount=amount,
                       quantity=quantity, payment_method=payment,
                       shipping_country=shipping, billing_country=billing,
                       express_shipping=express, returns_30d=returns30,
                       sku=sku, on_hand=onhand))

# ---------------- 2) Strong Business Features ----------------
df["addr_mismatch"] = (df["shipping_country"]!=df["billing_country"]).astype(int)
df["payment_risky"] = df["payment_method"].isin(["paypal","gift_card"]).astype(int)

p90 = df.groupby("sku_category")["order_amount"].transform(lambda s: s.quantile(0.90))
mean = df.groupby("sku_category")["order_amount"].transform("mean")
std  = df.groupby("sku_category")["order_amount"].transform("std").replace(0,1.0)
z = (df["order_amount"]-mean)/std

df["high_amount_cat"] = (df["order_amount"]>=p90).astype(int)
df["price_high_anom"] = (z>=2).astype(int)
df["price_low_anom"]  = (z<=-2).astype(int)
df["oversell_flag"]   = (df["quantity"]>0.6*df["on_hand"]).astype(int)
df["hoarding_flag"]   = (df["quantity"]>=4).astype(int)
df["return_whiplash"] = (df["returns_30d"]>=2).astype(int)
df["express_high_amt"]= ((df["express_shipping"]==1)&(df["high_amount_cat"]==1)).astype(int)
df["log_amount"]      = np.log1p(df["order_amount"])

# ---------------- 3) Simulated Fraud Label ----------------
p = (0.04 + 0.22*df["addr_mismatch"] + 0.18*df["payment_risky"] +
     0.20*df["high_amount_cat"] + 0.08*df["return_whiplash"] +
     0.10*df["express_high_amt"] + 0.06*df["oversell_flag"] + 0.05*df["hoarding_flag"])
df["fraud_flag"] = (np.random.rand(N)<np.clip(p,0,0.95)).astype(int)

# ---------------- 4) Model Training ----------------
features = ["channel","sku_category","payment_method","shipping_country","billing_country",
            "quantity","order_amount","log_amount","express_shipping","returns_30d",
            "addr_mismatch","payment_risky","high_amount_cat","price_high_anom",
            "price_low_anom","oversell_flag","hoarding_flag","return_whiplash",
            "express_high_amt","on_hand"]
X, y = df[features], df["fraud_flag"]

Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

num = ["quantity","order_amount","log_amount","express_shipping","returns_30d",
       "addr_mismatch","payment_risky","high_amount_cat","price_high_anom",
       "price_low_anom","oversell_flag","hoarding_flag","return_whiplash",
       "express_high_amt","on_hand"]
cat = ["channel","sku_category","payment_method","shipping_country","billing_country"]

pre = ColumnTransformer([
    ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
    ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                     ("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)
])

clf = Pipeline([("pre",pre),("lr",LogisticRegression(max_iter=300,class_weight="balanced"))])
clf.fit(Xtr,ytr)

yprob = clf.predict_proba(Xte)[:,1]  # fraud_score

# ---------------- 5) Threshold & Evaluation ----------------
prec,rec,thr = precision_recall_curve(yte,yprob)
f1s = (2*prec*rec)/(prec+rec+1e-9)
best_idx = np.argmax(f1s)
best_thr = thr[best_idx] if best_idx<len(thr) else 0.5
ypred = (yprob>=best_thr).astype(int)

acc = accuracy_score(yte,ypred)
pr  = precision_score(yte,ypred)
rc  = recall_score(yte,ypred)
f1  = f1_score(yte,ypred)
roc = roc_auc_score(yte,yprob)
pr_auc = auc(rec,prec)

# ---------------- 6) Business Summary ----------------
print("\nUNIFIED ML-DRIVEN FRAUD DETECTION FOR RETAIL POS")
print("- 4,000 transactions generated (Online + In-Store)")
print("- Model: Logistic Regression (explainable, outputs fraud_score probability)")
print("- fraud_score: probability that an order is fraudulent")
print(f"- Best Threshold (by F1): {best_thr:.2f}")
print(f"  Accuracy={acc:.2%}, Precision={pr:.2%}, Recall={rc:.2%}, F1={f1:.2%}, ROC-AUC={roc:.2%}, PR-AUC={pr_auc:.2%}")
print("\nStrong Features (business perspective):")
signals = {
 "addr_mismatch":"Shipping vs Billing mismatch",
 "payment_risky":"PayPal (online) or Gift Cards (in-store)",
 "high_amount_cat":"Above 90th percentile spend in category",
 "price_high_anom":"Price unusually high (possible fraud)",
 "price_low_anom":"Price unusually low (discount abuse)",
 "express_high_amt":"Express + high value → suspicious",
 "return_whiplash":"Multiple recent returns",
 "oversell_flag":"Quantity > 60% of on-hand stock",
 "hoarding_flag":"Bulk purchase (>=4 items)"
}
for k,v in signals.items(): print(f" • {k}: {v}")

# ---------------- 7) Score New Order ----------------
def score_new_order(order:dict)->float:
    r=pd.DataFrame([order])
    cp=df[df["sku_category"]==r["sku_category"].iloc[0]]
    if cp.empty: cp=df
    p90=cp["order_amount"].quantile(0.90)
    mu=cp["order_amount"].mean()
    sd=max(cp["order_amount"].std(),1.0)
    z=(r["order_amount"]-mu)/sd
    r["addr_mismatch"]=(r["shipping_country"]!=r["billing_country"]).astype(int)
    r["payment_risky"]=r["payment_method"].isin(["paypal","gift_card"]).astype(int)
    r["high_amount_cat"]=(r["order_amount"]>=p90).astype(int)
    r["price_high_anom"]=(z>=2).astype(int); r["price_low_anom"]=(z<=-2).astype(int)
    r["express_high_amt"]=((r["express_shipping"]==1)&(r["high_amount_cat"]==1)).astype(int)
    r["return_whiplash"]=(r["returns_30d"]>=2).astype(int)
    r["log_amount"]=np.log1p(r["order_amount"])
    for c in ["oversell_flag","hoarding_flag","on_hand"]: r.setdefault(c,0)
    return float(clf.predict_proba(r[features])[:,1])

example = dict(channel="Online",sku_category="Electronics",payment_method="paypal",
               shipping_country="US",billing_country="US",quantity=2,order_amount=1800,
               log_amount=np.log1p(1800),express_shipping=1,returns_30d=0,
               addr_mismatch=0,payment_risky=1,high_amount_cat=1,price_high_anom=0,
               price_low_anom=0,oversell_flag=0,hoarding_flag=0,return_whiplash=0,
               express_high_amt=1,on_hand=80)
fs = score_new_order(example)
print(f"\nNew order fraud_score={fs:.2f} → Decision at {best_thr:.2f}: {'ALERT' if fs>=best_thr else 'PASS'}")
