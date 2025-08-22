# app.py — run with: streamlit run app.py
import streamlit as st, pandas as pd, numpy as np, altair as alt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config("Retail Fraud", layout="wide")
alt.renderers.set_embed_options(actions=False)
RND = 42

# ───────────────── Sidebar (read-only source) ─────────────────
st.sidebar.header("Data source")
st.sidebar.text_input("Project", "mss-data-engineer-sandbox", disabled=True)
st.sidebar.text_input("Dataset", "retail", disabled=True)
st.sidebar.text_input("Raw table", "mss-data-engineer-sandbox.retail.transaction_data", disabled=True)

# ───────────────── Step 1. Load fixed 4k dataset ─────────────────
@st.cache_data
def load_data():
    np.random.seed(RND)
    rows = 4000
    df = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(rows)],
        "ts": pd.date_range("2023-01-01", periods=rows, freq="H"),
        "customer_id": np.random.choice([f"c{i}" for i in range(200)], rows),
        "store_id": np.random.choice([f"s{i}" for i in range(10)], rows),
        "device_id": np.random.choice([f"d{i}" for i in range(50)], rows),
        "sku_id": np.random.choice([f"sku{i}" for i in range(100)], rows),
        "sku_category": np.random.choice(["grocery","electronics","apparel","home","toys"], rows),
        "quantity": np.random.randint(1,5,rows),
        "unit_price": np.random.uniform(5,500,rows).round(2),
        "payment_channel": np.random.choice(["Card","DigitalWallet","GiftCard"], rows),
        "ship_country": np.random.choice(["US","CA","UK","DE","IN"], rows),
        "ip_country": np.random.choice(["US","CA","UK","DE","IN"], rows),
        "account_age_days": np.random.randint(0,1000,rows),
        "coupon_discount": np.random.choice([0,0,0,10,20,50], rows),
        "gift_balance_used": np.random.choice([0,0,0,5,25,100], rows),
    })
    # Fraud label (synthetic): based on riskier combinations
    df["fraud_flag"] = (
        ((df["ship_country"] != df["ip_country"]) & (df["unit_price"] > 300)) |
        ((df["coupon_discount"] > 20) & (df["gift_balance_used"] > 50)) |
        ((df["quantity"] >= 3) & (df["unit_price"] > 200))
    ).astype(int)
    return df

raw = load_data()

# ───────────────── Step 2. Strong Features ─────────────────
df = raw.copy()
df["order_amount"] = df["quantity"] * df["unit_price"]
cat_avg = df.groupby("sku_category")["unit_price"].transform("mean").replace(0,np.nan)
df["price_ratio"] = (df["unit_price"]/cat_avg).fillna(1.0)
df["geo_mismatch"] = (df["ship_country"] != df["ip_country"]).astype(int)
p90 = float(np.nanpercentile(df["order_amount"], 90))
df["s_geo_highvalue"] = ((df["geo_mismatch"]==1) & (df["order_amount"]>=p90)).astype(int)
df["coupon_pct"] = (df["coupon_discount"]/df["order_amount"].replace(0,np.nan)).fillna(0)
df["gift_pct"]   = (df["gift_balance_used"]/df["order_amount"].replace(0,np.nan)).fillna(0)
df["s_discount_gift_stack"] = ((df["coupon_pct"]>=0.3)&(df["gift_pct"]>=0.5)).astype(int)
df["s_price_bulk"] = ((df["price_ratio"].sub(1).abs()>=0.5)&(df["quantity"]>=3)).astype(int)

# ───────────────── Step 3. Train / Test Split ─────────────────
cut = int(len(df)*0.75)
train, test = df.iloc[:cut], df.iloc[cut:]
feat_num = ["order_amount","quantity","unit_price","account_age_days","price_ratio",
            "coupon_pct","gift_pct","s_geo_highvalue","s_discount_gift_stack","s_price_bulk","geo_mismatch"]
feat_cat = ["payment_channel","sku_category","ship_country","ip_country","store_id"]

Xtr = pd.concat([train[feat_num].fillna(0),
                 pd.get_dummies(train[feat_cat].astype(str))], axis=1)
ytr = train["fraud_flag"].values
Xte = pd.concat([test[feat_num].fillna(0),
                 pd.get_dummies(test[feat_cat].astype(str))], axis=1).reindex(columns=Xtr.columns, fill_value=0)
yte = test["fraud_flag"].values
cat_cols_model = Xtr.columns

clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=RND)
clf.fit(Xtr,ytr)

probs = clf.predict_proba(Xte)[:,1]
best_t = 0.3
yhat = (probs >= best_t).astype(int)

# ───────────────── Dashboard Layout ─────────────────
st.title("Retail Fraud Detection")

# Raw data
st.subheader("📦 Raw 4,000 Transactions Snapshot")
st.dataframe(raw.head(200), use_container_width=True, height=300)
st.caption("Showing first 200 of 4,000 for readability.")

# Channel mix
st.subheader("💳 Channel Mix")
mix = df.groupby("payment_channel")["order_id"].count().reset_index()
st.bar_chart(mix.set_index("payment_channel"))

# Model evaluation
st.subheader("📊 Model Evaluation (test set)")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Accuracy",  f"{accuracy_score(yte,yhat):.2%}")
c2.metric("Precision", f"{precision_score(yte,yhat,zero_division=0):.2%}")
c3.metric("Recall",    f"{recall_score(yte,yhat,zero_division=0):.2%}")
c4.metric("F1-score",  f"{f1_score(yte,yhat,zero_division=0):.2%}")

# ───────────────── New Order ─────────────────
st.subheader("🛒 New Order – Instant Decision")

c1,c2,c3 = st.columns(3)
cat = c1.selectbox("Category", df["sku_category"].unique())
channel = c2.selectbox("Payment channel", ["Card","DigitalWallet","GiftCard"])
ship = c3.selectbox("Shipping country", df["ship_country"].unique())
c4,c5,c6 = st.columns(3)
ip   = c4.selectbox("Network country", df["ip_country"].unique())
qty  = c5.number_input("Quantity",1,10,1)
price= c6.number_input("Unit price",1.0,1000.0,100.0)
c7,c8,c9 = st.columns(3)
disc = c7.number_input("Discount",0.0,500.0,0.0)
gift = c8.number_input("Gift balance",0.0,500.0,0.0)
age  = c9.number_input("Account age (days)",0,3650,100)

rec = {
    "sku_category":cat, "payment_channel":channel,"ship_country":ship,"ip_country":ip,
    "quantity":qty,"unit_price":price,"coupon_discount":disc,"gift_balance_used":gift,
    "account_age_days":age,"store_id":"s_manual"
}
rec["order_amount"]=rec["quantity"]*rec["unit_price"]
rec["price_ratio"]=rec["unit_price"]/cat_avg.mean()
rec["geo_mismatch"]=int(rec["ip_country"]!=rec["ship_country"])
rec["s_geo_highvalue"]=int(rec["geo_mismatch"]==1 and rec["order_amount"]>=p90)
rec["coupon_pct"]=rec["coupon_discount"]/rec["order_amount"] if rec["order_amount"] else 0
rec["gift_pct"]=rec["gift_balance_used"]/rec["order_amount"] if rec["order_amount"] else 0
rec["s_discount_gift_stack"]=int(rec["coupon_pct"]>=0.3 and rec["gift_pct"]>=0.5)
rec["s_price_bulk"]=int(abs(rec["price_ratio"]-1)>=0.5 and rec["quantity"]>=3)

num={k:rec[k] for k in feat_num}
catr={"payment_channel":rec["payment_channel"],"sku_category":rec["sku_category"],
      "ship_country":rec["ship_country"],"ip_country":rec["ip_country"],"store_id":rec["store_id"]}
Xnum=pd.DataFrame([num]).fillna(0)
Xcat=pd.get_dummies(pd.DataFrame([catr]).astype(str))
Xin=pd.concat([Xnum,Xcat],axis=1).reindex(columns=cat_cols_model,fill_value=0)

if st.button("Check Fraud"):
    pred=int((clf.predict_proba(Xin)[:,1]>=best_t)[0])
    label="Fraud" if pred else "Not Fraud"
    st.markdown(f"### Decision: **{label}**")

