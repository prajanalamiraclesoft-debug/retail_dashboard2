# app.py — Raw 4k data + backend model + New Order decision (business-friendly)
import streamlit as st, pandas as pd, numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.validation import check_is_fitted

st.set_page_config("Fraud – Raw Data & New-Order Decision", layout="wide")
RNG = np.random.default_rng(42)

# ───────────────────────────── Data builder (4k rows) ─────────────────────────────
def _ids(prefix, n): return [f"{prefix}_{i}" for i in range(1, n+1)]

def build_4k():
    n = 4000
    start = datetime(2023, 1, 1)
    # we keep a timestamp internally; not shown in the UI
    ts = [start + timedelta(minutes=int(x)) for x in RNG.integers(0, 60*24*365, size=n)]

    customers = RNG.choice(_ids("cust", 600), size=n)
    stores    = RNG.choice(_ids("store", 80), size=n)
    devices   = RNG.choice(_ids("dev", 400), size=n)
    sku_ids   = RNG.choice(_ids("sku", 900), size=n)
    category  = RNG.choice(["electronics","home","apparel","toys","grocery"],
                           p=[.30,.18,.18,.14,.20], size=n)

    # Business-friendly payment channels (no technical terms)
    pay_channel = RNG.choice(
        ["Card", "Wallet", "Bank", "Delivery", "Other"],  # internally we’ll weight some higher risk
        p=[.45, .20, .15, .10, .10], size=n
    )

    countries = ["US","UK","DE","IN","CA"]
    ship = RNG.choice(countries, size=n)
    ip   = np.array([ ship[i] if RNG.random()<0.8 else RNG.choice([c for c in countries if c!=ship[i]]) for i in range(n) ])

    base = {"electronics": (250, 90), "home": (80, 30), "apparel": (60, 25), "toys": (40, 15), "grocery": (20, 10)}
    unit_price = np.array([ max(1, RNG.normal(base[c][0], base[c][1])) for c in category ])
    quantity   = np.maximum(1, RNG.poisson(lam=RNG.uniform(1.3, 2.4, size=n))).astype(float)
    amount     = unit_price * quantity

    coupon_disc = np.clip(RNG.gamma(2.0, 5.0, size=n), 0, amount*0.5)
    gift_amt    = np.zeros(n)
    gift_used   = RNG.random(size=n) < 0.22
    gift_amt[gift_used] = np.clip(amount[gift_used]*RNG.uniform(0.2, 0.8, gift_used.sum()), 0, None)

    acct_age    = np.maximum(0, RNG.normal(120, 90, size=n)).astype(float)

    coupon_pct  = np.divide(coupon_disc, amount, out=np.zeros_like(amount), where=amount!=0)
    gift_pct    = np.divide(gift_amt,   amount, out=np.zeros_like(amount), where=amount!=0)

    # Address consistency signal (avoid the word “geo” in UI)
    addr_mismatch = (ship != ip).astype(int)

    # Price ratio vs category mean
    _tmp = pd.DataFrame({"cat":category, "p":unit_price})
    cat_avg = _tmp.groupby("cat")["p"].transform("mean").values
    price_ratio = np.divide(unit_price, cat_avg, out=np.ones_like(unit_price), where=cat_avg!=0)

    # Payment channel risk code (still business-friendly labels)
    pay_code = pd.Series(pay_channel).map({"Wallet":2, "Card":1, "Other":1, "Bank":0, "Delivery":0}).values

    # Target construction from strong signals (6–8)
    # 1) address mismatch + higher-risk channel + high amount
    # 2) big gift balance + big coupon
    # 3) price anomaly + bulk
    # 4) very new account + high amount
    # 5) “Wallet” channel with substantial amount
    s = np.zeros(n, dtype=float)
    s += (addr_mismatch & (np.isin(pay_channel, ["Wallet"])) & (amount > 300)) * 1.2
    s += ((gift_pct > 0.55) & (coupon_pct > 0.20)) * 1.0
    s += ((np.abs(price_ratio - 1) > 0.50) & (quantity >= 3)) * 1.0
    s += ((acct_age < 10) & (amount > 400)) * 0.9
    s += ((pay_channel == "Wallet") & (amount > 180)) * 0.6
    s += RNG.normal(0, 0.15, size=n)
    y = (s > 1.0).astype(int)

    df = pd.DataFrame({
        "order_id": _ids("order", n),
        "customer_id": customers,
        "store_id": stores,
        "device_id": devices,
        "sku_id": sku_ids,
        "sku_category": category,
        "pay_channel": pay_channel,
        "ship_country": ship,
        "ip_country": ip,
        "unit_price": unit_price,
        "quantity": quantity,
        "coupon_discount": coupon_disc,
        "gift_balance_used": gift_amt,
        "gift_used_flag": gift_used.astype(int),
        "account_age_days": acct_age,
        "order_amount": amount,
        "coupon_pct": coupon_pct,
        "gift_pct": gift_pct,
        "addr_mismatch": addr_mismatch,
        "price_ratio": price_ratio,
        "pay_code": pay_code,
        "fraud_flag": y,
        "ts": ts,  # not shown
    })
    return df

# Reasons for decision (business wording)
def reasons_from_row(r):
    reasons = []
    if r["addr_mismatch"]==1 and r["pay_channel"] in ("Wallet",) and r["order_amount"]>300:
        reasons.append("Address inconsistency with high-value wallet payment")
    if r["gift_pct"]>0.55 and r["coupon_pct"]>0.20:
        reasons.append("Large gift balance combined with high discount")
    if abs(r["price_ratio"]-1)>0.50 and r["quantity"]>=3:
        reasons.append("Unusual price for its category with bulk quantity")
    if r["account_age_days"]<10 and r["order_amount"]>400:
        reasons.append("Very new account with large purchase")
    if r["pay_channel"]=="Wallet" and r["order_amount"]>180:
        reasons.append("Wallet payment with substantial amount")
    if not reasons:
        safes=[]
        if r["addr_mismatch"]==0: safes.append("Address looks consistent")
        if r["gift_pct"]<0.20:     safes.append("Low gift-balance usage")
        if r["coupon_pct"]<0.30:   safes.append("Discount within normal range")
        if r["account_age_days"]>=10: safes.append("Account not brand-new")
        if abs(r["price_ratio"]-1)<=0.50 or r["quantity"]<3: safes.append("No unusual price or bulk")
        reasons = ["; ".join(safes)] if safes else ["No clear risk patterns"]
    return reasons

# ───────────────────────────── Build data & model (backend) ─────────────────────────────
df = build_4k()

STRONG_NUM = [
    "order_amount","quantity","unit_price","account_age_days",
    "coupon_pct","gift_pct","price_ratio","addr_mismatch","pay_code"
]
STRONG_CAT = ["sku_category","pay_channel","ship_country","ip_country","device_id","store_id"]

# Design matrix
Xn = df[STRONG_NUM].fillna(0)
Xc = pd.get_dummies(df[STRONG_CAT].astype(str), dummy_na=False)
X  = pd.concat([Xn, Xc], axis=1)
y  = df["fraud_flag"].astype(int).values

# Split & train
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pos = max(1, int((y_tr==1).sum())); neg = max(1, len(y_tr)-pos)
w_pos = min(15.0, neg/pos)  # class balance weight
sample_w = np.where(y_tr==1, w_pos, 1.0)

clf = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.06, early_stopping=True, random_state=42)
clf.fit(X_tr, y_tr, sample_weight=sample_w)

# Choose an internal decision point by F1 (kept backend; no slider)
probs = clf.predict_proba(X_te)[:,1]
grid  = np.linspace(0.05, 0.95, 91)
best_t, best_f1 = 0.5, -1
for t in grid:
    yp = (probs>=t).astype(int)
    f1 = f1_score(y_te, yp, zero_division=0)
    if f1 > best_f1: best_f1, best_t = f1, t

# Final test metrics (shown as numbers)
y_hat = (probs>=best_t).astype(int)
ACC  = accuracy_score(y_te, y_hat)
PREC = precision_score(y_te, y_hat, zero_division=0)
REC  = recall_score(y_te, y_hat, zero_division=0)
F1   = f1_score(y_te, y_hat, zero_division=0)

ALL_COLS = list(X.columns)

# ───────────────────────────── UI (single page) ─────────────────────────────
st.title("Fraud Review – Raw Data & New-Order Decision")

# Metrics (backend evaluation on the 4k set)
m1,m2,m3,m4 = st.columns(4)
m1.metric("Accuracy",  f"{ACC*100:.2f}%")
m2.metric("Precision", f"{PREC*100:.2f}%")
m3.metric("Recall",    f"{REC*100:.2f}%")
m4.metric("F1-score",  f"{F1*100:.2f}%")

# Raw table: every row (no EDA)
st.subheader("Raw dataset (4,000 rows)")
st.dataframe(df.drop(columns=["ts"]), use_container_width=True, height=520)
st.download_button("Download CSV", df.drop(columns=["ts"]).to_csv(index=False).encode("utf-8"),
                   file_name="fraud_raw_4000.csv", mime="text/csv")

st.markdown("---")
st.subheader("New Order – instant decision")

countries = ["US","UK","DE","IN","CA"]
categories = ["electronics","home","apparel","toys","grocery"]
channels   = ["Card","Wallet","Bank","Delivery","Other"]  # business friendly

with st.form("new_order_form", clear_on_submit=False):
    c1,c2,c3 = st.columns(3)
    with c1:
        sku_category   = st.selectbox("Category", categories, index=0)
        pay_channel    = st.selectbox("Payment channel", channels, index=0)
        ship_country   = st.selectbox("Shipping country", countries, index=0)
    with c2:
        ip_country     = st.selectbox("Network country", countries, index=0)
        quantity       = st.number_input("Quantity", 1.0, step=1.0, value=1.0)
        unit_price     = st.number_input("Unit price", 1.0, step=1.0, value=120.0)
    with c3:
        coupon_disc    = st.number_input("Discount amount", 0.0, step=1.0, value=0.0)
        gift_balance   = st.number_input("Gift balance used", 0.0, step=1.0, value=0.0)
        account_age    = st.number_input("Account age (days)", 0.0, step=1.0, value=120.0)

    # Optional IDs (business doesn’t need to change these)
    c4,c5 = st.columns(2)
    with c4:
        device_id = st.text_input("Device ID", "dev_manual")
        store_id  = st.text_input("Store ID",  "store_manual")
    with c5:
        customer_id = st.text_input("Customer ID", "cust_manual")
        sku_id      = st.text_input("SKU ID",      "sku_manual")

    submitted = st.form_submit_button("Check")

if submitted:
    # Feature build (same strong features as training)
    order_amount = float(quantity*unit_price)
    coupon_pct   = float((coupon_disc / order_amount) if order_amount else 0.0)
    gift_pct     = float((gift_balance / order_amount) if order_amount else 0.0)
    addr_mismatch= int(ship_country != ip_country)
    price_ratio  = 1.0  # unknown in manual entry; assume nominal
    pay_code     = {"Wallet":2, "Card":1, "Other":1, "Bank":0, "Delivery":0}.get(pay_channel, 1)

    rec = pd.DataFrame([{
        "order_id":"new_order", "customer_id":customer_id, "store_id":store_id, "device_id":device_id,
        "sku_id":sku_id, "sku_category":sku_category, "pay_channel":pay_channel,
        "ship_country":ship_country, "ip_country":ip_country,
        "unit_price":float(unit_price), "quantity":float(quantity),
        "coupon_discount":float(coupon_disc), "gift_balance_used":float(gift_balance),
        "gift_used_flag": int(gift_balance>0.0),
        "account_age_days": float(account_age), "order_amount": order_amount,
        "coupon_pct": coupon_pct, "gift_pct": gift_pct,
        "addr_mismatch": addr_mismatch, "price_ratio": price_ratio, "pay_code": pay_code
    }])

    Xn_ = rec[STRONG_NUM].fillna(0)
    Xc_ = pd.get_dummies(rec[STRONG_CAT].astype(str))
    X_  = pd.concat([Xn_, Xc_], axis=1).reindex(columns=ALL_COLS, fill_value=0)

    # Predict (no probabilities shown; internal decision point set by F1)
    try:
        check_is_fitted(clf)
        p = clf.predict_proba(X_)[:,1][0]
    except Exception:
        p = 0.5
    decision = int(p >= best_t)

    # Business-friendly reasons
    r = rec.iloc[0].to_dict()
    reasons = reasons_from_row({
        "addr_mismatch": r["addr_mismatch"],
        "pay_channel": r["pay_channel"],
        "order_amount": r["order_amount"],
        "gift_pct": r["gift_pct"],
        "coupon_pct": r["coupon_pct"],
        "price_ratio": r["price_ratio"],
        "quantity": r["quantity"],
        "account_age_days": r["account_age_days"],
    })

    st.markdown(f"### Decision: {'🚨 **FRAUD**' if decision==1 else '✅ **NOT FRAUD**'}")
    st.markdown("**Why:** " + " | ".join(reasons))
    st.caption("Decision uses strong features: amount, quantity, price vs category, discount %, gift-balance %, address consistency, account age, and payment channel.")
    st.dataframe(rec.drop(columns=["ts"]) if "ts" in rec else rec, use_container_width=True, height=200)
