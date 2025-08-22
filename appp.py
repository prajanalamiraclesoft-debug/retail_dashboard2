# app.py — local 4k data • EDA • model • predict with reasons (no probabilities)
import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance

st.set_page_config("Fraud: Data → EDA → Model → Predict", layout="wide")
alt.renderers.set_embed_options(actions=False)
RNG = np.random.default_rng(42)

# ───────────────────────── Utilities ─────────────────────────
def make_ids(prefix, n):
    return [f"{prefix}_{i}" for i in range(1, n+1)]

def build_synthetic_4k():
    """Make a realistic, learnable 4k dataset with patterns the model can capture well."""
    n = 4000
    start = datetime(2023,1,1)
    ts = [start + timedelta(minutes=int(x)) for x in RNG.integers(0, 60*24*365, size=n)]

    customers = RNG.choice(make_ids("cust", 600), size=n, replace=True)
    stores    = RNG.choice(make_ids("store", 80), size=n, replace=True)
    devices   = RNG.choice(make_ids("dev", 400), size=n, replace=True)
    sku_ids   = RNG.choice(make_ids("sku", 900), size=n, replace=True)
    categories= RNG.choice(["electronics","home","apparel","toys","grocery"], p=[.30,.18,.18,.14,.20], size=n)

    pay_methods = RNG.choice(
        ["credit_card","debit_card","paypal","bank_transfer","cod","crypto"],
        p=[.40,.15,.15,.10,.10,.10], size=n
    )
    countries = ["US","UK","DE","IN","CA"]
    ship = RNG.choice(countries, size=n, replace=True)
    ip   = []
    for i in range(n):
        if RNG.random() < 0.80: ip.append(ship[i])
        else: ip.append(RNG.choice([c for c in countries if c != ship[i]]))
    ip = np.array(ip)

    base_price = {"electronics": (250, 90),
                  "home":        (80,  30),
                  "apparel":     (60,  25),
                  "toys":        (40,  15),
                  "grocery":     (20,  10)}
    unit_price = np.array([
        max(1, RNG.normal(base_price[c][0], base_price[c][1])) for c in categories
    ])
    quantity = np.maximum(1, RNG.poisson(lam=RNG.uniform(1.3, 2.4, size=n))).astype(float)

    coupon_discount = np.clip(RNG.gamma(2.0, 5.0, size=n), 0, unit_price*quantity*0.5)
    gift_card_amount = np.zeros(n)
    gift_used = RNG.random(size=n) < 0.22
    gift_card_amount[gift_used] = np.clip(unit_price[gift_used]*quantity[gift_used]*RNG.uniform(0.2,0.8, gift_used.sum()), 0, None)

    acct_age_days = np.maximum(0, RNG.normal(120, 90, size=n)).astype(float)

    order_amount = unit_price*quantity
    coupon_pct = np.divide(coupon_discount, order_amount, out=np.zeros_like(order_amount), where=order_amount!=0)
    gift_pct   = np.divide(gift_card_amount, order_amount, out=np.zeros_like(order_amount), where=order_amount!=0)
    geo_mismatch = (ship != ip).astype(int)

    cat_df = pd.DataFrame({"cat":categories, "p":unit_price})
    cat_avg = cat_df.groupby("cat")["p"].transform("mean").values
    price_ratio = np.divide(unit_price, cat_avg, out=np.ones_like(unit_price), where=cat_avg!=0)

    pay_risk = pd.Series(pay_methods).map({"crypto":3,"paypal":2,"credit_card":2,"debit_card":1,"bank_transfer":0,"cod":0}).fillna(1).values

    # fraud signal (learnable rules)
    s = np.zeros(n, dtype=float)
    s += (geo_mismatch & (np.isin(pay_methods, ["crypto","paypal"])) & (order_amount > 300)) * 1.2
    s += ((gift_pct > 0.55) & (coupon_pct > 0.20)) * 1.0
    s += ((np.abs(price_ratio - 1) > 0.50) & (quantity >= 3)) * 1.0
    s += ((acct_age_days < 10) & (order_amount > 400)) * 0.9
    s += ((pay_methods == "crypto") & (order_amount > 180)) * 0.6
    s += RNG.normal(0, 0.15, size=n)
    y = (s > 1.0).astype(int)

    df = pd.DataFrame({
        "order_id": make_ids("order", n),
        "customer_id": customers,
        "store_id": stores,
        "device_id": devices,
        "sku_id": sku_ids,
        "sku_category": categories,
        "payment_method": pay_methods,
        "shipping_country": ship,
        "ip_country": ip,
        "unit_price": unit_price,
        "quantity": quantity,
        "coupon_discount": coupon_discount,
        "gift_card_amount": gift_card_amount,
        "gift_card_used": gift_used.astype(int),
        "account_age_days": acct_age_days,
        "order_amount": order_amount,
        "coupon_pct": coupon_pct,
        "gift_pct": gift_pct,
        "geo_mismatch": geo_mismatch,
        "price_ratio": price_ratio,
        "pay_risk": pay_risk,
        "fraud_flag": y,
        "ts": ts  # not shown in UI
    })
    return df

def iqr_outliers(x):
    x = np.array(x)
    q1, q3 = np.nanpercentile(x, 25), np.nanpercentile(x, 75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    return int(((x<lo)|(x>hi)).sum())

def reasons_for(row):
    reasons = []
    if row["geo_mismatch"]==1 and row["payment_method"] in ("crypto","paypal") and row["order_amount"]>300:
        reasons.append("Geo mismatch + high-risk pay + high amount")
    if row["gift_pct"]>0.55 and row["coupon_pct"]>0.20:
        reasons.append("Gift card stack with big coupon")
    if abs(row["price_ratio"]-1)>0.50 and row["quantity"]>=3:
        reasons.append("Price anomaly + bulk quantity")
    if row["account_age_days"]<10 and row["order_amount"]>400:
        reasons.append("Very new account with high spend")
    if row["payment_method"]=="crypto" and row["order_amount"]>180:
        reasons.append("Crypto with substantial spend")
    if not reasons:
        safes = []
        if row["geo_mismatch"]==0: safes.append("Shipping & IP countries match")
        if row["gift_pct"]<0.2:     safes.append("Low gift-card usage")
        if row["coupon_pct"]<0.3:   safes.append("Discounts within normal range")
        if row["account_age_days"]>=10: safes.append("Account not brand-new")
        if abs(row["price_ratio"]-1)<=0.50 or row["quantity"]<3: safes.append("No price anomaly / bulk")
        reasons = ["; ".join(safes)] if safes else ["No risk patterns detected"]
    return reasons

# ───────────────────────── Data ─────────────────────────
df = build_synthetic_4k()

tab1, tab2, tab3, tab4 = st.tabs(["1) Raw", "2) EDA", "3) Model", "4) Predict"])

# ───────────────────────── 1) RAW ─────────────────────────
with tab1:
    st.subheader("Raw (4,000 rows • generated)")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Fraud prevalence", f"{df['fraud_flag'].mean()*100:.1f}%")
    c3.metric("Avg amount", f"${df['order_amount'].mean():.2f}")
    c4.metric("Geo mismatch", f"{df['geo_mismatch'].mean()*100:.1f}%")
    st.dataframe(df.drop(columns=["ts"]).head(300), use_container_width=True, height=380)

# ───────────────────────── 2) EDA ─────────────────────────
with tab2:
    st.subheader("Skewness & outliers (key numeric)")
    num_cols = ["order_amount","unit_price","quantity","account_age_days","coupon_pct","gift_pct","price_ratio"]
    skew_tbl = pd.DataFrame({
        "feature": num_cols,
        "skewness": [df[c].dropna().skew() for c in num_cols],
        "outliers (IQR)": [iqr_outliers(df[c].values) for c in num_cols]
    }).sort_values("skewness", ascending=False)
    st.dataframe(skew_tbl, use_container_width=True, height=240)

    st.subheader("Distributions")
    for col in num_cols:
        st.altair_chart(
            alt.Chart(df).mark_bar().encode(
                x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=50), title=col),
                y=alt.Y("count():Q", title="Rows")
            ).properties(height=140),
            use_container_width=True
        )

# ───────────────────────── 3) MODEL ─────────────────────────
with tab3:
    st.subheader("Training & evaluation")
    cat_cols = ["sku_category","payment_method","shipping_country","ip_country","device_id","store_id"]
    num_cols = ["order_amount","quantity","unit_price","account_age_days","coupon_pct","gift_pct",
                "price_ratio","geo_mismatch","pay_risk"]

    Xn = df[num_cols].fillna(0)
    Xc = pd.get_dummies(df[cat_cols].astype(str), dummy_na=False)
    X  = pd.concat([Xn, Xc], axis=1)
    y  = df["fraud_flag"].astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pos = max(1, int((y_tr==1).sum())); neg = max(1, len(y_tr)-pos)
    w_pos = min(15.0, neg/pos)
    sw = np.where(y_tr==1, w_pos, 1.0)

    clf = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.06,
                                         early_stopping=True, random_state=42)
    clf.fit(X_tr, y_tr, sample_weight=sw)

    pr = clf.predict_proba(X_te)[:,1]
    ts = np.linspace(0.05,0.95,91)
    best_t, best_f1 = 0.5, -1
    for t in ts:
        yp = (pr>=t).astype(int)
        f1 = f1_score(y_te, yp, zero_division=0)
        if f1 > best_f1: best_f1, best_t = f1, t
    y_pred = (pr>=best_t).astype(int)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc*100:.2f}%")
    c2.metric("Precision", f"{prec*100:.2f}%")
    c3.metric("Recall",    f"{rec*100:.2f}%")
    c4.metric("F1-score",  f"{f1*100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred, labels=[0,1])
    cm_df = pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]).reset_index().melt(
        id_vars="index", var_name="Predicted", value_name="Count").rename(columns={"index":"Actual"})
    st.altair_chart(
        alt.Chart(cm_df).mark_rect().encode(
            x="Predicted:N", y="Actual:N",
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Actual","Predicted","Count"]
        ).properties(height=180),
        use_container_width=True
    )

    # Feature importances (permutation, robust across sklearn versions)
    st.subheader("Top feature importances (permutation)")
    perm = permutation_importance(clf, X_te, y_te, n_repeats=10, random_state=42, n_jobs=None)
    imp = (pd.Series(perm.importances_mean, index=X.columns)
             .sort_values(ascending=False)
             .head(15)
             .reset_index())
    imp.columns = ["feature","importance"]
    st.dataframe(imp, use_container_width=True, height=300)

    st.session_state["clf"] = clf
    st.session_state["best_t"] = float(best_t)
    st.session_state["all_cols"] = list(X.columns)
    st.info("Model trained. Use the Predict tab to test new orders.")

# ───────────────────────── 4) PREDICT ─────────────────────────
with tab4:
    st.subheader("New order → Fraud or Not (with reasons)")
    if "clf" not in st.session_state:
        st.warning("Train the model first (see Model tab).")
        st.stop()

    clf = st.session_state["clf"]
    best_t = st.session_state["best_t"]
    all_cols = st.session_state["all_cols"]

    countries = ["US","UK","DE","IN","CA"]
    categories = ["electronics","home","apparel","toys","grocery"]
    pays = ["credit_card","debit_card","paypal","bank_transfer","cod","crypto"]

    with st.form("new_form", clear_on_submit=False):
        colA, colB, colC = st.columns(3)
        with colA:
            sku_category = st.selectbox("Category", categories, index=0)
            payment_method = st.selectbox("Payment method", pays, index=0)
            shipping_country = st.selectbox("Shipping country", countries, index=0)
        with colB:
            ip_country = st.selectbox("IP country", countries, index=0)
            quantity = st.number_input("Quantity", 1.0, step=1.0, value=1.0)
            unit_price = st.number_input("Unit price", 1.0, step=1.0, value=120.0)
        with colC:
            coupon_discount = st.number_input("Coupon discount", 0.0, step=1.0, value=0.0)
            gift_card_amount = st.number_input("Gift card amount", 0.0, step=1.0, value=0.0)
            account_age_days = st.number_input("Account age (days)", 0.0, step=1.0, value=120.0)

        colD, colE = st.columns(2)
        with colD:
            device_id = st.text_input("Device ID", "dev_manual")
            store_id  = st.text_input("Store ID",  "store_manual")
        with colE:
            customer_id = st.text_input("Customer ID", "cust_manual")
            sku_id      = st.text_input("SKU ID",      "sku_manual")

        submitted = st.form_submit_button("Check")

    if submitted:
        order_amount = quantity*unit_price
        coupon_pct = (coupon_discount / order_amount) if order_amount else 0.0
        gift_pct   = (gift_card_amount / order_amount) if order_amount else 0.0
        geo_mismatch = int(shipping_country != ip_country)
        price_ratio = 1.0
        pay_risk = {"crypto":3,"paypal":2,"credit_card":2,"debit_card":1,"bank_transfer":0,"cod":0}.get(payment_method,1)

        rec = pd.DataFrame([{
            "order_id":"new_order", "customer_id":customer_id, "store_id":store_id, "device_id":device_id,
            "sku_id":sku_id, "sku_category":sku_category, "payment_method":payment_method,
            "shipping_country":shipping_country, "ip_country":ip_country,
            "unit_price":float(unit_price), "quantity":float(quantity),
            "coupon_discount":float(coupon_discount), "gift_card_amount":float(gift_card_amount),
            "gift_card_used": int(gift_card_amount>0.0),
            "account_age_days": float(account_age_days), "order_amount": float(order_amount),
            "coupon_pct": float(coupon_pct), "gift_pct": float(gift_pct),
            "geo_mismatch": int(geo_mismatch), "price_ratio": float(price_ratio),
            "pay_risk": int(pay_risk)
        }])

        Xn = rec[["order_amount","quantity","unit_price","account_age_days","coupon_pct","gift_pct",
                  "price_ratio","geo_mismatch","pay_risk"]].fillna(0)
        Xc = pd.get_dummies(rec[["sku_category","payment_method","shipping_country","ip_country","device_id","store_id"]].astype(str))
        X  = pd.concat([Xn, Xc], axis=1).reindex(columns=all_cols, fill_value=0)

        p = clf.predict_proba(X)[:,1][0]
        decision = int(p >= best_t)

        r = rec.iloc[0].to_dict()
        r_list = reasons_for({
            "geo_mismatch": r["geo_mismatch"],
            "payment_method": r["payment_method"],
            "order_amount": r["order_amount"],
            "gift_pct": r["gift_pct"],
            "coupon_pct": r["coupon_pct"],
            "price_ratio": r["price_ratio"],
            "quantity": r["quantity"],
            "account_age_days": r["account_age_days"]
        })

        st.markdown(f"### Decision: {'🚨 **FRAUD**' if decision==1 else '✅ **NOT FRAUD**'}")
        st.markdown("**Why:** " + " | ".join(r_list))
        st.caption("Note: the model uses many signals; this explanation highlights the main ones that match risk/safe patterns.")
        st.dataframe(rec.drop(columns=["coupon_discount","gift_card_amount"]), use_container_width=True, height=200)
