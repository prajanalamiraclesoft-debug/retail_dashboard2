# app.py  —  streamlit run app.py
import streamlit as st, pandas as pd, numpy as np
import altair as alt
from datetime import timedelta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------- App setup -----------------------------
st.set_page_config("Retail Fraud – Executive Dashboard", layout="wide")
alt.renderers.set_embed_options(actions=False)
RND = 42
np.random.seed(RND)

# ---------------------- 4k retail-like sample -----------------------
@st.cache_data
def make_data(n=4000):
    categories = ["grocery","electronics","apparel","home","toys"]
    channels   = ["Card","GiftCard","StoreCredit"]          # no wallet / cod
    countries  = ["US","CA","UK","DE","IN"]

    df = pd.DataFrame({
        "order_id":       [f"o{i}" for i in range(n)],
        "ts":             pd.date_range("2023-01-01", periods=n, freq="H"),
        "customer_id":    np.random.choice([f"c{i}" for i in range(600)], n),
        "device_id":      np.random.choice([f"d{i}" for i in range(250)], n),
        "store_id":       np.random.choice([f"s{i}" for i in range(20)],  n),
        "sku_id":         np.random.choice([f"sku{i}" for i in range(500)], n),
        "sku_category":   np.random.choice(categories, n, p=[.30,.25,.18,.17,.10]),
        "payment_channel":np.random.choice(channels,   n, p=[.75,.15,.10]),
        "ship_country":   np.random.choice(countries,  n, p=[.55,.12,.12,.10,.11]),
        "ip_country":     np.random.choice(countries,  n, p=[.60,.10,.10,.10,.10]),
        "quantity":       np.random.randint(1,6,n),
        "unit_price":     np.round(np.random.lognormal(mean=np.log(60), sigma=.9, size=n),2),
        "account_age_days":np.random.exponential(300, n).astype(int),
        "coupon_discount": np.random.choice([0,0,0,10,20,30,50], n, p=[.35,.25,.10,.10,.10,.07,.03]),
        "gift_balance_used":np.random.choice([0,0,0,5,25,100,150], n, p=[.40,.25,.10,.10,.10,.04,.01]),
    })

    # Feature scaffold for velocity (time-aware)
    df = df.sort_values("ts")
    for gcol, out in [("customer_id","cust_24h"), ("device_id","dev_24h")]:
        parts = []
        for _, g in df.groupby(gcol, sort=False):
            t = g["ts"].values
            left = np.searchsorted(t, t - np.timedelta64(24,'h'))
            idx = np.arange(len(g))
            parts.append(pd.Series(idx - left, index=g.index))
        df[out] = pd.concat(parts).sort_index()

    # Base features
    df["order_amount"]   = df["quantity"] * df["unit_price"]
    cat_mean             = df.groupby("sku_category")["unit_price"].transform("mean").replace(0, np.nan)
    df["price_ratio"]    = (df["unit_price"]/cat_mean).fillna(1.0)
    df["price_deviation"]= df["price_ratio"].sub(1).abs()                       # |Δ vs category mean|
    df["geo_mismatch"]   = (df["ship_country"] != df["ip_country"]).astype(int)
    df["discount_pct"]   = (df["coupon_discount"]/df["order_amount"].replace(0,np.nan)).fillna(0)
    df["gift_pct"]       = (df["gift_balance_used"]/df["order_amount"].replace(0,np.nan)).fillna(0)

    # Customer baseline (time-aware)
    def cust_roll(g):
        r = g["order_amount"].shift().rolling(10, min_periods=1)
        g["cust_avg_amt_10"] = r.mean()
        g["cust_std_amt_10"] = r.std(ddof=0).replace(0,np.nan)
        g["cust_amt_z"]      = ((g["order_amount"]-g["cust_avg_amt_10"])/g["cust_std_amt_10"]).fillna(0)
        return g
    df = df.groupby("customer_id", sort=False, group_keys=False).apply(cust_roll)

    # Synthetic label from meaningful patterns (used only for demo)
    hv     = (df["order_amount"] > np.quantile(df["order_amount"], 0.9)).astype(int)
    young  = (df["account_age_days"] < 60).astype(int)
    ch_risk= df["payment_channel"].isin(["GiftCard","StoreCredit"]).astype(int)

    r = (
        -3.2
        + 1.2*df["geo_mismatch"]
        + 1.0*hv
        + 0.9*ch_risk
        + 0.7*young
        + 0.8*(df["discount_pct"]>=0.30)
        + 0.8*(df["gift_pct"]>=0.50)
        + 0.6*(df["price_deviation"]>=0.50)
        + 0.6*(df["cust_24h"]>=3)
        + 0.5*(df["dev_24h"]>=3)
        + np.random.normal(0,0.4,len(df))
    )
    p = 1/(1+np.exp(-r))
    df["fraud_flag"] = (np.random.rand(len(df)) < p).astype(int)
    return df

df = make_data()
cat_means_global = df.groupby("sku_category")["unit_price"].mean()

# -------------------- Train / test (time split) ---------------------
cut = int(len(df)*0.75)
train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

num_feats = [
    "order_amount","quantity","unit_price","account_age_days",
    "price_deviation","discount_pct","gift_pct","geo_mismatch",
    "cust_24h","dev_24h","cust_amt_z"
]
cat_feats = ["payment_channel","sku_category","ship_country","ip_country","store_id"]

Xtr = pd.concat([train[num_feats],
                 pd.get_dummies(train[cat_feats].astype(str), drop_first=False)], axis=1)
ytr = train["fraud_flag"].values

# Use a small time-ordered validation slice to choose the best threshold
val_cut = int(len(train)*0.80)
val = train.iloc[val_cut:].copy()
Xval = pd.concat([val[num_feats],
                  pd.get_dummies(val[cat_feats].astype(str), drop_first=False)], axis=1).reindex(columns=Xtr.columns, fill_value=0)
yval = val["fraud_flag"].values

# Fit on full training window
clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, random_state=RND)
clf.fit(Xtr, ytr)

# Threshold search on validation (not shown on screen)
proba_val = clf.predict_proba(Xval)[:,1]
grid = np.linspace(0.01, 0.99, 99)
best_t, best_f1 = 0.5, -1.0
for t in grid:
    yv = (proba_val >= t).astype(int)
    f1 = f1_score(yval, yv, zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

# Evaluation on held-out test
Xte = pd.concat([test[num_feats],
                 pd.get_dummies(test[cat_feats].astype(str), drop_first=False)], axis=1).reindex(columns=Xtr.columns, fill_value=0)
yte   = test["fraud_flag"].values
proba = clf.predict_proba(Xte)[:,1]
yhat  = (proba >= best_t).astype(int)

# ------------------------------ UI ---------------------------------
st.title("Retail Fraud – Executive Dashboard")

tab_overview, tab_data, tab_decide = st.tabs(["Overview", "Raw Data (4,000 rows)", "New Order Decision"])

with tab_overview:
    st.subheader("Model metrics (time-split test)")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy",  f"{accuracy_score(yte,yhat):.2%}")
    c2.metric("Precision", f"{precision_score(yte,yhat,zero_division=0):.2%}")
    c3.metric("Recall",    f"{recall_score(yte,yhat,zero_division=0):.2%}")
    c4.metric("F1-score",  f"{f1_score(yte,yhat,zero_division=0):.2%}")

    st.subheader("Top features")
    try:
        imp = pd.Series(clf.feature_importances_, index=Xtr.columns).sort_values(ascending=False).head(12).reset_index()
        imp.columns = ["feature","importance"]
        st.altair_chart(
            alt.Chart(imp).mark_bar().encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("feature:N", sort="-x", title=None),
                tooltip=["feature","importance"]
            ),
            use_container_width=True
        )
    except Exception:
        st.caption("Feature importances are not available for this model version.")

with tab_data:
    st.caption("All 4,000 transactions are included below (scroll to view).")
    st.dataframe(df.drop(columns=["fraud_flag"]).sort_values("ts"), use_container_width=True, height=540)

with tab_decide:
    st.subheader("Enter order details")
    a,b,c = st.columns(3)
    cat    = a.selectbox("Category", sorted(df["sku_category"].unique()))
    chan   = b.selectbox("Payment channel", ["Card","GiftCard","StoreCredit"])
    ship   = c.selectbox("Shipping country", sorted(df["ship_country"].unique()))

    d,e,f = st.columns(3)
    ip     = d.selectbox("Network country", sorted(df["ip_country"].unique()))
    qty    = e.number_input("Quantity", 1, 10, 1)
    price  = f.number_input("Unit price", 1.0, 2000.0, 120.0, step=1.0)

    g,h,i = st.columns(3)
    disc   = g.number_input("Coupon discount ($)", 0.0, 500.0, 0.0, step=1.0)
    gift   = h.number_input("Gift balance used ($)", 0.0, 1000.0, 0.0, step=1.0)
    age    = i.number_input("Account age (days)", 0, 3650, 180)

    j,k = st.columns(2)
    cust  = j.text_input("Customer ID (optional)", "c_demo")
    dev   = k.text_input("Device ID (optional)", "d_demo")

    # Build features for this record using the same logic
    now = df["ts"].max() + timedelta(hours=1)
    order_amount   = qty*price
    price_ratio    = price/float(cat_means_global.get(cat, price))
    price_deviation= abs(price_ratio-1)
    geo_mismatch   = int(ship != ip)
    discount_pct   = (disc/order_amount) if order_amount else 0
    gift_pct       = (gift/order_amount) if order_amount else 0

    recent = df[(df["ts"]>now-timedelta(hours=24))&(df["ts"]<now)]
    cust_24h = int(recent[recent["customer_id"]==cust].shape[0])
    dev_24h  = int(recent[recent["device_id"]==dev].shape[0])

    hist = df[df["customer_id"]==cust].sort_values("ts")
    if len(hist):
        cm = hist["order_amount"].mean()
        cs = hist["order_amount"].std(ddof=0) or np.nan
        cust_amt_z = (order_amount-cm)/(cs if cs else np.nan)
        if np.isnan(cust_amt_z): cust_amt_z = 0
    else:
        cust_amt_z = 0

    xnum = pd.DataFrame([{
        "order_amount":order_amount, "quantity":qty, "unit_price":price, "account_age_days":age,
        "price_deviation":price_deviation, "discount_pct":discount_pct, "gift_pct":gift_pct,
        "geo_mismatch":geo_mismatch, "cust_24h":cust_24h, "dev_24h":dev_24h, "cust_amt_z":cust_amt_z
    }])
    xcat = pd.get_dummies(pd.DataFrame([{
        "payment_channel":chan, "sku_category":cat, "ship_country":ship, "ip_country":ip, "store_id":"s_manual"
    }]).astype(str))
    xin  = pd.concat([xnum, xcat], axis=1).reindex(columns=Xtr.columns, fill_value=0)

    if st.button("Check decision"):
        score = float(clf.predict_proba(xin)[:,1])
        pred  = int(score >= best_t)
        st.markdown(f"### Decision: {'Fraud' if pred else 'Not Fraud'}")

        # Simple human explanation
        reasons = []
        if geo_mismatch: reasons.append("Network vs shipping country mismatch")
        if order_amount >= df['order_amount'].quantile(.90): reasons.append("High order value")
        if chan in ("GiftCard","StoreCredit"): reasons.append("Higher-risk payment channel")
        if age < 60: reasons.append("New account")
        if discount_pct >= .30 and gift_pct >= .50: reasons.append("Coupon + gift stacking")
        if price_deviation >= .50: reasons.append("Large price anomaly for this category")
        if cust_24h >= 3 or dev_24h >= 3: reasons.append("Unusual 24h order velocity")
        if not reasons: reasons.append("No strong risk patterns triggered")
        st.markdown("**Why:** " + " · ".join(reasons))

# --------------------------- Notes ---------------------------
st.caption(
    "Notes: 4,000-row sample; time-aware split; threshold is chosen automatically on a validation slice to maximize F1 "
    "and is used internally for decisions. Strong features include order value, price anomaly vs category, geo mismatch, "
    "discount/gift stacking, short-term velocity (customer/device), account age, and customer spend baseline."
)
