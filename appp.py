# app.py  —  streamlit run app.py
import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import timedelta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------- App look & constants -------------
st.set_page_config("Retail Fraud – Executive Dashboard", layout="wide")
alt.renderers.set_embed_options(actions=False)
RND, THRESH = 42, 0.30
np.random.seed(RND)

# ------------- 4k Retail-like sample (fixed) -------------
@st.cache_data
def make_data(n=4000):
    cats = ["grocery","electronics","apparel","home","toys"]
    chans = ["Card","Wallet","GiftCard"]     
    cntrs = ["US","CA","UK","DE","IN"]

    df = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "ts": pd.date_range("2023-01-01", periods=n, freq="H"),
        "customer_id": np.random.choice([f"c{i}" for i in range(600)], n, p=None),
        "device_id":   np.random.choice([f"d{i}" for i in range(250)], n),
        "store_id":    np.random.choice([f"s{i}" for i in range(20)], n),
        "sku_id":      np.random.choice([f"sku{i}" for i in range(500)], n),
        "sku_category":np.random.choice(cats, n, p=[.30,.25,.18,.17,.10]),
        "payment_channel": np.random.choice(chans, n, p=[.70,.20,.10]),
        "ship_country": np.random.choice(cntrs, n, p=[.55,.12,.12,.10,.11]),
        "ip_country":   np.random.choice(cntrs, n, p=[.60,.10,.10,.10,.10]),
        "quantity": np.random.randint(1,6,n),
        "unit_price": np.round(np.random.lognormal(mean=np.log(60), sigma=.9, size=n),2),
        "account_age_days": np.random.exponential(300, n).astype(int),
        "coupon_discount": np.random.choice([0,0,0,10,20,30,50], n, p=[.35,.25,.10,.10,.10,.07,.03]),
        "gift_balance_used": np.random.choice([0,0,0,5,25,100,150], n, p=[.40,.25,.10,.10,.10,.04,.01]),
    })

    # --- Customer/device short-term velocity (last 24h) for label realism only ---
    df = df.sort_values("ts")
    for gcol, out in [("customer_id","cust_24h"), ("device_id","dev_24h")]:
        # count in previous 24h per group (time-aware, no leakage)
        cnt = []
        for _, g in df.groupby(gcol, sort=False):
            t = g["ts"].values
            left = np.searchsorted(t, t - np.timedelta64(24,'h'))
            idx = np.arange(len(g))
            cnt.append(pd.Series(idx - left, index=g.index))
        df[out] = pd.concat(cnt).sort_index()

    # Order value
    df["order_amount"] = df["quantity"]*df["unit_price"]

    # Category price anchors & anomaly
    cat_mean = df.groupby("sku_category")["unit_price"].transform("mean").replace(0,np.nan)
    df["price_ratio"] = (df["unit_price"]/cat_mean).fillna(1.0)
    df["price_deviation"] = df["price_ratio"].sub(1).abs()                  # |Δprice|

    # Geo mismatch & stacking
    df["geo_mismatch"] = (df["ship_country"]!=df["ip_country"]).astype(int)
    df["discount_pct"]  = (df["coupon_discount"]/df["order_amount"].replace(0,np.nan)).fillna(0)
    df["gift_pct"]      = (df["gift_balance_used"]/df["order_amount"].replace(0,np.nan)).fillna(0)

    # Customer rolling baseline (last 10, time-aware)
    def cust_roll(g):
        r = g["order_amount"].shift().rolling(10, min_periods=1)
        g["cust_avg_amt_10"] = r.mean()
        g["cust_std_amt_10"] = r.std(ddof=0).replace(0,np.nan)
        g["cust_amt_z"] = ((g["order_amount"]-g["cust_avg_amt_10"])/g["cust_std_amt_10"]).fillna(0)
        return g
    df = df.groupby("customer_id", sort=False, group_keys=False).apply(cust_roll)

    # --- Probabilistic label from meaningful signals (no leakage in training pipeline) ---
    # Risk score (log-odds style), then sample Bernoulli(sigmoid(r))
    hv = (df["order_amount"] > np.quantile(df["order_amount"], 0.9)).astype(int)
    young = (df["account_age_days"] < 60).astype(int)
    ch_risky = df["payment_channel"].isin(["Wallet","GiftCard"]).astype(int)
    r = (
        -3.2
        + 1.2*df["geo_mismatch"]
        + 1.0*hv
        + 0.9*ch_risky
        + 0.7*young
        + 0.8*(df["discount_pct"]>0.30)
        + 0.8*(df["gift_pct"]>0.50)
        + 0.6*(df["price_deviation"]>0.50)
        + 0.6*(df["cust_24h"]>=3)
        + 0.5*(df["dev_24h"]>=3)
        + np.random.normal(0,0.4,len(df))   # noise
    )
    p = 1/(1+np.exp(-r))
    df["fraud_flag"] = (np.random.rand(len(df)) < p).astype(int)
    return df

df = make_data()
cat_means_global = df.groupby("sku_category")["unit_price"].mean()

# ------------- Train / test (time split, no leakage) -------------
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
Xte = pd.concat([test[num_feats],
                 pd.get_dummies(test[cat_feats].astype(str), drop_first=False)], axis=1).reindex(columns=Xtr.columns, fill_value=0)
yte = test["fraud_flag"].values

clf = HistGradientBoostingClassifier(max_depth=None, max_iter=250, learning_rate=0.08,
                                     l2_regularization=0.0, random_state=RND)
clf.fit(Xtr, ytr)
proba = clf.predict_proba(Xte)[:,1]
yhat  = (proba >= THRESH).astype(int)

# ------------- KPI header -------------
st.title("Retail Fraud – Executive Dashboard")
k1,k2,k3,k4 = st.columns(4)
k1.metric("Total transactions", f"{len(df):,}")
k2.metric("Fraud rate (overall)", f"{df['fraud_flag'].mean():.1%}")
k3.metric("Avg order value", f"${df['order_amount'].mean():,.2f}")
k4.metric("Production threshold", f"{THRESH:.2f}")

# ------------- Tabs for Overview / Data / Decision -------------
tab_overview, tab_data, tab_decide = st.tabs(["📈 Overview", "📄 Raw 4,000 Rows", "⚡ New Order Decision"])

with tab_overview:
    # Daily fraud rate trend
    daily = df.assign(day=df["ts"].dt.date).groupby("day").agg(
        txns=("order_id","count"), frauds=("fraud_flag","sum")
    ).reset_index()
    daily["fraud_rate"] = daily["frauds"]/daily["txns"]
    tr = alt.Chart(daily).mark_line(point=True).encode(
        x="day:T", y=alt.Y("fraud_rate:Q", axis=alt.Axis(format="%")),
        tooltip=["day:T","txns:Q",alt.Tooltip("fraud_rate:Q",format=".1%")]
    ).properties(height=220)
    st.subheader("Daily fraud rate")
    st.altair_chart(tr, use_container_width=True)

    c1,c2 = st.columns([1,1])
    with c1:
        # Channel mix
        mix = df.groupby("payment_channel", as_index=False)["order_id"].count().rename(columns={"order_id":"orders"})
        st.subheader("Payment channel mix")
        st.altair_chart(alt.Chart(mix).mark_bar().encode(
            x="payment_channel:N", y="orders:Q", tooltip=["payment_channel","orders"]), use_container_width=True)

    with c2:
        # Geo mismatch heatmap
        heat = df.groupby(["ship_country","ip_country"], as_index=False)["order_id"].count()
        st.subheader("Ship vs. Network country")
        st.altair_chart(alt.Chart(heat).mark_rect().encode(
            x="ip_country:N", y="ship_country:N", color=alt.Color("order_id:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["ship_country","ip_country","order_id"]), use_container_width=True)

    # Model metrics
    st.subheader("Model evaluation (time-split test, threshold = 0.30)")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Accuracy",  f"{accuracy_score(yte,yhat):.2%}")
    m2.metric("Precision", f"{precision_score(yte,yhat,zero_division=0):.2%}")
    m3.metric("Recall",    f"{recall_score(yte,yhat,zero_division=0):.2%}")
    m4.metric("F1-score",  f"{f1_score(yte,yhat,zero_division=0):.2%}")

    # Feature importance
    st.subheader("Top features driving decisions")
    try:
        imp = pd.Series(clf.feature_importances_, index=Xtr.columns).sort_values(ascending=False).head(12).reset_index()
        imp.columns = ["feature","importance"]
        st.altair_chart(alt.Chart(imp).mark_bar().encode(
            x=alt.X("importance:Q", title="Importance"), y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature","importance"]), use_container_width=True)
    except Exception:
        st.caption("Feature importances unavailable for this model/version.")

with tab_data:
    st.caption("Scroll to view all rows (exactly 4,000).")
    st.dataframe(df.drop(columns=["fraud_flag"]).sort_values("ts"), use_container_width=True, height=520)
    st.caption("Label column is hidden here to keep the raw view clean.")

with tab_decide:
    st.subheader("Enter order details")
    # Left side: inputs
    a,b,c = st.columns(3)
    cat    = a.selectbox("Category", sorted(df["sku_category"].unique()))
    chan   = b.selectbox("Payment channel", ["Card","Wallet","GiftCard"])
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
    cust   = j.text_input("Customer ID (optional)", "c_demo")
    dev    = k.text_input("Device ID (optional)", "d_demo")

    # Build features for this order (using same pipeline)
    now = df["ts"].max() + timedelta(hours=1)
    order_amount = qty*price
    price_ratio  = price / float(cat_means_global.get(cat, price))
    price_deviation = abs(price_ratio-1)
    geo_mismatch = int(ship!=ip)
    discount_pct = disc / order_amount if order_amount else 0
    gift_pct     = gift / order_amount if order_amount else 0

    # Velocity from last 24h in our sample (if ID exists, else 0)
    w = df[(df["ts"]>now-timedelta(hours=24))&(df["ts"]<now)]
    cust_24h = int(w[w["customer_id"]==cust].shape[0])
    dev_24h  = int(w[w["device_id"]==dev].shape[0])

    # Customer baseline (use expanding history up to "now")
    hist = df[df["customer_id"]==cust].sort_values("ts")
    if len(hist):
        cm = hist["order_amount"].mean()
        cs = hist["order_amount"].std(ddof=0) or np.nan
        cust_amt_z = (order_amount - cm)/(cs if cs else np.nan)
        if np.isnan(cust_amt_z): cust_amt_z=0
    else:
        cust_amt_z = 0

    # Compose inference matrix with same columns
    xnum = pd.DataFrame([{
        "order_amount":order_amount, "quantity":qty, "unit_price":price, "account_age_days":age,
        "price_deviation":price_deviation, "discount_pct":discount_pct, "gift_pct":gift_pct,
        "geo_mismatch":geo_mismatch, "cust_24h":cust_24h, "dev_24h":dev_24h, "cust_amt_z":cust_amt_z
    }])
    xcat = pd.get_dummies(pd.DataFrame([{
        "payment_channel":chan, "sku_category":cat, "ship_country":ship, "ip_country":ip, "store_id":"s_manual"
    }]).astype(str))
    xin = pd.concat([xnum, xcat], axis=1).reindex(columns=Xtr.columns, fill_value=0)

    st.divider()
    lcol, rcol = st.columns([1,1])
    with lcol:
        if st.button("Check decision"):
            score = float(clf.predict_proba(xin)[:,1])
            pred  = int(score>=THRESH)
            st.markdown(f"### Decision: **{'Fraud' if pred else 'Not Fraud'}**  (score={score:.2f}, threshold={THRESH:.2f})")
            # Reasons (simple rule-based explanation)
            reasons = []
            if geo_mismatch: reasons.append("Network & ship countries differ")
            if order_amount > df['order_amount'].quantile(.90): reasons.append("High order value")
            if chan in ("Wallet","GiftCard"): reasons.append("Higher-risk payment channel")
            if age < 60: reasons.append("New account")
            if discount_pct >= .30 and gift_pct >= .50: reasons.append("Heavy coupon + gift stacking")
            if price_deviation >= .50: reasons.append("Large price anomaly vs category")
            if cust_24h >= 3 or dev_24h >= 3: reasons.append("Velocity burst (24h)")
            if not reasons: reasons.append("No strong risk patterns triggered")
            st.markdown("**Why:** " + " · ".join(reasons))

    with rcol:
        # What-if slider (does not affect the production threshold)
        what_if = st.slider("What-if threshold (for view only)", 0.01, 0.99, 0.30, 0.01)
        yhat_if = (proba>=what_if).astype(int)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Acc (what-if)", f"{accuracy_score(yte,yhat_if):.2%}")
        c2.metric("Prec (what-if)", f"{precision_score(yte,yhat_if,zero_division=0):.2%}")
        c3.metric("Recall (what-if)", f"{recall_score(yte,yhat_if,zero_division=0):.2%}")
        c4.metric("F1 (what-if)", f"{f1_score(yte,yhat_if,zero_division=0):.2%}")

# ------------- Footer (one-liners for the room) -------------
st.caption(
    "Notes: 4,000-row sample; time-aware split; threshold fixed at 0.30 for decisions. "
    "Features: value, price anomaly vs category, geo mismatch, discount/gift stacking, velocity (24h), account age, customer stability."
)
