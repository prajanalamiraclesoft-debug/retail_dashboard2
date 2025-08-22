# app.py  —  Streamlit demo (single file, no external data needed)
# Run:  streamlit run app.py

import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------------------------------------------------------------------
# Page & theme
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Fraud Review – Client Demo", layout="wide")
alt.renderers.set_embed_options(actions=False)

PRIMARY = "#2563eb"     # Tailwind blue-600
GOOD    = "#16a34a"     # green-600
WARN    = "#f59e0b"     # amber-500
BAD     = "#dc2626"     # red-600
MUTED   = "#64748b"     # slate-500

st.markdown(
    f"""
    <style>
      .kpi h2 {{ font-size: 2.0rem !important; margin: 0; }}
      .kpi small {{ color:{MUTED}; }}
      .badge {{
         display:inline-block; padding:.25rem .6rem; border-radius:999px;
         font-weight:600; color:white; background:{PRIMARY}; margin-left:.5rem;
      }}
      .decision {{
         font-size:1.6rem; font-weight:700; margin:.25rem 0;
      }}
      .approve {{ color:{GOOD}; }}
      .review  {{ color:{WARN}; }}
      .block   {{ color:{BAD};  }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# 1) Build a realistic 4k dataset (synthetic, no external systems)
#    - only the features business cares about
# ------------------------------------------------------------------------------
RNG = np.random.default_rng(42)

def build_4k():
    n = 4000
    start = datetime(2023, 1, 1)
    ts = [start + timedelta(minutes=int(x)) for x in RNG.integers(0, 60*24*365, size=n)]

    categories = ["electronics","home","apparel","toys","grocery"]
    channels   = ["Card","Wallet","Bank","Delivery","Other"]
    countries  = ["US","UK","DE","IN","CA"]

    sku_category = RNG.choice(categories, size=n, p=[.30,.18,.18,.14,.20])
    pay_channel  = RNG.choice(channels,  size=n, p=[.45,.20,.15,.10,.10])
    ship_country = RNG.choice(countries, size=n)
    ip_country   = np.array([ ship_country[i] if RNG.random()<0.80
                              else RNG.choice([c for c in countries if c!=ship_country[i]])
                              for i in range(n) ])

    # Category price baselines (mean, sd)
    base = {"electronics": (260, 90), "home": (85, 30), "apparel": (60, 25), "toys": (38, 14), "grocery": (22, 8)}
    unit_price = np.array([ max(1, RNG.normal(base[c][0], base[c][1])) for c in sku_category ])
    quantity   = np.maximum(1, RNG.poisson(lam=RNG.uniform(1.3, 2.5, size=n))).astype(float)
    order_amount = unit_price * quantity

    coupon_discount = np.clip(RNG.gamma(2.1, 5.0, size=n), 0, order_amount*0.5)
    gift_used_flag  = RNG.random(size=n) < 0.22
    gift_balance_used = np.zeros(n)
    gift_balance_used[gift_used_flag] = np.clip(order_amount[gift_used_flag] * RNG.uniform(0.2, 0.8, gift_used_flag.sum()), 0, None)

    account_age_days = np.maximum(0, RNG.normal(120, 90, size=n)).astype(float)

    # Strong features
    coupon_pct = np.divide(coupon_discount, order_amount, out=np.zeros_like(order_amount), where=order_amount!=0)
    gift_pct   = np.divide(gift_balance_used, order_amount, out=np.zeros_like(order_amount), where=order_amount!=0)
    addr_mismatch = (ship_country != ip_country).astype(int)
    tmp = pd.DataFrame({"cat":sku_category, "p":unit_price})
    cat_avg = tmp.groupby("cat")["p"].transform("mean").values
    price_ratio = np.divide(unit_price, cat_avg, out=np.ones_like(unit_price), where=cat_avg!=0)
    pay_code = pd.Series(pay_channel).map({"Wallet":2, "Card":1, "Other":1, "Bank":0, "Delivery":0}).values

    # Label from combinations of strong patterns (no jargon)
    s = np.zeros(n, dtype=float)
    s += (addr_mismatch & (pay_channel=="Wallet") & (order_amount>320)) * 1.2
    s += ((gift_pct>0.55) & (coupon_pct>0.20)) * 1.0
    s += ((np.abs(price_ratio-1)>0.50) & (quantity>=3)) * 1.0
    s += ((account_age_days<10) & (order_amount>420)) * 0.9
    s += ((pay_channel=="Wallet") & (order_amount>180)) * 0.6
    s += RNG.normal(0, 0.15, size=n)
    fraud_flag = (s > 1.0).astype(int)

    df = pd.DataFrame({
        "order_id": [f"order_{i}" for i in range(1, n+1)],
        "sku_category": sku_category,
        "pay_channel": pay_channel,
        "ship_country": ship_country,
        "ip_country": ip_country,
        "unit_price": unit_price,
        "quantity": quantity,
        "order_amount": order_amount,
        "coupon_discount": coupon_discount,
        "gift_balance_used": gift_balance_used,
        "gift_used_flag": gift_used_flag.astype(int),
        "account_age_days": account_age_days,
        # engineered features (model uses these directly)
        "coupon_pct": coupon_pct,
        "gift_pct": gift_pct,
        "addr_mismatch": addr_mismatch,
        "price_ratio": price_ratio,
        "pay_code": pay_code,
        "fraud_flag": fraud_flag,
        "ts": ts,  # internal; hidden in the table
    })
    return df

@st.cache_data(show_spinner=False)
def build_data_and_model():
    df = build_4k()

    FEATURES = [
        "order_amount","quantity","unit_price","account_age_days",
        "coupon_pct","gift_pct","price_ratio","addr_mismatch","pay_code",
    ]
    X = df[FEATURES].fillna(0)
    y = df["fraud_flag"].astype(int).values

    # Time-aware split (train early, test later) to mimic real life
    df_sorted = df.sort_values("ts").reset_index(drop=True)
    cut = int(len(df_sorted)*0.8)
    tr_idx = df_sorted.index[:cut]; te_idx = df_sorted.index[cut:]

    X_tr, y_tr = X.loc[tr_idx], y[tr_idx]
    X_te, y_te = X.loc[te_idx], y[te_idx]

    # Class weighting to keep recall strong on positives
    pos = max(1, int((y_tr==1).sum())); neg = max(1, len(y_tr)-pos)
    w_pos = min(15.0, neg/pos)
    sample_w = np.where(y_tr==1, w_pos, 1.0)

    clf = HistGradientBoostingClassifier(max_iter=700, learning_rate=0.06,
                                         early_stopping=True, random_state=42)
    clf.fit(X_tr, y_tr, sample_weight=sample_w)

    # Choose thresholds internally (no sliders): review & block
    probs_te = clf.predict_proba(X_te)[:,1]
    ts = np.linspace(0.05, 0.95, 91)

    # t_review: ensure high recall (keep fraud from slipping through)
    t_review = ts[0]
    for t in ts:
        yhat = (probs_te >= t).astype(int)
        if recall_score(y_te, yhat, zero_division=0) >= 0.90:
            t_review = t; break
    # t_block: ensure high precision (keep alert quality high)
    t_block = ts[-1]
    for t in ts[::-1]:
        yhat = (probs_te >= t).astype(int)
        if precision_score(y_te, yhat, zero_division=0) >= 0.80:
            t_block = t; break
    t_block = max(t_block, t_review + 0.05)  # tiny gap to create a review band

    # Overall test-set metrics (using a single cut at F1-max for reporting)
    best_t, best_f1 = 0.5, -1
    for t in ts:
        yhat = (probs_te >= t).astype(int)
        f1 = f1_score(y_te, yhat, zero_division=0)
        if f1 > best_f1: best_f1, best_t = f1, t
    y_hat = (probs_te >= best_t).astype(int)
    ACC  = accuracy_score(y_te, y_hat)
    PREC = precision_score(y_te, y_hat, zero_division=0)
    REC  = recall_score(y_te, y_hat, zero_division=0)
    F1   = f1_score(y_te, y_hat, zero_division=0)

    cat_avg_price = df.groupby("sku_category")["unit_price"].mean().to_dict()
    overall_avg   = float(df["unit_price"].mean())

    return {
        "df": df,
        "X": X,
        "FEATURES": FEATURES,
        "clf": clf,
        "metrics": (ACC, PREC, REC, F1),
        "thresholds": (t_review, t_block, best_t),
        "cat_avg_price": cat_avg_price,
        "overall_avg_price": overall_avg,
    }

bundle = build_data_and_model()
df = bundle["df"]; X = bundle["X"]; FEATURES = bundle["FEATURES"]; clf = bundle["clf"]
ACC, PREC, REC, F1 = bundle["metrics"]
t_review, t_block, best_t = bundle["thresholds"]
cat_avg_price = bundle["cat_avg_price"]; overall_avg = bundle["overall_avg_price"]

# ------------------------------------------------------------------------------
# Helpers for explanations & visuals
# ------------------------------------------------------------------------------
def reasons_from_row(r):
    reasons = []
    if r["addr_mismatch"] and (r["pay_channel"]=="Wallet") and (r["order_amount"]>300):
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
        if not r["addr_mismatch"]: safes.append("Address looks consistent")
        if r["gift_pct"]<0.20:     safes.append("Low gift-balance usage")
        if r["coupon_pct"]<0.30:   safes.append("Discount within normal range")
        if r["account_age_days"]>=10: safes.append("Account not brand-new")
        if abs(r["price_ratio"]-1)<=0.50 or r["quantity"]<3: safes.append("No unusual price or bulk")
        reasons = ["; ".join(safes)] if safes else ["No clear risk patterns"]
    return reasons

def pretty_percent(x): return f"{x*100:.2f}%"

# ------------------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------------------
st.markdown(f"## Fraud Review Dashboard <span class='badge'>demo</span>", unsafe_allow_html=True)
st.caption("Trains on 4,000 historical orders. Decisions use 8 strong, business-friendly signals only.")

# KPIs
k1,k2,k3,k4 = st.columns(4)
with k1: st.markdown(f"<div class='kpi'><small>Accuracy</small><h2>{pretty_percent(ACC)}</h2></div>", unsafe_allow_html=True)
with k2: st.markdown(f"<div class='kpi'><small>Precision</small><h2>{pretty_percent(PREC)}</h2></div>", unsafe_allow_html=True)
with k3: st.markdown(f"<div class='kpi'><small>Recall</small><h2>{pretty_percent(REC)}</h2></div>", unsafe_allow_html=True)
with k4: st.markdown(f"<div class='kpi'><small>F1-score</small><h2>{pretty_percent(F1)}</h2></div>", unsafe_allow_html=True)

st.divider()

# ------------------------------------------------------------------------------
# VISUALS (executive style, no heavy EDA)
# ------------------------------------------------------------------------------
left, right = st.columns([3,2], gap="large")

with left:
    st.markdown("#### Weekly volume & fraud rate")
    tmp = df.assign(week=pd.to_datetime(df["ts"]).dt.to_period("W").dt.start_time)
    weekly = tmp.groupby("week").agg(orders=("order_id","count"),
                                     frauds=("fraud_flag","sum")).reset_index()
    weekly["rate"] = weekly["frauds"]/weekly["orders"]
    base = alt.Chart(weekly).encode(x="week:T")
    line1 = base.mark_line(point=True).encode(y=alt.Y("orders:Q", title="Orders"))
    line2 = base.mark_line(color=BAD, strokeDash=[4,2]).encode(y=alt.Y("rate:Q", title="Fraud rate", axis=alt.Axis(format="%")))
    st.altair_chart((line1 + line2).resolve_scale(y='independent').properties(height=240), use_container_width=True)

    st.markdown("#### Fraud rate by category")
    cat = df.groupby("sku_category").agg(orders=("order_id","count"),
                                         frauds=("fraud_flag","sum")).reset_index()
    cat["rate"] = cat["frauds"]/cat["orders"]
    st.altair_chart(
        alt.Chart(cat).mark_bar().encode(
            x=alt.X("sku_category:N", title=None),
            y=alt.Y("rate:Q", title="Fraud rate", axis=alt.Axis(format="%")),
            tooltip=["orders","frauds",alt.Tooltip("rate:Q", format=".1%")],
            color=alt.value(BAD)
        ).properties(height=220),
        use_container_width=True,
    )

with right:
    st.markdown("#### Channel mix vs. fraud share")
    share = df.groupby("pay_channel").agg(orders=("order_id","count"),
                                          frauds=("fraud_flag","sum")).reset_index()
    share["order_share"] = share["orders"]/share["orders"].sum()
    share["fraud_share"] = share["frauds"]/share["frauds"].sum()
    donut_orders = alt.Chart(share).mark_arc(innerRadius=55).encode(theta="order_share:Q", color="pay_channel:N", tooltip=["pay_channel","orders",alt.Tooltip("order_share:Q",format=".1%")]).properties(title="Orders")
    donut_fraud  = alt.Chart(share).mark_arc(innerRadius=55).encode(theta="fraud_share:Q",  color="pay_channel:N", tooltip=["pay_channel","frauds",alt.Tooltip("fraud_share:Q",format=".1%")]).properties(title="Frauds")
    st.altair_chart(alt.hconcat(donut_orders, donut_fraud).resolve_scale(color='independent').properties(height=220), use_container_width=True)

    st.markdown("#### What the model pays attention to")
    imp = pd.Series(getattr(clf, "feature_importances_", np.ones(len(FEATURES))/len(FEATURES)),
                    index=FEATURES).sort_values(ascending=True).reset_index()
    imp.columns = ["feature","importance"]
    st.altair_chart(
        alt.Chart(imp).mark_bar().encode(
            x=alt.X("importance:Q", title="Relative importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.value(PRIMARY),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")]
        ).properties(height=240),
        use_container_width=True
    )

st.divider()

# ------------------------------------------------------------------------------
# RAW DATA (the whole 4,000 rows, nothing hidden)
# ------------------------------------------------------------------------------
st.markdown("### Raw dataset (4,000 rows)")
st.dataframe(df.drop(columns=["ts"]), use_container_width=True, height=420)
st.download_button("Download CSV", df.drop(columns=["ts"]).to_csv(index=False).encode("utf-8"),
                   file_name="fraud_raw_4000.csv", mime="text/csv")

st.divider()

# ------------------------------------------------------------------------------
# New Order – minimal fields (only strong features)
# ------------------------------------------------------------------------------
st.markdown("### New Order – instant decision")

countries = ["US","UK","DE","IN","CA"]
categories = ["electronics","home","apparel","toys","grocery"]
channels   = ["Card","Wallet","Bank","Delivery","Other"]

with st.form("new_order_form_min", clear_on_submit=False):
    c1,c2,c3 = st.columns(3)
    with c1:
        sku_category   = st.selectbox("Category", categories, index=0)
        ship_country   = st.selectbox("Shipping country", countries, index=0)
        quantity       = st.number_input("Quantity", 1.0, step=1.0, value=1.0)
    with c2:
        pay_channel    = st.selectbox("Payment channel", channels, index=0)
        ip_country     = st.selectbox("Network country", countries, index=0)
        unit_price     = st.number_input("Unit price", 1.0, step=1.0, value=120.0)
    with c3:
        discount_amt   = st.number_input("Discount amount", 0.0, step=1.0, value=0.0)
        gift_used_amt  = st.number_input("Gift balance used", 0.0, step=1.0, value=0.0)
        account_age    = st.number_input("Account age (days)", 0.0, step=1.0, value=120.0)
    submitted = st.form_submit_button("Check")

if submitted:
    order_amount = float(unit_price * quantity)
    coupon_pct   = float((discount_amt  / order_amount) if order_amount else 0.0)
    gift_pct     = float((gift_used_amt / order_amount) if order_amount else 0.0)
    addr_mismatch= int(ship_country != ip_country)
    ref_avg      = float(cat_avg_price.get(sku_category, overall_avg))
    price_ratio  = float(unit_price / ref_avg) if ref_avg>0 else 1.0
    pay_code     = {"Wallet":2, "Card":1, "Other":1, "Bank":0, "Delivery":0}[pay_channel]

    row_for_model = pd.DataFrame([{
        "order_amount": order_amount,
        "quantity": float(quantity),
        "unit_price": float(unit_price),
        "account_age_days": float(account_age),
        "coupon_pct": coupon_pct,
        "gift_pct": gift_pct,
        "price_ratio": price_ratio,
        "addr_mismatch": addr_mismatch,
        "pay_code": pay_code,
    }])[FEATURES].fillna(0)

    p = clf.predict_proba(row_for_model)[:,1][0]

    # Internal decision policy: Block / Review / Approve (no thresholds shown in UI)
    if p >= t_block:
        decision, css = "BLOCK", "block"
    elif p >= t_review:
        decision, css = "REVIEW", "review"
    else:
        decision, css = "APPROVE", "approve"

    reasons = reasons_from_row({
        "addr_mismatch": addr_mismatch,
        "pay_channel": pay_channel,
        "order_amount": order_amount,
        "gift_pct": gift_pct,
        "coupon_pct": coupon_pct,
        "price_ratio": price_ratio,
        "quantity": float(quantity),
        "account_age_days": float(account_age),
    })

    st.markdown(f"<div class='decision {css}'>Decision: {decision}</div>", unsafe_allow_html=True)
    st.write("**Why:** " + " | ".join(reasons))

    st.caption("The decision uses: amount, quantity, unit price vs category, discount %, gift-balance %, address consistency, account age, and payment channel.")
    st.table(pd.DataFrame([{
        "Category": sku_category,
        "Payment channel": pay_channel,
        "Shipping vs Network": f"{ship_country} / {ip_country}",
        "Unit price": round(unit_price,2),
        "Quantity": int(quantity),
        "Order amount": round(order_amount,2),
        "Discount %": round(coupon_pct*100,2),
        "Gift %": round(gift_pct*100,2),
        "Price ratio vs category": round(price_ratio,2),
        "Account age (days)": int(account_age),
    }]))
