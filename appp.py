# run: streamlit run app.py
import streamlit as st, pandas as pd, numpy as np, altair as alt
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config("Unified Fraud, Pricing & Inventory Risk Detection Dashboard", layout="wide")
alt.renderers.set_embed_options(actions=False)

# ------------------------- Sidebar: only table info -------------------------
with st.sidebar:
    st.header("BQ TABLE INFO")
    P  = st.text_input("Project", "mss-data-engineer-sandbox")
    D  = st.text_input("Dataset", "retail")
    PT = st.text_input("Predictions table", f"{P}.{D}.predictions_latest")
    FT = st.text_input("Features table",   f"{P}.{D}.features_signals_v4")

# ------------------------- BigQuery client ---------------------------------
sa = dict(st.secrets["gcp_service_account"])
sa["private_key"] = sa["private_key"].replace("\\n", "\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq = bigquery.Client(credentials=creds, project=creds.project_id)

def cols_of(t):
    p,d,x = t.split(".")
    q = f"SELECT column_name FROM `{p}.{d}.INFORMATION_SCHEMA.COLUMNS` WHERE table_name=@t"
    job = bq.query(q, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("t","STRING",x)]
    ))
    return set(job.result().to_dataframe().column_name.str.lower())

@st.cache_data(show_spinner=True)
def load(pred_t, feat_t):
    pcols, fcols = cols_of(pred_t), cols_of(feat_t)

    base = [
        "order_id","timestamp","fraud_score","customer_id","store_id","sku_id",
        "sku_category","order_amount","quantity","payment_method",
        "shipping_country","ip_country"
    ]
    strong = [
        "strong_tri_mismatch_high_value","strong_high_value_express_geo",
        "strong_burst_multi_device","strong_price_drop_bulk","strong_giftcard_geo",
        "strong_return_whiplash","strong_price_inventory_stress","strong_country_flip_express",
        "high_price_anomaly","low_price_anomaly","oversell_flag",
        "stockout_risk_flag","hoarding_flag","fraud_flag"
    ]

    sel = [
        (f"p.{c}" if c in pcols else
         f"CAST(NULL AS {'FLOAT64' if c=='fraud_score' else 'STRING'}) AS {c}")
        for c in base
    ]

    join = ""
    if fcols:
        join = f"LEFT JOIN `{feat_t}` s USING(order_id)"
        sel += [
            (f"s.{c}" if c in fcols else
             ("CAST(NULL AS INT64) AS fraud_flag" if c=="fraud_flag" else f"CAST(0 AS INT64) AS {c}"))
            for c in strong
        ]
    else:
        sel += [("CAST(NULL AS INT64) AS fraud_flag" if c=="fraud_flag" else f"CAST(0 AS INT64) AS {c}")
                for c in strong]

    sql = f"SELECT {', '.join(sel)} FROM `{pred_t}` p {join} ORDER BY timestamp"
    df  = bq.query(sql).result().to_dataframe()
    df["timestamp"]  = pd.to_datetime(df["timestamp"], errors="coerce")
    df["fraud_score"]= pd.to_numeric(df["fraud_score"], errors="coerce").fillna(0.0)
    return df

df = load(PT, FT)
if df.empty:
    st.warning("No rows available.")
    st.stop()

# ------------------------- Header + Threshold (on the RIGHT) ---------------
st.title("Unified Fraud, Pricing & Inventory Risk Detection")

k1, k2, k3, k4 = st.columns([1,1,1,1.4])
with k4:
    TH = st.slider("Decision threshold", 0.00, 1.00, 0.50, 0.01, help=(
        "Orders with fraud_score ≥ threshold are marked as **suspicious**. "
        "Move right to tighten control (fewer alerts, higher precision), "
        "move left to widen the net (more alerts, higher recall)."
    ))

# Compute alerts AFTER threshold is chosen
df["is_alert"] = (df["fraud_score"] >= TH).astype(int)

k1.metric("TOTAL ROWS", len(df))
k2.metric("SUSPICIOUS ORDERS", int(df["is_alert"].sum()))
k3.metric("ALERT RATE", f"{df['is_alert'].mean():.2%}")
if df.timestamp.notna().any():
    st.caption(f"TABLE WINDOW: {df.timestamp.min()} → {df.timestamp.max()}")

# ------------------------- Retail transactions (no fraud_score) ------------
st.subheader("Retail transactions")
show_cols = [c for c in [
    "order_id","timestamp","customer_id","store_id","sku_id","sku_category",
    "order_amount","quantity","payment_method","shipping_country","ip_country"
] if c in df.columns]
st.dataframe(df.loc[:, show_cols].head(4000), use_container_width=True, height=340)

# ------------------------- Fraud score distribution (with explainer) -------
st.subheader("Fraud score distribution")
st.caption("Histogram of model scores across all orders. Bars to the right indicate higher risk. "
           "The vertical line shows your current decision threshold.")

hist = alt.Chart(pd.DataFrame({"score": df["fraud_score"]})).mark_bar().encode(
    x=alt.X("score:Q", bin=alt.Bin(maxbins=50), title="Fraud score"),
    y=alt.Y("count():Q", title="Orders")
)
rule = alt.Chart(pd.DataFrame({"t":[TH]})).mark_rule(strokeDash=[6,3]).encode(x="t:Q")
st.altair_chart((hist + rule).properties(height=220), use_container_width=True)

# ------------------------- Strong Signals (with clear explanation) ----------
st.subheader("Strong signals driving alerts")
st.caption(
    "These signals are patterns we monitor. The left chart shows how common they are **among alerts**; "
    "the right shows the same for **cleared (not fraud) orders**. This helps explain **why** orders are flagged."
)

def prevalence(cols, title):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        st.info(f"No fields for {title}.")
        return
    z = df[["is_alert"] + cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    a  = max(1, int(z["is_alert"].sum()))
    na = max(1, int((1 - z["is_alert"]).sum()))
    rows = [{"signal":c, "% in alerts": z.loc[z.is_alert==1, c].sum()/a,
                       "% in non-alerts": z.loc[z.is_alert==0, c].sum()/na}
            for c in cols]
    dd = (pd.DataFrame(rows)
          .sort_values("% in alerts", ascending=False)
          .melt(id_vars="signal", var_name="group", value_name="value"))
    st.altair_chart(
        alt.Chart(dd).mark_bar().encode(
            x=alt.X("value:Q", axis=alt.Axis(format="%"), title="Prevalence"),
            y=alt.Y("signal:N", sort="-x", title=None),
            color="group:N",
            tooltip=["signal", alt.Tooltip("value:Q", format=".1%"), "group"]
        ).properties(height=300, title=title),
        use_container_width=True
    )

left, right = st.columns(2)
with left:
    prevalence([
        "strong_tri_mismatch_high_value","strong_high_value_express_geo","strong_burst_multi_device",
        "strong_price_drop_bulk","strong_giftcard_geo","strong_return_whiplash",
        "strong_price_inventory_stress","strong_country_flip_express"
    ], "Fraud-pattern signals")
with right:
    prevalence([
        "high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"
    ], "Pricing & inventory context")

# ------------------------- Suspicious orders (with score) -------------------
st.subheader("ML-detected suspicious orders")
cols = [c for c in [
    "order_id","timestamp","customer_id","store_id","sku_id","sku_category",
    "order_amount","quantity","payment_method","shipping_country","ip_country","fraud_score"
] if c in df.columns]
st.dataframe(
    df[df["is_alert"]==1].sort_values(["fraud_score","timestamp"], ascending=[False, False])[cols],
    use_container_width=True,
    height=min(720, 28*min(len(df), 20) + 120)
)

# ------------------------- Model Evaluation --------------------------------
st.subheader("Model evaluation")
labs = [c for c in ["fraud_flag","is_fraud","label","ground_truth","gt","y"] if c in df.columns]
y     = df[labs[0]].fillna(0).astype(int).values if labs else df["is_alert"].values
if labs:
    st.caption(f"Using label column: `{labs[0]}`")

yhat  = df["is_alert"].values
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy",  f"{accuracy_score(y, yhat):.2%}")
c2.metric("Precision", f"{precision_score(y, yhat, zero_division=0):.2%}")
c3.metric("Recall",    f"{recall_score(y, yhat, zero_division=0):.2%}")
c4.metric("F1-score",  f"{f1_score(y, yhat, zero_division=0):.2%}")

# ------------------------- Instant Decision --------------------------------
st.markdown("### New order — instant decision")
st.caption("Scores a single order using the same model logic. Result depends on the threshold above.")

cats  = sorted(df.get("sku_category", pd.Series(["apparel"])).dropna().unique())
ships = sorted(df.get("shipping_country", pd.Series(["US"])).dropna().unique())
ips   = sorted(df.get("ip_country",      pd.Series(["US"])).dropna().unique())

c1,c2,c3 = st.columns(3)
cat = c1.selectbox("SKU category", cats)
pay = c2.selectbox("Payment method", ["credit_card","debit_card","gift_card","paypal","apple_pay"])
qty = c3.number_input("Quantity", 1, 99, 1)

c4,c5 = st.columns(2)
price = c4.number_input("Unit price", 1.0, 10000.0, 100.0, 1.0)
ship  = c5.selectbox("Shipping country", ships)
ip    = st.selectbox("IP country", ips)

PAY = {"gift_card":0.35,"paypal":0.20,"apple_pay":0.12,"credit_card":0.10,"debit_card":0.08}

cs = (
    df[["sku_category","order_amount"]].dropna()
      .groupby("sku_category")["order_amount"]
      .agg(cat_mean="mean", cat_p90=lambda s: float(np.nanpercentile(s,90))).reset_index()
    if "order_amount" in df and "sku_category" in df else
    pd.DataFrame({"sku_category":cats or ["apparel"], "cat_mean":[500.0], "cat_p90":[1500.0]})
)

def score(cat, pay, qty, price, ship, ip):
    r   = cs[cs.sku_category==cat].iloc[:1]
    amt = qty*price
    geo = (ship != ip)
    high= bool(amt >= float(r.cat_p90))
    s   = 0.08 + (0.35 if geo else 0) + (0.30 if high else 0) + PAY.get(pay, 0.10)
    if geo and pay in ("gift_card","paypal"): s = max(s, 0.65)
    if geo and high and pay in ("gift_card","paypal"): s = max(s, 0.85)
    if high and pay in ("gift_card","paypal"): s = max(s, 0.70)
    return min(s, 0.99), {"amount":amt, "mean":float(r.cat_mean), "p90":float(r.cat_p90),
                          "geo":geo, "high":high}

if st.button("Score order"):
    s, e = score(cat, pay, int(qty), float(price), ship, ip)
    st.markdown(f"**Decision:** {'🚩 Suspicious' if s>=TH else '✅ Cleared (not fraud)'} • Score ≈ **{s:.2f}** • Threshold {TH:.2f}")
    why = [
        "Billing/Shipping mismatch" if e["geo"] else "Billing/Shipping consistent",
        "Amount high for this category" if e["high"] else "Amount typical for this category",
        f"Payment channel: {pay.replace('_',' ')}"
    ]
    st.write("**Signals:** " + " · ".join(why))
    st.caption(f"Amount ≈ {e['amount']:,.2f} • Category mean ≈ {e['mean']:,.2f} • P90 ≈ {e['p90']:,.2f}")
