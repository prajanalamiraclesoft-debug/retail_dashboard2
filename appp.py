# run: streamlit run app.py
import streamlit as st, pandas as pd, numpy as np, altair as alt
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config("Unified ML-Driven Fraud Detection for Retail POS", layout="wide")
alt.renderers.set_embed_options(actions=False)

with st.sidebar:
    st.header("BQ TABLE INFO")
    P  = st.text_input("Project", "mss-data-engineer-sandbox")
    D  = st.text_input("Dataset", "retail")
    PT = st.text_input("Predictions table", f"{P}.{D}.predictions_latest")
    FT = st.text_input("Features table",   f"{P}.{D}.features_signals_v4")
    st.caption("Provide fully-qualified table names. The app adapts to actual schemas.")

# BigQuery client
sa = dict(st.secrets["gcp_service_account"]); sa["private_key"] = sa["private_key"].replace("\\n","\n")
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

    base = ["order_id","timestamp","fraud_score","customer_id","store_id","sku_id",
            "sku_category","order_amount","quantity","payment_method","shipping_country","ip_country"]
    strong = ["strong_tri_mismatch_high_value","strong_high_value_express_geo","strong_burst_multi_device",
              "strong_price_drop_bulk","strong_giftcard_geo","strong_return_whiplash","strong_price_inventory_stress",
              "strong_country_flip_express","high_price_anomaly","low_price_anomaly","oversell_flag",
              "stockout_risk_flag","hoarding_flag","fraud_flag"]

    sel = [(f"p.{c}" if c in pcols else f"CAST(NULL AS {'FLOAT64' if c=='fraud_score' else 'STRING'}) AS {c}") for c in base]
    join = ""
    if fcols:
        join = f"LEFT JOIN `{feat_t}` s USING(order_id)"
        sel += [(f"s.{c}" if c in fcols else ("CAST(NULL AS INT64) AS fraud_flag" if c=="fraud_flag" else f"CAST(0 AS INT64) AS {c}")) for c in strong]
    else:
        sel += [("CAST(NULL AS INT64) AS fraud_flag" if c=="fraud_flag" else f"CAST(0 AS INT64) AS {c}") for c in strong]

    sql = f"SELECT {', '.join(sel)} FROM `{pred_t}` p {join} ORDER BY timestamp"
    df  = bq.query(sql).result().to_dataframe()

    # --- Clean types ---
    df["timestamp"]    = pd.to_datetime(df["timestamp"], errors="coerce")
    df["fraud_score"]  = pd.to_numeric(df["fraud_score"], errors="coerce").fillna(0.0)

    # --- IMPORTANT: Remove crypto, map to apple_pay; normalize payment names ---
    if "payment_method" in df.columns:
        pm = (df["payment_method"].astype(str).str.strip().str.lower()
              .str.replace(" ", "_").str.replace("-", "_"))
        crypto_like = pm.str.contains(r"\b(crypto|bitcoin|btc|ethereum|eth|usdt|usdc|doge|sol)\b", regex=True)
        pm = pm.mask(crypto_like, "apple_pay")
        # keep only known methods; unknowns → credit_card
        allowed = {"credit_card","debit_card","apple_pay","gift_card","paypal"}
        pm = pm.where(pm.isin(allowed), "credit_card")
        df["payment_method"] = pm

    return df

df = load(PT, FT)
if df.empty:
    st.warning("No rows available."); st.stop()

# ---------------- Header + Threshold (right) ----------------
st.title("Unified ML-Driven Fraud Detection for Retail POS")
st.caption("Bridging Behavioral Risk, Pricing Anomalies and Inventory Stress")
k1,k2,k3,k4 = st.columns([1,1,1,1.8])
with k4:
    TH = st.slider("Decision threshold", 0.00, 1.00, 0.50, 0.01,
                   help=("Orders with fraud_score at or above this value are marked as suspicious. "
                         "Move right to reduce alerts (higher precision). Move left to catch more (higher recall)."))

df["is_alert"] = (df["fraud_score"] >= TH).astype(int)
k1.metric("Transactions", f"{len(df):,}")
k2.metric("Suspicious orders", int(df["is_alert"].sum()))
k3.metric("Alert rate", f"{df['is_alert'].mean():.2%}")
if df.timestamp.notna().any():
    st.caption(f"TABLE WINDOW: {df.timestamp.min()} → {df.timestamp.max()}")

# --------------- Retail transactions (NO fraud_score) ---------------
st.subheader("Retail transactions")
tx_cols = [c for c in ["order_id","timestamp","customer_id","store_id","sku_id","sku_category",
                       "order_amount","quantity","payment_method","shipping_country","ip_country"] if c in df.columns]
st.dataframe(df.loc[:, tx_cols].head(4000), use_container_width=True, height=340)

# --------------- Fraud score distribution ---------------------------
st.subheader("Fraud score distribution")
st.caption("Distribution of ML-assigned risk scores across all orders; dashed line is your current decision threshold.")
hist = alt.Chart(pd.DataFrame({"score": df["fraud_score"]})).mark_bar().encode(
    x=alt.X("score:Q", bin=alt.Bin(maxbins=50), title="Fraud score"),
    y=alt.Y("count():Q", title="Orders")
)
rule = alt.Chart(pd.DataFrame({"t":[TH]})).mark_rule(strokeDash=[6,3]).encode(x="t:Q")
st.altair_chart((hist + rule).properties(height=220), use_container_width=True)

# --------------- Strong signals ---------------------------
st.subheader("Strong signals driving alerts")
st.caption("Patterns the system monitors, shown in flagged (suspicious) vs cleared (not fraud) orders—explaining why items are flagged.")

def prevalence(cols, title):
    cols = [c for c in cols if c in df.columns]
    if not cols: st.info(f"No fields for {title}."); return
    z  = df[["is_alert"] + cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    a  = max(1, int(z["is_alert"].sum()))
    na = max(1, int((1 - z["is_alert"]).sum()))
    rows = [{"signal":c, "% in flagged": z.loc[z.is_alert==1, c].sum()/a,
                       "% in cleared": z.loc[z.is_alert==0, c].sum()/na} for c in cols]
    dd = (pd.DataFrame(rows).sort_values("% in flagged", ascending=False)
          .melt(id_vars="signal", var_name="group", value_name="value"))
    st.altair_chart(
        alt.Chart(dd).mark_bar().encode(
            x=alt.X("value:Q", axis=alt.Axis(format="%"), title="Share of orders"),
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
    prevalence(["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"],
               "Pricing & inventory context")

# --------------- Suspicious orders (with score) ---------------------
st.subheader("ML-detected suspicious orders")
sus_cols = [c for c in ["order_id","timestamp","customer_id","store_id","sku_id","sku_category",
                        "order_amount","quantity","payment_method","shipping_country","ip_country","fraud_score"] if c in df.columns]
st.dataframe(df[df["is_alert"]==1].sort_values(["fraud_score","timestamp"], ascending=[False, False])[sus_cols],
             use_container_width=True, height=min(720, 28*min(len(df), 20) + 120))

# --------------- Model evaluation (optional) ------------------------
st.subheader("Model evaluation")
labs = [c for c in ["fraud_flag","is_fraud","label","ground_truth","gt","y"] if c in df.columns]
if labs:
    y, yhat = df[labs[0]].fillna(0).astype(int).values, df["is_alert"].values
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy",  f"{accuracy_score(y, yhat):.2%}")
    c2.metric("Precision", f"{precision_score(y, yhat, zero_division=0):.2%}")
    c3.metric("Recall",    f"{recall_score(y, yhat, zero_division=0):.2%}")
    c4.metric("F1-score",  f"{f1_score(y, yhat, zero_division=0):.2%}")
    st.caption(f"Using label column: `{labs[0]}` to evaluate current decisions.")
else:
    st.info("No label column found. Connect a ground-truth field (e.g., fraud_flag) to display evaluation metrics.")

# --------------- Instant decision (Apple Pay kept; no crypto) -------
st.markdown("### New order — instant decision")
st.caption("Scores a single order using the same logic. Result depends on the threshold above.")

cats  = sorted(df.get("sku_category", pd.Series(["apparel"])).dropna().unique())
ships = sorted(df.get("shipping_country", pd.Series(["US"])).dropna().unique())
ips   = sorted(df.get("ip_country",      pd.Series(["US"])).dropna().unique())

c1,c2,c3 = st.columns(3)
cat = c1.selectbox("SKU category", cats)
pay = c2.selectbox("Payment method", ["credit_card","debit_card","apple_pay","gift_card","paypal"])  # apple_pay included; no crypto
qty = c3.number_input("Quantity", 1, 99, 1)

c4,c5 = st.columns(2)
price = c4.number_input("Unit price", 1.0, 10000.0, 100.0, 1.0)
ship  = c5.selectbox("Shipping country", ships)
ip    = st.selectbox("IP country", ips)

PAY = {"gift_card":0.35,"paypal":0.20,"apple_pay":0.12,"credit_card":0.10,"debit_card":0.08}

cs = (df[["sku_category","order_amount"]].dropna()
        .groupby("sku_category")["order_amount"]
        .agg(cat_mean="mean", cat_p90=lambda s: float(np.nanpercentile(s, 90))).reset_index()
      if "order_amount" in df and "sku_category" in df else
      pd.DataFrame({"sku_category":cats or ["apparel"], "cat_mean":[500.0], "cat_p90":[1500.0]}))

def score_one(cat, pay, qty, price, ship, ip):
    row = cs[cs.sku_category==cat].iloc[:1]
    amt = qty * price
    billing_shipping_diff = (ship != ip)
    high_for_category     = bool(amt >= float(row.cat_p90))
    s = 0.08 + (0.35 if billing_shipping_diff else 0) + (0.30 if high_for_category else 0) + PAY.get(pay, 0.10)
    if billing_shipping_diff and pay in ("gift_card","paypal"): s = max(s, 0.65)
    if billing_shipping_diff and high_for_category and pay in ("gift_card","paypal"): s = max(s, 0.85)
    if high_for_category and pay in ("gift_card","paypal"): s = max(s, 0.70)
    s = min(s, 0.99)
    return s, {"amount": amt, "mean": float(row.cat_mean), "p90": float(row.cat_p90),
               "billing_shipping_diff": billing_shipping_diff, "high_for_category": high_for_category}

if st.button("Score order"):
    s, e = score_one(cat, pay, int(qty), float(price), ship, ip)
    st.markdown(f"**Decision:** {'🚩 Suspicious' if s>=TH else '✅ Cleared (not fraud)'} • Score ≈ **{s:.2f}** • Threshold {TH:.2f}")
    why = [
        "Shipping and IP countries differ" if e["billing_shipping_diff"] else "Shipping and IP countries match",
        "Order amount unusually high for this category" if e["high_for_category"] else "Order amount typical for this category",
        f"Payment method: {pay.replace('_',' ')}"
    ]
    st.write("**Signals:** " + " · ".join(why))
    st.caption(f"Order amount ≈ {e['amount']:,.2f} • Category mean ≈ {e['mean']:,.2f} • P90 ≈ {e['p90']:,.2f}")
