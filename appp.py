# app.py — Retail Fraud Dashboard (fixed internal threshold, never shown)
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, auc
)

# ───────────────────────────── Page / Altair
st.set_page_config("Retail Fraud Dashboard", layout="wide")
alt.renderers.set_embed_options(actions=False)

# Fixed decision threshold (kept internal only; never rendered)
TH = 0.50

# ───────────────────────────── Sidebar (no date, no threshold controls)
with st.sidebar:
    st.header("BQ TABLE INFO")
    P  = st.text_input("Project", "mss-data-engineer-sandbox")
    D  = st.text_input("Dataset", "retail")
    PT = st.text_input("Predictions table", f"{P}.{D}.predictions_latest")
    FT = st.text_input("Features table",   f"{P}.{D}.features_signals_v4")

# ───────────────────────────── BigQuery client
sa = dict(st.secrets["gcp_service_account"])
sa["private_key"] = sa["private_key"].replace("\\n", "\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq = bigquery.Client(credentials=creds, project=creds.project_id)

# ───────────────────────────── Helpers (schema-aware)
def _split_table_id(table_id: str):
    parts = table_id.split(".")
    if len(parts) != 3:
        raise ValueError(f"Bad table id: {table_id} (use project.dataset.table)")
    return parts[0], parts[1], parts[2]

@st.cache_data(show_spinner=False)
def get_columns(table_id: str) -> set:
    proj, dset, tbl = _split_table_id(table_id)
    sql = f"""
      SELECT column_name
      FROM `{proj}.{dset}.INFORMATION_SCHEMA.COLUMNS`
      WHERE table_name = @T
    """
    job = bq.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("T", "STRING", tbl)]
        ),
    )
    return set(job.result().to_dataframe()["column_name"].str.lower().tolist())

@st.cache_data(show_spinner=True)
def load_df(pred_table: str, feat_table: str) -> pd.DataFrame:
    pcols = get_columns(pred_table)
    fcols = get_columns(feat_table)

    sel = ["p.order_id"]
    sel.append("p.timestamp" if "timestamp" in pcols else "TIMESTAMP(NULL) AS timestamp")
    sel.append("CAST(p.fraud_score AS FLOAT64) AS fraud_score" if "fraud_score" in pcols
               else "CAST(0.0 AS FLOAT64) AS fraud_score")

    info_cols = ["customer_id","store_id","sku_id","sku_category",
                 "order_amount","quantity","payment_method",
                 "shipping_country","ip_country"]
    for c in info_cols:
        sel.append(f"p.{c}" if c in pcols else f"CAST(NULL AS STRING) AS {c}")

    strongs = [
        "strong_tri_mismatch_high_value","strong_high_value_express_geo",
        "strong_burst_multi_device","strong_price_drop_bulk","strong_giftcard_geo",
        "strong_return_whiplash","strong_price_inventory_stress","strong_country_flip_express",
        "high_price_anomaly","low_price_anomaly",
        "oversell_flag","stockout_risk_flag","hoarding_flag",
        "fraud_flag"
    ]

    joins = ""
    if get_columns(feat_table):
        joins = f"LEFT JOIN `{feat_table}` s USING(order_id)"
        for c in strongs:
            sel.append(f"s.{c}" if c in get_columns(feat_table) else
                       ("CAST(NULL AS INT64) AS fraud_flag" if c=="fraud_flag" else f"CAST(0 AS INT64) AS {c}"))
    else:
        for c in strongs:
            sel.append(("CAST(NULL AS INT64) AS fraud_flag" if c=="fraud_flag" else f"CAST(0 AS INT64) AS {c}"))

    sql = f"""
      SELECT {", ".join(sel)}
      FROM `{pred_table}` p
      {joins}
      ORDER BY timestamp
    """
    d = bq.query(sql).result().to_dataframe()
    d["timestamp"]   = pd.to_datetime(d["timestamp"], errors="coerce").dt.tz_localize(None)
    d["fraud_score"] = pd.to_numeric(d["fraud_score"], errors="coerce").fillna(0.0)

    # simple geo mismatch derived feature for prevalence & new-order baseline
    if {"shipping_country","ip_country"}.issubset(d.columns):
        d["geo_mismatch"] = (d["shipping_country"].astype(str) != d["ip_country"].astype(str)).astype(int)
    else:
        d["geo_mismatch"] = 0

    return d

# ───────────────────────────── Load & prepare
df = load_df(PT, FT)
if df.empty:
    st.warning("No rows available.")
    st.stop()

df["is_alert"] = (df["fraud_score"] >= TH).astype(int)

# ───────────────────────────── Header KPIs
st.title("Retail Fraud Dashboard")

k1, k2, k3, k4 = st.columns([1,1,1,2])
TOT = len(df); AL = int(df["is_alert"].sum())
k1.metric("TOTAL ROWS", TOT)
k2.metric("ALERTS", AL)
k3.metric("ALERT RATE", f"{(AL/TOT if TOT else 0):.2%}")
win_text = f"{df['timestamp'].min()} → {df['timestamp'].max()}" if df["timestamp"].notna().any() else ""
k4.caption(f"TABLE WINDOW: {win_text}")
st.markdown("---")

# ───────────────────────────── Strong features (plain English, one sentence each)
st.subheader("STRONG FEATURES (what they mean)")
st.markdown("""
- **strong_tri_mismatch_high_value** — the order is in the top spend band and the shipping country does not match the IP country.  
- **strong_high_value_express_geo** — top spend combined with a geo pattern seen in fast-moving fraud runs.  
- **strong_burst_multi_device** — many orders in a short time from multiple devices tied to the same identity.  
- **strong_price_drop_bulk** — high quantity while the unit price is much lower than the category’s typical level.  
- **strong_giftcard_geo** — gift-card usage together with a location pattern historically linked to abuse.  
- **strong_return_whiplash** — unusual spike in returns following purchases.  
- **strong_price_inventory_stress** — purchase timing and price behavior aligned with inventory stress.  
- **strong_country_flip_express** — recent country change combined with rapid ordering behavior.  
- **high_price_anomaly** — total amount is unusually high versus the category baseline.  
- **low_price_anomaly** — total amount is unusually low versus the category baseline.  
- **oversell_flag / stockout_risk_flag / hoarding_flag** — buying pattern that adds inventory-risk context.  
- **geo_mismatch** — shipping country and IP country are different on the order.
""")

# ───────────────────────────── Fraud score distribution
st.subheader("Fraud Score Distribution")
st.altair_chart(
    alt.Chart(df).mark_bar().encode(
        x=alt.X("fraud_score:Q", bin=alt.Bin(maxbins=50), title="Fraud score"),
        y=alt.Y("count():Q", title="Rows"),
        tooltip=[alt.Tooltip("count()", title="Rows")]
    ).properties(height=220),
    use_container_width=True
)

# ───────────────────────────── Strong-signal prevalence
st.subheader("Strong Signal Prevalence in Alerts vs Non-alerts")

def prevalence(cols, title):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        st.info(f"No columns found for: {title}")
        return
    z  = df[["is_alert"] + cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    a  = int(z["is_alert"].sum()) or 1
    na = int((1 - z["is_alert"]).sum()) or 1
    rows = [{"signal": c,
             "% in alerts":     z.loc[z.is_alert == 1, c].sum()/a,
             "% in non-alerts": z.loc[z.is_alert == 0, c].sum()/na}
            for c in cols]
    dd = (pd.DataFrame(rows)
          .sort_values("% in alerts", ascending=False)
          .melt(id_vars="signal", var_name="group", value_name="value"))
    chart = alt.Chart(dd).mark_bar().encode(
        x=alt.X("value:Q", axis=alt.Axis(format="%"), title="Prevalence"),
        y=alt.Y("signal:N", sort="-x", title=None),
        color="group:N",
        tooltip=["signal:N", alt.Tooltip("value:Q", format=".1%"), "group:N"]
    ).properties(height=300, title=title)
    st.altair_chart(chart, use_container_width=True)

left, right = st.columns(2)
with left:
    prevalence(
        [
            "strong_tri_mismatch_high_value","strong_high_value_express_geo",
            "strong_burst_multi_device","strong_price_drop_bulk","strong_giftcard_geo",
            "strong_return_whiplash","strong_price_inventory_stress","strong_country_flip_express",
        ],
        "Fraud pattern flags"
    )
with right:
    prevalence(
        ["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag","geo_mismatch"],
        "Pricing / Inventory / Geo context"
    )

# ───────────────────────────── Alerts table (entire set)
st.subheader("All Alerts")
cols_show = [c for c in [
    "order_id","timestamp","customer_id","store_id","sku_id","sku_category",
    "order_amount","quantity","payment_method","shipping_country","ip_country","fraud_score",
    "strong_tri_mismatch_high_value","strong_high_value_express_geo","strong_burst_multi_device",
    "strong_price_drop_bulk","strong_giftcard_geo","strong_return_whiplash",
    "strong_price_inventory_stress","strong_country_flip_express",
    "high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag",
    "geo_mismatch"
] if c in df.columns]
alerts = df[df["is_alert"] == 1].sort_values(["fraud_score","timestamp"], ascending=[False, False])
st.caption(f"Showing **{len(alerts)}** alert(s). Download below.")
st.dataframe(alerts.loc[:, cols_show], use_container_width=True, height=min(700, 28*max(8, len(alerts))))
st.download_button("Download alerts (.csv)", alerts.loc[:, cols_show].to_csv(index=False).encode("utf-8"),
                   file_name="alerts.csv")

# ───────────────────────────── Model evaluation (uses label if available)
st.subheader("Model Evaluation")
label_candidates = [c for c in ["fraud_flag","is_fraud","label","ground_truth","gt","y"] if c in df.columns]
if label_candidates:
    lab = st.selectbox("Ground truth (1=fraud, 0=legit)", label_candidates, index=0)
    y_true = df[lab].fillna(0).astype(int).values
    st.caption(f"Using label column: `{lab}`")
else:
    st.warning("No label column found; using alert decision as a proxy for demo.")
    y_true = df["is_alert"].values

y_pred  = df["is_alert"].values
y_score = df["fraud_score"].values

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy",  f"{accuracy_score(y_true, y_pred):.2%}")
m2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.2%}")
m3.metric("Recall",    f"{recall_score(y_true, y_pred, zero_division=0):.2%}")
m4.metric("F1-score",  f"{f1_score(y_true, y_pred, zero_division=0):.2%}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0,1])
cm_long = (pd.DataFrame(cm, index=["Actual: 0","Actual: 1"], columns=["Pred: 0","Pred: 1"])
           .reset_index().melt(id_vars="index", var_name="Predicted", value_name="Count")
           .rename(columns={"index":"Actual"}))
st.altair_chart(
    alt.Chart(cm_long).mark_rect().encode(
        x="Predicted:N", y="Actual:N",
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["Actual","Predicted","Count"]
    ).properties(height=180),
    use_container_width=True
)

# ROC and PR curves (based on score)
try:
    auc_roc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    st.altair_chart(
        alt.Chart(pd.DataFrame({"fpr": fpr, "tpr": tpr}))
        .mark_line().encode(x=alt.X("fpr:Q", title="False Positive Rate"),
                            y=alt.Y("tpr:Q", title="True Positive Rate"))
        .properties(height=220, title=f"ROC (AUC = {auc_roc:.3f})"),
        use_container_width=True
    )
except Exception:
    pass

prec, rec, _ = precision_recall_curve(y_true, y_score)
ap = auc(rec, prec)
st.altair_chart(
    alt.Chart(pd.DataFrame({"recall": rec, "precision": prec}))
    .mark_line().encode(x="recall:Q", y="precision:Q")
    .properties(height=220, title=f"Precision–Recall (AP ≈ {ap:.3f})"),
    use_container_width=True
)

# ───────────────────────────── New Order — Instant Decision
st.markdown("---")
st.header("New Order — Instant Decision")

# Payment methods allowed for client demo
allowed_methods = ["credit_card", "debit_card", "gift_card", "paypal", "apple_pay"]

# Controls
c1, c2, c3 = st.columns(3)
cat  = c1.selectbox("SKU category", sorted(df["sku_category"].dropna().astype(str).unique().tolist() or ["electronics"]))
pay  = c2.selectbox("Payment method", allowed_methods)
qty  = int(c3.number_input("Quantity", min_value=1, max_value=50, value=1, step=1))

c4, c5, c6 = st.columns(3)
price = float(c4.number_input("Unit price", min_value=0.01, max_value=10000.0, value=100.0, step=1.0))
ship  = c5.selectbox("Shipping country", sorted(df["shipping_country"].dropna().astype(str).unique().tolist() or ["US"]))
ip    = c6.selectbox("IP country", sorted(df["ip_country"].dropna().astype(str).unique().tolist() or ["US"]))

# Category baselines
cat_df = df[df["sku_category"].astype(str)==str(cat)]
cat_mean = float(cat_df["order_amount"].fillna(0).mean() if "order_amount" in cat_df else 0)
cat_p90  = float(np.nanpercentile(cat_df["order_amount"].dropna(), 90)) if ("order_amount" in cat_df and cat_df["order_amount"].notna().any()) else 0

# Heuristic scorer (transparent signals → score in [0,1])
def score_new_order(cat, qty, price, ship, ip, pay, cat_mean, cat_p90):
    amount = qty * price
    signals = []

    # Geo mismatch
    if ship != ip:
        signals.append(("geo_mismatch", 0.25, "Shipping country and IP country are different."))

    # High spend vs category p90
    if cat_p90 and amount >= cat_p90:
        signals.append(("high_spend_vs_category", 0.20, "Order amount is in the top spend band for this category."))

    # Bulk while cheap vs category
    unit_price_ratio = (price / (cat_mean/ max(1, qty))) if (cat_mean and qty) else (price/100.0)
    # Safer: compare unit price to category mean per unit if order_amount and qty exist
    cat_unit_mean = (cat_mean / max(1, qty)) if cat_mean else 0
    if cat_unit_mean and (price < 0.7*cat_unit_mean) and qty >= 3:
        signals.append(("bulk_cheap", 0.20, "Large quantity while unit price is well below typical category price."))

    # Payment risk contribution
    pay_weight = {"gift_card":0.20, "credit_card":0.12, "debit_card":0.10, "paypal":0.10, "apple_pay":0.05}
    if pay in pay_weight:
        signals.append((f"payment_{pay}", pay_weight[pay], f"Payment channel is {pay.replace('_',' ')}."))

    # High or low amount vs category mean
    if cat_mean:
        if amount >= 1.8*cat_mean:
            signals.append(("high_price_anomaly", 0.15, "Total amount is much higher than the category baseline."))
        elif amount <= 0.4*cat_mean:
            signals.append(("low_price_anomaly", 0.10, "Total amount is much lower than the category baseline."))

    # Aggregate
    score = max(0.0, min(1.0, sum(w for _, w, _ in signals)))
    return score, amount, signals

risk_score, amount, reason_signals = score_new_order(
    cat=cat, qty=qty, price=price, ship=ship, ip=ip, pay=pay, cat_mean=cat_mean, cat_p90=cat_p90
)
decision = "Fraud" if risk_score >= TH else "Not fraud"

if st.button("Score order"):
    st.markdown(f"### Decision: **{decision}** · Score ≈ **{risk_score:.2f}**")
    # Plain-English reasons
    if reason_signals:
        bullets = [f"- {desc}" for (_, _, desc) in reason_signals]
        st.markdown("**Why:**\n" + "\n".join(bullets))
    else:
        st.markdown("**Why:** Pattern looks normal compared to category norms and geo/payment context.")

    # Helpful context line (no threshold revealed)
    st.caption(
        f"Order amount ≈ {amount:,.2f} · Category mean ≈ {cat_mean:,.2f} · P90 amount ≈ {cat_p90:,.2f} "
        f"· Geo mismatch: {'Yes' if ship!=ip else 'No'}."
    )
