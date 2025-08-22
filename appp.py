# app.py — run with: streamlit run app.py
# Retail Fraud Dashboard (fixed internal threshold, no date controls, no threshold shown)

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

# ───────────────────────── Page / Altair ─────────────────────────
st.set_page_config("Retail Fraud Dashboard", layout="wide")
alt.renderers.set_embed_options(actions=False)

# ───────────────────────── Sidebar (no dates, no threshold) ─────────────────────────
with st.sidebar:
    st.header("BQ TABLE INFO")
    P  = st.text_input("Project", "mss-data-engineer-sandbox")
    D  = st.text_input("Dataset", "retail")
    PT = st.text_input("Predictions table", f"{P}.{D}.predictions_latest")
    FT = st.text_input("Features table",   f"{P}.{D}.features_signals_v4")
    st.caption("The app automatically adapts to each table’s actual schema.")

# ───────────────────────── BigQuery client ─────────────────────────
sa = dict(st.secrets["gcp_service_account"])
sa["private_key"] = sa["private_key"].replace("\\n", "\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq = bigquery.Client(credentials=creds, project=creds.project_id)

# ───────────────────────── Helpers (schema-aware & cache-safe) ─────────────────────────
def _split_table_id(table_id: str):
    parts = table_id.split(".")
    if len(parts) != 3:
        raise ValueError(f"Bad table id: {table_id} (use project.dataset.table)")
    return parts[0], parts[1], parts[2]

@st.cache_data(show_spinner=False)
def get_columns(table_id: str) -> set:
    """Return lowercase set of columns for a table (uses INFO_SCHEMA)."""
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
    """SAFE SELECT: only references columns that exist. No date filter."""
    pcols = get_columns(pred_table)
    fcols = get_columns(feat_table)

    sel = ["p.order_id"]

    if "timestamp" in pcols:
        sel.append("p.timestamp")
    else:
        sel.append("TIMESTAMP(NULL) AS timestamp")

    sel.append("CAST(p.fraud_score AS FLOAT64) AS fraud_score" if "fraud_score" in pcols
               else "CAST(0.0 AS FLOAT64) AS fraud_score")

    for c in ["customer_id","store_id","sku_id","sku_category",
              "order_amount","quantity","payment_method",
              "shipping_country","ip_country"]:
        sel.append(f"p.{c}" if c in pcols else f"CAST(NULL AS STRING) AS {c}")

    joins = ""
    strongs = [
        "strong_tri_mismatch_high_value","strong_high_value_express_geo",
        "strong_burst_multi_device","strong_price_drop_bulk","strong_giftcard_geo",
        "strong_return_whiplash","strong_price_inventory_stress","strong_country_flip_express",
        "high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag",
        "fraud_flag"
    ]
    if fcols:
        joins = f"LEFT JOIN `{feat_table}` s USING(order_id)"
        for c in strongs:
            if c in fcols:
                sel.append(f"s.{c}")
            else:
                sel.append(("CAST(NULL AS INT64) AS fraud_flag") if c == "fraud_flag"
                           else f"CAST(0 AS INT64) AS {c}")
    else:
        for c in strongs:
            sel.append(("CAST(NULL AS INT64) AS fraud_flag") if c == "fraud_flag"
                       else f"CAST(0 AS INT64) AS {c}")

    sql = f"""
      SELECT {", ".join(sel)}
      FROM `{pred_table}` p
      {joins}
      ORDER BY timestamp
    """
    d = bq.query(sql).result().to_dataframe()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce").dt.tz_localize(None)
    d["fraud_score"] = pd.to_numeric(d["fraud_score"], errors="coerce").fillna(0.0)
    return d

# ───────────────────────── Load & prepare ─────────────────────────
df = load_df(PT, FT)
if df.empty:
    st.warning("No rows available.")
    st.stop()

# Internal fixed decision threshold (NOT displayed anywhere)
TH = 0.50
df["is_alert"] = (df["fraud_score"] >= TH).astype(int)

# ───────────────────────── Header KPIs ─────────────────────────
st.title("Retail Fraud Dashboard")

k1, k2, k3, k4 = st.columns([1,1,1,2])
TOT = len(df)
AL  = int(df["is_alert"].sum())
k1.metric("TOTAL ROWS", TOT)
k2.metric("ALERTS", AL)
k3.metric("ALERT RATE", f"{(AL/TOT if TOT else 0):.2%}")
win_text = ""
if "timestamp" in df.columns and df["timestamp"].notna().any():
    win_text = f"{df['timestamp'].min()} → {df['timestamp'].max()}"
k4.caption(f"TABLE WINDOW: {win_text}")
st.markdown("---")

# ───────────────────────── Raw data (first 4,000 shown for speed) ─────────────────────────
st.subheader("Raw data (sample)")
sample = df.head(4000) if len(df) > 4000 else df
st.dataframe(sample, use_container_width=True, height=320)

# ───────────────────────── Fraud Score Distribution ─────────────────────────
st.subheader("Fraud Score Distribution")
st.altair_chart(
    alt.Chart(df).mark_bar().encode(
        x=alt.X("fraud_score:Q", bin=alt.Bin(maxbins=50), title="Fraud score"),
        y=alt.Y("count():Q", title="Rows"),
        tooltip=[alt.Tooltip("count()", title="Rows")]
    ).properties(height=220),
    use_container_width=True
)

# ───────────────────────── Strong signal prevalence ─────────────────────────
st.subheader("Strong Signal Prevalence")

def prevalence(cols, title):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        st.info(f"No columns found for: {title}")
        return
    z = df[["is_alert"] + cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    a = int(z["is_alert"].sum()) or 1
    na = int((1 - z["is_alert"]).sum()) or 1
    rows = [{"signal": c,
             "% in alerts": z.loc[z.is_alert == 1, c].sum() / a,
             "% in non-alerts": z.loc[z.is_alert == 0, c].sum() / na}
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

c1, c2 = st.columns(2)
with c1:
    prevalence(
        [
            "strong_tri_mismatch_high_value","strong_high_value_express_geo",
            "strong_burst_multi_device","strong_price_drop_bulk","strong_giftcard_geo",
            "strong_return_whiplash","strong_price_inventory_stress","strong_country_flip_express",
        ],
        "Fraud Pattern Flags"
    )
with c2:
    prevalence(
        ["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"],
        "Pricing & Inventory Context"
    )

# ───────────────────────── Alerts (ALL rows) ─────────────────────────
st.subheader("Alerts (all)")
cols_alerts = [c for c in [
    "order_id","timestamp","customer_id","store_id","sku_id","sku_category",
    "order_amount","quantity","payment_method","shipping_country","ip_country","fraud_score"
] if c in df.columns]
alerts = df[df["is_alert"] == 1].sort_values(["fraud_score","timestamp"], ascending=[False, False]).loc[:, cols_alerts]
h = min(720, max(240, 28 * (len(alerts) if len(alerts) < 20 else 20)))
st.dataframe(alerts, use_container_width=True, height=h)

# ───────────────────────── Model Evaluation ─────────────────────────
# ───────────────────────── Model Evaluation ─────────────────────────
st.subheader("Model Evaluation")

# Prefer a real ground-truth column if present; otherwise fall back to decisions
label_candidates = [c for c in ["fraud_flag","is_fraud","label","ground_truth","gt","y"] if c in df.columns]
if label_candidates:
    lab = st.selectbox("Ground truth (1=fraud, 0=legit)", label_candidates, index=0)
    y_true = df[lab].fillna(0).astype(int).values
    st.caption(f"Using label column: `{lab}`")
else:
    st.warning("No label column found; using current decisions as a proxy.")
    y_true = df["is_alert"].values

y_pred  = df["is_alert"].values
y_score = df["fraud_score"].values  # kept for possible future diagnostics

# Headline KPIs
c_acc, c_prec, c_rec, c_f1 = st.columns(4)
c_acc.metric("Accuracy",  f"{accuracy_score(y_true, y_pred):.2%}")
c_prec.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.2%}")
c_rec.metric("Recall",    f"{recall_score(y_true, y_pred, zero_division=0):.2%}")
c_f1.metric("F1-score",   f"{f1_score(y_true, y_pred, zero_division=0):.2%}")

# Short, plain-English explanations (business-facing)
st.caption(
    "• **Accuracy**: share of all orders where the decision matched the label.  "
    "• **Precision**: among the orders we flagged, how many were truly fraud (cleanliness of alerts).  "
    "• **Recall**: share of all true fraud that we actually caught (miss-rate complement).  "
    "• **F1**: single score balancing precision and recall when classes are imbalanced."
)

# Confusion matrix heatmap (kept)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
cm_long = (
    pd.DataFrame(cm, index=["Actual: 0 (legit)", "Actual: 1 (fraud)"], columns=["Pred: 0", "Pred: 1"])
      .reset_index()
      .melt(id_vars="index", var_name="Predicted", value_name="Count")
      .rename(columns={"index": "Actual"})
)

st.altair_chart(
    alt.Chart(cm_long).mark_rect().encode(
        x="Predicted:N",
        y="Actual:N",
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["Actual", "Predicted", "Count"]
    ).properties(height=180),
    use_container_width=True
)

# Note: intentionally removed ROC/AUC and Precision–Recall curves for a cleaner, simpler panel.

# ───────────────────────── New Order — Instant Decision ─────────────────────────
st.markdown("## New Order — Instant Decision")

# Short, clear feature help (bold names, one short sentence each)
st.markdown("""
- **Geo mismatch** — shipping country and IP country are different.
- **High amount for category** — order amount is in the top 10% of amounts for that SKU category.
- **Payment channel risk** — gift card and PayPal are riskier; Apple Pay moderate; card/debit are standard risk.
""")

# UI choices (taken from the data where possible, with safe fallbacks)
cats = sorted([c for c in df.get("sku_category", pd.Series(["apparel"])).dropna().unique()])
ships = sorted([c for c in df.get("shipping_country", pd.Series(["US"])).dropna().unique()])
ips   = sorted([c for c in df.get("ip_country", pd.Series(["US"])).dropna().unique()])

c1, c2, c3 = st.columns(3)
sku_category  = c1.selectbox("SKU category", options=cats, index=0)
payment_method = c2.selectbox(
    "Payment method",
    options=["credit_card", "debit_card", "gift_card", "paypal", "apple_pay"],
    index=0
)
quantity = c3.number_input("Quantity", min_value=1, max_value=99, value=1, step=1)

c4, c5 = st.columns(2)
unit_price = c4.number_input("Unit price", min_value=1.0, max_value=10_000.0, value=100.0, step=1.0)
ship_country = c5.selectbox("Shipping country", options=ships, index=0)

ip_country = st.selectbox("IP country", options=ips, index=0)

# Build category statistics for “high amount” detection
if "order_amount" in df.columns and "sku_category" in df.columns and df["order_amount"].notna().any():
    cat_stats = (
        df[["sku_category", "order_amount"]]
        .dropna()
        .groupby("sku_category")["order_amount"]
        .agg(cat_mean="mean", cat_p90=lambda s: float(np.nanpercentile(s, 90)))
        .reset_index()
    )
else:
    # fallbacks
    cat_stats = pd.DataFrame({"sku_category": cats or ["apparel"],
                              "cat_mean": [500.0],
                              "cat_p90": [1500.0]})

# Payment method risk weights (tuned for business clarity)
PAY_WEIGHTS = {
    "gift_card": 0.35,
    "paypal": 0.20,
    "apple_pay": 0.12,
    "credit_card": 0.10,
    "debit_card": 0.08,
}

def score_new_order(cat: str, pay: str, qty: int, price: float, ship: str, ip: str):
    amt = qty * price

    row = cat_stats[cat_stats["sku_category"] == cat]
    if row.empty:
        row = cat_stats.iloc[:1]
    cat_mean = float(row["cat_mean"].values[0])
    cat_p90  = float(row["cat_p90"].values[0])

    geo_mismatch = (ship != ip)
    high_amount  = bool(amt >= cat_p90)
    pay_weight   = PAY_WEIGHTS.get(pay, 0.10)

    # Weighted risk with synergy bumps.
    base = 0.08
    score = base
    if geo_mismatch:
        score += 0.35
    if high_amount:
        score += 0.30
    score += pay_weight

    # Synergy escalations for realistic outcomes:
    if geo_mismatch and pay in ("gift_card", "paypal"):
        score = max(score, 0.65)
    if geo_mismatch and high_amount and pay in ("gift_card", "paypal"):
        score = max(score, 0.85)
    if high_amount and pay in ("gift_card", "paypal"):
        score = max(score, 0.70)

    return min(score, 0.99), {
        "amount": amt,
        "cat_mean": cat_mean,
        "cat_p90": cat_p90,
        "geo_mismatch": geo_mismatch,
        "high_amount": high_amount,
        "payment": pay,
    }

if st.button("Score order"):
    score, expl = score_new_order(
        sku_category, payment_method, int(quantity), float(unit_price),
        ship_country, ip_country
    )
    decision = "Fraud" if score >= TH else "Not fraud"

    why = []
    if expl["geo_mismatch"]:
        why.append("Shipping country and IP country are different.")
    else:
        why.append("Shipping country and IP country match.")

    if expl["high_amount"]:
        why.append("Order amount is high for this category (top 10%).")
    else:
        why.append("Order amount is typical for this category.")

    if payment_method in ("gift_card", "paypal"):
        why.append(f"Payment channel is {payment_method.replace('_',' ')} (higher risk).")
    elif payment_method == "apple_pay":
        why.append("Payment channel is Apple Pay (moderate risk).")
    else:
        why.append(f"Payment channel is {payment_method.replace('_',' ')} (standard risk).")

    st.markdown(f"### Decision: **{decision}** · Score ≈ **{score:.2f}**")
    st.markdown("**Why:**")
    st.markdown("\n".join([f"- {w}" for w in why]))
    st.caption(
        f"Order amount ≈ {expl['amount']:,.2f} · "
        f"Category mean ≈ {expl['cat_mean']:,.2f} · "
        f"P90 amount ≈ {expl['cat_p90']:,.2f} · "
        f"Geo mismatch: {'Yes' if expl['geo_mismatch'] else 'No'}."
    )

