# app.py — run with: streamlit run app.py

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

# ───────────────────────────────────────────────────────── Config / constants
st.set_page_config("Retail Fraud Dashboard", layout="wide")
alt.renderers.set_embed_options(actions=False)

# ───────────────────────────────────────────────────────── Sidebar (no slider)
with st.sidebar:
    st.header("BQ TABLE INFO")
    P  = st.text_input("Project", "mss-data-engineer-sandbox")
    D  = st.text_input("Dataset", "retail")
    PT = st.text_input("Predictions table", f"{P}.{D}.predictions_latest")
    FT = st.text_input("Features table",   f"{P}.{D}.features_signals_v4")

# ───────────────────────────────────────────────────────── BigQuery client
sa = dict(st.secrets["gcp_service_account"])
sa["private_key"] = sa["private_key"].replace("\\n", "\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq = bigquery.Client(credentials=creds, project=creds.project_id)

# ───────────────────────────────────────────────────────── Helpers (schema-aware)
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
    d["timestamp"]   = pd.to_datetime(d["timestamp"], errors="coerce").dt.tz_localize(None)
    d["fraud_score"] = pd.to_numeric(d["fraud_score"], errors="coerce").fillna(0.0)
    if {"shipping_country","ip_country"}.issubset(d.columns):
        d["geo_mismatch"] = (d["shipping_country"] != d["ip_country"]).astype(int)
    else:
        d["geo_mismatch"] = 0
    return d

# ───────────────────────────────────────────────────────── Load & prepare
df = load_df(PT, FT)
if df.empty:
    st.warning("No rows available.")
    st.stop()

df["is_alert"] = (df["fraud_score"] >= TH).astype(int)

# ───────────────────────────────────────────────────────── Header KPIs
st.title("Retail Fraud Dashboard")

k1, k2, k3, k4 = st.columns([1,1,1,2])
TOT = len(df); AL = int(df["is_alert"].sum())
k1.metric("TOTAL ROWS", TOT)
k2.metric("ALERTS (≥ 0.50)", AL)
k3.metric("ALERT RATE", f"{(AL/TOT if TOT else 0):.2%}")
win_text = f"{df['timestamp'].min()} → {df['timestamp'].max()}" if df["timestamp"].notna().any() else ""
k4.caption(f"TABLE WINDOW: {win_text}  |  THRESHOLD = 0.50 (fixed)")
st.markdown("---")

# ───────────────────────────────────────────────────────── STRONG FEATURES (bold + meaning)
st.subheader("STRONG FEATURES USED")
st.markdown("""
- **strong_tri_mismatch_high_value** — high-value order **and** shipping vs IP country mismatch  
- **strong_high_value_express_geo** — high-value + express/geo risk pattern  
- **strong_burst_multi_device** — **rapid bursts** of orders across devices  
- **strong_price_drop_bulk** — bulk quantity with **large price drop**  
- **strong_giftcard_geo** — gift card usage with geo risk  
- **strong_return_whiplash** — spiky returns pattern  
- **strong_price_inventory_stress** — price + inventory stress combined signal  
- **strong_country_flip_express** — sudden **country flip** with express behavior  
- **high_price_anomaly / low_price_anomaly** — unusual price vs norm  
- **oversell_flag / stockout_risk_flag / hoarding_flag** — inventory risk context  
- **geo_mismatch** — (derived) shipping vs IP country mismatch
""")

# ───────────────────────────────────────────────────────── Score distribution
st.subheader("Fraud Score Distribution (red line at 0.50)")
hist = alt.Chart(df).mark_bar().encode(
    x=alt.X("fraud_score:Q", bin=alt.Bin(maxbins=50), title="Fraud score"),
    y=alt.Y("count():Q", title="Rows"),
    tooltip=[alt.Tooltip("count()", title="Rows")]
).properties(height=220)
rule = alt.Chart(pd.DataFrame({"x":[TH]})).mark_rule(color="crimson").encode(x="x")
st.altair_chart(hist + rule, use_container_width=True)

# ───────────────────────────────────────────────────────── Strong-signal prevalence
st.subheader("Strong Signal Prevalence by Alerts")

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
        ["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag","geo_mismatch"],
        "Pricing / Inventory / Geo Context"
    )

# ───────────────────────────────────────────────────────── ALL alerts (download)
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
st.caption(f"Showing **{len(alerts)}** alert(s).")
st.dataframe(alerts.loc[:, cols_show], use_container_width=True, height=min(700, 28*max(8, len(alerts))))

csv = alerts.loc[:, cols_show].to_csv(index=False).encode("utf-8")
st.download_button("Download alerts (.csv)", csv, file_name="alerts.csv")

# ───────────────────────────────────────────────────────── Model Evaluation
st.subheader("Model Evaluation")
label_candidates = [c for c in ["fraud_flag","is_fraud","label","ground_truth","gt","y"] if c in df.columns]
if label_candidates:
    lab = st.selectbox("Ground truth (1=fraud, 0=legit)", label_candidates, index=0)
    y_true = df[lab].fillna(0).astype(int).values
    st.caption(f"Using label column: `{lab}`")
else:
    st.warning("No label column found; using alert decision as a proxy for demo.")
    y_true = df["is_alert"].values

y_score = df["fraud_score"].values

def best_threshold_f1(y_true, scores):
    ts = np.linspace(0.01, 0.99, 99)
    best_t, best_f = 0.5, -1
    for t in ts:
        y_hat = (scores >= t).astype(int)
        f = f1_score(y_true, y_hat, zero_division=0)
        if f > best_f: best_f, best_t = f, t
    return float(best_t), float(best_f)

t_star, f1_star = best_threshold_f1(y_true, y_score)
st.caption(f"Recommended threshold for **max F1** (for reference): **{t_star:.2f}** (F1 ≈ {f1_star:.2%}).**.")

y_pred = (y_score >= TH).astype(int)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy",  f"{accuracy_score(y_true, y_pred):.2%}")
m2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.2%}")
m3.metric("Recall",    f"{recall_score(y_true, y_pred, zero_division=0):.2%}")
m4.metric("F1-score",  f"{f1_score(y_true, y_pred, zero_division=0):.2%}")

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

# ───────────────────────────────────────────────────────── New Order — Instant Decision
st.subheader("New Order — Instant Decision (TH = 0.50)")

# baselines for auto-feature logic
p90_amt = float(np.nanpercentile(df["order_amount"], 90)) if "order_amount" in df else 200.0
cat_means = df.groupby("sku_category")["order_amount"].mean().to_dict() if "order_amount" in df and "sku_category" in df else {}

with st.form("new_order"):
    c1, c2, c3 = st.columns(3)
    sku_cat = c1.selectbox("SKU category", sorted(df["sku_category"].dropna().unique()) if "sku_category" in df else ["electronics","grocery","apparel","home","toys"])
    paym    = c2.selectbox("Payment method", sorted(df["payment_method"].dropna().unique()) if "payment_method" in df else ["card","credit_card","paypal","apple_pay","gift_card"])
    qty     = c3.number_input("Quantity", min_value=1, max_value=20, value=1)

    c4, c5, c6 = st.columns(3)
    price   = c4.number_input("Unit price", min_value=0.0, value=100.0, step=1.0)
    ship_c  = c5.selectbox("Shipping country", sorted(df["shipping_country"].dropna().unique()) if "shipping_country" in df else ["US","CA","UK","DE","IN"])
    ip_c    = c6.selectbox("IP country", sorted(df["ip_country"].dropna().unique()) if "ip_country" in df else ["US","CA","UK","DE","IN"])

    # Optional advanced overrides (kept hidden unless needed)
    with st.expander("Advanced (optional): manually flag observed strong signals"):
        adv_cols = ["strong_burst_multi_device","strong_giftcard_geo","strong_return_whiplash",
                    "strong_price_inventory_stress","strong_country_flip_express"]
        adv = {c: st.checkbox(c.replace("_"," "), False) for c in adv_cols}

    submitted = st.form_submit_button("Score order")

if submitted:
    order_amount = qty * price
    # auto-derived signals
    geo_mismatch = int(ship_c != ip_c)
    tri_mismatch_high_value = int(geo_mismatch == 1 and order_amount >= p90_amt)

    # category-based price anomaly (if we have baseline)
    cat_mean = cat_means.get(sku_cat, max(1.0, np.nanmean(list(cat_means.values())) if cat_means else 100.0))
    high_price_anomaly = int(order_amount >= 1.5 * cat_mean)
    low_price_anomaly  = int(order_amount <= 0.5 * cat_mean)

    # price drop bulk proxy
    price_drop_bulk = int(qty >= 3 and low_price_anomaly == 1)

    # compile all signals
    signals = {
        "geo_mismatch": geo_mismatch,
        "tri_mismatch_high_value": tri_mismatch_high_value,
        "high_price_anomaly": high_price_anomaly,
        "low_price_anomaly": low_price_anomaly,
        "price_drop_bulk": price_drop_bulk,
        # optional manual flags from expander
        "burst_multi_device": int('adv' in locals() and adv.get("strong_burst_multi_device", False)),
        "giftcard_geo": int('adv' in locals() and adv.get("strong_giftcard_geo", False)),
        "return_whiplash": int('adv' in locals() and adv.get("strong_return_whiplash", False)),
        "price_inventory_stress": int('adv' in locals() and adv.get("strong_price_inventory_stress", False)),
        "country_flip_express": int('adv' in locals() and adv.get("strong_country_flip_express", False)),
    }

    # simple transparent risk score based on signals
    fired = [k for k,v in signals.items() if v == 1]
    score = min(1.0, 0.10*signals["geo_mismatch"] +
                      0.20*signals["tri_mismatch_high_value"] +
                      0.15*signals["high_price_anomaly"] +
                      0.15*signals["low_price_anomaly"] +
                      0.15*signals["price_drop_bulk"] +
                      0.25*int(any(signals[k] for k in ["burst_multi_device","giftcard_geo","return_whiplash",
                                                        "price_inventory_stress","country_flip_express"])))
    decision = "Fraud" if score >= TH else "Not fraud"

    st.markdown(f"### Decision: **{decision}**  ·  Score ≈ **{score:.2f}**  (TH = 0.50)")
    st.caption(
        f"Signals fired: {', '.join(fired) if fired else 'none'}  ·  "
        f"Order amount: {order_amount:,.2f}  ·  "
        f"Geo mismatch: {bool(geo_mismatch)}  ·  "
        f"Category mean ≈ {cat_mean:,.2f}  ·  P90 amount ≈ {p90_amt:,.0f}"
    )

