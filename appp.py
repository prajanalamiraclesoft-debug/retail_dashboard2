# app.py — streamlit run app.py
import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import date
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

# ─────────────────────────── PAGE / ALTAIR ───────────────────────────
st.set_page_config("Retail Dashboard: Fraud & Inventory", layout="wide")
alt.data_transformers.enable("default", max_rows=None)
alt.renderers.set_embed_options(actions=False)

# ───────────────────── INTERNAL OPERATING THRESHOLD ──────────────────
# ALL DECISIONS AND METRICS USE THIS VALUE (NOT THE SLIDER).
TH_INTERNAL = 0.02

# ─────────────────────────── SIDEBAR (BQ) ────────────────────────────
with st.sidebar:
    st.header("BQ TABLE INFO")
    P  = st.text_input("Project", "mss-data-engineer-sandbox")
    D  = st.text_input("Dataset", "retail")
    PT = st.text_input("Predictions", f"{P}.{D}.predictions_latest")
    FT = st.text_input("Features",    f"{P}.{D}.features_signals_v4")
    MT = st.text_input("Metrics (optional)", f"{P}.{D}.predictions_daily_metrics")
    S  = st.date_input("Start", date(2023, 1, 1))
    E  = st.date_input("End",   date(2024,12,31))

    # WHAT-IF slider: used only for charts (distribution/ops helper)
    TH_VIEW = st.slider("WHAT-IF THRESHOLD ", 0.00, 1.00, 0.30, 0.01)
# ───────────────────────── BIGQUERY CLIENT ───────────────────────────
sa = dict(st.secrets["gcp_service_account"])
sa["private_key"] = sa["private_key"].replace("\\n", "\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq    = bigquery.Client(credentials=creds, project=creds.project_id)

# ───────────────────────── DATA LOADER (BQ) ──────────────────────────
@st.cache_data(show_spinner=True)
def load_df(PT, FT, S, E):
    sql = f"""
    SELECT
      p.order_id, p.timestamp, p.customer_id, p.store_id, p.sku_id, p.sku_category,
      p.order_amount, p.quantity, p.payment_method, p.shipping_country, p.ip_country,
      CAST(p.fraud_score AS FLOAT64) AS fraud_score,

      -- STRONG SIGNALS (ALREADY ENGINEERED IN YOUR FEATURES TABLE)
      s.strong_tri_mismatch_high_value,
      s.strong_high_value_express_geo,
      s.strong_burst_multi_device,
      s.strong_price_drop_bulk,
      s.strong_giftcard_geo,
      s.strong_return_whiplash,
      s.strong_price_inventory_stress,
      s.strong_country_flip_express,

      -- PRICING / INVENTORY CONTEXT
      s.high_price_anomaly,
      s.low_price_anomaly,
      s.oversell_flag,
      s.stockout_risk_flag,
      s.hoarding_flag,

      SAFE_CAST(s.fraud_flag AS INT64) AS fraud_flag
    FROM `{PT}` p
    LEFT JOIN `{FT}` s USING(order_id)
    WHERE DATE(p.timestamp) BETWEEN @S AND @E
    ORDER BY p.timestamp
    """
    job = bq.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("S", "DATE", str(S)),
                bigquery.ScalarQueryParameter("E", "DATE", str(E)),
            ]
        ),
    )
    d = job.result().to_dataframe()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce").dt.tz_localize(None)
    return d

df = load_df(PT, FT, S, E)
if df.empty:
    st.warning("NO ROWS IN THIS WINDOW.")
    st.stop()

# ───────────────────────── BASIC PREP / LABELS ───────────────────────
df["fraud_score"] = pd.to_numeric(df["fraud_score"], errors="coerce").fillna(0.0)
df["day"]         = df["timestamp"].dt.floor("D")

# DECISION FLAG FOR KPI/EVAL USES INTERNAL THRESHOLD
df["is_alert"] = (df["fraud_score"] >= TH_INTERNAL).astype(int)

# ────────────────────────────── KPIs ─────────────────────────────────
st.markdown("### EXECUTIVE KPIS")
st.caption("WHAT THIS SHOWS: OVERALL THROUGHPUT AND HOW MANY ORDERS WERE FLAGGED AT THE FIXED OPERATING POINT.")
c1, c2, c3, c4 = st.columns([1,1,1,2])
TOT = len(df)
AL  = int(df["is_alert"].sum())
c1.metric("TOTAL SCORED", TOT)
c2.metric("TOTAL ALERTS", AL)
c3.metric("ALERT RATE", f"{(AL/TOT if TOT else 0):.2%}")
c4.caption(
    f"WINDOW: {df['timestamp'].min()} → {df['timestamp'].max()} | "
    f"DECISION THRESHOLD (INTERNAL) = {TH_INTERNAL:.2f}"
)
st.markdown("---")

# ───────────────────────── DAILY TREND (OPTIONAL) ────────────────────
st.subheader("DAILY TREND")
st.caption("WHAT THIS SHOWS: HOW MANY ORDERS ENTER THE SYSTEM EACH DAY AND HOW MANY BECOME ALERTS.")
trend = df.groupby("day").agg(scored=("order_id","count"), alerts=("is_alert","sum")).reset_index()
if len(trend):
    tl = trend.melt("day", ["scored","alerts"], "series", "value")
    st.altair_chart(
        alt.Chart(tl).mark_line(point=True).encode(
            x="day:T", y="value:Q", color="series:N",
            tooltip=[alt.Tooltip("day:T"), "series:N", alt.Tooltip("value:Q")]
        ).properties(height=260),
        use_container_width=True
    )
else:
    st.info("NO ACTIVITY IN THIS WINDOW.")

# ───────────────────── FRAUD-SCORE DISTRIBUTION ─────────────────────
st.subheader("FRAUD-SCORE DISTRIBUTION")
st.caption("WHAT THIS SHOWS: SCORE SHAPE; THE RED LINE IS YOUR WHAT-IF SLIDER, THE BLACK LINE IS THE FIXED OPERATING POINT (0.02).")
hist = alt.Chart(df).mark_bar().encode(
    x=alt.X("fraud_score:Q", bin=alt.Bin(maxbins=50), title="Fraud score"),
    y=alt.Y("count():Q", title="Rows"),
    tooltip=[alt.Tooltip("count()", title="Rows")]
).properties(height=220)

rule_view = alt.Chart(pd.DataFrame({"x":[TH_VIEW]})).mark_rule(color="crimson").encode(x="x")
rule_fix  = alt.Chart(pd.DataFrame({"x":[TH_INTERNAL]})).mark_rule(color="black", strokeDash=[4,4]).encode(x="x")

st.altair_chart(hist + rule_view + rule_fix, use_container_width=True)

# ───────────────────── SIGNAL PREVALENCE COMPARISON ─────────────────
st.subheader("CONTEXT SIGNALS")
st.caption("WHAT THIS SHOWS: HOW OFTEN KEY SIGNALS APPEAR IN ALERTS VS NON-ALERTS.")

def prevalence(cols, title):
    cols = [c for c in cols if c in df]
    if not cols:
        st.info(f"NO SIGNALS FOUND FOR: {title}")
        return
    z   = df[["is_alert"] + cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    a   = int(z["is_alert"].sum()) or 1
    na  = int((1 - z["is_alert"]).sum()) or 1
    rows = []
    for c in cols:
        rows.append({
            "signal": c,
            "% in alerts":     z.loc[z.is_alert==1, c].sum()/a,
            "% in non-alerts": z.loc[z.is_alert==0, c].sum()/na
        })
    dd = pd.DataFrame(rows).sort_values("% in alerts", ascending=False).melt(
        id_vars="signal", var_name="group", value_name="value"
    )
    st.altair_chart(
        alt.Chart(dd).mark_bar().encode(
            x=alt.X("value:Q", axis=alt.Axis(format="%"), title="Prevalence"),
            y=alt.Y("signal:N", sort="-x", title=None),
            color="group:N",
            tooltip=["signal:N", alt.Tooltip("value:Q", format=".1%"), "group:N"]
        ).properties(title=title, height=300),
        use_container_width=True
    )

left, right = st.columns(2)
with left:
    prevalence(
        [
            "strong_tri_mismatch_high_value","strong_high_value_express_geo",
            "strong_burst_multi_device","strong_price_drop_bulk","strong_giftcard_geo",
            "strong_return_whiplash","strong_price_inventory_stress","strong_country_flip_express"
        ],
        "STRONG-SIGNAL PREVALENCE"
    )
with right:
    prevalence(
        ["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"],
        "PRICING / INVENTORY PREVALENCE"
    )

# ───────────────────── PRICE / INVENTORY CORRELATION ─────────────────
st.caption("CORRELATION OF FRAUD_SCORE WITH PRICING/INVENTORY SIGNALS (PEARSON).")
CTX = [c for c in ["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"] if c in df]
if CTX:
    co = df[["fraud_score"]+CTX].apply(pd.to_numeric, errors="coerce").fillna(0).corr().loc[CTX, "fraud_score"].reset_index()
    co.columns = ["signal", "corr"]
    st.altair_chart(
        alt.Chart(co).mark_bar().encode(
            x="corr:Q", y=alt.Y("signal:N", sort="x", title=None),
            tooltip=[alt.Tooltip("corr:Q", format=".3f")]
        ).properties(height=120),
        use_container_width=True
    )
else:
    st.info("NO PRICING/INVENTORY COLUMNS TO CORRELATE.")

# ───────────────────────────── TOP ALERTS ────────────────────────────
st.subheader("TOP ALERTS")
st.caption("WHAT THIS SHOWS: HIGHEST-SCORING ORDERS FOR REVIEW.")
cols = [c for c in [
    "order_id","timestamp","customer_id","store_id","sku_id","sku_category",
    "order_amount","quantity","payment_method","shipping_country","ip_country","fraud_score"
] if c in df]
st.dataframe(
    df.sort_values(["fraud_score","timestamp"], ascending=[False, False]).loc[:, cols].head(50),
    use_container_width=True, height=320
)

# ───────────────────────── MODEL EVALUATION ──────────────────────────
st.subheader("MODEL EVALUATION")
st.caption("WHAT THIS SHOWS: METRICS AT THE FIXED OPERATING POINT (INTERNAL THRESHOLD = 0.02).")

# PREFERRED LABEL ORDER (AUTO-DETECT)
LABELS = [c for c in ["fraud_flag","is_fraud","label","ground_truth","gt","y"] if c in df]
if LABELS:
    lab = st.selectbox("GROUND-TRUTH COLUMN (1 = FRAUD, 0 = LEGIT)", LABELS, 0)
    st.caption(f"USING: `{lab}`")
    y_true = df[lab].fillna(0).astype(int).values
else:
    st.warning("NO LABEL COLUMN FOUND; USING DECISION AS PROXY.")
    y_true = (df["fraud_score"] >= TH_INTERNAL).astype(int).values

# PREDICTIONS ALWAYS USE INTERNAL THRESHOLD
y_pred  = (df["fraud_score"] >= TH_INTERNAL).astype(int).values
y_score = df["fraud_score"].values

m1, m2, m3, m4 = st.columns(4)
m1.metric("ACCURACY",  f"{accuracy_score(y_true, y_pred):.2%}")
m2.metric("PRECISION", f"{precision_score(y_true, y_pred, zero_division=0):.2%}")
m3.metric("RECALL",    f"{recall_score(y_true, y_pred, zero_division=0):.2%}")
m4.metric("F1-SCORE",  f"{f1_score(y_true, y_pred, zero_division=0):.2%}")

cm = confusion_matrix(y_true, y_pred, labels=[0,1])
cm_long = pd.DataFrame(cm, index=["Actual: 0","Actual: 1"], columns=["Pred: 0","Pred: 1"])\
            .reset_index().melt(id_vars="index", var_name="Predicted", value_name="Count")\
            .rename(columns={"index":"Actual"})

st.altair_chart(
    alt.Chart(cm_long).mark_rect().encode(
        x="Predicted:N", y="Actual:N",
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["Actual","Predicted","Count"]
    ).properties(height=180),
    use_container_width=True
)

# ROC & PR CURVES (REFERENCE)
try:
    AUC = roc_auc_score(y_true, y_score)
except Exception:
    AUC = float("nan")

fpr, tpr, _ = roc_curve(y_true, y_score)
st.altair_chart(
    alt.Chart(pd.DataFrame({"fpr":fpr,"tpr":tpr})).mark_line().encode(
        x=alt.X("fpr:Q", title="FALSE POSITIVE RATE"),
        y=alt.Y("tpr:Q", title="TRUE POSITIVE RATE")
    ).properties(height=200, title=f"ROC (AUC = {AUC:.3f})"),
    use_container_width=True
)

P, R, _ = precision_recall_curve(y_true, y_score)
AP = auc(R, P)
st.altair_chart(
    alt.Chart(pd.DataFrame({"recall":R, "precision":P})).mark_line().encode(
        x=alt.X("recall:Q", title="RECALL"),
        y=alt.Y("precision:Q", title="PRECISION")
    ).properties(height=200, title=f"PRECISION–RECALL (AP ≈ {AP:.3f})"),
    use_container_width=True
)

# ───────────────────── OPERATING POINT HELPER ────────────────────────
st.subheader("OPERATING-POINT HELPER")
st.caption("WHAT THIS SHOWS: HOW PRECISION/RECALL MOVE IF YOU CHANGED THE THRESHOLD (REFERENCE ONLY).")

@st.cache_data(show_spinner=False)
def load_ops(MT, S, E):
    sql = f"""
      SELECT threshold, AVG(precision) AS precision, AVG(recall) AS recall
      FROM `{MT}` WHERE dt BETWEEN @S AND @E
      GROUP BY threshold ORDER BY threshold
    """
    return bq.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("S","DATE",str(S)),
                bigquery.ScalarQueryParameter("E","DATE",str(E)),
            ]
        )
    ).result().to_dataframe()

use_bq = True
try:
    OP = load_ops(MT, S, E)
    use_bq = not OP.empty
except Exception:
    use_bq = False

if not use_bq:
    # Local fallback sweep (does not affect KPIs)
    grid = np.round(np.linspace(0.01, 0.99, 33), 3)
    pos  = max(1, (df["fraud_score"] >= 0).sum())
    OP = []
    y1 = (y_true == 1)
    for t in grid:
        yh = (df["fraud_score"] >= t)
        prec = ((yh & y1).sum()/max(1, yh.sum()))
        rec  = ((yh & y1).sum()/max(1, y1.sum()))
        OP.append({"threshold": t, "precision": prec, "recall": rec})
    OP = pd.DataFrame(OP)

st.altair_chart(
    alt.Chart(OP.melt(id_vars="threshold", value_vars=["precision","recall"],
                      var_name="metric", value_name="value")).mark_line(point=True).encode(
        x="threshold:Q",
        y=alt.Y("value:Q", axis=alt.Axis(format="%")),
        color="metric:N"
    ).properties(height=200, title=("BIGQUERY METRICS" if use_bq else "LOCAL FALLBACK")),
    use_container_width=True
)

