# app.py — run with: streamlit run app.py
import streamlit as st, pandas as pd, numpy as np, altair as alt
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config("Retail Fraud Dashboard", layout="wide")
alt.renderers.set_embed_options(actions=False)

# ───────── Sidebar ─────────
with st.sidebar:
    st.header("BQ TABLE INFO")
    P  = st.text_input("Project", "mss-data-engineer-sandbox")
    D  = st.text_input("Dataset", "retail")
    PT = st.text_input("Predictions table", f"{P}.{D}.predictions_latest")
    FT = st.text_input("Features table",   f"{P}.{D}.features_signals_v4")
    TH = st.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01)
    st.caption("Tables are schema-aware; threshold applies everywhere.")

# ───────── BQ client + loaders (schema-aware) ─────────
sa = dict(st.secrets["gcp_service_account"]); sa["private_key"]=sa["private_key"].replace("\\n","\n")
creds = service_account.Credentials.from_service_account_info(sa)
bq = bigquery.Client(credentials=creds, project=creds.project_id)

def cols_of(t):
    p,d,x = t.split(".")
    q = f"SELECT column_name FROM `{p}.{d}.INFORMATION_SCHEMA.COLUMNS` WHERE table_name=@t"
    j = bq.query(q, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("t","STRING",x)]))
    return set(j.result().to_dataframe()["column_name"].str.lower())

@st.cache_data(show_spinner=True)
def load(pred_t, feat_t):
    pcols, fcols = cols_of(pred_t), cols_of(feat_t)
    base = ["order_id","timestamp","fraud_score","customer_id","store_id","sku_id","sku_category",
            "order_amount","quantity","payment_method","shipping_country","ip_country"]
    strong = ["strong_tri_mismatch_high_value","strong_high_value_express_geo","strong_burst_multi_device",
              "strong_price_drop_bulk","strong_giftcard_geo","strong_return_whiplash","strong_price_inventory_stress",
              "strong_country_flip_express","high_price_anomaly","low_price_anomaly","oversell_flag",
              "stockout_risk_flag","hoarding_flag","fraud_flag"]
    sel = [f"p.{c}" if c in pcols else f"CAST(NULL AS {'FLOAT64' if c=='fraud_score' else 'STRING'}) AS {c}" for c in base]
    join = ""
    if fcols:
        join = f"LEFT JOIN `{feat_t}` s USING(order_id)"
        sel += [f"s.{c}" if c in fcols else (f"CAST(NULL AS INT64) AS {c}" if c=='fraud_flag' else f"CAST(0 AS INT64) AS {c}") for c in strong]
    else:
        sel += [f"CAST(0 AS INT64) AS {c}" if c!='fraud_flag' else "CAST(NULL AS INT64) AS fraud_flag" for c in strong]
    sql = f"SELECT {', '.join(sel)} FROM `{pred_t}` p {join} ORDER BY timestamp"
    df = bq.query(sql).result().to_dataframe()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["fraud_score"] = pd.to_numeric(df["fraud_score"], errors="coerce").fillna(0.0)
    return df

df = load(PT, FT)
if df.empty: st.warning("No rows available."); st.stop()
df["is_alert"] = (df["fraud_score"] >= TH).astype(int)

# ───────── KPIs ─────────
st.title("Retail Fraud Dashboard")
k1,k2,k3,k4 = st.columns([1,1,1,2])
TOT, AL = len(df), int(df["is_alert"].sum())
k1.metric("TOTAL ROWS", TOT); k2.metric("ALERTS", AL)
k3.metric("ALERT RATE", f"{(AL/TOT if TOT else 0):.2%}")
if df["timestamp"].notna().any():
    k4.caption(f"TABLE WINDOW: {df['timestamp'].min()} → {df['timestamp'].max()}")

# ───────── Raw data ─────────
st.subheader("Raw data (sample)")
st.dataframe(df.head(4000), use_container_width=True, height=320)

# ───────── Strong features ─────────
st.subheader("Strong Signal Prevalence")
def prevalence(cols, title):
    cols=[c for c in cols if c in df.columns]
    if not cols: st.info(f"No fields for {title}."); return
    z = df[["is_alert"]+cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    a,na = max(1,int(z.is_alert.sum())), max(1,int((1-z.is_alert).sum()))
    rows=[{"signal":c,"% in alerts":z.loc[z.is_alert==1,c].sum()/a,"% in non-alerts":z.loc[z.is_alert==0,c].sum()/na} for c in cols]
    dd=(pd.DataFrame(rows).sort_values("% in alerts",ascending=False).melt("signal","group","value"))
    st.altair_chart(alt.Chart(dd).mark_bar().encode(
        x=alt.X("value:Q", axis=alt.Axis(format="%"), title="Prevalence"),
        y=alt.Y("signal:N", sort="-x", title=None), color="group:N",
        tooltip=["signal",alt.Tooltip("value:Q",format=".1%"),"group"]
    ).properties(height=300,title=title), use_container_width=True)
c1,c2 = st.columns(2)
with c1: prevalence(
    ["strong_tri_mismatch_high_value","strong_high_value_express_geo","strong_burst_multi_device",
     "strong_price_drop_bulk","strong_giftcard_geo","strong_return_whiplash","strong_price_inventory_stress",
     "strong_country_flip_express"], "Fraud Pattern Flags")
with c2: prevalence(["high_price_anomaly","low_price_anomaly","oversell_flag","stockout_risk_flag","hoarding_flag"],
                    "Pricing & Inventory Context")

# ───────── Fraud distribution ─────────
st.subheader("Fraud Score Distribution")
st.altair_chart(alt.Chart(df).mark_bar().encode(
    x=alt.X("fraud_score:Q", bin=alt.Bin(maxbins=50), title="Fraud score"),
    y=alt.Y("count():Q", title="Rows"), tooltip=[alt.Tooltip("count()", title="Rows")]
).properties(height=220), use_container_width=True)

# ───────── Alerts ─────────
st.subheader("Alerts (all)")
cols=[c for c in ["order_id","timestamp","customer_id","store_id","sku_id","sku_category",
                  "order_amount","quantity","payment_method","shipping_country","ip_country","fraud_score"] if c in df.columns]
alerts=df[df.is_alert==1].sort_values(["fraud_score","timestamp"], ascending=[False,False])[cols]
st.dataframe(alerts, use_container_width=True, height=min(720, 28*min(len(alerts),20)+120))

# ───────── Model evaluation ─────────
st.subheader("Model Evaluation")
labs=[c for c in ["fraud_flag","is_fraud","label","ground_truth","gt","y"] if c in df.columns]
if labs: lab=st.selectbox("Ground truth (1=fraud, 0=legit)", labs, 0); y=df[lab].fillna(0).astype(int).values; st.caption(f"Using label column: `{lab}`")
else:    st.warning("No label column; using current decisions as proxy."); y=df["is_alert"].values
yhat=df["is_alert"].values
a,p,r,f = accuracy_score(y,yhat), precision_score(y,yhat,zero_division=0), recall_score(y,yhat,zero_division=0), f1_score(y,yhat,zero_division=0)
m1,m2,m3,m4=st.columns(4); m1.metric("Accuracy",f"{a:.2%}"); m2.metric("Precision",f"{p:.2%}"); m3.metric("Recall",f"{r:.2%}"); m4.metric("F1-score",f"{f:.2%}")
st.caption("Accuracy: overall match · Precision: cleanliness of alerts · Recall: % of true fraud caught · F1: balance.")

# ───────── New order — instant decision ─────────
st.markdown("## New Order — Instant Decision")
st.markdown("- **Geo mismatch** — shipping and IP countries differ.\n- **High amount for category** — top 10% for that SKU.\n- **Payment channel risk** — gift card/paypal higher; Apple Pay moderate; card/debit standard.")
cats=sorted(df.get("sku_category",pd.Series(["apparel"])).dropna().unique().tolist())
ships=sorted(df.get("shipping_country",pd.Series(["US"])).dropna().unique().tolist())
ips =sorted(df.get("ip_country",pd.Series(["US"])).dropna().unique().tolist())
c1,c2,c3=st.columns(3); cat=c1.selectbox("SKU category",cats); pay=c2.selectbox("Payment method",["credit_card","debit_card","gift_card","paypal","apple_pay"]); qty=c3.number_input("Quantity",1,99,1)
c4,c5=st.columns(2); price=c4.number_input("Unit price",1.0,10000.0,100.0,1.0); ship=c5.selectbox("Shipping country",ships); ip=st.selectbox("IP country",ips)
PAY_W={"gift_card":0.35,"paypal":0.20,"apple_pay":0.12,"credit_card":0.10,"debit_card":0.08}
cs=(df[["sku_category","order_amount"]].dropna().groupby("sku_category")["order_amount"]
     .agg(cat_mean="mean",cat_p90=lambda s: float(np.nanpercentile(s,90))).reset_index()
     if "order_amount" in df and "sku_category" in df else pd.DataFrame({"sku_category":cats or ["apparel"],"cat_mean":[500.0],"cat_p90":[1500.0]}))
def score(cat,pay,qty,price,ship,ip):
    r=cs[cs.sku_category==cat].iloc[:1]; amt=qty*price; geo=ship!=ip; high=bool(amt>=float(r.cat_p90)); s=0.08+(0.35 if geo else 0)+(0.30 if high else 0)+PAY_W.get(pay,0.10)
    if geo and pay in("gift_card","paypal"): s=max(s,0.65)
    if geo and high and pay in("gift_card","paypal"): s=max(s,0.85)
    if high and pay in("gift_card","paypal"): s=max(s,0.70)
    return min(s,0.99), dict(amount=amt,mean=float(r.cat_mean),p90=float(r.cat_p90),geo=geo,high=high)
if st.button("Score order"):
    s,e=score(cat,pay,int(qty),float(price),ship,ip)
    st.markdown(f"### Decision: **{'Fraud' if s>=TH else 'Not fraud'}** · Score ≈ **{s:.2f}**")
    why=["Shipping & IP differ." if e["geo"] else "Shipping & IP match.",
         "Amount is high for this category." if e["high"] else "Amount is typical for category.",
         f"Payment channel: {pay.replace('_',' ')}."]
    st.markdown("**Why:**\n- " + "\n- ".join(why))
    st.caption(f"Order amount ≈ {e['amount']:,.2f} · Category mean ≈ {e['mean']:,.2f} · P90 ≈ {e['p90']:,.2f}")
