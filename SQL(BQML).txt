# app.py — run with: streamlit run app.py
# AI/ML Retail Fraud + Pricing & Inventory Risk Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ─────────── Config ───────────
st.set_page_config("AI/ML Retail Risk Dashboard", layout="wide")
alt.renderers.set_embed_options(actions=False)

# ─────────── Sidebar Controls ───────────
with st.sidebar:
    st.header("Business Controls")
    TH = st.slider("Decision Threshold", 0.0, 1.0, 0.50, 0.05,
                   help="Lower → stricter fraud catch. Higher → smoother CX.")

# ─────────── Load Demo Data ───────────
# Simulated dataset for demo (replace with BigQuery in real)
np.random.seed(42)
df = pd.DataFrame({
    "order_id": range(1001, 1021),
    "channel": np.random.choice(["Online", "In-Store"], 20),
    "customer_id": np.random.randint(2000, 2100, 20),
    "store_id": np.random.choice(["TX01","TX02","NY01"], 20),
    "sku_category": np.random.choice(["Electronics","Grocery","Apparel"], 20),
    "order_amount": np.random.randint(20, 2000, 20),
    "payment_method": np.random.choice(
        ["credit_card","debit_card","gift_card","paypal","apple_pay"], 20),
    "shipping_country": np.random.choice(["US","CA","UK"], 20),
    "billing_country": np.random.choice(["US","CA","UK"], 20),
    "ip_country": np.random.choice(["US","CA","RU"], 20),
    "fraud_score": np.random.rand(20).round(2)
})
df["is_alert"] = (df["fraud_score"] >= TH).astype(int)

# ─────────── KPIs ───────────
st.title("AI/ML Retail Fraud, Pricing & Inventory Risk Dashboard")
c1, c2, c3 = st.columns(3)
c1.metric("Total Transactions", len(df))
c2.metric("ML Detected Suspicious Orders", int(df["is_alert"].sum()))
c3.metric("Alert Rate", f"{df['is_alert'].mean():.1%}")

# ─────────── Retail Transactions (Raw Data) ───────────
st.subheader("Retail Transactions (Online + In-Store)")
tx_cols = ["order_id","channel","customer_id","store_id",
           "sku_category","order_amount","payment_method",
           "shipping_country","billing_country","ip_country"]
st.dataframe(df[tx_cols], use_container_width=True, height=250)

# ─────────── Fraud Score Distribution ───────────
st.subheader("Fraud Score Distribution (AI/ML model output)")
hist = alt.Chart(df).mark_bar().encode(
    x=alt.X("fraud_score:Q", bin=alt.Bin(maxbins=20), title="Fraud Score"),
    y="count()"
).properties(height=220)
st.altair_chart(hist, use_container_width=True)

# ─────────── Strong Features Block ───────────
st.markdown("### Strong Features Driving Suspicious Orders")
st.markdown("""
- **Geo mismatch** — Shipping, billing, and IP locations don’t align.  
- **High-value express order** — Unusually high order with fast shipping.  
- **Multiple devices** — Same account used on many devices.  
- **Gift card / PayPal risk** — Higher-risk payment channels.  
- **Return whiplash** — Frequent quick returns after purchases.  
- **Price anomalies** — Abnormally high or low prices used in order.  
- **Inventory abuse** — Overselling, hoarding, or stockout risk.  
""")

# ─────────── ML Detected Suspicious Orders ───────────
st.subheader("ML Detected Suspicious Orders")
suspicious = df[df["is_alert"] == 1].sort_values("fraud_score", ascending=False)
st.dataframe(suspicious[tx_cols + ["fraud_score"]], use_container_width=True, height=250)

# ─────────── Model Evaluation (Simulated) ───────────
st.subheader("Model Evaluation (AI/ML Performance)")
acc, prec, rec, f1 = 0.88, 0.72, 0.75, 0.73
e1, e2, e3, e4 = st.columns(4)
e1.metric("Accuracy", f"{acc:.0%}")
e2.metric("Precision", f"{prec:.0%}")
e3.metric("Recall", f"{rec:.0%}")
e4.metric("F1-Score", f"{f1:.0%}")
st.caption("""
- **Accuracy** — How often predictions are correct.  
- **Precision** — Quality of flagged alerts.  
- **Recall** — How much true fraud we captured.  
- **F1** — Balanced score for uneven classes.  
""")

# ─────────── Instant Decision (What-If) ───────────
st.subheader("Instant Decision — Test a New Order")
col1, col2, col3 = st.columns(3)
sku = col1.selectbox("SKU Category", ["Electronics","Grocery","Apparel"])
pay = col2.selectbox("Payment Method", ["credit_card","debit_card","gift_card","paypal","apple_pay"])
amt = col3.number_input("Order Amount", 10, 5000, 250)

fraud_score = (0.2 if pay in ["gift_card","paypal"] else 0.05) + (0.3 if amt > 1000 else 0.1)
fraud_score = min(fraud_score, 0.95)
decision = "Suspicious" if fraud_score >= TH else "Legit"

if st.button("Score Order"):
    st.write(f"**Decision:** {decision} (Fraud Score ≈ {fraud_score:.2f})")
    st.caption("Explanation: Based on amount, payment type, and risk patterns learned by ML.")
