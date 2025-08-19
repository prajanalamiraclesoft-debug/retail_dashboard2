import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# ---------- Data Load ----------
@st.cache_data
def load_data():
    return pd.read_csv("transactions_raw_4k.csv", parse_dates=["date"])

# ---------- Fraud Detection ----------
def fraud_score(df, th=0.7):
    df["fraud_score"] = (df["price"] / df["quantity"]).rank(pct=True)
    df["alert"] = df["fraud_score"] > th
    return df

# ---------- Inventory Flags ----------
def inventory_flags(df):
    df["low_price_anomaly"] = df["price"] < df["price"].quantile(0.1)
    df["high_price_anomaly"] = df["price"] > df["price"].quantile(0.9)
    df["hoarding_flag"] = df["quantity"] > df["quantity"].quantile(0.95)
    df["stockout_risk_flag"] = df["quantity"] < 2
    df["overall_sell_flag"] = ~df["stockout_risk_flag"]
    return df

# ---------- Model Evaluation ----------
def evaluate(y_true, y_pred):
    st.write("**Classification Report**")
    st.text(classification_report(y_true, y_pred))
    st.write("**Confusion Matrix**")
    st.write(confusion_matrix(y_true, y_pred))
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.2f}")
    plt.legend(); st.pyplot()

# ---------- Streamlit UI ----------
st.title("Retail Dashboard: Fraud + Inventory + Pricing")

df = load_data()
threshold = st.sidebar.slider("Fraud Threshold", 0.1, 0.9, 0.7)
df = fraud_score(inventory_flags(df), threshold)

st.metric("Total Txns", len(df))
st.metric("Alerts", df["alert"].sum())
st.metric("Alert %", round(100*df["alert"].mean(),2))

st.line_chart(df.groupby("date")["alert"].mean())  # Daily trend
st.bar_chart(df.groupby("product")["alert"].mean())  # Strong signals

top_alerts = df[df["alert"]].sort_values("fraud_score", ascending=False).head(10)
st.subheader("Top Alerts for Review")
st.write(top_alerts[["date","product","price","quantity","fraud_score"]])

# Model evaluation example
y_true, y_pred = df["fraud_label"], df["alert"]  # assuming fraud_label is known
evaluate(y_true, y_pred)
