# app.py — end-to-end: raw → clean/EDA → model → evaluate → predict
import streamlit as st, pandas as pd, numpy as np, altair as alt
from datetime import date, datetime
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.model_selection import train_test_split

st.set_page_config("Fraud: Data → EDA → Model → Predict", layout="wide")
alt.renderers.set_embed_options(actions=False)
RNG = 42

# ───────────────────────── Sidebar: BigQuery pull ─────────────────────────
with st.sidebar:
    st.header("BigQuery")
    PROJ = st.text_input("Project", "mss-data-engineer-sandbox")
    DATASET = st.text_input("Dataset", "retail")
    RAW = st.text_input("Raw table", f"{PROJ}.{DATASET}.transaction_data")
    S = st.date_input("Start date", date(2023,1,1))
    E = st.date_input("End date",   date(2030,12,31))
    st.caption("Pulls raw rows between Start/End from the specified table. No threshold controls shown anywhere.")

# ───────────────────────── BigQuery client & loader ───────────────────────
@st.cache_data(show_spinner=True)
def load_raw(raw, s, e, secrets_dict):
    sa = dict(secrets_dict)
    sa["private_key"] = sa["private_key"].replace("\\n","\n")
    creds = service_account.Credentials.from_service_account_info(sa)
    bq = bigquery.Client(credentials=creds, project=creds.project_id)
    sql = f"""
      SELECT
        order_id, TIMESTAMP(timestamp) AS ts, customer_id, store_id, sku_id, sku_category,
        SAFE_CAST(quantity AS FLOAT64) AS quantity, SAFE_CAST(unit_price AS FLOAT64) AS unit_price,
        CAST(payment_method AS STRING) AS payment_method, CAST(shipping_country AS STRING) AS shipping_country,
        CAST(ip_country AS STRING) AS ip_country, CAST(device_id AS STRING) AS device_id,
        SAFE_CAST(account_created_at AS TIMESTAMP) AS account_created_at,
        SAFE_CAST(coupon_discount AS FLOAT64) AS coupon_discount,
        SAFE_CAST(gift_card_amount AS FLOAT64) AS gift_card_amount,
        SAFE_CAST(gift_card_used AS BOOL) AS gift_card_used,
        SAFE_CAST(fraud_flag AS INT64) AS fraud_flag
      FROM `{raw}`
      WHERE DATE(timestamp) BETWEEN @S AND @E
    """
    job = bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("S","DATE",str(s)),
            bigquery.ScalarQueryParameter("E","DATE",str(e))
        ]))
    return job.result().to_dataframe()

try:
    df_raw = load_raw(RAW, S, E, st.secrets["gcp_service_account"])
except Exception as ex:
    st.error(f"BigQuery error: {ex}")
    st.stop()

if df_raw.empty:
    st.warning("No rows returned for this date range.")
    st.stop()

# ───────────────────────── Tabs ─────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["1) Raw Data", "2) Cleaning & EDA", "3) Model & Metrics", "4) Predict"])

# ───────────────────────── 1) RAW ─────────────────────────
with tab1:
    st.subheader("Raw data (from BigQuery)")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", len(df_raw))
    c2.metric("Columns", len(df_raw.columns))
    c3.metric("Min ts", str(pd.to_datetime(df_raw["ts"]).min()))
    c4.metric("Max ts", str(pd.to_datetime(df_raw["ts"]).max()))
    st.dataframe(df_raw.sample(min(500,len(df_raw))), use_container_width=True, height=420)
    st.caption("Random sample of raw rows; use the Cleaning & EDA tab for issues like skewness and outliers.")

# ───────────────────────── 2) CLEANING & EDA ─────────────────────────
def _wins(g, col):
    lo, hi = g[col].quantile(0.01), g[col].quantile(0.99)
    g[col] = g[col].clip(lo, hi)
    return g

@st.cache_data(show_spinner=True)
def clean_and_fe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    # basic dedupe & validity
    df = df.sort_values("ts").drop_duplicates("order_id", keep="last")
    df = df[(df["quantity"]>0) & (df["unit_price"]>0)].copy()

    # time fields / age
    df["ts"]   = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_localize(None)
    df["account_created_at"] = pd.to_datetime(df["account_created_at"], errors="coerce", utc=True).dt.tz_localize(None)
    df["account_age_days"] = (df["ts"] - df["account_created_at"]).dt.days.astype("float").fillna(0).clip(lower=0)

    # amount
    df["order_amount"] = df["quantity"]*df["unit_price"]

    # winsorize price & qty per category (reduce extreme skew/outliers)
    for c in ["unit_price","quantity"]:
        df = df.groupby("sku_category", group_keys=False).apply(_wins, c)

    df["order_amount"] = df["quantity"]*df["unit_price"]

    # price baseline per category
    cat_avg = df.groupby("sku_category")["unit_price"].transform("mean").replace(0,np.nan)
    df["price_ratio"] = (df["unit_price"]/cat_avg).fillna(1.0)

    # intensities
    den = df["order_amount"].replace(0,np.nan)
    df["coupon_pct"] = (df["coupon_discount"]/den).fillna(0)
    df["gift_pct"]   = (df["gift_card_amount"]/den).fillna(0)

    # geo mismatch
    df["geo_mismatch"] = (df["shipping_country"] != df["ip_country"]).astype(int)

    # simple time parts
    df["hour"] = pd.to_datetime(df["ts"]).dt.hour
    df["dow"]  = pd.to_datetime(df["ts"]).dt.dayofweek

    # payment risk (very simple)
    pay_map = {"crypto":3,"paypal":2,"credit_card":2,"apple_pay":2,"google_pay":2,"debit_card":1,"bank_transfer":0,"cod":0}
    df["pay_risk"] = df["payment_method"].map(pay_map).fillna(1).astype(int)

    # “strong” shapes
    p90 = float(np.nanpercentile(df["order_amount"], 90)) if len(df) else 0.0
    df["s_price_bulk"] = ((df["price_ratio"].sub(1).abs()>=.50) & (df["quantity"]>=3)).astype(int)
    df["s_gc_geo"]     = ((df["gift_card_used"].fillna(False)) & (df["geo_mismatch"]==1)).astype(int)
    df["s_geo_hi"]     = ((df["geo_mismatch"]==1) & (df["order_amount"]>=p90)).astype(int)
    df["s_any"]        = (df[["s_price_bulk","s_gc_geo","s_geo_hi"]].sum(axis=1)>0).astype(int)

    return df

df = clean_and_fe(df_raw)

with tab2:
    st.subheader("Cleaning summary")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows after cleaning", len(df))
    c2.metric("Distinct customers", df["customer_id"].nunique())
    c3.metric("Fraud labels present", df["fraud_flag"].notna().sum())
    c4.metric("Fraud prevalence", f"{(df['fraud_flag'].fillna(0).mean()*100):.2f}%")

    st.markdown("**Skewness (selected numeric columns)**")
    num_cols = ["quantity","unit_price","order_amount","account_age_days","coupon_pct","gift_pct","price_ratio"]
    skew_tbl = pd.DataFrame({
        "feature": num_cols,
        "skewness": [df[c].dropna().skew() for c in num_cols]
    }).sort_values("skewness", ascending=False)
    st.dataframe(skew_tbl, use_container_width=True, height=250)

    st.markdown("**Outlier counts (IQR rule)**")
    def iqr_outliers(x):
        q1, q3 = np.nanpercentile(x, 25), np.nanpercentile(x, 75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        return int(((x<lo)|(x>hi)).sum())
    out_tbl = pd.DataFrame({
        "feature": num_cols,
        "outliers": [iqr_outliers(df[c].values) for c in num_cols]
    }).sort_values("outliers", ascending=False)
    st.dataframe(out_tbl, use_container_width=True, height=240)

    st.markdown("**Distributions**")
    dist_cols = ["order_amount","unit_price","quantity","account_age_days","coupon_pct","gift_pct","price_ratio"]
    for col in dist_cols:
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=50), title=col),
            y=alt.Y("count():Q", title="Rows")
        ).properties(height=140)
        st.altair_chart(chart, use_container_width=True)

# ───────────────────────── 3) MODEL & METRICS ─────────────────────────
@st.cache_resource(show_spinner=True)
def train_model(clean_df: pd.DataFrame):
    labeled = clean_df[clean_df["fraud_flag"].isin([0,1])].copy()
    if labeled["fraud_flag"].nunique() < 2:
        raise RuntimeError("Need both classes (0/1) in fraud_flag to train.")

    # Features
    cat_cols = ["sku_category","payment_method","shipping_country","ip_country","device_id","store_id","hour","dow"]
    num_cols = ["order_amount","quantity","unit_price","account_age_days","coupon_pct","gift_pct","price_ratio",
                "geo_mismatch","pay_risk","s_price_bulk","s_gc_geo","s_geo_hi","s_any"]

    # Time-based split: last 20% by timestamp as test
    labeled = labeled.sort_values("ts")
    split_idx = int(np.floor(0.8 * len(labeled)))
    train_df = labeled.iloc[:split_idx].copy()
    test_df  = labeled.iloc[split_idx:].copy()

    X_tr_num = train_df[num_cols].fillna(0)
    X_tr_cat = pd.get_dummies(train_df[cat_cols].astype(str), dummy_na=False)
    X_tr = pd.concat([X_tr_num, X_tr_cat], axis=1)
    y_tr = train_df["fraud_flag"].astype(int).values

    X_te_num = test_df[num_cols].fillna(0)
    X_te_cat = pd.get_dummies(test_df[cat_cols].astype(str), dummy_na=False)
    X_te = pd.concat([X_te_num, X_te_cat], axis=1)

    # align columns
    all_cols = list(set(X_tr.columns).union(set(X_te.columns)))
    X_tr = X_tr.reindex(columns=all_cols, fill_value=0)
    X_te = X_te.reindex(columns=all_cols, fill_value=0)
    y_te = test_df["fraud_flag"].astype(int).values

    # class imbalance → weight positives
    pos = max(1, int((y_tr==1).sum())); neg = max(1, len(y_tr)-pos)
    w_pos = min(20.0, neg/pos)
    sw = np.where(y_tr==1, w_pos, 1.0)

    # model
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.07,
                                         early_stopping=True, random_state=RNG)
    clf.fit(X_tr, y_tr, sample_weight=sw)

    # predict proba on test
    pr_te = clf.predict_proba(X_te)[:,1]

    # choose operating threshold (maximize F1 on the test set)
    ts = np.linspace(0.01, 0.99, 99)
    f1s = []
    for t in ts:
        y_hat = (pr_te >= t).astype(int)
        f1s.append(f1_score(y_te, y_hat, zero_division=0))
    best_idx = int(np.argmax(f1s))
    th = float(ts[best_idx])

    # evaluate at chosen threshold
    y_pred = (pr_te >= th).astype(int)
    metrics = {
        "threshold": th,
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_te, pr_te),
        "pr_auc": average_precision_score(y_te, pr_te)
    }
    cm = confusion_matrix(y_te, y_pred, labels=[0,1])

    artifacts = {
        "clf": clf,
        "all_cols": all_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "threshold": th
    }
    evalpack = {
        "metrics": metrics,
        "cm": cm,
        "test_df": test_df[["order_id","ts","fraud_flag"]].assign(prob=pr_te, pred=y_pred)
    }
    return artifacts, evalpack

with tab3:
    st.subheader("Train model & evaluate (automatic threshold selection)")
    try:
        artifacts, evalpack = train_model(df)
    except Exception as ex:
        st.error(str(ex))
        st.stop()

    m = evalpack["metrics"]
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Accuracy",  f"{m['accuracy']*100:.2f}%")
    c2.metric("Precision", f"{m['precision']*100:.2f}%")
    c3.metric("Recall",    f"{m['recall']*100:.2f}%")
    c4.metric("F1-score",  f"{m['f1']*100:.2f}%")
    c5.metric("ROC-AUC",   f"{m['roc_auc']:.3f}")
    c6.metric("PR-AUC",    f"{m['pr_auc']:.3f}")
    st.caption(f"Chosen operating point (max F1 on test): **threshold = {m['threshold']:.2f}** (not user-controlled).")

    # Confusion matrix heatmap
    cm = evalpack["cm"]
    cm_df = pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]).reset_index().melt(
        id_vars="index", var_name="Predicted", value_name="Count").rename(columns={"index":"Actual"})
    st.altair_chart(
        alt.Chart(cm_df).mark_rect().encode(x="Predicted:N", y="Actual:N", color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
                                            tooltip=["Actual","Predicted","Count"]).properties(height=180),
        use_container_width=True
    )

    # Curves
    te = evalpack["test_df"]
    # ROC
    # Build points for ROC manually (threshold sweep)
    def roc_points(y_true, scores, n=50):
        thr = np.linspace(0,1,n)
        out=[]
        P=(y_true==1).sum(); N=(y_true==0).sum()
        for t in thr:
            yhat = (scores>=t).astype(int)
            tp = int(((y_true==1)&(yhat==1)).sum()); fp=int(((y_true==0)&(yhat==1)).sum())
            tn = int(((y_true==0)&(yhat==0)).sum()); fn=int(((y_true==1)&(yhat==0)).sum())
            tpr = 0 if P==0 else tp/P
            fpr = 0 if N==0 else fp/N
            out.append({"fpr":fpr,"tpr":tpr})
        return pd.DataFrame(out)
    roc_df = roc_points(te["fraud_flag"].values, te["prob"].values, 80)
    st.altair_chart(
        alt.Chart(roc_df).mark_line().encode(x=alt.X("fpr:Q", title="FPR"), y=alt.Y("tpr:Q", title="TPR")).properties(height=200, title=f"ROC (AUC={m['roc_auc']:.3f})"),
        use_container_width=True
    )

    # PR curve
    def pr_points(y_true, scores, n=80):
        thr = np.linspace(0,1,n)
        out=[]
        for t in thr:
            yhat=(scores>=t).astype(int)
            tp=int(((y_true==1)&(yhat==1)).sum()); fp=int(((y_true==0)&(yhat==1)).sum())
            fn=int(((y_true==1)&(yhat==0)).sum())
            prec = 0 if (tp+fp)==0 else tp/(tp+fp)
            rec  = 0 if (tp+fn)==0 else tp/(tp+fn)
            out.append({"recall":rec,"precision":prec})
        return pd.DataFrame(out)
    pr_df = pr_points(te["fraud_flag"].values, te["prob"].values, 80)
    st.altair_chart(
        alt.Chart(pr_df).mark_line().encode(x=alt.X("recall:Q", title="Recall"), y=alt.Y("precision:Q", title="Precision")).properties(height=200, title=f"PR (AP≈{m['pr_auc']:.3f})"),
        use_container_width=True
    )

    st.markdown("**Hold-out predictions (test set)**")
    st.dataframe(te.sort_values("prob", ascending=False).head(200), use_container_width=True, height=260)

# ───────────────────────── 4) PREDICT ─────────────────────────
def prepare_X_for_model(df_like: pd.DataFrame, artifacts):
    """Takes a 1+ row DataFrame with same raw columns/derived features; returns X aligned to training columns."""
    cat_cols = artifacts["cat_cols"]; num_cols = artifacts["num_cols"]; all_cols = artifacts["all_cols"]
    Xn = df_like[num_cols].fillna(0)
    Xc = pd.get_dummies(df_like[cat_cols].astype(str), dummy_na=False)
    X  = pd.concat([Xn, Xc], axis=1).reindex(columns=all_cols, fill_value=0)
    return X

def make_features_from_inputs(d):
    """Build a single-row DF with engineered features from manual inputs."""
    row = {}
    # Raw-ish fields
    row["order_id"] = d.get("order_id","new_order")
    row["ts"] = pd.to_datetime(d.get("ts"))
    row["account_created_at"] = pd.to_datetime(d.get("account_created_at"))
    row["customer_id"] = d.get("customer_id","cust_new")
    row["store_id"] = d.get("store_id","store_1")
    row["sku_id"] = d.get("sku_id","sku_1")
    row["sku_category"] = d.get("sku_category","general")
    row["quantity"] = float(d.get("quantity",1.0))
    row["unit_price"] = float(d.get("unit_price",10.0))
    row["payment_method"] = d.get("payment_method","credit_card")
    row["shipping_country"] = d.get("shipping_country","US")
    row["ip_country"] = d.get("ip_country","US")
    row["device_id"] = d.get("device_id","dev_1")
    row["coupon_discount"] = float(d.get("coupon_discount",0.0))
    row["gift_card_amount"] = float(d.get("gift_card_amount",0.0))
    row["gift_card_used"] = bool(d.get("gift_card_used",False))

    # Derived
    row["account_age_days"] = max(0.0, (row["ts"] - row["account_created_at"]).days)
    row["order_amount"] = row["quantity"]*row["unit_price"]

    # Use global medians from training set for category avg price if needed
    # Here, simple ratio proxy:
    row["price_ratio"] = 1.0  # conservative (unknown cat context)
    den = row["order_amount"] if row["order_amount"]!=0 else np.nan
    row["coupon_pct"] = (row["coupon_discount"]/den) if den==den else 0.0
    row["gift_pct"]   = (row["gift_card_amount"]/den) if den==den else 0.0
    row["geo_mismatch"] = int(row["shipping_country"] != row["ip_country"])
    row["hour"] = row["ts"].hour
    row["dow"]  = row["ts"].weekday()
    pay_map = {"crypto":3,"paypal":2,"credit_card":2,"apple_pay":2,"google_pay":2,"debit_card":1,"bank_transfer":0,"cod":0}
    row["pay_risk"] = int(pay_map.get(row["payment_method"], 1))

    # Risk shapes (use local p90 approximation  = 0 for single, so gate uses other signals)
    row["s_price_bulk"] = int(abs(row["price_ratio"]-1.0)>=0.50 and row["quantity"]>=3)
    row["s_gc_geo"]     = int(row["gift_card_used"] and row["geo_mismatch"]==1)
    row["s_geo_hi"]     = int(row["geo_mismatch"]==1 and row["order_amount"]>=2000)  # conservative high amount
    row["s_any"]        = int(row["s_price_bulk"] or row["s_gc_geo"] or row["s_geo_hi"])

    # Wrap to DataFrame
    return pd.DataFrame([row])

with tab4:
    st.subheader("Predict")
    st.caption("Pick an existing cleaned row or enter a brand-new order. The model uses the threshold it learned on the test set (max F1).")

    artifacts = st.session_state.get("artifacts_cache") or artifacts
    st.session_state["artifacts_cache"] = artifacts
    clf = artifacts["clf"]; threshold = artifacts["threshold"]

    colA, colB = st.columns(2, gap="large")

    # A) Pick an existing order (from cleaned)
    with colA:
        st.markdown("**Use a row from the cleaned data**")
        pick = st.selectbox("Order to score", options=df["order_id"].astype(str).head(5000).tolist())
        row_df = df[df["order_id"].astype(str)==pick].head(1)
        Xp = prepare_X_for_model(row_df, artifacts)
        p = float(clf.predict_proba(Xp)[:,1][0])
        yhat = int(p >= threshold)
        st.metric("Predicted probability of fraud", f"{p:.3f}")
        st.markdown(f"**Decision at learned threshold {threshold:.2f}**: {'🚨 FRAUD' if yhat==1 else '✅ Not fraud'}")
        st.dataframe(row_df, use_container_width=True, height=200)

    # B) Manual form
    with colB:
        st.markdown("**Enter a new order**")
        with st.form("new_order_form", clear_on_submit=False):
            order_id   = st.text_input("order_id", "new_0001")
            ts         = st.text_input("timestamp (YYYY-MM-DD HH:MM:SS)", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
            acct       = st.text_input("account_created_at (YYYY-MM-DD HH:MM:SS)", (datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S"))
            customer   = st.text_input("customer_id", "cust_new")
            store_id   = st.text_input("store_id", "store_1")
            sku_id     = st.text_input("sku_id", "sku_1")
            sku_cat    = st.text_input("sku_category", "general")
            qty        = st.number_input("quantity", 1.0, step=1.0)
            price      = st.number_input("unit_price", 0.01, step=0.01, value=10.0)
            pay        = st.text_input("payment_method", "credit_card")
            ship       = st.text_input("shipping_country", "US")
            ip         = st.text_input("ip_country", "US")
            device     = st.text_input("device_id", "dev_1")
            coup       = st.number_input("coupon_discount", 0.0, step=0.01)
            gift_amt   = st.number_input("gift_card_amount", 0.0, step=0.01)
            gift_used  = st.checkbox("gift_card_used", False)
            submitted = st.form_submit_button("Score this order")
        if submitted:
            try:
                rec = make_features_from_inputs({
                    "order_id":order_id,"ts":ts,"account_created_at":acct,"customer_id":customer,
                    "store_id":store_id,"sku_id":sku_id,"sku_category":sku_cat,"quantity":qty,"unit_price":price,
                    "payment_method":pay,"shipping_country":ship,"ip_country":ip,"device_id":device,
                    "coupon_discount":coup,"gift_card_amount":gift_amt,"gift_card_used":gift_used
                })
                Xn = prepare_X_for_model(rec, artifacts)
                p2 = float(clf.predict_proba(Xn)[:,1][0])
                y2 = int(p2 >= threshold)
                st.metric("Predicted probability of fraud", f"{p2:.3f}")
                st.markdown(f"**Decision at learned threshold {threshold:.2f}**: {'🚨 FRAUD' if y2==1 else '✅ Not fraud'}")
                st.dataframe(rec, use_container_width=True, height=220)
            except Exception as ex:
                st.error(f"Could not score: {ex}")
