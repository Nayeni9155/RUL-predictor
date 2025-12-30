import streamlit as st
import joblib
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

from utils import (
    read_cmapss,
    build_test_X_for_model,
    get_alert
)

# ---------------------
# App Config
# ---------------------
st.set_page_config(page_title="RUL Predictor", layout="wide")
st.title("🔧 Remaining Useful Life (RUL) Predictor")

SEQ_LEN = 50
CAP_RUL = 125
DOMAIN_MAP = {"FD001": 0, "FD002": 1, "FD003": 2, "FD004": 3}

# ---------------------
# Sidebar – Load artifacts
# ---------------------
st.sidebar.header("Model Artifacts")

model = load_model("grudomain_fd_all.h5", compile=False)
scaler = joblib.load("scaler_fd_all.joblib")
sensor_cols = joblib.load("sensor_cols_fd_all.joblib")

st.sidebar.success("✅ Model & scaler loaded")

# ---------------------
# Upload test files
# ---------------------
st.header("📂 Upload Test Data")

test_file = st.file_uploader("Upload test_FDxxx.txt", type=["txt"])
rul_file  = st.file_uploader("Upload RUL_FDxxx.txt", type=["txt"])
domain_key = st.selectbox("Select domain", list(DOMAIN_MAP.keys()))

# ---------------------
# Run inference
# ---------------------
if st.button("🚀 Run Prediction"):

    if test_file is None or rul_file is None:
        st.warning("Please upload both test and RUL files.")
        st.stop()

    domain_id = DOMAIN_MAP[domain_key]

    # Read data
    test_df = read_cmapss(test_file)
    y_true = pd.read_csv(rul_file, sep=r"\s+", header=None).values.ravel()
    y_true = np.clip(y_true, None, CAP_RUL)

    # Build model input
    X_test, units = build_test_X_for_model(
        test_df, sensor_cols, scaler,
        seq_len=SEQ_LEN,
        domain_id=domain_id,
        num_domains=len(DOMAIN_MAP)
    )

    y_pred = model.predict(X_test).ravel()
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    # ---------------------
    # Results
    # ---------------------
    st.subheader("📊 Results")
    st.metric("RMSE", f"{rmse:.3f}")

    # Scatter plot
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([0, CAP_RUL], [0, CAP_RUL], "k--")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("True vs Predicted RUL")
    st.pyplot(fig)

    # ---------------------
    # Alerts table
    # ---------------------
    st.subheader("🚨 Alerts")

    alert_df = pd.DataFrame({
        "Unit": units,
        "True RUL": y_true,
        "Predicted RUL": y_pred,
    })
    alert_df["Alert"] = alert_df["Predicted RUL"].apply(get_alert)

    st.dataframe(alert_df.head(20))
