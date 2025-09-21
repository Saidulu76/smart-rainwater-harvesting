import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "smart_rainwater_data.csv")

# If data file is missing, try parent directory (common when running from repo root)
if not os.path.exists(DATA_PATH):
    alt = os.path.join(BASE_DIR, "..", "smart_rainwater_data.csv")
    if os.path.exists(alt):
        DATA_PATH = os.path.abspath(alt)

# Provide clearer error for missing data
if not os.path.exists(DATA_PATH):
    st.warning(f"Data file not found at {DATA_PATH}. Please upload `smart_rainwater_data.csv` or set the correct path.")
else:
    df = pd.read_csv(DATA_PATH)

# Detect rainfall column automatically
rain_cols = [c for c in df.columns if 'rain' in c.lower() or 'precip' in c.lower() or 'mm' in c.lower()]
if rain_cols:
    df = df.rename(columns={rain_cols[0]: 'rainfall'})
else:
    num_cols = df.select_dtypes('number').columns
    if len(num_cols) > 0:
        df = df.rename(columns={num_cols[0]: 'rainfall'})
    else:
        st.warning("No rainfall-like column found in dataset. Please check your CSV.")


df = pd.read_csv(DATA_PATH)

# Detect rainfall column automatically
rain_cols = [c for c in df.columns if 'rain' in c.lower() or 'precip' in c.lower() or 'mm' in c.lower()]
if rain_cols:
    df = df.rename(columns={rain_cols[0]: 'rainfall'})
else:
    num_cols = df.select_dtypes('number').columns
    if len(num_cols) > 0:
        df = df.rename(columns={num_cols[0]: 'rainfall'})
    else:
        st.warning("No rainfall-like column found in dataset. Please check your CSV.")


st.set_page_config(page_title="Smart Rainwater Dashboard", layout="wide")

# Sidebar navigation
menu = st.sidebar.radio(
    "📊 Dashboard Navigation",
    ["Overview", "Rainfall Data", "Storage Utilization", "Forecasting", "Optimization (GA)", "Reports"]
)

# ---------------- Overview ----------------
if menu == "Overview":
    st.title("🌧️ Smart Rainwater Harvesting Dashboard")
    st.write("Monitor rainfall, tank storage, forecasts, and optimization insights.")
    st.metric("Total Rainfall", f"{df['rainfall'].sum():.2f} mm")
    st.metric("Average Rainfall", f"{df['rainfall'].mean():.2f} mm")

# ---------------- Rainfall Data ----------------
elif menu == "Rainfall Data":
    st.header("📈 Rainfall Data Visualization")
    fig, ax = plt.subplots()
    df['rainfall'].plot(ax=ax, title="Rainfall Over Time")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    df['rainfall'].hist(ax=ax2, bins=20)
    ax2.set_title("Rainfall Distribution")
    st.pyplot(fig2)

# ---------------- Storage Utilization ----------------
elif menu == "Storage Utilization":
    st.header("💧 Tank Storage & Usage")
    if 'storage' in df.columns:
        fig, ax = plt.subplots()
        df[['storage', 'usage']].plot(ax=ax)
        ax.set_title("Storage vs Usage")
        st.pyplot(fig)

        st.write("### Water Reuse Ratio")
        reuse = df['usage'].sum()
        wasted = df['storage'].sum() - reuse
        fig2, ax2 = plt.subplots()
        ax2.pie([reuse, wasted], labels=["Reused", "Wasted"], autopct='%1.1f%%')
        st.pyplot(fig2)
    else:
        st.warning("⚠️ No storage/usage data available in dataset.")

# ---------------- Forecasting (LSTM) ----------------
elif menu == "Forecasting":
    st.header("🤖 LSTM Rainfall Prediction")
    if os.path.exists("predictions.csv"):
        pred_df = pd.read_csv("predictions.csv")
        fig, ax = plt.subplots()
        ax.plot(pred_df['actual'], label="Actual")
        ax.plot(pred_df['predicted'], label="Predicted")
        ax.set_title("Rainfall Forecast")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("⚠️ No prediction data found. Please run model training.")

# ---------------- Optimization (GA) ----------------
elif menu == "Optimization (GA)":
    st.header("🧬 Genetic Algorithm Optimization")
    if os.path.exists("ga_results.csv"):
        ga_df = pd.read_csv("ga_results.csv")
        fig, ax = plt.subplots()
        ax.plot(ga_df['generation'], ga_df['best_fitness'])
        ax.set_title("Best Fitness Over Generations")
        st.pyplot(fig)
    else:
        st.warning("⚠️ No GA optimization results found.")

# ---------------- Reports ----------------
elif menu == "Reports":
    st.header("📄 Export Reports")
    st.write("Download system performance reports as PDF/Word.")
    st.download_button("⬇️ Download PDF", data="PDF content here", file_name="report.pdf")
    st.download_button("⬇️ Download Word", data="Word content here", file_name="report.docx")
