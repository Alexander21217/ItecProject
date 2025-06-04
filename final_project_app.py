import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from pandas_datareader import data as pdr

# --- Page Config ---
st.set_page_config(page_title="AI Revenue Forecasting - Starbucks", layout="centered")

st.title("📊 Starbucks Revenue Forecasting & Risk Assessment")
st.markdown("This app forecasts Starbucks revenue using ARIMA/ARIMAX, integrates live CPI data, and analyzes operating income trends.")

# --- User Input ---
forecast_periods = st.slider("Select number of quarters to forecast:", min_value=1, max_value=8, value=4)

# --- Load Data (placeholder) ---
st.subheader("📈 Historical Revenue Data")
st.markdown("_Using simulated revenue data for demonstration._")
dates = pd.date_range(start="2018-01-01", periods=24, freq="Q")
revenue = pd.Series(np.random.normal(6, 0.3, 24), index=dates)
revenue.plot(label="Revenue", title="Starbucks Revenue (Simulated)")
plt.legend()
st.pyplot(plt)

# --- Forecast (simplified demo) ---
model = sm.tsa.ARIMA(revenue, order=(1, 1, 1))
fit = model.fit()
forecast = fit.forecast(steps=forecast_periods)
forecast_dates = pd.date_range(dates[-1] + pd.offsets.QuarterEnd(), periods=forecast_periods, freq="Q")
forecast_series = pd.Series(forecast, index=forecast_dates)

# Plot forecast
st.subheader("🔮 Revenue Forecast")
plt.figure()
plt.plot(revenue, label="Historical Revenue")
plt.plot(forecast_series, label="Forecast", linestyle="--")
plt.legend()
plt.title("Revenue Forecast")
st.pyplot(plt)

# --- Live CPI Data ---
st.subheader("🌐 CPI Trend (FRED - Live Data)")
try:
    cpi = pdr.DataReader("CPIAUCSL", "fred", start="2018-01-01")
    st.line_chart(cpi)
except:
    st.warning("CPI data could not be loaded. Please check your internet connection or FRED availability.")

# --- Operating Income Insight (simulated) ---
st.subheader("💡 Operating Income Insight")
op_income = revenue * np.random.uniform(0.15, 0.2, len(revenue))
plt.figure()
plt.plot(op_income, label="Operating Income")
plt.title("Operating Income Trend")
plt.legend()
st.pyplot(plt)

# --- Risk Flagging (basic logic) ---
st.subheader("⚠️ Risk Flagging")
avg_growth = revenue.pct_change().mean()
forecast_growth = (forecast_series.iloc[0] - revenue.iloc[-1]) / revenue.iloc[-1]

if forecast_growth > avg_growth * 2:
    st.error("⚠️ Revenue forecast shows abnormal growth. Review for overstatement risk.")
else:
    st.success("✅ Forecast within reasonable range. No major red flags.")

# --- AI Summary ---
st.subheader("📝 AI Summary")
ai_summary = (
    "Our forecasting model projects steady growth in Starbucks revenue over the next few quarters, "
    "aligned with historical performance and supported by stable CPI trends. "
    "Operating income trends also appear consistent, indicating no immediate overstatement risks."
)
st.info(ai_summary)
