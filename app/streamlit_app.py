import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
import datetime
import matplotlib.pyplot as plt

def fetch_live_gold_data():
    """Fetch live Gold Futures (GC=F) prices for the past 3 months."""
    data = yf.download('GC=F', period='3mo')
    
    # Handle potentially multi-indexed columns from yfinance 0.2.x+
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.levels[0]:
            data = data['Adj Close']['GC=F']
        elif 'Close' in data.columns.levels[0]:
            data = data['Close']['GC=F']
    else:
        if 'Adj Close' in data.columns:
            data = data['Adj Close']
        else:
            data = data['Close']
            
    return data

@st.cache_data
def load_static_data():
    """Load historical static data from the original given dataset."""
    df = pd.read_csv("data/raw/Gold_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

@st.cache_data
def load_live_data():
    """Fetch all historical gold data from yfinance (GC=F) to train the model up to today."""
    df = yf.download('GC=F', start='2016-01-01')
    
    # Handle potentially multi-indexed columns from yfinance 0.2.x+
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.levels[0]:
            df = df['Adj Close']['GC=F']
        elif 'Close' in df.columns.levels[0]:
            df = df['Close']['GC=F']
    else:
        if 'Adj Close' in df.columns:
            df = df['Adj Close']
        else:
            df = df['Close']
            
    # Convert Series to DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame()
        
    df = df.reset_index()
    df.columns = ['date', 'price']
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
    df.set_index('date', inplace=True)
    
    # Forward fill any missing values
    df = df.ffill()
    return df

@st.cache_resource
def train_and_forecast_prophet(historical_data, forecast_days):
    """Train Prophet model and generate future bounds."""
    df = historical_data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    df = df.reset_index().rename(columns={'date': 'ds', 'price': 'y'})
    
    # Initialize and fit model
    m = Prophet(daily_seasonality=False, yearly_seasonality=True, changepoint_prior_scale=0.05)
    m.fit(df)
    
    # Generate forecast dates
    future = m.make_future_dataframe(periods=forecast_days)
    forecast = m.predict(future)
    
    return m, forecast

def render_forecast_ui(historical_df, forecast_period, is_live=False):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Historical Training Data")
        st.line_chart(historical_df['price'])
        
    with col2:
        if is_live:
            st.subheader("Live Market Context (GC=F)")
            try:
                live_data = fetch_live_gold_data()
                st.line_chart(live_data)
                st.success(f"Latest Live Gold Price: **${live_data.iloc[-1]:.2f}**")
            except Exception as e:
                st.warning("Could not fetch live data from Yahoo Finance.")
        else:
            st.subheader("Dataset Context")
            st.info("This model is trained on the fixed, static historical dataset that ends in 2021.")
            
    st.markdown("---")
    
    st.markdown("💡 **Tip:** Click the button below to train the Prophet model on the displayed data and generate a future forecast with uncertainty bounds.")
    if st.button('GENERATE EXPERT FORECAST', key=f"btn_{is_live}", use_container_width=True):
        with st.spinner('Training Prophet model and generating statistical bounds...'):
            model, forecast = train_and_forecast_prophet(historical_df, forecast_period)
            
            st.subheader(f"Forecast Overview (Next {forecast_period} Days)")
            
            # Extract historical and future split
            future_forecast = forecast.tail(forecast_period)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot recent historical data for visual context (last 180 days)
            recent_hist = historical_df.iloc[-180:]
            ax.plot(recent_hist.index, recent_hist['price'], label="Historical Data", color='#1f77b4', linewidth=2)
            
            # Plot forecast line
            ax.plot(future_forecast['ds'], future_forecast['yhat'], label="Prophet Forecast", color='#ff7f0e', linewidth=2)
            
            # Plot uncertainty bounds
            ax.fill_between(future_forecast['ds'], 
                            future_forecast['yhat_lower'], 
                            future_forecast['yhat_upper'], 
                            color='#ff7f0e', alpha=0.3, label="95% Uncertainty Interval")
                            
            ax.set_title("Future Gold Prices with Confidence Intervals", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Price (USD)", fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Format dates nicely
            fig.autofmt_xdate()
            
            st.pyplot(fig)
            
            st.markdown("### Forecast Data Table")
            display_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            formatted_df = future_forecast[display_cols].copy()
            formatted_df.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
            formatted_df.set_index('Date', inplace=True)
            formatted_df = formatted_df.round(2)
            
            st.dataframe(formatted_df, use_container_width=True)

def main():
    st.set_page_config(page_title="Gold Price Prediction", layout="wide", page_icon="📈")
    
    st.title('📈 Gold Price Prediction App')
    st.markdown("""
    This application predicts future gold prices using **Facebook Prophet**, an industry-standard time-series forecasting model. 
    It captures multiple seasonalities (e.g., year-end spikes, monthly trends) and provides statistical **uncertainty intervals** indicating forecast confidence.
    """)
    
    st.sidebar.header("Configuration")
    st.sidebar.markdown("Use this slider to change how many days into the future the model should forecast.")
    forecast_period = st.sidebar.slider("Forecast Period (Days)", 1, 365, 30)
    
    tab1, tab2 = st.tabs(["Fixed Data (Original)", "Live Data (Recent Upgrade)"])
    
    with tab1:
        st.markdown("### 📊 Original Dataset")
        st.write("This tab uses the original static dataset provided for the project, which contains data until the end of 2021.")
        historical_static_df = load_static_data()
        render_forecast_ui(historical_static_df, forecast_period, is_live=False)
        
    with tab2:
        st.markdown("### 🌐 Live Data Upgrade")
        st.write("This tab features a recent upgrade that dynamically fetches the entire dataset up to **today** from Yahoo Finance. The forecast will start from the present day.")
        historical_live_df = load_live_data()
        render_forecast_ui(historical_live_df, forecast_period, is_live=True)

if __name__ == '__main__':
    main()