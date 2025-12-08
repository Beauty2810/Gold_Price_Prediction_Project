import pandas as pd
import yfinance as yf
import numpy as np

def fetch_external_data(start_date='2016-01-01', end_date='2021-12-31'):
    """
    Fetches SPY, USO, and DXY data from yfinance and aligns their dates.
    """
    tickers = ['SPY', 'USO', 'DX-Y.NYB', 'INR=X']
    
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # yfinance version 0.2+ often returns a MultiIndex column DataFrame if multiple tickers are provided
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.levels[0]:
            data = data['Adj Close']
        elif 'Close' in data.columns.levels[0]:
            data = data['Close']
    else:
        # If not MultiIndex, maybe it's single (shouldn't be, but just in case)
        # Though we passed multiple tickers so it shouldn't happen unless we provided wrong params.
        pass

    # Rename columns to standard names
    rename_map = {'DX-Y.NYB': 'DXY', 'INR=X': 'USD_INR'}
    data = data.rename(columns=rename_map)
    
    # Make sure we have the expected columns in case something fails
    expected_cols = ['SPY', 'USO', 'DXY', 'USD_INR']
    for col in expected_cols:
        if col not in data.columns:
            print(f"Warning: {col} not found in yfinance data.")
            data[col] = np.nan
            
    # Forward fill missing values
    data = data.ffill()
    
    return data

def create_features(df):
    """
    Adds calendar, lag, and rolling features to the dataset.
    df must have 'price' column and a datetime index.
    """
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    # Calendar features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_year_end'] = df.index.is_year_end.astype(int)
    
    # Lag features based on 'price'
    if 'price' in df.columns:
        df['price_lag_1'] = df['price'].shift(1)
        df['price_lag_7'] = df['price'].shift(7)
        df['price_lag_30'] = df['price'].shift(30)
        
        # Rolling statistics (using shift to prevent data leakage)
        df['rolling_mean_7'] = df['price'].shift(1).rolling(window=7).mean()
        df['rolling_mean_30'] = df['price'].shift(1).rolling(window=30).mean()
        df['rolling_std_7'] = df['price'].shift(1).rolling(window=7).std()
        
    return df

def generate_full_dataset(gold_filepath, start_date='2016-01-01', end_date='2021-12-31'):
    """
    Reads local gold data, fetches external data, creates features, and merges them.
    """
    gold_df = pd.read_csv(gold_filepath)
    gold_df['date'] = pd.to_datetime(gold_df['date'])
    gold_df.set_index('date', inplace=True)
    
    # Add 1 day to end_date as yf end date is exclusive
    end_date_yf = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    end_date_yf = end_date_yf.strftime('%Y-%m-%d')
    
    ext_data = fetch_external_data(start_date, end_date_yf)
    
    # Merge external data with gold data
    # Left join to keep all gold dates
    gold_df.index = gold_df.index.tz_localize(None)
    ext_data.index = ext_data.index.tz_localize(None)
    
    merged_df = gold_df.join(ext_data, how='left')
    
    # Forward fill any NaNs from non-trading days in external data
    merged_df = merged_df.ffill()
    # Backfill in case the very first days are NaNs
    merged_df = merged_df.bfill()
    
    # Create final features
    final_df = create_features(merged_df)
    
    # Drop rows with NaNs introduced by lagging/rolling
    final_df.dropna(inplace=True)
    
    return final_df
