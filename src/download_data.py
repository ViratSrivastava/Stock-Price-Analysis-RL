import yfinance as yf
import os
import datetime
import pandas as pd
import sys
import time
import random
import numpy as np

# Get project root directory (handle running from src/ or project root)
def get_project_root():
    """Get absolute path to project root directory"""
    script_path = os.path.abspath(__file__)
    if os.path.basename(os.path.dirname(script_path)) == 'src':
        return os.path.dirname(os.path.dirname(script_path))
    return os.path.dirname(script_path)

# Create data directory using absolute path
data_dir = os.path.join(get_project_root(), "data")
os.makedirs(data_dir, exist_ok=True)

# Symbols to download
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']

def verify_and_fix_data(file_path):
    """Verify the CSV has the required columns for the model and fix if needed"""
    expected_columns = ['open', 'high', 'low', 'close', 'volume']
    try:
        df = pd.read_csv(file_path)
        
        # If 'Date' is a column and not the index, set it as the index
        if 'Date' in df.columns:
            df = df.set_index('Date')
            
        # Case insensitive check - map all columns to lowercase for comparison
        column_mapping = {}
        found_columns = []
        
        # Match columns case-insensitively
        for col in df.columns:
            col_lower = col.lower()
            for expected in expected_columns:
                if expected in col_lower or col_lower in expected:
                    column_mapping[col] = expected
                    found_columns.append(expected)
                    break
        
        # Check if we found all required columns
        missing = [col for col in expected_columns if col not in found_columns]
        if missing:
            return False, f"Missing columns after mapping: {missing}"
            
        # Rename columns to expected lowercase format
        df = df.rename(columns=column_mapping)
        
        # Save fixed data
        df.to_csv(file_path)
        return True, len(df)
        
    except Exception as e:
        return False, f"Error: {e}"

def download_with_retry(sym, start=None, interval=None, max_retries=3):
    """Download data with retry logic"""
    for attempt in range(max_retries):
        try:
            if interval:
                df = yf.download(sym, start=start, interval=interval, progress=False)
            else:
                df = yf.download(sym, start=start, progress=False)
                
            if not df.empty:
                return df
                
        except Exception as e:
            print(f"  Attempt {attempt+1}/{max_retries} failed: {e}")
            time.sleep(2 + random.uniform(0, 2))  # Backoff with jitter
    
    # All attempts failed, return empty DataFrame
    return pd.DataFrame()

def generate_dummy_data(symbol, is_intraday=False):
    """Generate dummy stock data if download fails"""
    print(f"  Generating synthetic data for {symbol}...")
    
    if is_intraday:
        # Create 7 days of 5-minute data (1008 rows)
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=7)
        index = pd.date_range(start=start, end=end, freq='5min')
        rows = len(index)
    else:
        # Create ~5 years of daily data
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=5*365)
        index = pd.date_range(start=start, end=end, freq='D')
        rows = len(index)
    
    # Generate plausible stock data with random walk
    base_price = random.uniform(50, 500)  # Random starting price
    
    np.random.seed(sum(ord(c) for c in symbol))  # Seed based on symbol name
    
    # Create price series with random walk
    daily_returns = np.random.normal(0.0005, 0.015, rows).cumsum()
    prices = base_price * (1 + daily_returns)
    
    # Generate OHLC data
    data = {
        'open': prices * np.random.uniform(0.99, 1.01, rows),
        'high': prices * np.random.uniform(1.01, 1.03, rows),
        'low': prices * np.random.uniform(0.97, 0.99, rows),
        'close': prices,
        'volume': np.random.randint(100000, 10000000, rows),
    }
    
    # Ensure high is highest, low is lowest
    for i in range(rows):
        data['high'][i] = max(data['open'][i], data['close'][i], data['high'][i])
        data['low'][i] = min(data['open'][i], data['close'][i], data['low'][i])
    
    df = pd.DataFrame(data, index=index)
    return df

# 1. INTRADAY DATA - Short timeframe (7 days maximum for 5m data)
def download_intraday():
    interval = '5m'
    # For 5m data, Yahoo only allows ~7 days of history
    start_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    
    for sym in symbols:
        print(f"Downloading intraday data for {sym} ({interval} interval)...")
        
        # Try downloading data
        df = download_with_retry(sym, start=start_date, interval=interval)
        
        if df.empty:
            print(f"⚠️ Download failed for {sym}, generating synthetic data")
            df = generate_dummy_data(sym, is_intraday=True)
        
        # Ensure column names are lowercase
        df.columns = [col.lower() if isinstance(col, str) else 
                     (col[1].lower() if isinstance(col, tuple) and len(col) > 1 else 'unknown')
                     for col in df.columns]
        
        # Save to data folder
        file_path = os.path.join(data_dir, f"{sym}_intraday_{interval}.csv")
        df.to_csv(file_path)
        
        # Verify data is usable
        valid, details = verify_and_fix_data(file_path)
        if valid:
            print(f"✓ Saved {file_path} with {details} rows (verified)")
        else:
            print(f"⚠️ Data may not be compatible: {details}")

# 2. DAILY DATA - Long timeframe (multiple years)
def download_daily():
    for sym in symbols:
        print(f"Downloading daily data for {sym}...")
        
        # Try downloading data
        df = download_with_retry(sym, start="2015-01-01")
        
        if df.empty:
            print(f"⚠️ Download failed for {sym}, generating synthetic data")
            df = generate_dummy_data(sym, is_intraday=False)
        
        # Ensure column names are lowercase
        df.columns = [col.lower() if isinstance(col, str) else 
                     (col[1].lower() if isinstance(col, tuple) and len(col) > 1 else 'unknown')
                     for col in df.columns]
        
        # Save to data folder
        file_path = os.path.join(data_dir, f"{sym}_daily.csv")
        df.to_csv(file_path)
        
        # Verify data is usable
        valid, details = verify_and_fix_data(file_path)
        if valid:
            print(f"✓ Saved {file_path} with {details} rows (verified)")
        else:
            print(f"⚠️ Data may not be compatible: {details}")

if __name__ == "__main__":
    # Allow command line arguments to select which data to download
    if len(sys.argv) > 1 and sys.argv[1] == "intraday":
        print("=== Downloading Intraday Data (5m) ===")
        download_intraday()
    elif len(sys.argv) > 1 and sys.argv[1] == "daily":
        print("=== Downloading Daily Data ===")
        download_daily()
    else:
        # Default: download both
        print("=== Downloading Intraday Data (5m) ===")
        download_intraday()
        print("\n=== Downloading Daily Data ===")
        download_daily()
    
    print("\nData download complete!")
    print(f"Files saved to: {data_dir}")
    print("\nTo use with your model:")
    print("  python src/main.py train --symbol AAPL --episodes 1000")