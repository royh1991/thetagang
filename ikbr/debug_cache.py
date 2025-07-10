#!/usr/bin/env python3
"""Debug cached data to see date ranges"""

import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime

cache_dir = Path("backtest/cache")

# Check AAPL cache files
aapl_files = sorted(cache_dir.glob("AAPL_*.pkl"))

print("AAPL Cache Files:")
print("-" * 80)

for cache_file in aapl_files:
    print(f"\nFile: {cache_file.name}")
    
    # Parse dates from filename
    parts = cache_file.stem.split('_')
    start_date = parts[1]
    end_date = parts[2]
    
    print(f"Filename dates: {start_date} to {end_date}")
    
    # Load and check actual data
    try:
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        
        if not df.empty:
            actual_start = df.index.min()
            actual_end = df.index.max()
            print(f"Actual data dates: {actual_start} to {actual_end}")
            print(f"Number of bars: {len(df)}")
            print(f"First few bars:")
            print(df.head(3))
            print(f"Last few bars:")
            print(df.tail(3))
        else:
            print("DataFrame is empty!")
    except Exception as e:
        print(f"Error loading file: {e}")
    
    print("-" * 80)

# Now check what would be requested for 10-day and 100-day backtests
print("\nExpected cache keys for backtests:")
print("-" * 80)

# Today is 2025-07-06 based on env
today = datetime(2025, 7, 6)  # Adjust based on actual date

# 10-day backtest
start_10 = today - pd.Timedelta(days=10)
print(f"10-day backtest: {start_10.strftime('%Y%m%d')} to {today.strftime('%Y%m%d')}")
expected_10 = f"AAPL_{start_10.strftime('%Y%m%d')}_{today.strftime('%Y%m%d')}_5_mins_TRADES.pkl"
print(f"Expected cache file: {expected_10}")
print(f"Exists: {(cache_dir / expected_10).exists()}")

# 100-day backtest  
start_100 = today - pd.Timedelta(days=100)
print(f"\n100-day backtest: {start_100.strftime('%Y%m%d')} to {today.strftime('%Y%m%d')}")
expected_100 = f"AAPL_{start_100.strftime('%Y%m%d')}_{today.strftime('%Y%m%d')}_5_mins_TRADES.pkl"
print(f"Expected cache file: {expected_100}")
print(f"Exists: {(cache_dir / expected_100).exists()}")