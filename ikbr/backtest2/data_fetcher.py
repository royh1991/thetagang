"""
Simple data fetcher for IB historical data
"""
import os
import csv
from datetime import datetime, timedelta
from typing import List, Dict
from ib_async import IB, Stock, util
import pandas as pd
from loguru import logger


class DataFetcher:
    """Fetches historical data from IB and caches it locally"""
    
    def __init__(self, cache_dir: str = "backtest2/data_cache"):
        self.cache_dir = cache_dir
        self.ib = None
        os.makedirs(cache_dir, exist_ok=True)
    
    def connect(self, host: str = 'localhost', port: int = 4102, client_id: int = 1):
        """Connect to IB Gateway"""
        if self.ib and self.ib.isConnected():
            return
        
        self.ib = IB()
        self.ib.connect(host, port, clientId=client_id)
        logger.info(f"Connected to IB Gateway at {host}:{port}")
    
    def disconnect(self):
        """Disconnect from IB Gateway"""
        if self.ib:
            self.ib.disconnect()
            logger.info("Disconnected from IB Gateway")
    
    def fetch_historical_data(self, symbol: str, days: int, bar_size: str = '5 mins') -> pd.DataFrame:
        """
        Fetch historical data from IB
        
        Args:
            symbol: Stock symbol (e.g., 'SPY')
            days: Number of days of history
            bar_size: Bar size ('1 min', '5 mins', '1 hour', '1 day')
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Check cache first
        cache_file = self._get_cache_filename(symbol, days, bar_size)
        if os.path.exists(cache_file):
            logger.info(f"Loading {symbol} data from cache: {cache_file}")
            return pd.read_csv(cache_file, parse_dates=['timestamp'])
        
        # Connect if needed
        if not self.ib or not self.ib.isConnected():
            self.connect()
        
        # Create contract
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        
        # Fetch data
        end_datetime = ''  # Use current time
        duration_str = f'{days} D'
        
        logger.info(f"Fetching {days} days of {bar_size} bars for {symbol}")
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime,
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True,  # Regular trading hours only
            formatDate=1
        )
        
        if not bars:
            logger.error(f"No data received for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = util.df(bars)
        df = df.rename(columns={'date': 'timestamp'})
        
        # Save to cache
        df.to_csv(cache_file, index=False)
        logger.info(f"Saved {len(df)} bars to cache: {cache_file}")
        
        return df
    
    def _get_cache_filename(self, symbol: str, days: int, bar_size: str) -> str:
        """Generate cache filename"""
        bar_size_clean = bar_size.replace(' ', '_')
        today = datetime.now().strftime('%Y%m%d')
        return os.path.join(self.cache_dir, f"{symbol}_{days}d_{bar_size_clean}_{today}.csv")
    
    def clear_cache(self):
        """Clear all cached data files"""
        import glob
        files = glob.glob(os.path.join(self.cache_dir, "*.csv"))
        for f in files:
            os.remove(f)
        logger.info(f"Cleared {len(files)} cache files")


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = DataFetcher()
    try:
        # Fetch SPY data
        df = fetcher.fetch_historical_data('SPY', days=5, bar_size='5 mins')
        print(f"\nFetched {len(df)} bars")
        print("\nFirst few rows:")
        print(df.head())
        print("\nLast few rows:")
        print(df.tail())
        print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    finally:
        fetcher.disconnect()