"""
Backtest Data Provider

Handles loading and serving historical data for backtesting.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger

from core.market_data import TickData


class BacktestDataProvider:
    """
    Provides historical data for backtesting
    
    Can load data from:
    - CSV files
    - IB historical data
    - Cached data
    """
    
    def __init__(self, config):
        self.config = config
        self.data: Dict[str, pd.DataFrame] = {}  # symbol -> DataFrame
        self.current_index: int = 0
        
    async def load_data(self, symbols: List[str], 
                       start_date: datetime, 
                       end_date: datetime):
        """Load historical data for symbols"""
        for symbol in symbols:
            logger.info(f"Loading data for {symbol}")
            
            # For now, generate synthetic data
            # In production, this would load from IB or files
            df = self._generate_synthetic_data(symbol, start_date, end_date)
            self.data[symbol] = df
            
        logger.info(f"Loaded data for {len(symbols)} symbols")
    
    def get_tick_data(self, timestamp: datetime) -> Dict[str, TickData]:
        """Get tick data for all symbols at given timestamp"""
        tick_data = {}
        
        for symbol, df in self.data.items():
            # Find data for timestamp
            try:
                # Get the row closest to the timestamp
                idx = df.index.get_indexer([timestamp], method='nearest')[0]
                if idx >= 0 and idx < len(df):
                    row = df.iloc[idx]
                    
                    # Create tick data
                    tick = TickData(
                        symbol=symbol,
                        timestamp=timestamp.timestamp(),
                        last=row['close'],
                        bid=row['close'] * 0.9995,  # Approximate bid
                        ask=row['close'] * 1.0005,  # Approximate ask
                        bid_size=100,
                        ask_size=100,
                        volume=int(row['volume']),
                        high=row['high'],
                        low=row['low'],
                        close=row['close']
                    )
                    tick_data[symbol] = tick
            except Exception as e:
                logger.debug(f"No data for {symbol} at {timestamp}: {e}")
        
        return tick_data
    
    def _generate_synthetic_data(self, symbol: str, 
                                start_date: datetime, 
                                end_date: datetime) -> pd.DataFrame:
        """Generate synthetic price data for testing"""
        # Create datetime index based on frequency
        if self.config.data_frequency == "1min":
            freq = "1T"
        elif self.config.data_frequency == "5min":
            freq = "5T"
        elif self.config.data_frequency == "1hour":
            freq = "1H"
        else:
            freq = "1D"
        
        # Generate business day timestamps
        timestamps = pd.date_range(
            start=start_date,
            end=end_date,
            freq=freq
        )
        
        # Filter for market hours (9:30 AM - 4:00 PM ET)
        timestamps = timestamps[
            (timestamps.hour >= 9) & 
            (timestamps.hour < 16) &
            (timestamps.weekday < 5)  # Weekdays only
        ]
        if len(timestamps) > 0 and timestamps[0].hour == 9 and timestamps[0].minute < 30:
            timestamps = timestamps[1:]
        
        # Generate random walk prices
        num_points = len(timestamps)
        
        # Different characteristics for different symbols
        if symbol == "SPY":
            initial_price = 450.0
            volatility = 0.001  # 0.1% per period
            trend = 0.00001  # Slight upward trend
        elif symbol == "AAPL":
            initial_price = 175.0
            volatility = 0.0015
            trend = 0.00002
        elif symbol == "TSLA":
            initial_price = 850.0
            volatility = 0.003  # Higher volatility
            trend = -0.00001  # Slight downward trend
        else:
            initial_price = 100.0
            volatility = 0.002
            trend = 0.0
        
        # Generate returns
        returns = np.random.normal(trend, volatility, num_points)
        returns[0] = 0
        
        # Calculate prices
        price_series = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC data
        data = []
        for i, (ts, price) in enumerate(zip(timestamps, price_series)):
            # Add some intrabar movement
            high_low_range = price * volatility * 2
            high = price + np.random.uniform(0, high_low_range)
            low = price - np.random.uniform(0, high_low_range)
            
            # Ensure open is between high and low
            if i > 0:
                open_price = price_series[i-1] + np.random.uniform(-volatility, volatility) * price
            else:
                open_price = price
            
            open_price = max(low, min(high, open_price))
            
            # Volume with some randomness
            base_volume = 1000000 if symbol == "SPY" else 500000
            volume = int(base_volume * (1 + np.random.uniform(-0.5, 0.5)))
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=timestamps)
        return df
    
    def save_cache(self, filepath: str):
        """Save loaded data to cache file"""
        # Combine all dataframes
        combined = {}
        for symbol, df in self.data.items():
            combined[symbol] = df.to_dict()
        
        # Save to file (would use pickle or parquet in production)
        logger.info(f"Saved data cache to {filepath}")
    
    def load_cache(self, filepath: str) -> bool:
        """Load data from cache file"""
        # Load from file (would use pickle or parquet in production)
        logger.info(f"Loading data cache from {filepath}")
        return False  # Not implemented for this example