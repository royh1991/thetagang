"""
IB Data Provider for Backtesting

Fetches real historical data from Interactive Brokers for backtesting.
Implements caching to avoid repeated API calls.
"""

import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import pandas as pd
from loguru import logger
from ib_async import IB, Stock, Contract, BarData
import asyncio
from pathlib import Path

from core.market_data import TickData


@dataclass
class HistoricalDataRequest:
    """Request for historical data"""
    symbol: str
    start_date: datetime
    end_date: datetime
    bar_size: str = "1 min"
    what_to_show: str = "TRADES"
    use_rth: bool = True
    
    def cache_key(self) -> str:
        """Generate unique cache key"""
        return f"{self.symbol}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}_{self.bar_size.replace(' ', '_')}_{self.what_to_show}"


class IBDataProvider:
    """
    Fetches historical data from Interactive Brokers
    
    Features:
    - Real historical data from IB
    - Local caching to minimize API calls
    - Automatic connection management
    - Bar to tick conversion for backtesting
    """
    
    def __init__(self, cache_dir: str = "backtest/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ib: Optional[IB] = None
        self._contracts_cache: Dict[str, Contract] = {}
        
    async def connect(self, host: str = "localhost", port: int = 4102, client_id: int = 99):
        """Connect to IB Gateway"""
        if self.ib and self.ib.isConnected():
            return
            
        self.ib = IB()
        try:
            await self.ib.connectAsync(host, port, clientId=client_id)
            logger.info(f"Connected to IB Gateway at {host}:{port}")
            
            # Request market data type
            self.ib.reqMarketDataType(4)  # Delayed frozen data for backtesting
            
        except Exception as e:
            logger.error(f"Failed to connect to IB Gateway: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from IB Gateway"""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB Gateway")
    
    async def get_contract(self, symbol: str) -> Contract:
        """Get or create contract for symbol"""
        if symbol in self._contracts_cache:
            return self._contracts_cache[symbol]
        
        # Create stock contract
        contract = Stock(symbol, 'SMART', 'USD')
        
        # Qualify contract to get conId
        await self.ib.qualifyContractsAsync(contract)
        
        if not contract.conId:
            raise ValueError(f"Failed to qualify contract for {symbol}")
        
        self._contracts_cache[symbol] = contract
        return contract
    
    async def fetch_historical_data(self, request: HistoricalDataRequest) -> pd.DataFrame:
        """
        Fetch historical data from IB
        
        Returns DataFrame with columns: date, open, high, low, close, volume
        """
        # Check cache first
        cache_file = self.cache_dir / f"{request.cache_key()}.pkl"
        if cache_file.exists():
            logger.info(f"Loading cached data for {request.symbol}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Connect if needed
        if not self.ib or not self.ib.isConnected():
            await self.connect()
        
        # Get contract
        contract = await self.get_contract(request.symbol)
        
        # Calculate duration
        duration_days = (request.end_date - request.start_date).days
        if duration_days <= 1:
            duration_str = f"{duration_days + 1} D"
        elif duration_days <= 30:
            duration_str = f"{duration_days} D"
        elif duration_days <= 365:
            duration_str = f"{duration_days // 30} M"
        else:
            duration_str = f"{duration_days // 365} Y"
        
        logger.info(f"Fetching {duration_str} of {request.bar_size} bars for {request.symbol}")
        
        try:
            # Fetch data from IB
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime=request.end_date,
                durationStr=duration_str,
                barSizeSetting=request.bar_size,
                whatToShow=request.what_to_show,
                useRTH=request.use_rth,
                formatDate=1  # Return as datetime
            )
            
            if not bars:
                logger.warning(f"No data returned for {request.symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'date': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'average': bar.average,
                'barCount': bar.barCount
            } for bar in bars])
            
            # Set date as index
            df.set_index('date', inplace=True)
            
            # Cache the data
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            logger.info(f"Cached {len(df)} bars for {request.symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {request.symbol}: {e}")
            raise
    
    async def fetch_multiple_symbols(self, symbols: List[str], 
                                   start_date: datetime, 
                                   end_date: datetime,
                                   bar_size: str = "1 min") -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols"""
        data = {}
        
        for symbol in symbols:
            try:
                request = HistoricalDataRequest(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    bar_size=bar_size
                )
                df = await self.fetch_historical_data(request)
                if not df.empty:
                    data[symbol] = df
                    
                # Rate limiting - IB has restrictions
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        return data
    
    def bars_to_ticks(self, bars_df: pd.DataFrame, symbol: str) -> List[TickData]:
        """
        Convert bar data to tick data for backtesting
        
        Uses OHLC to generate realistic tick sequence:
        - Open tick at bar start
        - Low/High ticks (order depends on close position)
        - Close tick at bar end
        """
        ticks = []
        
        for timestamp, bar in bars_df.iterrows():
            # Determine if bar moved up or down
            up_bar = bar['close'] >= bar['open']
            
            # Generate tick sequence
            # Always start with open
            ticks.append(TickData(
                symbol=symbol,
                timestamp=timestamp.timestamp() if hasattr(timestamp, 'timestamp') else float(timestamp),
                bid=bar['open'] - 0.01,  # Simulate spread
                ask=bar['open'] + 0.01,
                last=bar['open'],
                bid_size=100,
                ask_size=100,
                volume=int(bar['volume'] / 4) if bar['volume'] > 0 else 100,
                high=bar['high'],
                low=bar['low'],
                close=bar['close']
            ))
            
            # Add high/low in realistic order
            if up_bar:
                # For up bars: low then high
                if bar['low'] < bar['open']:
                    ticks.append(TickData(
                        symbol=symbol,
                        timestamp=(timestamp + timedelta(seconds=15)).timestamp() if hasattr(timestamp, 'timestamp') else float(timestamp) + 15,
                        bid=bar['low'] - 0.01,
                        ask=bar['low'] + 0.01,
                        last=bar['low'],
                        bid_size=100,
                        ask_size=100,
                        volume=int(bar['volume'] / 4) if bar['volume'] > 0 else 100,
                        high=bar['high'],
                        low=bar['low'],
                        close=bar['close']
                    ))
                
                if bar['high'] > bar['open']:
                    ticks.append(TickData(
                        symbol=symbol,
                        timestamp=(timestamp + timedelta(seconds=30)).timestamp() if hasattr(timestamp, 'timestamp') else float(timestamp) + 30,
                        bid=bar['high'] - 0.01,
                        ask=bar['high'] + 0.01,
                        last=bar['high'],
                        bid_size=100,
                        ask_size=100,
                        volume=int(bar['volume'] / 4) if bar['volume'] > 0 else 100,
                        high=bar['high'],
                        low=bar['low'],
                        close=bar['close']
                    ))
            else:
                # For down bars: high then low
                if bar['high'] > bar['open']:
                    ticks.append(TickData(
                        symbol=symbol,
                        timestamp=(timestamp + timedelta(seconds=15)).timestamp() if hasattr(timestamp, 'timestamp') else float(timestamp) + 15,
                        bid=bar['high'] - 0.01,
                        ask=bar['high'] + 0.01,
                        last=bar['high'],
                        bid_size=100,
                        ask_size=100,
                        volume=int(bar['volume'] / 4) if bar['volume'] > 0 else 100,
                        high=bar['high'],
                        low=bar['low'],
                        close=bar['close']
                    ))
                
                if bar['low'] < bar['open']:
                    ticks.append(TickData(
                        symbol=symbol,
                        timestamp=(timestamp + timedelta(seconds=30)).timestamp() if hasattr(timestamp, 'timestamp') else float(timestamp) + 30,
                        bid=bar['low'] - 0.01,
                        ask=bar['low'] + 0.01,
                        last=bar['low'],
                        bid_size=100,
                        ask_size=100,
                        volume=int(bar['volume'] / 4) if bar['volume'] > 0 else 100,
                        high=bar['high'],
                        low=bar['low'],
                        close=bar['close']
                    ))
            
            # Always end with close
            if bar['close'] != bar['open']:
                ticks.append(TickData(
                    symbol=symbol,
                    timestamp=(timestamp + timedelta(seconds=45)).timestamp() if hasattr(timestamp, 'timestamp') else float(timestamp) + 45,
                    bid=bar['close'] - 0.01,
                    ask=bar['close'] + 0.01,
                    last=bar['close'],
                    bid_size=100,
                    ask_size=100,
                    volume=int(bar['volume'] / 4) if bar['volume'] > 0 else 100,
                    high=bar['high'],
                    low=bar['low'],
                    close=bar['close']
                ))
        
        return ticks
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cache for symbol or all symbols"""
        if symbol:
            # Clear specific symbol
            for cache_file in self.cache_dir.glob(f"{symbol}_*.pkl"):
                cache_file.unlink()
                logger.info(f"Cleared cache for {symbol}")
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cleared all cache")


# Standalone data fetching function for testing
async def fetch_sample_data():
    """Fetch sample data for testing"""
    provider = IBDataProvider()
    
    try:
        await provider.connect()
        
        # Fetch last 5 days of SPY data
        request = HistoricalDataRequest(
            symbol="SPY",
            start_date=datetime.now() - timedelta(days=5),
            end_date=datetime.now(),
            bar_size="5 mins"
        )
        
        df = await provider.fetch_historical_data(request)
        print(f"Fetched {len(df)} bars")
        print(df.tail())
        
        # Convert to ticks
        ticks = provider.bars_to_ticks(df, "SPY")
        print(f"Generated {len(ticks)} ticks")
        
    finally:
        await provider.disconnect()


if __name__ == "__main__":
    # Test the data provider
    asyncio.run(fetch_sample_data())