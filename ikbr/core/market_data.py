"""
Market Data Manager for IBKR Trading System

Handles real-time and historical data management with caching,
normalization, and efficient distribution via the event bus.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any
import time
import numpy as np
from ib_async import *
from loguru import logger

from .event_bus import EventBus, Event, EventTypes, get_event_bus


@dataclass
class MarketDataConfig:
    """Configuration for market data manager"""
    tick_buffer_size: int = 1000
    bar_history_days: int = 30
    cache_ttl_seconds: int = 300
    delayed_data_fallback: bool = True
    snapshot_mode: bool = False  # Use snapshots instead of streaming
    max_concurrent_requests: int = 50
    request_timeout: int = 10


@dataclass
class TickData:
    """Enhanced tick data with metadata"""
    symbol: str
    timestamp: float
    last: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    volume: Optional[int] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread"""
        if self.bid and self.ask:
            return self.ask - self.bid
        return None
    
    @property
    def mid(self) -> Optional[float]:
        """Calculate mid price"""
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.last


class TickBuffer:
    """High-performance circular buffer for tick data"""
    
    def __init__(self, size: int = 1000):
        self.size = size
        self.timestamps = np.zeros(size, dtype=np.float64)
        self.prices = np.zeros(size, dtype=np.float64)
        self.volumes = np.zeros(size, dtype=np.int64)
        self.index = 0
        self.count = 0
    
    def add(self, timestamp: float, price: float, volume: int):
        """Add tick to buffer"""
        self.timestamps[self.index] = timestamp
        self.prices[self.index] = price
        self.volumes[self.index] = volume
        self.index = (self.index + 1) % self.size
        self.count = min(self.count + 1, self.size)
    
    def get_recent(self, n: int) -> tuple:
        """Get most recent n ticks"""
        if n > self.count:
            n = self.count
        
        if self.count < self.size:
            # Buffer not full yet
            return (
                self.timestamps[:n],
                self.prices[:n],
                self.volumes[:n]
            )
        else:
            # Circular buffer logic
            start_idx = (self.index - n) % self.size
            if start_idx < self.index:
                return (
                    self.timestamps[start_idx:self.index],
                    self.prices[start_idx:self.index],
                    self.volumes[start_idx:self.index]
                )
            else:
                return (
                    np.concatenate([self.timestamps[start_idx:], self.timestamps[:self.index]]),
                    np.concatenate([self.prices[start_idx:], self.prices[:self.index]]),
                    np.concatenate([self.volumes[start_idx:], self.volumes[:self.index]])
                )


class MarketDataManager:
    """
    Manages market data subscriptions and distribution
    
    Features:
    - Real-time tick data streaming
    - Historical data caching
    - Automatic reconnection
    - Rate limiting
    - Data normalization
    """
    
    def __init__(self, ib: IB, config: Optional[MarketDataConfig] = None):
        self.ib = ib
        self.config = config or MarketDataConfig()
        self.event_bus = get_event_bus()
        
        # Data storage
        self._tick_buffers: Dict[str, TickBuffer] = {}
        self._latest_ticks: Dict[str, TickData] = {}
        self._bar_cache: Dict[str, List[BarData]] = {}
        self._subscriptions: Dict[Contract, Ticker] = {}
        
        # Rate limiting
        self._request_queue = asyncio.Queue()
        self._request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # State
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self.ib.pendingTickersEvent += self._on_pending_tickers
        
    async def start(self):
        """Start the market data manager"""
        if not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self._process_requests())
            await self.event_bus.start()
            logger.info("MarketDataManager started")
    
    async def stop(self):
        """Stop the market data manager"""
        self._running = False
        
        # Cancel all subscriptions
        for contract, ticker in self._subscriptions.items():
            self.ib.cancelMktData(contract)
        
        self._subscriptions.clear()
        
        if self._worker_task:
            await self._request_queue.put(None)
            await self._worker_task
        
        logger.info("MarketDataManager stopped")
    
    async def subscribe_ticker(self, symbol: str, exchange: str = 'SMART', 
                              currency: str = 'USD') -> bool:
        """
        Subscribe to real-time market data for a symbol
        
        Returns:
            bool: True if subscription successful
        """
        try:
            contract = Stock(symbol, exchange, currency)
            
            # Qualify contract
            self.ib.qualifyContracts(contract)
            
            if contract in self._subscriptions:
                logger.debug(f"Already subscribed to {symbol}")
                return True
            
            # Request market data
            ticker = self.ib.reqMktData(
                contract, 
                genericTickList='', 
                snapshot=self.config.snapshot_mode,
                regulatorySnapshot=False
            )
            
            # Store subscription
            self._subscriptions[contract] = ticker
            
            # Initialize tick buffer
            if symbol not in self._tick_buffers:
                self._tick_buffers[symbol] = TickBuffer(self.config.tick_buffer_size)
            
            logger.info(f"Subscribed to market data for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            return False
    
    async def unsubscribe_ticker(self, symbol: str):
        """Unsubscribe from market data"""
        contract_to_remove = None
        
        for contract, ticker in self._subscriptions.items():
            if contract.symbol == symbol:
                self.ib.cancelMktData(contract)
                contract_to_remove = contract
                break
        
        if contract_to_remove:
            del self._subscriptions[contract_to_remove]
            logger.info(f"Unsubscribed from {symbol}")
    
    async def get_historical_bars(self, symbol: str, duration: str = '1 D',
                                 bar_size: str = '1 min', what_to_show: str = 'TRADES',
                                 use_rth: bool = True) -> Optional[List[BarData]]:
        """
        Get historical bar data with caching
        
        Args:
            symbol: Stock symbol
            duration: Time period (e.g., '1 D', '1 W', '1 M')
            bar_size: Bar size (e.g., '1 min', '5 mins', '1 hour')
            what_to_show: Data type (TRADES, BID, ASK, etc.)
            use_rth: Use regular trading hours only
            
        Returns:
            List of BarData objects or None
        """
        cache_key = f"{symbol}:{duration}:{bar_size}:{what_to_show}"
        
        # Check cache
        if cache_key in self._bar_cache:
            cached_data, cache_time = self._bar_cache[cache_key]
            if time.time() - cache_time < self.config.cache_ttl_seconds:
                logger.debug(f"Using cached data for {symbol}")
                return cached_data
        
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Request historical data
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1
            )
            
            # Cache the result
            self._bar_cache[cache_key] = (bars, time.time())
            
            # Emit event
            await self.event_bus.emit(Event(
                EventTypes.BAR,
                {
                    'symbol': symbol,
                    'bars': bars,
                    'bar_size': bar_size
                }
            ))
            
            return bars
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None
    
    def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """Get latest tick data for a symbol"""
        return self._latest_ticks.get(symbol)
    
    def get_tick_buffer(self, symbol: str) -> Optional[TickBuffer]:
        """Get tick buffer for a symbol"""
        return self._tick_buffers.get(symbol)
    
    def _on_pending_tickers(self, tickers: Set[Ticker]):
        """Handle incoming tick updates"""
        for ticker in tickers:
            if ticker.contract and ticker.contract.symbol:
                symbol = ticker.contract.symbol
                
                # Create tick data
                tick_data = TickData(
                    symbol=symbol,
                    timestamp=time.time(),
                    last=ticker.last if not np.isnan(ticker.last) else None,
                    bid=ticker.bid if not np.isnan(ticker.bid) else None,
                    ask=ticker.ask if not np.isnan(ticker.ask) else None,
                    bid_size=ticker.bidSize if ticker.bidSize > 0 else None,
                    ask_size=ticker.askSize if ticker.askSize > 0 else None,
                    volume=ticker.volume if ticker.volume > 0 else None,
                    high=ticker.high if not np.isnan(ticker.high) else None,
                    low=ticker.low if not np.isnan(ticker.low) else None,
                    close=ticker.close if not np.isnan(ticker.close) else None
                )
                
                # Update latest tick
                self._latest_ticks[symbol] = tick_data
                
                # Add to buffer
                if symbol in self._tick_buffers and tick_data.last:
                    self._tick_buffers[symbol].add(
                        tick_data.timestamp,
                        tick_data.last,
                        tick_data.volume or 0
                    )
                
                # Emit tick event asynchronously
                asyncio.create_task(self._emit_tick_event(tick_data))
    
    async def _emit_tick_event(self, tick_data: TickData):
        """Emit tick event to event bus"""
        await self.event_bus.emit(Event(
            EventTypes.TICK,
            tick_data,
            source="MarketDataManager"
        ))
    
    async def _process_requests(self):
        """Worker to process queued requests with rate limiting"""
        while self._running:
            try:
                request = await self._request_queue.get()
                if request is None:
                    break
                
                async with self._request_semaphore:
                    await request()
                    
            except Exception as e:
                logger.error(f"Error processing request: {e}")
    
    def get_all_subscribed_symbols(self) -> List[str]:
        """Get list of all subscribed symbols"""
        return [contract.symbol for contract in self._subscriptions.keys()]
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        # This is a simplified check - in production you'd want to
        # check actual exchange hours
        now = datetime.now()
        weekday = now.weekday()
        
        # Basic US market hours check (9:30 AM - 4:00 PM ET)
        if weekday >= 5:  # Weekend
            return False
        
        # Convert to ET (simplified - doesn't handle DST properly)
        hour = now.hour
        minute = now.minute
        
        market_open = (hour == 9 and minute >= 30) or (hour > 9 and hour < 16)
        return market_open
    
    async def wait_for_data(self, symbol: str, timeout: float = 5.0) -> bool:
        """
        Wait for market data to be available for a symbol
        
        Args:
            symbol: Stock symbol
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if data received within timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            tick = self.get_latest_tick(symbol)
            if tick and tick.last is not None:
                return True
            await asyncio.sleep(0.1)
        
        return False


# Market data utilities
def calculate_vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Calculate volume-weighted average price"""
    if len(prices) == 0 or len(volumes) == 0:
        return np.nan
    total_volume = np.sum(volumes)
    if total_volume == 0:
        return np.mean(prices)
    return np.sum(prices * volumes) / total_volume


def calculate_spread_metrics(tick_buffer: TickBuffer, window: int = 100) -> Dict[str, float]:
    """Calculate spread statistics from recent ticks"""
    # This is a placeholder - would need bid/ask data in buffer
    return {
        'avg_spread': 0.0,
        'max_spread': 0.0,
        'min_spread': 0.0,
        'spread_volatility': 0.0
    }