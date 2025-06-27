"""
Unit tests for MarketDataManager
"""

import asyncio
import time
import pytest
import pytest_asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ib_async import Stock, Ticker, BarData, IB
from core.market_data import (
    MarketDataManager, MarketDataConfig, TickData, TickBuffer,
    calculate_vwap
)
from core.event_bus import EventBus, Event, EventTypes


class TestTickBuffer:
    """Test the circular tick buffer"""
    
    def test_basic_operations(self):
        """Test basic buffer operations"""
        buffer = TickBuffer(size=5)
        
        # Add some ticks
        for i in range(3):
            buffer.add(timestamp=i, price=100.0 + i, volume=1000 * (i + 1))
        
        assert buffer.count == 3
        assert buffer.index == 3
        
        # Get recent ticks
        timestamps, prices, volumes = buffer.get_recent(2)
        assert len(timestamps) == 2
        assert prices[0] == 100.0
        assert prices[1] == 101.0
        assert volumes[0] == 1000
        assert volumes[1] == 2000
    
    def test_circular_behavior(self):
        """Test circular buffer wrapping"""
        buffer = TickBuffer(size=3)
        
        # Fill buffer beyond capacity
        for i in range(5):
            buffer.add(timestamp=i, price=100.0 + i, volume=1000)
        
        assert buffer.count == 3  # Should not exceed size
        assert buffer.index == 2  # Should wrap around
        
        # Most recent 3 should be indices 2, 3, 4
        timestamps, prices, _ = buffer.get_recent(3)
        assert len(prices) == 3
        assert prices[0] == 102.0  # Index 2
        assert prices[1] == 103.0  # Index 3
        assert prices[2] == 104.0  # Index 4


class TestTickData:
    """Test TickData functionality"""
    
    def test_spread_calculation(self):
        """Test bid-ask spread calculation"""
        tick = TickData(
            symbol="TEST",
            timestamp=time.time(),
            bid=100.0,
            ask=100.05
        )
        assert abs(tick.spread - 0.05) < 0.0001  # Use epsilon comparison for floats
        
        # Test with missing data
        tick2 = TickData(symbol="TEST", timestamp=time.time())
        assert tick2.spread is None
    
    def test_mid_price(self):
        """Test mid price calculation"""
        tick = TickData(
            symbol="TEST",
            timestamp=time.time(),
            bid=100.0,
            ask=100.10,
            last=100.05
        )
        assert tick.mid == 100.05
        
        # Falls back to last price
        tick2 = TickData(
            symbol="TEST",
            timestamp=time.time(),
            last=100.0
        )
        assert tick2.mid == 100.0


class TestMarketDataManager:
    """Test MarketDataManager functionality"""
    
    @pytest_asyncio.fixture
    async def setup(self):
        """Setup test environment"""
        # Mock IB connection
        mock_ib = Mock(spec=IB)
        mock_ib.pendingTickersEvent = Mock()
        mock_ib.qualifyContracts = Mock()
        mock_ib.reqMktData = Mock()
        mock_ib.cancelMktData = Mock()
        mock_ib.reqHistoricalDataAsync = AsyncMock()
        
        # Create event bus
        event_bus = EventBus("test")
        await event_bus.start()
        
        # Create manager
        config = MarketDataConfig(
            tick_buffer_size=100,
            cache_ttl_seconds=5
        )
        manager = MarketDataManager(mock_ib, config)
        
        yield {
            'mock_ib': mock_ib,
            'event_bus': event_bus,
            'manager': manager
        }
        
        # Cleanup
        await manager.stop()
        await event_bus.stop()
    
    @pytest.mark.asyncio
    async def test_subscribe_ticker(self, setup):
        """Test subscribing to market data"""
        manager = setup['manager']
        mock_ib = setup['mock_ib']
        
        # Mock ticker
        mock_ticker = Mock(spec=Ticker)
        mock_ticker.contract = Stock('SPY', 'SMART', 'USD')
        mock_ib.reqMktData.return_value = mock_ticker
        
        await manager.start()
        
        # Subscribe to ticker
        result = await manager.subscribe_ticker('SPY')
        assert result is True
        
        # Verify IB calls
        mock_ib.qualifyContracts.assert_called_once()
        mock_ib.reqMktData.assert_called_once()
        
        # Check subscription stored
        assert len(manager._subscriptions) == 1
        assert 'SPY' in manager._tick_buffers
    
    @pytest.mark.asyncio
    async def test_duplicate_subscription(self, setup):
        """Test duplicate subscription handling"""
        manager = setup['manager']
        mock_ib = setup['mock_ib']
        
        mock_ticker = Mock(spec=Ticker)
        mock_ticker.contract = Stock('AAPL', 'SMART', 'USD')
        mock_ib.reqMktData.return_value = mock_ticker
        
        await manager.start()
        
        # Subscribe twice
        await manager.subscribe_ticker('AAPL')
        await manager.subscribe_ticker('AAPL')
        
        # Should only call once
        assert mock_ib.reqMktData.call_count == 1
    
    @pytest.mark.asyncio
    async def test_unsubscribe_ticker(self, setup):
        """Test unsubscribing from market data"""
        manager = setup['manager']
        mock_ib = setup['mock_ib']
        
        # Setup subscription
        contract = Stock('TSLA', 'SMART', 'USD')
        mock_ticker = Mock(spec=Ticker)
        mock_ticker.contract = contract
        mock_ib.reqMktData.return_value = mock_ticker
        
        await manager.start()
        await manager.subscribe_ticker('TSLA')
        
        # Unsubscribe
        await manager.unsubscribe_ticker('TSLA')
        
        # Verify cancellation
        mock_ib.cancelMktData.assert_called_once()
        assert len(manager._subscriptions) == 0
    
    @pytest.mark.asyncio
    async def test_tick_processing(self, setup):
        """Test processing incoming ticks"""
        manager = setup['manager']
        mock_ib = setup['mock_ib']
        
        await manager.start()
        
        # Create mock ticker with data
        mock_ticker = Mock(spec=Ticker)
        mock_ticker.contract = Stock('IBM', 'SMART', 'USD')
        mock_ticker.contract.symbol = 'IBM'
        mock_ticker.last = 150.0
        mock_ticker.bid = 149.95
        mock_ticker.ask = 150.05
        mock_ticker.bidSize = 100
        mock_ticker.askSize = 200
        mock_ticker.volume = 1000000
        mock_ticker.high = 151.0
        mock_ticker.low = 149.0
        mock_ticker.close = 149.50
        
        # Subscribe first
        mock_ib.reqMktData.return_value = mock_ticker
        await manager.subscribe_ticker('IBM')
        
        # Simulate tick update
        manager._on_pending_tickers({mock_ticker})
        
        # Give time for async processing
        await asyncio.sleep(0.1)
        
        # Check tick was processed
        latest_tick = manager.get_latest_tick('IBM')
        assert latest_tick is not None
        assert latest_tick.symbol == 'IBM'
        assert latest_tick.last == 150.0
        assert latest_tick.bid == 149.95
        assert latest_tick.ask == 150.05
        assert latest_tick.spread == 0.10
        
        # Check buffer updated
        buffer = manager.get_tick_buffer('IBM')
        assert buffer is not None
        assert buffer.count == 1
    
    @pytest.mark.asyncio
    async def test_historical_data_request(self, setup):
        """Test historical data request with caching"""
        manager = setup['manager']
        mock_ib = setup['mock_ib']
        
        # Mock historical data
        mock_bars = [
            Mock(date='2024-01-01', open=100, high=101, low=99, close=100.5, volume=1000),
            Mock(date='2024-01-02', open=100.5, high=102, low=100, close=101.5, volume=1500)
        ]
        mock_ib.reqHistoricalDataAsync.return_value = mock_bars
        
        await manager.start()
        
        # First request
        bars1 = await manager.get_historical_bars('SPY', '1 D', '1 hour')
        assert bars1 == mock_bars
        assert mock_ib.reqHistoricalDataAsync.call_count == 1
        
        # Second request should use cache
        bars2 = await manager.get_historical_bars('SPY', '1 D', '1 hour')
        assert bars2 == mock_bars
        assert mock_ib.reqHistoricalDataAsync.call_count == 1  # No additional call
        
        # Different parameters should make new request
        bars3 = await manager.get_historical_bars('SPY', '2 D', '1 hour')
        assert mock_ib.reqHistoricalDataAsync.call_count == 2
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, setup):
        """Test cache TTL expiration"""
        manager = setup['manager']
        mock_ib = setup['mock_ib']
        
        # Set short TTL
        manager.config.cache_ttl_seconds = 0.1
        
        mock_bars = [Mock()]
        mock_ib.reqHistoricalDataAsync.return_value = mock_bars
        
        await manager.start()
        
        # First request
        await manager.get_historical_bars('AAPL', '1 D', '1 min')
        assert mock_ib.reqHistoricalDataAsync.call_count == 1
        
        # Wait for cache to expire
        await asyncio.sleep(0.2)
        
        # Second request should fetch again
        await manager.get_historical_bars('AAPL', '1 D', '1 min')
        assert mock_ib.reqHistoricalDataAsync.call_count == 2
    
    @pytest.mark.asyncio
    async def test_wait_for_data(self, setup):
        """Test waiting for data availability"""
        manager = setup['manager']
        
        await manager.start()
        
        # Start wait task
        wait_task = asyncio.create_task(
            manager.wait_for_data('TEST', timeout=1.0)
        )
        
        # Simulate data arrival after delay
        await asyncio.sleep(0.1)
        manager._latest_ticks['TEST'] = TickData(
            symbol='TEST',
            timestamp=time.time(),
            last=100.0
        )
        
        # Should return True
        result = await wait_task
        assert result is True
        
        # Test timeout
        result2 = await manager.wait_for_data('NODATA', timeout=0.1)
        assert result2 is False
    
    def test_market_open_check(self, setup):
        """Test market open checking"""
        manager = setup['manager']
        
        # This test would need to be more sophisticated in production
        # For now just verify the method exists and returns a bool
        is_open = manager.is_market_open()
        assert isinstance(is_open, bool)
    
    @pytest.mark.asyncio
    async def test_event_emission(self, setup):
        """Test event emission on tick updates"""
        manager = setup['manager']
        mock_ib = setup['mock_ib']
        event_bus = setup['event_bus']
        
        # Track emitted events
        received_events = []
        
        async def event_handler(event: Event):
            received_events.append(event)
        
        event_bus.subscribe(EventTypes.TICK, event_handler)
        
        await manager.start()
        
        # Create mock ticker
        mock_ticker = Mock(spec=Ticker)
        mock_ticker.contract = Stock('MSFT', 'SMART', 'USD')
        mock_ticker.contract.symbol = 'MSFT'
        mock_ticker.last = 400.0
        mock_ticker.bid = 399.95
        mock_ticker.ask = 400.05
        mock_ticker.bidSize = 100
        mock_ticker.askSize = 100
        mock_ticker.volume = 5000000
        mock_ticker.high = float('nan')
        mock_ticker.low = float('nan')
        mock_ticker.close = float('nan')
        
        # Subscribe and trigger tick
        mock_ib.reqMktData.return_value = mock_ticker
        await manager.subscribe_ticker('MSFT')
        manager._on_pending_tickers({mock_ticker})
        
        # Wait for async processing
        await asyncio.sleep(0.1)
        
        # Check event was emitted
        assert len(received_events) == 1
        event = received_events[0]
        assert event.event_type == EventTypes.TICK
        assert isinstance(event.data, TickData)
        assert event.data.symbol == 'MSFT'
        assert event.data.last == 400.0


class TestMarketDataUtilities:
    """Test utility functions"""
    
    def test_calculate_vwap(self):
        """Test VWAP calculation"""
        prices = np.array([100.0, 101.0, 102.0])
        volumes = np.array([1000, 2000, 3000])
        
        vwap = calculate_vwap(prices, volumes)
        expected = (100*1000 + 101*2000 + 102*3000) / 6000
        assert abs(vwap - expected) < 0.01
        
        # Test edge cases
        assert np.isnan(calculate_vwap(np.array([]), np.array([])))
        assert calculate_vwap(prices, np.zeros(3)) == np.mean(prices)


class TestMarketDataIntegration:
    """Integration tests with real IB connection (skipped by default)"""
    
    @pytest.mark.skip(reason="Requires live IB connection")
    @pytest.mark.asyncio
    async def test_real_connection(self):
        """Test with real IB Gateway connection"""
        ib = IB()
        ib.connect('localhost', 4002, clientId=99)
        
        try:
            config = MarketDataConfig()
            manager = MarketDataManager(ib, config)
            await manager.start()
            
            # Subscribe to SPY
            success = await manager.subscribe_ticker('SPY')
            assert success
            
            # Wait for data
            got_data = await manager.wait_for_data('SPY', timeout=10.0)
            assert got_data
            
            # Check tick
            tick = manager.get_latest_tick('SPY')
            assert tick is not None
            assert tick.symbol == 'SPY'
            
            # Get historical data
            bars = await manager.get_historical_bars('SPY', '1 D', '5 mins')
            assert bars is not None
            assert len(bars) > 0
            
        finally:
            await manager.stop()
            ib.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])