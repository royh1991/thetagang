"""
Unit tests for Order Manager
"""

import pytest
import asyncio
import time
import uuid
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from ib_async import IB, Stock, Order, Trade, OrderStatus as IBOrderStatus, Fill, CommissionReport

from core.order_manager import (
    OrderManager, OrderConfig, Signal, OrderInfo, OrderStatus,
    calculate_order_size, validate_signal
)
from core.event_bus import EventBus, Event, EventTypes


@pytest.fixture
async def mock_ib():
    """Create a mock IB connection"""
    mock = Mock(spec=IB)
    mock.orderStatusEvent = Mock()
    mock.orderStatusEvent.__iadd__ = Mock(return_value=mock.orderStatusEvent)
    mock.execDetailsEvent = Mock()
    mock.execDetailsEvent.__iadd__ = Mock(return_value=mock.execDetailsEvent)
    mock.errorEvent = Mock()
    mock.errorEvent.__iadd__ = Mock(return_value=mock.errorEvent)
    mock.newOrderEvent = Mock()
    mock.newOrderEvent.__iadd__ = Mock(return_value=mock.newOrderEvent)
    mock.qualifyContracts = Mock()
    mock.placeOrder = Mock()
    mock.cancelOrder = Mock()
    return mock


@pytest.fixture
async def event_bus():
    """Create and start an event bus for testing"""
    bus = EventBus("test")
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
async def order_manager(mock_ib, event_bus):
    """Create an order manager with mocked IB"""
    config = OrderConfig(
        default_timeout=5,
        use_adaptive_orders=False,
        enable_bracket_orders=True
    )
    
    with patch('ikbr.core.order_manager.get_event_bus', return_value=event_bus):
        manager = OrderManager(mock_ib, config)
        yield manager


class TestSignal:
    """Test Signal dataclass"""
    
    def test_signal_creation(self):
        """Test creating a signal"""
        signal = Signal(
            action="BUY",
            symbol="SPY",
            quantity=100,
            order_type="LIMIT",
            limit_price=450.50,
            stop_loss=445.00,
            take_profit=460.00
        )
        
        assert signal.is_buy is True
        assert signal.is_sell is False
        assert signal.symbol == "SPY"
        assert signal.quantity == 100
        assert signal.limit_price == 450.50
    
    def test_signal_validation(self):
        """Test signal validation"""
        # Valid signal
        signal = Signal("BUY", "SPY", 100, "MARKET")
        assert validate_signal(signal) is True
        
        # Invalid action
        signal = Signal("BUYY", "SPY", 100, "MARKET")
        assert validate_signal(signal) is False
        
        # Invalid quantity
        signal = Signal("BUY", "SPY", 0, "MARKET")
        assert validate_signal(signal) is False
        
        # Limit order without price
        signal = Signal("BUY", "SPY", 100, "LIMIT")
        assert validate_signal(signal) is False
        
        # Valid limit order
        signal = Signal("BUY", "SPY", 100, "LIMIT", limit_price=450.50)
        assert validate_signal(signal) is True


class TestOrderManager:
    """Test OrderManager functionality"""
    
    @pytest.mark.asyncio
    async def test_submit_market_order(self, order_manager, mock_ib):
        """Test submitting a market order"""
        # Setup mock
        mock_trade = Mock(spec=Trade)
        mock_order = Mock(spec=Order)
        mock_order.orderId = 123
        mock_trade.order = mock_order
        mock_trade.contract = Stock("SPY", "SMART", "USD")
        mock_ib.placeOrder.return_value = mock_trade
        
        # Create signal
        signal = Signal("BUY", "SPY", 100, "MARKET")
        
        # Submit order
        order_info = await order_manager.submit_order(signal)
        
        assert order_info is not None
        assert order_info.signal == signal
        assert order_info.status == OrderStatus.PENDING
        assert mock_ib.placeOrder.called
        assert order_info.order_id in order_manager._orders
        assert order_info.order_id in order_manager._active_orders
    
    @pytest.mark.asyncio
    async def test_submit_limit_order(self, order_manager, mock_ib):
        """Test submitting a limit order"""
        # Setup mock
        mock_trade = Mock(spec=Trade)
        mock_order = Mock(spec=Order)
        mock_order.orderId = 124
        mock_trade.order = mock_order
        mock_ib.placeOrder.return_value = mock_trade
        
        # Create signal
        signal = Signal("SELL", "AAPL", 50, "LIMIT", limit_price=175.50)
        
        # Submit order
        order_info = await order_manager.submit_order(signal)
        
        assert order_info is not None
        assert order_info.ib_order.lmtPrice == 175.50
        assert order_info.ib_order.totalQuantity == 50
        assert order_info.ib_order.action == "SELL"
    
    @pytest.mark.asyncio
    async def test_risk_check_rejection(self, order_manager, mock_ib):
        """Test order rejected by risk manager"""
        # Setup mock risk manager
        mock_risk_manager = Mock()
        mock_risk_manager.check_order = AsyncMock(return_value=False)
        order_manager.set_risk_manager(mock_risk_manager)
        
        # Create signal
        signal = Signal("BUY", "TSLA", 1000, "MARKET")
        
        # Submit order
        order_info = await order_manager.submit_order(signal)
        
        assert order_info is None
        assert mock_risk_manager.check_order.called
        assert not mock_ib.placeOrder.called
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, order_manager, mock_ib):
        """Test cancelling an order"""
        # Setup and submit order
        mock_trade = Mock(spec=Trade)
        mock_order = Mock(spec=Order)
        mock_order.orderId = 125
        mock_trade.order = mock_order
        mock_ib.placeOrder.return_value = mock_trade
        
        signal = Signal("BUY", "SPY", 100, "MARKET")
        order_info = await order_manager.submit_order(signal)
        
        # Cancel order
        result = await order_manager.cancel_order(order_info.order_id)
        
        assert result is True
        assert mock_ib.cancelOrder.called_with(mock_order)
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self, order_manager):
        """Test cancelling a non-existent order"""
        result = await order_manager.cancel_order("fake-order-id")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_modify_order(self, order_manager, mock_ib):
        """Test modifying an order"""
        # Setup and submit order
        mock_trade = Mock(spec=Trade)
        mock_order = Mock(spec=Order)
        mock_order.orderId = 126
        mock_trade.order = mock_order
        mock_trade.contract = Stock("SPY", "SMART", "USD")
        mock_ib.placeOrder.return_value = mock_trade
        
        signal = Signal("BUY", "SPY", 100, "LIMIT", limit_price=450.00)
        order_info = await order_manager.submit_order(signal)
        
        # Modify order
        result = await order_manager.modify_order(
            order_info.order_id,
            limit_price=451.00,
            quantity=150
        )
        
        assert result is True
        assert order_info.ib_order.lmtPrice == 451.00
        assert order_info.ib_order.totalQuantity == 150
        assert mock_ib.placeOrder.call_count == 2  # Original + modification
    
    @pytest.mark.asyncio
    async def test_order_status_updates(self, order_manager, mock_ib):
        """Test handling order status updates"""
        # Setup and submit order
        mock_trade = Mock(spec=Trade)
        mock_order = Mock(spec=Order)
        mock_order.orderId = 127
        mock_trade.order = mock_order
        mock_ib.placeOrder.return_value = mock_trade
        
        signal = Signal("BUY", "AAPL", 100, "MARKET")
        order_info = await order_manager.submit_order(signal)
        
        # Simulate status update to SUBMITTED
        mock_order_status = Mock()
        mock_order_status.status = "Submitted"
        mock_trade.orderStatus = mock_order_status
        
        order_manager._on_order_status(mock_trade)
        
        assert order_info.status == OrderStatus.SUBMITTED
        
        # Simulate fill
        mock_order_status.status = "Filled"
        mock_order_status.avgFillPrice = 175.25
        order_manager._on_order_status(mock_trade)
        
        assert order_info.status == OrderStatus.FILLED
        assert order_info.fill_price == 175.25
        assert order_info.fill_time is not None
        assert order_info.order_id not in order_manager._active_orders
    
    @pytest.mark.asyncio
    async def test_execution_details(self, order_manager, mock_ib):
        """Test handling execution details"""
        # Setup and submit order
        mock_trade = Mock(spec=Trade)
        mock_order = Mock(spec=Order)
        mock_order.orderId = 128
        mock_trade.order = mock_order
        mock_ib.placeOrder.return_value = mock_trade
        
        signal = Signal("SELL", "TSLA", 50, "MARKET")
        order_info = await order_manager.submit_order(signal)
        
        # Simulate execution details
        mock_fill = Mock(spec=Fill)
        mock_fill.execution = Mock(price=850.50, shares=50)
        mock_fill.commissionReport = Mock(spec=CommissionReport, commission=1.50)
        
        order_manager._on_exec_details(mock_trade, mock_fill)
        
        assert order_info.commission == 1.50
        assert order_manager._metrics['total_commission'] == 1.50
    
    @pytest.mark.asyncio
    async def test_error_handling(self, order_manager, mock_ib):
        """Test handling order errors"""
        # Setup and submit order
        mock_trade = Mock(spec=Trade)
        mock_order = Mock(spec=Order)
        mock_order.orderId = 129
        mock_trade.order = mock_order
        mock_ib.placeOrder.return_value = mock_trade
        
        signal = Signal("BUY", "SPY", 100, "MARKET")
        order_info = await order_manager.submit_order(signal)
        
        # Simulate error
        order_manager._on_error(
            reqId=129,
            errorCode=201,
            errorString="Order rejected - insufficient funds",
            contract=Stock("SPY", "SMART", "USD")
        )
        
        assert order_info.status == OrderStatus.REJECTED
        assert "insufficient funds" in order_info.error_message
        assert order_info.order_id not in order_manager._active_orders
    
    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, order_manager, mock_ib):
        """Test cancelling all orders"""
        # Submit multiple orders
        mock_trade = Mock(spec=Trade)
        mock_ib.placeOrder.return_value = mock_trade
        
        orders = []
        for i, symbol in enumerate(["SPY", "AAPL", "TSLA"]):
            mock_order = Mock(spec=Order)
            mock_order.orderId = 130 + i
            mock_trade.order = mock_order
            
            signal = Signal("BUY", symbol, 100, "MARKET")
            order_info = await order_manager.submit_order(signal)
            orders.append(order_info)
        
        # Cancel all
        cancelled = await order_manager.cancel_all_orders()
        
        assert cancelled == 3
        assert mock_ib.cancelOrder.call_count == 3
    
    @pytest.mark.asyncio
    async def test_cancel_orders_by_symbol(self, order_manager, mock_ib):
        """Test cancelling orders for specific symbol"""
        # Submit orders for different symbols
        mock_trade = Mock(spec=Trade)
        mock_ib.placeOrder.return_value = mock_trade
        
        # Submit 2 SPY orders and 1 AAPL order
        for i, (symbol, qty) in enumerate([("SPY", 100), ("SPY", 200), ("AAPL", 50)]):
            mock_order = Mock(spec=Order)
            mock_order.orderId = 140 + i
            mock_trade.order = mock_order
            
            signal = Signal("BUY", symbol, qty, "MARKET")
            await order_manager.submit_order(signal)
        
        # Cancel only SPY orders
        cancelled = await order_manager.cancel_all_orders(symbol="SPY")
        
        assert cancelled == 2
        assert mock_ib.cancelOrder.call_count == 2
    
    def test_get_metrics(self, order_manager):
        """Test metrics calculation"""
        # Manually update metrics for testing
        order_manager._metrics['orders_submitted'] = 10
        order_manager._metrics['orders_filled'] = 8
        order_manager._metrics['orders_cancelled'] = 1
        order_manager._metrics['orders_rejected'] = 1
        order_manager._metrics['total_commission'] = 15.50
        order_manager._metrics['total_latency_ms'] = 500.0
        
        metrics = order_manager.get_metrics()
        
        assert metrics['orders_submitted'] == 10
        assert metrics['orders_filled'] == 8
        assert metrics['fill_rate'] == 0.8
        assert metrics['avg_submit_latency_ms'] == 50.0
        assert metrics['total_commission'] == 15.50


class TestOrderUtilities:
    """Test order utility functions"""
    
    def test_calculate_order_size(self):
        """Test position size calculation"""
        # 1% risk on $10,000 capital
        capital = 10000
        risk_pct = 0.01
        entry_price = 100
        stop_price = 98
        
        size = calculate_order_size(capital, risk_pct, entry_price, stop_price)
        
        # Risk amount = $100, price risk = $2, size = 50
        assert size == 50
    
    def test_calculate_order_size_zero_risk(self):
        """Test order size with zero price risk"""
        size = calculate_order_size(10000, 0.01, 100, 100)
        assert size == 0


@pytest.mark.asyncio
async def test_event_emission_integration(mock_ib, event_bus):
    """Test integration with event bus"""
    # Track events
    events_received = {
        EventTypes.ORDER_SUBMITTED: [],
        EventTypes.ORDER_FILLED: [],
        EventTypes.ORDER_CANCELLED: [],
        EventTypes.ORDER_REJECTED: []
    }
    
    async def capture_event(event: Event):
        if event.event_type in events_received:
            events_received[event.event_type].append(event)
    
    # Subscribe to all order events
    for event_type in events_received.keys():
        event_bus.subscribe(event_type, capture_event)
    
    # Create order manager
    with patch('ikbr.core.order_manager.get_event_bus', return_value=event_bus):
        manager = OrderManager(mock_ib)
        
        # Setup mock
        mock_trade = Mock(spec=Trade)
        mock_order = Mock(spec=Order)
        mock_order.orderId = 200
        mock_trade.order = mock_order
        mock_ib.placeOrder.return_value = mock_trade
        
        # Submit order
        signal = Signal("BUY", "SPY", 100, "MARKET")
        order_info = await manager.submit_order(signal)
        
        # Wait for event processing
        await asyncio.sleep(0.1)
        
        # Check submitted event
        assert len(events_received[EventTypes.ORDER_SUBMITTED]) == 1
        event = events_received[EventTypes.ORDER_SUBMITTED][0]
        assert event.data['order_id'] == order_info.order_id
        assert event.data['signal'] == signal
        
        # Simulate fill
        mock_order_status = Mock()
        mock_order_status.status = "Filled"
        mock_order_status.avgFillPrice = 450.25
        mock_trade.orderStatus = mock_order_status
        
        manager._on_order_status(mock_trade)
        await asyncio.sleep(0.1)
        
        # Check filled event
        assert len(events_received[EventTypes.ORDER_FILLED]) == 1
        event = events_received[EventTypes.ORDER_FILLED][0]
        assert event.data['fill_price'] == 450.25