"""
Unit tests for Risk Manager
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from ib_async import IB, Position, AccountValue, Contract

from core.risk_manager import (
    RiskManager, RiskLimits, RiskLevel, PositionRisk, PortfolioRisk
)
from core.order_manager import Signal
from core.event_bus import EventBus, Event, EventTypes


@pytest.fixture
async def mock_ib():
    """Create a mock IB connection"""
    mock = Mock(spec=IB)
    mock.positions = Mock(return_value=[])
    mock.accountValues = Mock(return_value=[])
    return mock


@pytest.fixture
async def event_bus():
    """Create and start an event bus for testing"""
    bus = EventBus("test")
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
async def risk_manager(mock_ib, event_bus):
    """Create a risk manager with mocked IB"""
    limits = RiskLimits(
        max_position_size=10000,
        max_positions=5,
        max_total_exposure=50000,
        max_daily_loss=1000,
        max_drawdown_pct=0.10,
        risk_per_trade_pct=0.01
    )
    
    with patch('ikbr.core.risk_manager.get_event_bus', return_value=event_bus):
        manager = RiskManager(mock_ib, limits)
        yield manager


class TestRiskLimits:
    """Test RiskLimits configuration"""
    
    def test_default_limits(self):
        """Test default risk limits"""
        limits = RiskLimits()
        assert limits.max_position_size == 10000
        assert limits.max_positions == 10
        assert limits.max_leverage == 1.0
        assert limits.risk_per_trade_pct == 0.01
    
    def test_custom_limits(self):
        """Test custom risk limits"""
        limits = RiskLimits(
            max_position_size=20000,
            max_daily_trades=50,
            max_drawdown_pct=0.05
        )
        assert limits.max_position_size == 20000
        assert limits.max_daily_trades == 50
        assert limits.max_drawdown_pct == 0.05


class TestPositionRisk:
    """Test PositionRisk calculations"""
    
    def test_position_risk_calculations(self):
        """Test position risk metric calculations"""
        position = PositionRisk(
            symbol="SPY",
            quantity=100,
            market_value=45000,
            unrealized_pnl=500,
            realized_pnl=200,
            cost_basis=44300,
            current_price=450,
            position_pct=0.10
        )
        
        assert position.total_pnl == 700
        assert abs(position.pnl_pct - 0.0158) < 0.0001  # ~1.58%
    
    def test_position_risk_zero_cost_basis(self):
        """Test position risk with zero cost basis"""
        position = PositionRisk(
            symbol="SPY",
            quantity=0,
            market_value=0,
            unrealized_pnl=0,
            realized_pnl=100,
            cost_basis=0,
            current_price=450,
            position_pct=0
        )
        
        assert position.pnl_pct == 0.0


class TestPortfolioRisk:
    """Test PortfolioRisk calculations"""
    
    def test_risk_level_determination(self):
        """Test risk level calculation"""
        # Low risk
        portfolio = PortfolioRisk(
            total_value=100000,
            leverage=0.3,
            current_drawdown=0.02
        )
        assert portfolio.risk_level == RiskLevel.LOW
        
        # Medium risk
        portfolio = PortfolioRisk(
            total_value=100000,
            leverage=0.6,
            current_drawdown=0.04
        )
        assert portfolio.risk_level == RiskLevel.MEDIUM
        
        # High risk
        portfolio = PortfolioRisk(
            total_value=100000,
            leverage=0.8,
            current_drawdown=0.06
        )
        assert portfolio.risk_level == RiskLevel.HIGH
        
        # Critical risk
        portfolio = PortfolioRisk(
            total_value=100000,
            leverage=0.95,
            current_drawdown=0.09
        )
        assert portfolio.risk_level == RiskLevel.CRITICAL


class TestRiskManager:
    """Test RiskManager functionality"""
    
    @pytest.mark.asyncio
    async def test_position_limit_check(self, risk_manager):
        """Test position size limit checking"""
        # Mock get_current_price
        with patch.object(risk_manager, '_get_current_price', return_value=100.0):
            # Valid order
            signal = Signal("BUY", "SPY", 50, "MARKET")
            result = await risk_manager._check_position_limit(signal)
            assert result is True
            
            # Order too large
            signal = Signal("BUY", "SPY", 150, "MARKET")  # $15,000 > $10,000 limit
            result = await risk_manager._check_position_limit(signal)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_max_positions_check(self, risk_manager):
        """Test maximum positions limit"""
        # Fill up positions
        risk_manager._positions = {
            f"STOCK{i}": Mock() for i in range(5)  # Max is 5
        }
        
        # Try to add new position
        signal = Signal("BUY", "NEWSTOCK", 10, "MARKET")
        result = await risk_manager._check_position_limit(signal)
        assert result is False
        
        # Existing position is OK
        signal = Signal("BUY", "STOCK1", 10, "MARKET")
        with patch.object(risk_manager, '_get_current_price', return_value=100.0):
            result = await risk_manager._check_position_limit(signal)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_exposure_limit_check(self, risk_manager):
        """Test exposure limit checking"""
        # Setup mock portfolio
        mock_portfolio = PortfolioRisk(
            total_value=100000,
            total_exposure=40000  # $40k of $50k limit used
        )
        
        with patch.object(risk_manager, 'get_portfolio_risk', return_value=mock_portfolio):
            with patch.object(risk_manager, '_get_current_price', return_value=100.0):
                # Valid order
                signal = Signal("BUY", "SPY", 50, "MARKET")  # $5,000
                result = await risk_manager._check_exposure_limit(signal)
                assert result is True
                
                # Order exceeds limit
                signal = Signal("BUY", "SPY", 150, "MARKET")  # $15,000
                result = await risk_manager._check_exposure_limit(signal)
                assert result is False
    
    @pytest.mark.asyncio
    async def test_daily_loss_limit_check(self, risk_manager):
        """Test daily loss limit checking"""
        # Within limit
        risk_manager._daily_pnl = -500
        result = await risk_manager._check_daily_loss_limit()
        assert result is True
        
        # Exceeds limit
        risk_manager._daily_pnl = -1500  # Exceeds $1000 limit
        result = await risk_manager._check_daily_loss_limit()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_drawdown_limit_check(self, risk_manager):
        """Test drawdown limit checking"""
        # Within limit
        mock_portfolio = PortfolioRisk(current_drawdown=0.05)
        with patch.object(risk_manager, 'get_portfolio_risk', return_value=mock_portfolio):
            result = await risk_manager._check_drawdown_limit()
            assert result is True
        
        # Exceeds limit
        mock_portfolio = PortfolioRisk(current_drawdown=0.15)  # 15% > 10% limit
        with patch.object(risk_manager, 'get_portfolio_risk', return_value=mock_portfolio):
            result = await risk_manager._check_drawdown_limit()
            assert result is False
    
    @pytest.mark.asyncio
    async def test_order_frequency_check(self, risk_manager):
        """Test order frequency limiting"""
        signal = Signal("BUY", "SPY", 100, "MARKET")
        
        # First order is OK
        result = await risk_manager._check_order_frequency(signal)
        assert result is True
        
        # Set recent order time
        risk_manager._last_order_time["SPY"] = time.time() - 0.5  # 0.5 seconds ago
        
        # Too soon (min interval is 1 second)
        result = await risk_manager._check_order_frequency(signal)
        assert result is False
        
        # Wait and try again
        risk_manager._last_order_time["SPY"] = time.time() - 2.0  # 2 seconds ago
        result = await risk_manager._check_order_frequency(signal)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_daily_trade_limit_check(self, risk_manager):
        """Test daily trade limit"""
        risk_manager.limits.max_daily_trades = 3
        
        # Within limit
        risk_manager._daily_trades = 2
        result = await risk_manager._check_daily_trade_limit()
        assert result is True
        
        # At limit
        risk_manager._daily_trades = 3
        result = await risk_manager._check_daily_trade_limit()
        assert result is False
    
    @pytest.mark.asyncio
    async def test_leverage_limit_check(self, risk_manager):
        """Test leverage limit checking"""
        mock_portfolio = PortfolioRisk(
            total_value=100000,
            total_exposure=80000  # 0.8x leverage
        )
        
        with patch.object(risk_manager, 'get_portfolio_risk', return_value=mock_portfolio):
            with patch.object(risk_manager, '_get_current_price', return_value=100.0):
                # Valid order
                signal = Signal("BUY", "SPY", 100, "MARKET")  # $10k
                result = await risk_manager._check_leverage_limit(signal)
                assert result is True
                
                # Would exceed leverage
                signal = Signal("BUY", "SPY", 300, "MARKET")  # $30k would make 1.1x
                result = await risk_manager._check_leverage_limit(signal)
                assert result is False
    
    @pytest.mark.asyncio
    async def test_check_order_comprehensive(self, risk_manager):
        """Test comprehensive order checking"""
        # Setup mocks
        mock_portfolio = PortfolioRisk(
            total_value=100000,
            total_exposure=30000,
            current_drawdown=0.03,
            leverage=0.3
        )
        
        with patch.object(risk_manager, 'get_portfolio_risk', return_value=mock_portfolio):
            with patch.object(risk_manager, '_get_current_price', return_value=100.0):
                # Valid order
                signal = Signal("BUY", "SPY", 50, "MARKET")
                result = await risk_manager.check_order(signal)
                assert result is True
    
    @pytest.mark.asyncio
    async def test_calculate_position_size(self, risk_manager):
        """Test position size calculation"""
        # Setup mock portfolio
        mock_portfolio = PortfolioRisk(total_value=100000)
        
        with patch.object(risk_manager, 'get_portfolio_risk', return_value=mock_portfolio):
            with patch.object(risk_manager, '_get_current_price', return_value=100.0):
                # With stop loss
                signal = Signal(
                    "BUY", "SPY", 0, "MARKET",
                    stop_loss=98.0  # $2 risk per share
                )
                
                # 1% of $100k = $1000 risk, $2 per share = 500 shares
                size = await risk_manager.calculate_position_size(signal)
                assert size == 500
                
                # Without stop loss (uses 2% price risk)
                signal = Signal("BUY", "SPY", 0, "MARKET")
                # $1000 risk / ($100 * 0.02) = 500 shares
                size = await risk_manager.calculate_position_size(signal)
                assert size == 500
    
    @pytest.mark.asyncio
    async def test_update_positions(self, risk_manager, mock_ib):
        """Test position update from IB"""
        # Mock IB positions
        mock_contract = Mock()
        mock_contract.symbol = "SPY"
        
        mock_position = Mock(spec=Position)
        mock_position.contract = mock_contract
        mock_position.position = 100
        mock_position.marketValue = 45000
        mock_position.unrealizedPNL = 500
        mock_position.realizedPNL = 200
        mock_position.avgCost = 445.0
        mock_position.marketPrice = 450.0
        
        mock_ib.positions.return_value = [mock_position]
        
        # Mock cash balance
        mock_av = Mock(spec=AccountValue)
        mock_av.tag = "TotalCashBalance"
        mock_av.currency = "USD"
        mock_av.value = "55000"
        mock_ib.accountValues.return_value = [mock_av]
        
        # Update positions
        await risk_manager._update_positions()
        
        # Check position was updated
        assert "SPY" in risk_manager._positions
        position = risk_manager._positions["SPY"]
        assert position.quantity == 100
        assert position.market_value == 45000
        assert position.unrealized_pnl == 500
    
    @pytest.mark.asyncio
    async def test_get_portfolio_risk(self, risk_manager, mock_ib):
        """Test portfolio risk calculation"""
        # Setup positions
        risk_manager._positions = {
            "SPY": PositionRisk(
                symbol="SPY",
                quantity=100,
                market_value=45000,
                unrealized_pnl=500,
                realized_pnl=200,
                cost_basis=44300,
                current_price=450,
                position_pct=0.45
            ),
            "AAPL": PositionRisk(
                symbol="AAPL",
                quantity=50,
                market_value=8750,
                unrealized_pnl=-100,
                realized_pnl=50,
                cost_basis=8800,
                current_price=175,
                position_pct=0.0875
            )
        }
        
        # Mock cash balance
        with patch.object(risk_manager, '_get_cash_balance', return_value=46250):
            risk_manager._high_water_mark = 105000
            risk_manager._daily_pnl = 650
            
            portfolio = await risk_manager.get_portfolio_risk()
            
            assert portfolio.total_value == 100000  # 45000 + 8750 + 46250
            assert portfolio.cash_balance == 46250
            assert portfolio.total_exposure == 53750  # 45000 + 8750
            assert portfolio.leverage == 0.5375
            assert portfolio.daily_pnl == 650
            assert abs(portfolio.current_drawdown - 0.0476) < 0.0001  # ~4.76%
            assert len(portfolio.positions) == 2
    
    def test_reset_daily_metrics(self, risk_manager):
        """Test daily metrics reset"""
        # Set some values
        risk_manager._daily_trades = 10
        risk_manager._daily_pnl = -500
        
        # Reset
        risk_manager.reset_daily_metrics()
        
        assert risk_manager._daily_trades == 0
        assert risk_manager._daily_pnl == 0
        assert len(risk_manager._pnl_history) == 1
        assert risk_manager._pnl_history[0] == 0
    
    @pytest.mark.asyncio
    async def test_event_handling(self, risk_manager, event_bus):
        """Test event handler integration"""
        # Emit order filled event
        order_data = {
            'order_info': Mock(
                signal=Mock(symbol="SPY", action="BUY", quantity=100),
                fill_price=450.0,
                commission=1.0
            )
        }
        
        await event_bus.emit(Event(EventTypes.ORDER_FILLED, order_data))
        await asyncio.sleep(0.1)
        
        # Check updates
        assert risk_manager._daily_trades == 1
        assert "SPY" in risk_manager._last_order_time
        assert len(risk_manager._trade_history) == 1


@pytest.mark.asyncio
async def test_risk_monitoring_integration(mock_ib, event_bus):
    """Test risk monitoring with event emission"""
    # Track events
    risk_events = []
    
    async def capture_event(event: Event):
        risk_events.append(event)
    
    event_bus.subscribe(EventTypes.RISK_LIMIT_BREACHED, capture_event)
    event_bus.subscribe(EventTypes.RISK_CHECK_FAILED, capture_event)
    
    # Create risk manager with tight limits
    limits = RiskLimits(
        max_drawdown_pct=0.05,
        max_daily_loss=500
    )
    
    with patch('ikbr.core.risk_manager.get_event_bus', return_value=event_bus):
        manager = RiskManager(mock_ib, limits)
        
        # Simulate high drawdown
        manager._high_water_mark = 100000
        with patch.object(manager, '_get_cash_balance', return_value=50000):
            with patch.object(manager, '_positions', {"SPY": Mock(market_value=45000)}):
                await manager._update_positions()
                
                portfolio = await manager.get_portfolio_risk()
                assert portfolio.current_drawdown > 0.05
                
                # Check order should fail
                signal = Signal("BUY", "AAPL", 100, "MARKET")
                result = await manager.check_order(signal)
                assert result is False
                
                await asyncio.sleep(0.1)
                
                # Check events were emitted
                assert len(risk_events) > 0
                assert any(e.event_type == EventTypes.RISK_CHECK_FAILED for e in risk_events)