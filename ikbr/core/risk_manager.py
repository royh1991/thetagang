"""
Risk Management System for IBKR Trading Bot

Implements comprehensive risk controls including position limits, exposure management,
drawdown monitoring, and real-time risk calculations.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import numpy as np
from loguru import logger
from ib_async import IB, AccountValue, Position

from .event_bus import EventBus, Event, EventTypes, get_event_bus
from .order_manager import Signal, OrderInfo


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    # Position limits
    max_position_size: float = 10000  # Max $ per position
    max_positions: int = 10  # Max number of open positions
    max_symbol_exposure: float = 20000  # Max $ exposure per symbol
    max_sector_exposure: float = 50000  # Max $ exposure per sector
    
    # Portfolio limits
    max_total_exposure: float = 100000  # Max total $ exposure
    max_leverage: float = 1.0  # Max leverage ratio
    max_daily_loss: float = 5000  # Max daily loss
    max_drawdown_pct: float = 0.10  # Max drawdown percentage
    
    # Order limits
    max_order_size: float = 10000  # Max $ per order
    max_daily_trades: int = 100  # Max trades per day
    min_order_interval: float = 1.0  # Min seconds between orders
    
    # Risk per trade
    risk_per_trade_pct: float = 0.01  # 1% risk per trade
    max_correlation_exposure: float = 0.5  # Max correlated exposure


@dataclass
class PositionRisk:
    """Risk metrics for a position"""
    symbol: str
    quantity: int
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    cost_basis: float
    current_price: float
    position_pct: float  # Percentage of portfolio
    var_95: Optional[float] = None  # Value at Risk 95%
    beta: Optional[float] = None
    sector: Optional[str] = None
    
    @property
    def total_pnl(self) -> float:
        return self.unrealized_pnl + self.realized_pnl
    
    @property
    def pnl_pct(self) -> float:
        if self.cost_basis != 0:
            return self.total_pnl / abs(self.cost_basis)
        return 0.0


@dataclass
class PortfolioRisk:
    """Portfolio-wide risk metrics"""
    timestamp: float = field(default_factory=time.time)
    total_value: float = 0.0
    cash_balance: float = 0.0
    total_exposure: float = 0.0
    leverage: float = 0.0
    daily_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: Optional[float] = None
    var_95: Optional[float] = None
    positions: List[PositionRisk] = field(default_factory=list)
    
    @property
    def risk_level(self) -> RiskLevel:
        """Determine overall risk level"""
        if self.current_drawdown > 0.08 or self.leverage > 0.9:
            return RiskLevel.CRITICAL
        elif self.current_drawdown > 0.05 or self.leverage > 0.7:
            return RiskLevel.HIGH
        elif self.current_drawdown > 0.03 or self.leverage > 0.5:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


class RiskManager:
    """
    Comprehensive risk management system
    
    Features:
    - Real-time position and exposure monitoring
    - Pre-trade risk checks
    - Drawdown and loss limits
    - Correlation-based risk assessment
    - Dynamic position sizing
    - Risk alerts and notifications
    """
    
    def __init__(self, ib: IB, limits: Optional[RiskLimits] = None):
        self.ib = ib
        self.limits = limits or RiskLimits()
        self.event_bus = get_event_bus()
        
        # Risk tracking
        self._positions: Dict[str, PositionRisk] = {}
        self._daily_trades: int = 0
        self._daily_pnl: float = 0.0
        self._high_water_mark: float = 0.0
        self._last_order_time: Dict[str, float] = {}
        self._trade_history: List[Dict] = []
        
        # Historical data for calculations
        self._pnl_history: List[float] = []
        self._portfolio_values: List[Tuple[float, float]] = []  # (timestamp, value)
        
        # State
        self._last_update: float = 0.0
        self._update_interval: float = 5.0  # Update every 5 seconds
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Subscribe to events
        self._setup_event_handlers()
        
    def _setup_event_handlers(self):
        """Setup event handlers"""
        self.event_bus.subscribe(EventTypes.ORDER_FILLED, self._on_order_filled)
        self.event_bus.subscribe(EventTypes.POSITION_UPDATED, self._on_position_updated)
        
    async def start(self):
        """Start risk monitoring"""
        if not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._monitor_risk())
            await self._update_positions()
            logger.info("RiskManager started")
    
    async def stop(self):
        """Stop risk monitoring"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        logger.info("RiskManager stopped")
    
    async def check_order(self, signal: Signal) -> bool:
        """
        Pre-trade risk check
        
        Returns:
            bool: True if order passes all risk checks
        """
        # Update positions before checking
        if time.time() - self._last_update > 60:  # Update if stale
            await self._update_positions()
        
        # Run all risk checks
        checks = [
            self._check_position_limit(signal),
            self._check_exposure_limit(signal),
            self._check_daily_loss_limit(),
            self._check_drawdown_limit(),
            self._check_order_frequency(signal),
            self._check_daily_trade_limit(),
            self._check_leverage_limit(signal)
        ]
        
        results = await asyncio.gather(*checks)
        
        if not all(results):
            reasons = []
            check_names = [
                "position_limit", "exposure_limit", "daily_loss", 
                "drawdown", "order_frequency", "daily_trades", "leverage"
            ]
            for i, (result, name) in enumerate(zip(results, check_names)):
                if not result:
                    reasons.append(name)
            
            logger.warning(f"Risk check failed for {signal.symbol}: {', '.join(reasons)}")
            await self._emit_risk_check_failed(signal, reasons)
            return False
        
        return True
    
    async def _check_position_limit(self, signal: Signal) -> bool:
        """Check position size limits"""
        # Estimate order value
        if signal.limit_price:
            order_value = signal.quantity * signal.limit_price
        else:
            # Get current price for market orders
            current_price = await self._get_current_price(signal.symbol)
            if not current_price:
                return False
            order_value = signal.quantity * current_price
        
        # Check max order size
        if order_value > self.limits.max_order_size:
            logger.warning(f"Order size ${order_value:.2f} exceeds limit ${self.limits.max_order_size}")
            return False
        
        # Check max position size (including existing position)
        existing_position = self._positions.get(signal.symbol)
        if existing_position:
            if signal.is_buy == (existing_position.quantity > 0):
                # Adding to position
                total_value = abs(existing_position.market_value) + order_value
                if total_value > self.limits.max_position_size:
                    logger.warning(f"Total position ${total_value:.2f} would exceed limit")
                    return False
        
        # Check max number of positions
        if signal.symbol not in self._positions:
            if len(self._positions) >= self.limits.max_positions:
                logger.warning(f"Max positions limit reached ({self.limits.max_positions})")
                return False
        
        return True
    
    async def _check_exposure_limit(self, signal: Signal) -> bool:
        """Check exposure limits"""
        portfolio_risk = await self.get_portfolio_risk()
        
        # Estimate new exposure
        if signal.limit_price:
            order_value = signal.quantity * signal.limit_price
        else:
            current_price = await self._get_current_price(signal.symbol)
            if not current_price:
                return False
            order_value = signal.quantity * current_price
        
        # Check total exposure
        new_total_exposure = portfolio_risk.total_exposure + order_value
        if new_total_exposure > self.limits.max_total_exposure:
            logger.warning(f"Total exposure ${new_total_exposure:.2f} would exceed limit")
            return False
        
        # Check symbol exposure
        symbol_exposure = sum(p.market_value for p in portfolio_risk.positions 
                            if p.symbol == signal.symbol)
        new_symbol_exposure = abs(symbol_exposure) + order_value
        if new_symbol_exposure > self.limits.max_symbol_exposure:
            logger.warning(f"Symbol exposure ${new_symbol_exposure:.2f} would exceed limit")
            return False
        
        return True
    
    async def _check_daily_loss_limit(self) -> bool:
        """Check daily loss limit"""
        if self._daily_pnl < -self.limits.max_daily_loss:
            logger.warning(f"Daily loss ${-self._daily_pnl:.2f} exceeds limit")
            return False
        return True
    
    async def _check_drawdown_limit(self) -> bool:
        """Check drawdown limit"""
        portfolio_risk = await self.get_portfolio_risk()
        if portfolio_risk.current_drawdown > self.limits.max_drawdown_pct:
            logger.warning(f"Drawdown {portfolio_risk.current_drawdown:.1%} exceeds limit")
            return False
        return True
    
    async def _check_order_frequency(self, signal: Signal) -> bool:
        """Check minimum time between orders for same symbol"""
        last_order_time = self._last_order_time.get(signal.symbol, 0)
        time_since_last = time.time() - last_order_time
        
        if time_since_last < self.limits.min_order_interval:
            logger.warning(f"Order too soon after last order ({time_since_last:.1f}s)")
            return False
        
        return True
    
    async def _check_daily_trade_limit(self) -> bool:
        """Check daily trade limit"""
        if self._daily_trades >= self.limits.max_daily_trades:
            logger.warning(f"Daily trade limit reached ({self.limits.max_daily_trades})")
            return False
        return True
    
    async def _check_leverage_limit(self, signal: Signal) -> bool:
        """Check leverage limit"""
        portfolio_risk = await self.get_portfolio_risk()
        
        # Estimate new leverage
        if signal.limit_price:
            order_value = signal.quantity * signal.limit_price
        else:
            current_price = await self._get_current_price(signal.symbol)
            if not current_price:
                return False
            order_value = signal.quantity * current_price
        
        new_exposure = portfolio_risk.total_exposure + order_value
        new_leverage = new_exposure / max(portfolio_risk.total_value, 1)
        
        if new_leverage > self.limits.max_leverage:
            logger.warning(f"Leverage {new_leverage:.2f} would exceed limit")
            return False
        
        return True
    
    async def calculate_position_size(self, signal: Signal) -> int:
        """
        Calculate risk-adjusted position size
        
        Args:
            signal: Trading signal
            
        Returns:
            int: Recommended position size
        """
        portfolio_risk = await self.get_portfolio_risk()
        
        # Get account value
        account_value = portfolio_risk.total_value
        logger.debug(f"Portfolio value for position sizing: ${account_value}")
        
        # Calculate risk amount
        risk_amount = account_value * self.limits.risk_per_trade_pct
        logger.debug(f"Risk amount: ${risk_amount} ({self.limits.risk_per_trade_pct*100}% of ${account_value})")
        
        # Get current price
        if signal.limit_price:
            entry_price = signal.limit_price
        else:
            entry_price = await self._get_current_price(signal.symbol)
            logger.debug(f"Current price for {signal.symbol}: {entry_price}")
            if not entry_price:
                logger.warning(f"Could not get current price for {signal.symbol}")
                return 0
        
        # Calculate position size based on stop loss
        if signal.stop_loss:
            price_risk = abs(entry_price - signal.stop_loss)
            logger.debug(f"Stop loss: {signal.stop_loss}, Price risk: {price_risk}")
            if price_risk > 0:
                shares = int(risk_amount / price_risk)
                logger.debug(f"Shares based on stop loss: {shares}")
            else:
                shares = 0
        else:
            # Use fixed percentage if no stop loss
            shares = int(risk_amount / (entry_price * 0.02))  # 2% price risk
            logger.debug(f"No stop loss, using 2% risk. Shares: {shares}")
        
        # Apply limits
        max_shares = int(self.limits.max_order_size / entry_price)
        shares = min(shares, max_shares)
        logger.debug(f"After max order size limit ({self.limits.max_order_size}): {shares} shares")
        
        # Check existing position
        if signal.symbol in self._positions:
            existing = self._positions[signal.symbol]
            if signal.is_buy == (existing.quantity > 0):
                # Adding to position - check position limit
                max_additional = int(self.limits.max_position_size / entry_price) - abs(existing.quantity)
                shares = min(shares, max_additional)
                logger.debug(f"After position limit check: {shares} shares")
        
        final_shares = max(0, shares)
        logger.info(f"Final position size for {signal.symbol}: {final_shares} shares")
        return final_shares
    
    async def get_portfolio_risk(self) -> PortfolioRisk:
        """Get current portfolio risk metrics"""
        # Update if needed
        if time.time() - self._last_update > self._update_interval:
            await self._update_positions()
        
        # Calculate portfolio metrics
        total_value = sum(p.market_value for p in self._positions.values())
        
        # Get cash balance
        cash_balance = 0.0
        for av in self.ib.accountValues():
            if av.tag == "TotalCashBalance" and av.currency == "USD":
                cash_balance = float(av.value)
                break
        
        portfolio_value = total_value + cash_balance
        
        # Calculate exposure and leverage
        total_exposure = sum(abs(p.market_value) for p in self._positions.values())
        leverage = total_exposure / max(portfolio_value, 1)
        
        # Calculate drawdown
        if portfolio_value > self._high_water_mark:
            self._high_water_mark = portfolio_value
        
        current_drawdown = 0.0
        if self._high_water_mark > 0:
            current_drawdown = (self._high_water_mark - portfolio_value) / self._high_water_mark
        
        # Create portfolio risk object
        portfolio_risk = PortfolioRisk(
            total_value=portfolio_value,
            cash_balance=cash_balance,
            total_exposure=total_exposure,
            leverage=leverage,
            daily_pnl=self._daily_pnl,
            max_drawdown=self.limits.max_drawdown_pct,
            current_drawdown=current_drawdown,
            positions=list(self._positions.values())
        )
        
        # Calculate VaR if we have history
        if len(self._pnl_history) > 20:
            portfolio_risk.var_95 = np.percentile(self._pnl_history, 5)
        
        # Calculate Sharpe ratio if we have history
        if len(self._pnl_history) > 30:
            returns = np.array(self._pnl_history)
            if returns.std() > 0:
                portfolio_risk.sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        
        return portfolio_risk
    
    async def _update_positions(self):
        """Update position information from IB"""
        self._positions.clear()
        
        for ib_position in self.ib.positions():
            symbol = ib_position.contract.symbol
            
            # Calculate position risk metrics
            position_risk = PositionRisk(
                symbol=symbol,
                quantity=int(ib_position.position),
                market_value=ib_position.marketValue,
                unrealized_pnl=ib_position.unrealizedPNL,
                realized_pnl=ib_position.realizedPNL,
                cost_basis=ib_position.avgCost * abs(ib_position.position),
                current_price=ib_position.marketPrice,
                position_pct=0.0  # Will calculate after getting total
            )
            
            self._positions[symbol] = position_risk
        
        # Calculate position percentages
        total_value = sum(abs(p.market_value) for p in self._positions.values())
        if total_value > 0:
            for position in self._positions.values():
                position.position_pct = abs(position.market_value) / total_value
        
        self._last_update = time.time()
        
        # Update daily P&L
        daily_pnl = sum(p.unrealized_pnl + p.realized_pnl for p in self._positions.values())
        self._daily_pnl = daily_pnl
        
        # Store portfolio value history
        portfolio_value = total_value + self._get_cash_balance()
        self._portfolio_values.append((time.time(), portfolio_value))
        
        # Keep only last 1000 values
        if len(self._portfolio_values) > 1000:
            self._portfolio_values = self._portfolio_values[-1000:]
    
    def _get_cash_balance(self) -> float:
        """Get current cash balance"""
        for av in self.ib.accountValues():
            if av.tag == "TotalCashBalance" and av.currency == "USD":
                return float(av.value)
        return 0.0
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        # Try to get from IB ticker
        for ticker in self.ib.pendingTickers():
            if hasattr(ticker, 'contract') and ticker.contract.symbol == symbol:
                if ticker.last and not np.isnan(ticker.last):
                    logger.debug(f"Got price for {symbol} from ticker.last: {ticker.last}")
                    return ticker.last
                elif ticker.bid and ticker.ask and not np.isnan(ticker.bid) and not np.isnan(ticker.ask):
                    price = (ticker.bid + ticker.ask) / 2
                    logger.debug(f"Got price for {symbol} from bid/ask: {price}")
                    return price
        
        # If not found in tickers, check account positions
        for position in self.ib.positions():
            if position.contract.symbol == symbol:
                if position.marketPrice and position.marketPrice > 0:
                    logger.debug(f"Got price for {symbol} from position: {position.marketPrice}")
                    return position.marketPrice
        
        logger.warning(f"Could not find price for {symbol}")
        return None
    
    async def _monitor_risk(self):
        """Background task to monitor risk levels"""
        while True:
            try:
                await asyncio.sleep(self._update_interval)
                
                # Update positions
                await self._update_positions()
                
                # Check risk levels
                portfolio_risk = await self.get_portfolio_risk()
                
                # Emit alerts if needed
                if portfolio_risk.risk_level == RiskLevel.CRITICAL:
                    await self._emit_risk_alert("CRITICAL", portfolio_risk)
                elif portfolio_risk.risk_level == RiskLevel.HIGH:
                    await self._emit_risk_alert("HIGH", portfolio_risk)
                
                # Check specific limits
                if portfolio_risk.current_drawdown > self.limits.max_drawdown_pct * 0.8:
                    await self._emit_risk_limit_warning("drawdown", portfolio_risk.current_drawdown)
                
                if self._daily_pnl < -self.limits.max_daily_loss * 0.8:
                    await self._emit_risk_limit_warning("daily_loss", self._daily_pnl)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
    
    async def _on_order_filled(self, event: Event):
        """Handle order filled events"""
        order_info = event.data.get('order_info')
        if order_info:
            # Update trade count
            self._daily_trades += 1
            
            # Update last order time
            self._last_order_time[order_info.signal.symbol] = time.time()
            
            # Store trade history
            self._trade_history.append({
                'timestamp': time.time(),
                'symbol': order_info.signal.symbol,
                'action': order_info.signal.action,
                'quantity': order_info.signal.quantity,
                'price': order_info.fill_price,
                'commission': order_info.commission
            })
    
    async def _on_position_updated(self, event: Event):
        """Handle position update events"""
        # Trigger position update
        await self._update_positions()
    
    async def _emit_risk_check_failed(self, signal: Signal, reasons: List[str]):
        """Emit risk check failed event"""
        await self.event_bus.emit(Event(
            EventTypes.RISK_CHECK_FAILED,
            {
                'signal': signal,
                'reasons': reasons,
                'timestamp': time.time()
            },
            source="RiskManager"
        ))
    
    async def _emit_risk_alert(self, level: str, portfolio_risk: PortfolioRisk):
        """Emit risk alert"""
        await self.event_bus.emit(Event(
            EventTypes.RISK_LIMIT_BREACHED,
            {
                'level': level,
                'portfolio_risk': portfolio_risk,
                'message': f"Risk level {level}: Drawdown={portfolio_risk.current_drawdown:.1%}, "
                          f"Leverage={portfolio_risk.leverage:.2f}"
            },
            source="RiskManager"
        ))
    
    async def _emit_risk_limit_warning(self, limit_type: str, value: float):
        """Emit risk limit warning"""
        await self.event_bus.emit(Event(
            EventTypes.RISK_LIMIT_BREACHED,
            {
                'limit_type': limit_type,
                'value': value,
                'warning': f"Approaching {limit_type} limit: {value}"
            },
            source="RiskManager"
        ))
    
    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of trading day)"""
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._pnl_history.append(self._daily_pnl)
        
        # Keep only last 252 days (1 year)
        if len(self._pnl_history) > 252:
            self._pnl_history = self._pnl_history[-252:]
        
        logger.info("Daily risk metrics reset")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get risk management metrics"""
        portfolio_risk = asyncio.run(self.get_portfolio_risk())
        
        return {
            'portfolio_value': portfolio_risk.total_value,
            'total_exposure': portfolio_risk.total_exposure,
            'leverage': portfolio_risk.leverage,
            'daily_pnl': self._daily_pnl,
            'current_drawdown': portfolio_risk.current_drawdown,
            'daily_trades': self._daily_trades,
            'position_count': len(self._positions),
            'risk_level': portfolio_risk.risk_level.value,
            'var_95': portfolio_risk.var_95,
            'sharpe_ratio': portfolio_risk.sharpe_ratio
        }