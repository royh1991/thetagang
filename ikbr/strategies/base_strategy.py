"""
Base Strategy Abstract Class

Provides the foundation for all trading strategies with standard interfaces
for signal generation, position management, and performance tracking.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum
from loguru import logger

from core.event_bus import EventBus, Event, EventTypes, get_event_bus
from core.market_data import TickData, MarketDataManager
from core.order_manager import Signal, OrderInfo, OrderManager
from core.risk_manager import RiskManager


class StrategyState(Enum):
    """Strategy state enumeration"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class StrategyConfig:
    """Base configuration for strategies"""
    name: str
    symbols: List[str]
    enabled: bool = True
    max_positions: int = 5
    position_size_pct: float = 0.2  # 20% of capital per position
    stop_loss_pct: Optional[float] = 0.02  # 2% stop loss
    take_profit_pct: Optional[float] = 0.05  # 5% take profit
    max_holding_period: Optional[int] = None  # Days
    trade_start_time: str = "09:30"  # Market open
    trade_end_time: str = "15:30"  # 30 min before close
    cooldown_period: float = 60.0  # Seconds between signals
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyMetrics:
    """Strategy performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: Optional[float] = None
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    trades_today: int = 0
    pnl_today: float = 0.0
    
    def update_trade(self, pnl: float):
        """Update metrics with a completed trade"""
        self.total_trades += 1
        self.trades_today += 1
        self.total_pnl += pnl
        self.pnl_today += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            self.gross_profit += pnl
        else:
            self.losing_trades += 1
            self.gross_loss += abs(pnl)
        
        # Update calculated metrics
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        
        if self.winning_trades > 0:
            self.avg_win = self.gross_profit / self.winning_trades
        
        if self.losing_trades > 0:
            self.avg_loss = self.gross_loss / self.losing_trades
        
        if self.gross_loss > 0:
            self.profit_factor = self.gross_profit / self.gross_loss


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies
    
    Subclasses must implement:
    - on_start(): Initialize strategy
    - on_tick(): Process market data ticks
    - calculate_signals(): Generate trading signals
    - should_close_position(): Determine when to exit
    """
    
    def __init__(self, config: StrategyConfig,
                 market_data: MarketDataManager,
                 order_manager: OrderManager,
                 risk_manager: RiskManager):
        self.config = config
        self.market_data = market_data
        self.order_manager = order_manager
        self.risk_manager = risk_manager
        self.event_bus = get_event_bus()
        
        # Strategy state
        self.state = StrategyState.STOPPED
        self.metrics = StrategyMetrics()
        
        # Position tracking
        self._positions: Dict[str, OrderInfo] = {}  # symbol -> OrderInfo
        self._pending_orders: Set[str] = set()  # order_ids
        self._last_signal_time: Dict[str, float] = {}  # symbol -> timestamp
        self._processed_orders: Set[str] = set()  # Track processed order IDs to avoid duplicates
        
        # Strategy-specific data storage
        self._data: Dict[str, Any] = {}
        
        # Subscribe to events
        self._setup_event_handlers()
        
    def _setup_event_handlers(self):
        """Setup event handlers"""
        self.event_bus.subscribe(EventTypes.TICK, self._on_tick_event)
        self.event_bus.subscribe(EventTypes.ORDER_FILLED, self._on_order_filled)
        self.event_bus.subscribe(EventTypes.ORDER_REJECTED, self._on_order_rejected)
        self.event_bus.subscribe(EventTypes.ORDER_CANCELLED, self._on_order_cancelled)
    
    async def start(self):
        """Start the strategy"""
        if self.state != StrategyState.STOPPED:
            logger.warning(f"Strategy {self.config.name} already running")
            return
        
        try:
            self.state = StrategyState.STARTING
            logger.info(f"Starting strategy: {self.config.name}")
            
            # Subscribe to market data
            for symbol in self.config.symbols:
                success = await self.market_data.subscribe_ticker(symbol)
                if not success:
                    logger.error(f"Failed to subscribe to {symbol}")
                    self.state = StrategyState.ERROR
                    return
            
            # Call strategy-specific initialization
            await self.on_start()
            
            self.state = StrategyState.RUNNING
            logger.info(f"Strategy {self.config.name} started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start strategy: {e}")
            self.state = StrategyState.ERROR
            raise
    
    async def stop(self):
        """Stop the strategy"""
        if self.state == StrategyState.STOPPED:
            return
        
        logger.info(f"Stopping strategy: {self.config.name}")
        self.state = StrategyState.STOPPED
        
        # Cancel pending orders
        for order_id in list(self._pending_orders):
            await self.order_manager.cancel_order(order_id)
        
        # Unsubscribe from market data
        for symbol in self.config.symbols:
            await self.market_data.unsubscribe_ticker(symbol)
        
        # Call strategy-specific cleanup
        await self.on_stop()
        
        logger.info(f"Strategy {self.config.name} stopped")
    
    async def pause(self):
        """Pause the strategy (keeps positions but stops new signals)"""
        if self.state == StrategyState.RUNNING:
            self.state = StrategyState.PAUSED
            logger.info(f"Strategy {self.config.name} paused")
    
    async def resume(self):
        """Resume a paused strategy"""
        if self.state == StrategyState.PAUSED:
            self.state = StrategyState.RUNNING
            logger.info(f"Strategy {self.config.name} resumed")
    
    async def _on_tick_event(self, event: Event):
        """Handle market data tick events"""
        if self.state != StrategyState.RUNNING:
            return
        
        tick_data: TickData = event.data
        if tick_data.symbol not in self.config.symbols:
            return
        
        try:
            # Let strategy process the tick
            await self.on_tick(tick_data)
            
            # Check existing positions first before generating new signals
            await self._check_positions(tick_data)
            
            # Check if we should generate signals
            should_gen = self._should_generate_signal(tick_data.symbol)
            logger.debug(f"Should generate signal for {tick_data.symbol}: {should_gen}, positions: {list(self._positions.keys())}")
            if should_gen:
                signals = await self.calculate_signals(tick_data)
                if signals:
                    logger.info(f"Generated {len(signals)} signals for {tick_data.symbol}")
                    
                    for signal in signals:
                        logger.debug(f"Validating signal for {signal.symbol}")
                        if await self._validate_signal(signal):
                            logger.info(f"Signal validated, submitting order for {signal.symbol}")
                            await self._submit_signal(signal)
                        else:
                            logger.warning(f"Signal validation failed for {signal.symbol}")
            
        except Exception as e:
            logger.error(f"Error processing tick for {tick_data.symbol}: {e}", exc_info=True)
            await self._emit_strategy_error(str(e))
    
    def _should_generate_signal(self, symbol: str) -> bool:
        """Check if we should generate a signal for this symbol"""
        # Check cooldown period
        last_signal = self._last_signal_time.get(symbol, 0)
        time_since_last = time.time() - last_signal
        if time_since_last < self.config.cooldown_period:
            logger.debug(f"Cooldown active for {symbol}: {time_since_last:.1f}s < {self.config.cooldown_period}s")
            return False
        
        # Check trading hours
        if not self._is_trading_time():
            logger.debug(f"Outside trading hours for {symbol}")
            return False
        
        # Check if we already have a position
        if symbol in self._positions:
            logger.debug(f"Already have position in {symbol}")
            return False
        
        # Check position limits
        if len(self._positions) >= self.config.max_positions:
            logger.debug(f"Position limit reached: {len(self._positions)} >= {self.config.max_positions}")
            return False
        
        logger.debug(f"Signal generation allowed for {symbol}")
        return True
    
    def _is_trading_time(self) -> bool:
        """Check if current time is within trading hours"""
        # For backtesting, always return True since we're using historical data
        # In live trading, this would check actual market hours
        return True
    
    async def _validate_signal(self, signal: Signal) -> bool:
        """Validate a trading signal"""
        # Don't check quantity here as it will be calculated later
        
        # Check if we already have a position
        if signal.symbol in self._positions:
            logger.debug(f"Already have position in {signal.symbol}")
            return False
        
        # Add strategy metadata
        signal.strategy_id = self.config.name
        
        logger.debug(f"Signal validation passed for {signal.symbol}")
        return True
    
    async def _submit_signal(self, signal: Signal):
        """Submit a trading signal"""
        logger.debug(f"Starting _submit_signal for {signal.symbol}")
        
        # Calculate position size if not specified
        if signal.quantity == 0:
            logger.debug(f"Calculating position size for {signal.symbol}")
            try:
                signal.quantity = await self.risk_manager.calculate_position_size(signal)
                logger.info(f"Position size for {signal.symbol}: {signal.quantity}")
                if signal.quantity == 0:
                    logger.warning(f"Position size calculation returned 0 for {signal.symbol}")
                    return
            except Exception as e:
                logger.error(f"Error calculating position size for {signal.symbol}: {e}")
                return
        
        # Add stop loss and take profit if configured
        if self.config.stop_loss_pct and not signal.stop_loss:
            if signal.limit_price:
                base_price = signal.limit_price
            else:
                tick = self.market_data.get_latest_tick(signal.symbol)
                base_price = tick.last if tick else None
            
            if base_price:
                if signal.is_buy:
                    signal.stop_loss = base_price * (1 - self.config.stop_loss_pct)
                else:
                    signal.stop_loss = base_price * (1 + self.config.stop_loss_pct)
        
        if self.config.take_profit_pct and not signal.take_profit:
            if signal.limit_price:
                base_price = signal.limit_price
            else:
                tick = self.market_data.get_latest_tick(signal.symbol)
                base_price = tick.last if tick else None
            
            if base_price:
                if signal.is_buy:
                    signal.take_profit = base_price * (1 + self.config.take_profit_pct)
                else:
                    signal.take_profit = base_price * (1 - self.config.take_profit_pct)
        
        # Submit order
        logger.debug(f"About to call order_manager.submit_order for {signal.symbol}")
        order_info = await self.order_manager.submit_order(signal)
        
        if order_info:
            self._pending_orders.add(order_info.order_id)
            self._last_signal_time[signal.symbol] = time.time()
            logger.info(f"Strategy {self.config.name} submitted order for {signal.symbol}")
            await self._emit_signal_generated(signal)
        else:
            logger.warning(f"order_manager.submit_order returned None for {signal.symbol}")
    
    async def _check_positions(self, tick_data: TickData):
        """Check existing positions for exit conditions"""
        if tick_data.symbol not in self._positions:
            return
        
        # Check if we already have a pending close order for this symbol
        for order_id in self._pending_orders:
            order = self.order_manager.get_order(order_id)
            if order and order.signal.symbol == tick_data.symbol and order.signal.metadata.get("close_reason"):
                logger.debug(f"Already have pending close order for {tick_data.symbol}")
                return
        
        order_info = self._positions[tick_data.symbol]
        
        # Check if we should close the position
        should_close, reason = await self.should_close_position(tick_data, order_info)
        
        if should_close:
            logger.info(f"Position exit triggered for {tick_data.symbol}: {reason}")
            await self._close_position(tick_data.symbol, reason)
    
    async def _close_position(self, symbol: str, reason: str):
        """Close a position"""
        if symbol not in self._positions:
            return
        
        order_info = self._positions[symbol]
        
        # Create closing signal
        action = "SELL" if order_info.signal.is_buy else "BUY"
        close_signal = Signal(
            action=action,
            symbol=symbol,
            quantity=order_info.signal.quantity,
            order_type="MARKET",
            strategy_id=self.config.name,
            metadata={"close_reason": reason}
        )
        
        # Submit closing order
        close_order = await self.order_manager.submit_order(close_signal)
        
        if close_order:
            self._pending_orders.add(close_order.order_id)
            logger.info(f"Strategy {self.config.name} closing position in {symbol}: {reason}")
    
    async def _on_order_filled(self, event: Event):
        """Handle order filled events"""
        order_info: OrderInfo = event.data.get('order_info')
        if not order_info or order_info.signal.strategy_id != self.config.name:
            return
        
        # Check if we've already processed this order fill
        if order_info.order_id in self._processed_orders:
            logger.debug(f"Ignoring duplicate order filled event for {order_info.order_id}")
            return
        
        # Mark as processed
        self._processed_orders.add(order_info.order_id)
        
        logger.info(f"Order filled event for {order_info.signal.symbol}: {order_info.signal.action}, "
                   f"close_reason: {order_info.signal.metadata.get('close_reason')}")
        
        self._pending_orders.discard(order_info.order_id)
        
        # Check if this is an opening or closing order
        if order_info.signal.metadata.get("close_reason"):
            # Closing order
            symbol = order_info.signal.symbol
            if symbol in self._positions:
                # Calculate P&L
                open_order = self._positions[symbol]
                pnl = self._calculate_pnl(open_order, order_info)
                
                # Update metrics
                self.metrics.update_trade(pnl)
                
                # Remove position
                del self._positions[symbol]
                
                logger.info(f"Strategy {self.config.name} closed {symbol} "
                          f"P&L: ${pnl:.2f}")
            else:
                logger.warning(f"Closing order for {symbol} but no position found")
        else:
            # Opening order
            # Store entry price in the signal metadata for future reference
            order_info.signal.metadata['entry_price'] = order_info.fill_price
            self._positions[order_info.signal.symbol] = order_info
            logger.info(f"Strategy {self.config.name} opened position in "
                       f"{order_info.signal.symbol} at ${order_info.fill_price:.2f}, "
                       f"total positions: {len(self._positions)}")
    
    async def _on_order_rejected(self, event: Event):
        """Handle order rejected events"""
        order_id = event.data.get('order_id')
        if order_id in self._pending_orders:
            self._pending_orders.discard(order_id)
            signal = event.data.get('signal')
            if signal:
                self._last_signal_time[signal.symbol] = 0  # Reset cooldown
    
    async def _on_order_cancelled(self, event: Event):
        """Handle order cancelled events"""
        order_info: OrderInfo = event.data.get('order_info')
        if order_info and order_info.order_id in self._pending_orders:
            self._pending_orders.discard(order_info.order_id)
    
    def _calculate_pnl(self, open_order: OrderInfo, close_order: OrderInfo) -> float:
        """Calculate P&L for a closed position"""
        if open_order.signal.is_buy:
            pnl = (close_order.fill_price - open_order.fill_price) * open_order.signal.quantity
        else:
            pnl = (open_order.fill_price - close_order.fill_price) * open_order.signal.quantity
        
        # Subtract commissions
        pnl -= (open_order.commission or 0) + (close_order.commission or 0)
        
        return pnl
    
    async def _emit_signal_generated(self, signal: Signal):
        """Emit signal generated event"""
        await self.event_bus.emit(Event(
            EventTypes.SIGNAL_GENERATED,
            {
                'strategy': self.config.name,
                'signal': signal
            },
            source=f"Strategy.{self.config.name}"
        ))
    
    async def _emit_strategy_error(self, error: str):
        """Emit strategy error event"""
        await self.event_bus.emit(Event(
            EventTypes.STRATEGY_ERROR,
            {
                'strategy': self.config.name,
                'error': error,
                'state': self.state.value
            },
            source=f"Strategy.{self.config.name}"
        ))
    
    def get_positions(self) -> Dict[str, OrderInfo]:
        """Get current positions"""
        return self._positions.copy()
    
    def get_metrics(self) -> StrategyMetrics:
        """Get strategy metrics"""
        return self.metrics
    
    def reset_daily_metrics(self):
        """Reset daily metrics"""
        self.metrics.trades_today = 0
        self.metrics.pnl_today = 0.0
    
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    async def on_start(self):
        """Called when strategy starts"""
        pass
    
    @abstractmethod
    async def on_stop(self):
        """Called when strategy stops"""
        pass
    
    @abstractmethod
    async def on_tick(self, tick: TickData):
        """Process market data tick"""
        pass
    
    @abstractmethod
    async def calculate_signals(self, tick: TickData) -> List[Signal]:
        """Calculate trading signals based on current market data"""
        pass
    
    @abstractmethod
    async def should_close_position(self, tick: TickData, 
                                  position: OrderInfo) -> tuple[bool, Optional[str]]:
        """
        Determine if a position should be closed
        
        Returns:
            tuple: (should_close, reason)
        """
        pass