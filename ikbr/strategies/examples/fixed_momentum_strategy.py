"""
Fixed Momentum Trading Strategy

A simplified momentum strategy that actually generates trades throughout the backtest period.
"""

import numpy as np
from collections import deque
from datetime import datetime
from typing import List, Optional, Tuple
from loguru import logger

from strategies.base_strategy import BaseStrategy, StrategyConfig
from core.market_data import TickData
from core.order_manager import Signal, OrderInfo


class FixedMomentumStrategy(BaseStrategy):
    """Fixed momentum strategy that generates more signals"""
    
    def __init__(self, config: StrategyConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        # Strategy parameters
        self.lookback_period = config.metadata.get('lookback_period', 20)
        self.momentum_threshold = config.metadata.get('momentum_threshold', 0.005)  # 0.5%
        self.ma_period = config.metadata.get('ma_period', 20)
        
        # Data storage
        self.price_history = {}
        self.ma_values = {}
        self.momentum_values = {}
        self.last_signal_bar = {}  # Track when we last signaled
        
        # Stats
        self.total_bars = 0
        self.signals_checked = 0
    
    async def on_start(self):
        """Initialize strategy"""
        logger.info(f"Fixed Momentum strategy starting with symbols: {self.config.symbols}")
        
        # Initialize data structures
        for symbol in self.config.symbols:
            self.price_history[symbol] = deque(maxlen=1000)  # Store plenty of history
            self.ma_values[symbol] = 0.0
            self.momentum_values[symbol] = 0.0
            self.last_signal_bar[symbol] = -1000  # Can signal immediately
    
    async def on_stop(self):
        """Cleanup strategy"""
        logger.info(f"Fixed Momentum strategy stopping. Bars: {self.total_bars}, Signals checked: {self.signals_checked}")
    
    async def on_tick(self, tick: TickData):
        """Process market data tick"""
        if tick.last is None or tick.symbol not in self.config.symbols:
            return
        
        self.total_bars += 1
        symbol = tick.symbol
        
        # Update price history
        self.price_history[symbol].append(tick.last)
        
        # Calculate indicators if we have enough data
        if len(self.price_history[symbol]) >= self.lookback_period:
            self._calculate_indicators(symbol)
    
    async def calculate_signals(self, tick: TickData) -> List[Signal]:
        """Generate trading signals"""
        signals = []
        symbol = tick.symbol
        
        if symbol not in self.config.symbols:
            return signals
        
        self.signals_checked += 1
        
        # Need enough data
        if len(self.price_history[symbol]) < max(self.ma_period, self.lookback_period):
            return signals
        
        # Don't signal if we have a position
        if symbol in self._positions:
            return signals
        
        # Require some bars between signals (prevent rapid fire)
        bars_since_last = self.total_bars - self.last_signal_bar.get(symbol, -1000)
        if bars_since_last < 50:  # At least 50 bars between signals
            return signals
        
        # Get current values
        current_price = tick.last
        current_ma = self.ma_values.get(symbol, 0)
        current_momentum = self.momentum_values.get(symbol, 0)
        
        # Simple momentum strategy: buy when momentum is positive and price above MA
        if current_price > current_ma and current_momentum > self.momentum_threshold:
            
            logger.info(f"🚀 BUY SIGNAL for {symbol}: "
                       f"Price=${current_price:.2f}, MA=${current_ma:.2f}, "
                       f"Momentum={current_momentum:.4f}")
            
            # Simple fixed stops
            stop_loss = current_price * 0.97    # 3% stop
            take_profit = current_price * 1.05  # 5% target
            
            signal = Signal(
                action="BUY",
                symbol=symbol,
                quantity=0,  # Risk manager will calculate
                order_type="MARKET",
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'momentum': current_momentum,
                    'ma_ratio': current_price / current_ma,
                    'entry_price': current_price
                }
            )
            signals.append(signal)
            self.last_signal_bar[symbol] = self.total_bars
        
        return signals
    
    async def should_close_position(self, tick: TickData, 
                                  position: OrderInfo) -> Tuple[bool, Optional[str]]:
        """Determine if position should be closed"""
        symbol = tick.symbol
        
        if not tick.last:
            return False, None
        
        # Check fixed stop loss
        if position.signal.stop_loss:
            if position.signal.is_buy and tick.last <= position.signal.stop_loss:
                return True, "stop_loss_hit"
        
        # Check take profit
        if position.signal.take_profit:
            if position.signal.is_buy and tick.last >= position.signal.take_profit:
                return True, "take_profit_hit"
        
        # Exit if momentum turns strongly negative
        current_momentum = self.momentum_values.get(symbol, 0)
        if position.signal.is_buy and current_momentum < -self.momentum_threshold:
            return True, "momentum_reversal"
        
        # Exit if price drops below MA by significant amount
        current_ma = self.ma_values.get(symbol, tick.last)
        if position.signal.is_buy and tick.last < current_ma * 0.98:  # 2% below MA
            return True, "below_ma_exit"
        
        return False, None
    
    def _calculate_indicators(self, symbol: str):
        """Calculate technical indicators"""
        prices = list(self.price_history[symbol])
        
        # Calculate moving average
        if len(prices) >= self.ma_period:
            self.ma_values[symbol] = np.mean(prices[-self.ma_period:])
        
        # Calculate momentum (rate of change)
        if len(prices) >= self.lookback_period:
            old_price = prices[-self.lookback_period]
            current_price = prices[-1]
            self.momentum_values[symbol] = (current_price - old_price) / old_price


# Configuration helper
class FixedMomentumConfig(StrategyConfig):
    """Configuration for Fixed Momentum Strategy"""
    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'FixedMomentum')
        kwargs.setdefault('metadata', {}).update({
            'lookback_period': kwargs.get('lookback_period', 20),
            'momentum_threshold': kwargs.get('momentum_threshold', 0.005),
            'ma_period': kwargs.get('ma_period', 20)
        })
        super().__init__(**kwargs)