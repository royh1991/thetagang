"""
Simple Momentum Trading Strategy

A simplified momentum strategy that actually trades!
"""

import numpy as np
from collections import deque
from typing import List, Optional, Tuple
from loguru import logger

from strategies.base_strategy import BaseStrategy, StrategyConfig
from core.market_data import TickData
from core.order_manager import Signal, OrderInfo


class SimpleMomentumStrategy(BaseStrategy):
    """
    Simple momentum strategy - buys when price is rising
    
    Much simpler than the enhanced version:
    - No trading window restrictions
    - No market regime filter
    - Simple momentum calculation
    - Fixed stop/target levels
    """
    
    def __init__(self, config: StrategyConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        # Strategy parameters
        self.lookback_period = config.metadata.get('lookback_period', 20)
        self.momentum_threshold = config.metadata.get('momentum_threshold', 0.01)  # 1%
        self.ma_period = config.metadata.get('ma_period', 20)
        
        # Data storage
        self.price_history = {}  # symbol -> deque
        self.ma_values = {}  # symbol -> float
        self.momentum_values = {}  # symbol -> float
    
    async def on_start(self):
        """Initialize strategy"""
        logger.info(f"Simple Momentum strategy starting with symbols: {self.config.symbols}")
        
        # Initialize data structures
        for symbol in self.config.symbols:
            self.price_history[symbol] = deque(maxlen=max(self.ma_period, self.lookback_period))
            self.ma_values[symbol] = 0.0
            self.momentum_values[symbol] = 0.0
    
    async def on_stop(self):
        """Cleanup strategy"""
        logger.info("Simple Momentum strategy stopping")
        self.price_history.clear()
        self.ma_values.clear()
        self.momentum_values.clear()
    
    async def on_tick(self, tick: TickData):
        """Process market data tick"""
        if tick.last is None:
            return
        
        symbol = tick.symbol
        
        # Update price history
        if symbol not in self.price_history:
            return
            
        self.price_history[symbol].append(tick.last)
        
        # Calculate indicators
        if len(self.price_history[symbol]) >= self.lookback_period:
            self._calculate_indicators(symbol)
    
    async def calculate_signals(self, tick: TickData) -> List[Signal]:
        """Generate trading signals"""
        signals = []
        symbol = tick.symbol
        
        # Need enough data
        if len(self.price_history[symbol]) < self.ma_period:
            return signals
        
        # Get current values
        current_price = tick.last
        current_ma = self.ma_values.get(symbol, 0)
        current_momentum = self.momentum_values.get(symbol, 0)
        
        # Simple buy signal
        if (current_price > current_ma and 
            current_momentum > self.momentum_threshold):
            
            logger.info(f"📈 SIMPLE BUY SIGNAL for {symbol}: "
                       f"Price=${current_price:.2f}, MA=${current_ma:.2f}, "
                       f"Momentum={current_momentum:.4f}")
            
            # Fixed stops and targets
            stop_loss = current_price * (1 - self.config.stop_loss_pct)
            take_profit = current_price * (1 + self.config.take_profit_pct)
            
            signal = Signal(
                action="BUY",
                symbol=symbol,
                quantity=0,  # Will be calculated by risk manager
                order_type="MARKET",
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'momentum': current_momentum,
                    'ma_ratio': current_price / current_ma
                }
            )
            signals.append(signal)
        
        # Simple sell signal (if shorting enabled)
        elif (current_price < current_ma and 
              current_momentum < -self.momentum_threshold and
              self.config.metadata.get('allow_shorts', False)):
            
            logger.info(f"📉 SIMPLE SELL SIGNAL for {symbol}: "
                       f"Price=${current_price:.2f}, MA=${current_ma:.2f}, "
                       f"Momentum={current_momentum:.4f}")
            
            stop_loss = current_price * (1 + self.config.stop_loss_pct)
            take_profit = current_price * (1 - self.config.take_profit_pct)
            
            signal = Signal(
                action="SELL",
                symbol=symbol,
                quantity=0,
                order_type="MARKET",
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'momentum': current_momentum,
                    'ma_ratio': current_price / current_ma
                }
            )
            signals.append(signal)
        
        return signals
    
    async def should_close_position(self, tick: TickData, 
                                  position: OrderInfo) -> Tuple[bool, Optional[str]]:
        """Determine if position should be closed"""
        symbol = tick.symbol
        
        if not tick.last:
            return False, None
        
        # Check stop loss
        if position.signal.stop_loss:
            if position.signal.is_buy and tick.last <= position.signal.stop_loss:
                return True, "stop_loss_hit"
            elif position.signal.is_sell and tick.last >= position.signal.stop_loss:
                return True, "stop_loss_hit"
        
        # Check take profit
        if position.signal.take_profit:
            if position.signal.is_buy and tick.last >= position.signal.take_profit:
                return True, "take_profit_hit"
            elif position.signal.is_sell and tick.last <= position.signal.take_profit:
                return True, "take_profit_hit"
        
        # Exit if momentum reverses
        current_momentum = self.momentum_values.get(symbol, 0)
        if position.signal.is_buy and current_momentum < -self.momentum_threshold * 0.5:
            return True, "momentum_reversal"
        elif position.signal.is_sell and current_momentum > self.momentum_threshold * 0.5:
            return True, "momentum_reversal"
        
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
class SimpleMomentumConfig(StrategyConfig):
    """Configuration for Simple Momentum Strategy"""
    def __init__(self, **kwargs):
        # Set defaults
        kwargs.setdefault('name', 'SimpleMomentum')
        kwargs.setdefault('metadata', {}).update({
            'lookback_period': kwargs.get('lookback_period', 20),
            'momentum_threshold': kwargs.get('momentum_threshold', 0.01),
            'ma_period': kwargs.get('ma_period', 20),
            'allow_shorts': kwargs.get('allow_shorts', False)
        })
        super().__init__(**kwargs)