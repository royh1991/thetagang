"""
Momentum Trading Strategy

A simple momentum strategy that buys when price momentum is strong
and sells when momentum weakens.
"""

import numpy as np
from collections import deque
from typing import List, Optional, Tuple
from loguru import logger

from strategies.base_strategy import BaseStrategy, StrategyConfig
from core.market_data import TickData
from core.order_manager import Signal, OrderInfo


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy implementation
    
    Buy when:
    - Price is above moving average
    - Momentum (rate of change) is positive and accelerating
    - Volume is above average
    
    Sell when:
    - Stop loss hit
    - Momentum turns negative
    - Maximum holding period reached
    """
    
    def __init__(self, config: StrategyConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        # Strategy parameters
        self.lookback_period = config.metadata.get('lookback_period', 20)
        self.momentum_threshold = config.metadata.get('momentum_threshold', 0.02)
        self.volume_multiplier = config.metadata.get('volume_multiplier', 1.5)
        self.ma_period = config.metadata.get('ma_period', 50)
        
        # Data storage
        self.price_history = {}  # symbol -> deque
        self.volume_history = {}  # symbol -> deque
        self.ma_values = {}  # symbol -> float
        self.momentum_values = {}  # symbol -> float
    
    async def on_start(self):
        """Initialize strategy"""
        logger.info(f"Momentum strategy starting with symbols: {self.config.symbols}")
        
        # Initialize data structures
        for symbol in self.config.symbols:
            self.price_history[symbol] = deque(maxlen=self.ma_period)
            self.volume_history[symbol] = deque(maxlen=self.lookback_period)
            self.ma_values[symbol] = 0.0
            self.momentum_values[symbol] = 0.0
            
            # Load historical data if available
            await self._load_historical_data(symbol)
    
    async def on_stop(self):
        """Cleanup strategy"""
        logger.info("Momentum strategy stopping")
        # Clear data
        self.price_history.clear()
        self.volume_history.clear()
        self.ma_values.clear()
        self.momentum_values.clear()
    
    async def on_tick(self, tick: TickData):
        """Process market data tick"""
        if tick.last is None:
            return
        
        symbol = tick.symbol
        
        
        # Update price history
        if symbol not in self.price_history:
            logger.warning(f"Symbol {symbol} not in price_history! Available symbols: {list(self.price_history.keys())}")
            return
            
        self.price_history[symbol].append(tick.last)
        
        # Update volume history
        if tick.volume:
            self.volume_history[symbol].append(tick.volume)
        
        # Calculate indicators
        if len(self.price_history[symbol]) >= self.lookback_period:
            self._calculate_indicators(symbol)
    
    async def calculate_signals(self, tick: TickData) -> List[Signal]:
        """Generate trading signals"""
        signals = []
        symbol = tick.symbol
        
        # Need enough data
        # Need enough data
        if len(self.price_history[symbol]) < self.ma_period:
            return signals
        
        # Get current values
        current_price = tick.last
        current_ma = self.ma_values.get(symbol, 0)
        current_momentum = self.momentum_values.get(symbol, 0)
        
        
        # Check volume condition
        if self.volume_history[symbol]:
            avg_volume = np.mean(list(self.volume_history[symbol]))
            current_volume = tick.volume or 0
            volume_condition = current_volume > avg_volume * self.volume_multiplier
        else:
            volume_condition = True  # No volume data, ignore condition
        
            
        # Buy signal conditions
        if (current_price > current_ma and 
            current_momentum > self.momentum_threshold and
            volume_condition):
            
            logger.info(f"BUY SIGNAL for {symbol}: Price={current_price:.2f} > MA={current_ma:.2f}, Momentum={current_momentum:.4f} > {self.momentum_threshold}")
            
            # Calculate position size based on momentum strength
            position_score = min(current_momentum / self.momentum_threshold, 2.0)
            
            # Calculate stop loss and take profit
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
                    'ma_ratio': current_price / current_ma,
                    'position_score': position_score
                }
            )
            signals.append(signal)
            
            logger.info(f"Momentum BUY signal for {symbol}: "
                       f"momentum={current_momentum:.3f}, "
                       f"price/MA={current_price/current_ma:.3f}")
        
        # Sell/Short signal conditions
        elif (current_price < current_ma and 
              current_momentum < -self.momentum_threshold and
              volume_condition):
            
            logger.info(f"SELL SIGNAL for {symbol}: Price={current_price:.2f} < MA={current_ma:.2f}, Momentum={current_momentum:.4f} < {-self.momentum_threshold}")
            
            # Calculate position size based on momentum strength
            position_score = min(abs(current_momentum) / self.momentum_threshold, 2.0)
            
            # Calculate stop loss and take profit for short position
            stop_loss = current_price * (1 + self.config.stop_loss_pct)  # Higher for shorts
            take_profit = current_price * (1 - self.config.take_profit_pct)  # Lower for shorts
            
            signal = Signal(
                action="SELL",
                symbol=symbol,
                quantity=0,  # Will be calculated by risk manager
                order_type="MARKET",
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'momentum': current_momentum,
                    'ma_ratio': current_price / current_ma,
                    'position_score': position_score
                }
            )
            signals.append(signal)
            
            logger.info(f"Momentum SELL signal for {symbol}: "
                       f"momentum={current_momentum:.3f}, "
                       f"price/MA={current_price/current_ma:.3f}")
        
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
        
        # Check momentum reversal
        current_momentum = self.momentum_values.get(symbol, 0)
        if position.signal.is_buy and current_momentum < -self.momentum_threshold * 0.5:
            return True, "momentum_reversal"
        
        # Check maximum holding period
        if self.config.max_holding_period:
            holding_time = tick.timestamp - position.submit_time
            max_seconds = self.config.max_holding_period * 86400  # Convert days to seconds
            if holding_time > max_seconds:
                return True, "max_holding_period"
        
        # Check if price falls below MA
        current_ma = self.ma_values.get(symbol, 0)
        if current_ma > 0 and tick.last < current_ma * 0.98:  # 2% below MA
            return True, "below_moving_average"
        
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
    
    async def _load_historical_data(self, symbol: str):
        """Load historical data for indicators"""
        try:
            # Get historical bars
            bars = await self.market_data.get_historical_bars(
                symbol=symbol,
                duration='2 D',
                bar_size='5 mins'
            )
            
            if bars:
                # Populate price history
                for bar in bars[-self.ma_period:]:
                    self.price_history[symbol].append(bar.close)
                
                # Calculate initial indicators
                if len(self.price_history[symbol]) >= self.lookback_period:
                    self._calculate_indicators(symbol)
                
                logger.info(f"Loaded {len(bars)} historical bars for {symbol}")
        
        except Exception as e:
            logger.warning(f"Failed to load historical data for {symbol}: {e}")


class MomentumConfig(StrategyConfig):
    """Configuration specific to momentum strategy"""
    
    def __init__(self, symbols: List[str], **kwargs):
        # Set momentum-specific defaults
        metadata = kwargs.get('metadata', {})
        metadata.setdefault('lookback_period', 20)
        metadata.setdefault('momentum_threshold', 0.02)
        metadata.setdefault('volume_multiplier', 1.5)
        metadata.setdefault('ma_period', 50)
        
        # Remove conflicting kwargs
        kwargs.pop('metadata', None)
        kwargs.setdefault('stop_loss_pct', 0.02)
        kwargs.setdefault('take_profit_pct', 0.05)
        kwargs.setdefault('max_holding_period', 5)
        
        super().__init__(
            name="Momentum",
            symbols=symbols,
            metadata=metadata,
            **kwargs
        )