"""
Mean Reversion Trading Strategy

A mean reversion strategy that trades on the assumption that prices
will revert to their historical mean.
"""

import numpy as np
from collections import deque
from typing import List, Optional, Tuple
from loguru import logger

from strategies.base_strategy import BaseStrategy, StrategyConfig
from core.market_data import TickData
from core.order_manager import Signal, OrderInfo


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy implementation
    
    Buy when:
    - Price is significantly below moving average (oversold)
    - RSI is below oversold threshold
    - Bollinger Band lower band is breached
    
    Sell when:
    - Price reverts to mean or above
    - RSI reaches overbought levels
    - Stop loss or take profit hit
    """
    
    def __init__(self, config: StrategyConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        # Strategy parameters
        self.ma_period = config.metadata.get('ma_period', 20)
        self.bb_std = config.metadata.get('bb_std', 2.0)
        self.rsi_period = config.metadata.get('rsi_period', 14)
        self.rsi_oversold = config.metadata.get('rsi_oversold', 30)
        self.rsi_overbought = config.metadata.get('rsi_overbought', 70)
        self.entry_threshold = config.metadata.get('entry_threshold', 0.02)  # 2% below MA
        
        # Data storage
        self.price_history = {}  # symbol -> deque
        self.rsi_values = {}  # symbol -> float
        self.bb_upper = {}  # symbol -> float
        self.bb_lower = {}  # symbol -> float
        self.bb_middle = {}  # symbol -> float
    
    async def on_start(self):
        """Initialize strategy"""
        logger.info(f"Mean reversion strategy starting with symbols: {self.config.symbols}")
        
        # Initialize data structures
        for symbol in self.config.symbols:
            self.price_history[symbol] = deque(maxlen=max(self.ma_period, self.rsi_period) + 1)
            self.rsi_values[symbol] = 50.0  # Neutral RSI
            self.bb_upper[symbol] = 0.0
            self.bb_lower[symbol] = 0.0
            self.bb_middle[symbol] = 0.0
            
            # Load historical data
            await self._load_historical_data(symbol)
    
    async def on_stop(self):
        """Cleanup strategy"""
        logger.info("Mean reversion strategy stopping")
        # Clear data
        self.price_history.clear()
        self.rsi_values.clear()
        self.bb_upper.clear()
        self.bb_lower.clear()
        self.bb_middle.clear()
    
    async def on_tick(self, tick: TickData):
        """Process market data tick"""
        if tick.last is None:
            return
        
        symbol = tick.symbol
        
        # Update price history
        self.price_history[symbol].append(tick.last)
        
        # Calculate indicators
        if len(self.price_history[symbol]) >= self.ma_period:
            self._calculate_indicators(symbol)
    
    async def calculate_signals(self, tick: TickData) -> List[Signal]:
        """Generate trading signals"""
        signals = []
        symbol = tick.symbol
        
        # Need enough data
        if len(self.price_history[symbol]) < self.ma_period:
            return signals
        
        current_price = tick.last
        ma = self.bb_middle[symbol]
        upper_band = self.bb_upper[symbol]
        lower_band = self.bb_lower[symbol]
        rsi = self.rsi_values[symbol]
        
        # Calculate price deviation from mean
        if ma > 0:
            deviation = (current_price - ma) / ma
        else:
            return signals
        
        # Buy signal - oversold conditions
        if (current_price <= lower_band and 
            rsi < self.rsi_oversold and
            deviation < -self.entry_threshold):
            
            # Stronger signal if more oversold
            signal_strength = min(abs(deviation) / self.entry_threshold, 2.0)
            
            signal = Signal(
                action="BUY",
                symbol=symbol,
                quantity=0,  # Will be calculated by risk manager
                order_type="LIMIT",
                limit_price=current_price * 1.001,  # Slightly above market
                metadata={
                    'rsi': rsi,
                    'deviation': deviation,
                    'signal_strength': signal_strength,
                    'bb_position': 'below_lower'
                }
            )
            signals.append(signal)
            
            logger.info(f"Mean reversion BUY signal for {symbol}: "
                       f"RSI={rsi:.1f}, deviation={deviation:.3f}, "
                       f"price={current_price:.2f}, lower_band={lower_band:.2f}")
        
        # Short signal - overbought conditions
        elif (current_price >= upper_band and 
              rsi > self.rsi_overbought and
              deviation > self.entry_threshold):
            
            signal_strength = min(deviation / self.entry_threshold, 2.0)
            
            signal = Signal(
                action="SELL",
                symbol=symbol,
                quantity=0,
                order_type="LIMIT",
                limit_price=current_price * 0.999,  # Slightly below market
                metadata={
                    'rsi': rsi,
                    'deviation': deviation,
                    'signal_strength': signal_strength,
                    'bb_position': 'above_upper'
                }
            )
            signals.append(signal)
            
            logger.info(f"Mean reversion SELL signal for {symbol}: "
                       f"RSI={rsi:.1f}, deviation={deviation:.3f}, "
                       f"price={current_price:.2f}, upper_band={upper_band:.2f}")
        
        return signals
    
    async def should_close_position(self, tick: TickData, 
                                  position: OrderInfo) -> Tuple[bool, Optional[str]]:
        """Determine if position should be closed"""
        symbol = tick.symbol
        
        if not tick.last:
            return False, None
        
        # Check stop loss and take profit
        if position.signal.stop_loss:
            if position.signal.is_buy and tick.last <= position.signal.stop_loss:
                return True, "stop_loss_hit"
            elif position.signal.is_sell and tick.last >= position.signal.stop_loss:
                return True, "stop_loss_hit"
        
        if position.signal.take_profit:
            if position.signal.is_buy and tick.last >= position.signal.take_profit:
                return True, "take_profit_hit"
            elif position.signal.is_sell and tick.last <= position.signal.take_profit:
                return True, "take_profit_hit"
        
        # Check mean reversion completion
        ma = self.bb_middle.get(symbol, 0)
        rsi = self.rsi_values.get(symbol, 50)
        
        if ma > 0:
            if position.signal.is_buy:
                # Close long when price returns to mean or RSI overbought
                if tick.last >= ma * 0.995 or rsi > self.rsi_overbought - 5:
                    return True, "mean_reversion_complete"
            else:
                # Close short when price returns to mean or RSI oversold
                if tick.last <= ma * 1.005 or rsi < self.rsi_oversold + 5:
                    return True, "mean_reversion_complete"
        
        # Check maximum holding period
        if self.config.max_holding_period:
            holding_time = tick.timestamp - position.submit_time
            max_seconds = self.config.max_holding_period * 86400
            if holding_time > max_seconds:
                return True, "max_holding_period"
        
        return False, None
    
    def _calculate_indicators(self, symbol: str):
        """Calculate technical indicators"""
        prices = np.array(list(self.price_history[symbol]))
        
        # Calculate Bollinger Bands
        if len(prices) >= self.ma_period:
            ma = np.mean(prices[-self.ma_period:])
            std = np.std(prices[-self.ma_period:])
            
            self.bb_middle[symbol] = ma
            self.bb_upper[symbol] = ma + (self.bb_std * std)
            self.bb_lower[symbol] = ma - (self.bb_std * std)
        
        # Calculate RSI
        if len(prices) > self.rsi_period:
            self.rsi_values[symbol] = self._calculate_rsi(prices[-self.rsi_period-1:])
    
    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        gains = deltas.copy()
        losses = deltas.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def _load_historical_data(self, symbol: str):
        """Load historical data for indicators"""
        try:
            # Get historical bars
            bars = await self.market_data.get_historical_bars(
                symbol=symbol,
                duration='5 D',
                bar_size='30 mins'
            )
            
            if bars:
                # Populate price history
                for bar in bars[-(self.ma_period + self.rsi_period):]:
                    self.price_history[symbol].append(bar.close)
                
                # Calculate initial indicators
                if len(self.price_history[symbol]) >= self.ma_period:
                    self._calculate_indicators(symbol)
                
                logger.info(f"Loaded {len(bars)} historical bars for {symbol}")
        
        except Exception as e:
            logger.warning(f"Failed to load historical data for {symbol}: {e}")


class MeanReversionConfig(StrategyConfig):
    """Configuration specific to mean reversion strategy"""
    
    def __init__(self, symbols: List[str], **kwargs):
        # Set mean reversion specific defaults
        metadata = kwargs.get('metadata', {})
        metadata.setdefault('ma_period', 20)
        metadata.setdefault('bb_std', 2.0)
        metadata.setdefault('rsi_period', 14)
        metadata.setdefault('rsi_oversold', 30)
        metadata.setdefault('rsi_overbought', 70)
        metadata.setdefault('entry_threshold', 0.02)
        
        # Remove conflicting kwargs
        kwargs.pop('metadata', None)
        kwargs.setdefault('stop_loss_pct', 0.03)  # Wider stop for mean reversion
        kwargs.setdefault('take_profit_pct', 0.02)  # Smaller profit target
        kwargs.setdefault('max_holding_period', 3)  # Shorter holding period
        kwargs.setdefault('cooldown_period', 300)  # 5 min cooldown
        
        super().__init__(
            name="MeanReversion",
            symbols=symbols,
            metadata=metadata,
            **kwargs
        )