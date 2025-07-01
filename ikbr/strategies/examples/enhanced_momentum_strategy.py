"""
Enhanced Momentum Trading Strategy

An improved momentum strategy that:
- Focuses on high-volume periods (first/last 30 minutes)
- Uses market regime filtering
- Implements momentum acceleration detection
- Includes volatility-based position sizing
- Better entry/exit logic
"""

import numpy as np
from collections import deque
from datetime import datetime, time
from typing import List, Optional, Tuple
from loguru import logger

from strategies.base_strategy import BaseStrategy, StrategyConfig
from core.market_data import TickData
from core.order_manager import Signal, OrderInfo


class EnhancedMomentumStrategy(BaseStrategy):
    """
    Enhanced momentum strategy implementation
    
    Key improvements:
    - Trade during high-volume periods (market open/close)
    - Market regime filter using SPY as proxy
    - Momentum acceleration detection
    - Dynamic position sizing based on volatility
    - Trailing stop losses
    """
    
    def __init__(self, config: StrategyConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        # Strategy parameters
        self.lookback_period = config.metadata.get('lookback_period', 20)
        self.momentum_threshold = config.metadata.get('momentum_threshold', 0.015)  # 1.5%
        self.volume_multiplier = config.metadata.get('volume_multiplier', 1.2)
        self.ma_period = config.metadata.get('ma_period', 50)
        self.regime_ma_period = config.metadata.get('regime_ma_period', 200)
        self.volatility_period = config.metadata.get('volatility_period', 20)
        self.min_adr_pct = config.metadata.get('min_adr_pct', 0.015)  # 1.5% minimum daily range
        self.use_trading_windows = config.metadata.get('use_trading_windows', False)  # Make it optional
        
        # Trading time windows (EST) - only used if use_trading_windows is True
        self.morning_start = time(9, 30)
        self.morning_end = time(10, 0)
        self.afternoon_start = time(15, 30)
        self.afternoon_end = time(16, 0)
        
        # Data storage
        self.price_history = {}  # symbol -> deque
        self.volume_history = {}  # symbol -> deque
        self.ma_values = {}  # symbol -> float
        self.regime_ma_values = {}  # symbol -> float (200-day MA)
        self.momentum_values = {}  # symbol -> float
        self.momentum_acceleration = {}  # symbol -> float
        self.volatility = {}  # symbol -> float (ATR)
        self.spy_regime = None  # Market regime from SPY
        
        # Position management
        self.trailing_stops = {}  # symbol -> float
        self.position_high_water_marks = {}  # symbol -> float
    
    async def on_start(self):
        """Initialize strategy"""
        logger.info(f"Enhanced Momentum strategy starting with symbols: {self.config.symbols}")
        
        # Get all symbols including SPY for market regime
        all_symbols = list(self.config.symbols)
        if 'SPY' not in all_symbols:
            all_symbols.append('SPY')
            
        # Initialize data structures
        for symbol in all_symbols:
            self.price_history[symbol] = deque(maxlen=max(self.regime_ma_period, self.ma_period))
            self.volume_history[symbol] = deque(maxlen=self.lookback_period)
            self.ma_values[symbol] = 0.0
            self.regime_ma_values[symbol] = 0.0
            self.momentum_values[symbol] = 0.0
            self.momentum_acceleration[symbol] = 0.0
            self.volatility[symbol] = 0.0
            
            # Load historical data if available
            await self._load_historical_data(symbol)
            
        # Subscribe to SPY if not in trading symbols
        if 'SPY' not in self.config.symbols and self.market_data:
            try:
                await self.market_data.subscribe_ticker('SPY')
                logger.info("Subscribed to SPY for market regime detection")
            except Exception as e:
                logger.warning(f"Could not subscribe to SPY: {e}")
    
    async def on_stop(self):
        """Cleanup strategy"""
        logger.info("Enhanced Momentum strategy stopping")
        # Clear data
        self.price_history.clear()
        self.volume_history.clear()
        self.ma_values.clear()
        self.regime_ma_values.clear()
        self.momentum_values.clear()
        self.momentum_acceleration.clear()
        self.volatility.clear()
        self.trailing_stops.clear()
        self.position_high_water_marks.clear()
    
    def _is_trading_window(self, timestamp: float) -> bool:
        """Check if we're in a high-volume trading window"""
        # If trading windows are disabled, always return True
        if not self.use_trading_windows:
            return True
            
        # Convert Unix timestamp to datetime
        dt = datetime.fromtimestamp(timestamp)
        current_time = dt.time()
        
        # Morning session (first 30 minutes)
        if self.morning_start <= current_time <= self.morning_end:
            return True
        
        # Afternoon session (last 30 minutes)
        if self.afternoon_start <= current_time <= self.afternoon_end:
            return True
        
        return False
    
    async def on_tick(self, tick: TickData):
        """Process market data tick"""
        if tick.last is None:
            return
        
        symbol = tick.symbol
        
        # Update price history
        if symbol not in self.price_history:
            if symbol != 'SPY':  # SPY might be tracked separately
                return
            
        self.price_history[symbol].append(tick.last)
        
        # Update volume history
        if tick.volume and symbol != 'SPY':
            self.volume_history[symbol].append(tick.volume)
        
        # Calculate indicators
        if len(self.price_history[symbol]) >= self.lookback_period:
            self._calculate_indicators(symbol)
        
        # Update market regime using SPY
        if symbol == 'SPY' or 'SPY' in self.price_history:
            self._update_market_regime()
        
        # Update trailing stops for existing positions
        if symbol in self.position_high_water_marks:
            self._update_trailing_stop(symbol, tick.last)
    
    async def calculate_signals(self, tick: TickData) -> List[Signal]:
        """Generate trading signals"""
        signals = []
        symbol = tick.symbol
        
        # Skip if not in trading window
        if not self._is_trading_window(tick.timestamp):
            return signals
        
        # Skip SPY if it's only used for regime
        if symbol == 'SPY' and symbol not in self.config.symbols:
            return signals
        
        # Need enough data
        if len(self.price_history[symbol]) < self.ma_period:
            return signals
        
        # Check market regime
        if self.spy_regime is False:  # Bear market
            logger.debug(f"Skipping {symbol} - Bear market regime")
            return signals
        
        # Get current values
        current_price = tick.last
        current_ma = self.ma_values.get(symbol, 0)
        current_momentum = self.momentum_values.get(symbol, 0)
        momentum_accel = self.momentum_acceleration.get(symbol, 0)
        current_volatility = self.volatility.get(symbol, 0.20)  # Default 20% if not set
        
        # Check if stock has enough volatility
        if current_volatility < self.min_adr_pct:
            logger.debug(f"{symbol}: Low volatility {current_volatility:.3f} < {self.min_adr_pct}")
            return signals
        
        # Check volume condition
        volume_condition = True
        if self.volume_history[symbol]:
            avg_volume = np.mean(list(self.volume_history[symbol]))
            current_volume = tick.volume or 0
            volume_condition = current_volume > avg_volume * self.volume_multiplier
        
        # Enhanced buy signal conditions
        if (current_price > current_ma and 
            current_momentum > self.momentum_threshold and
            momentum_accel > 0 and  # Momentum is accelerating
            volume_condition):
            
            logger.info(f"🚀 ENHANCED BUY SIGNAL for {symbol}: "
                       f"Price=${current_price:.2f}, MA=${current_ma:.2f}, "
                       f"Momentum={current_momentum:.4f}, Accel={momentum_accel:.4f}")
            
            # Dynamic position sizing based on volatility
            volatility_factor = min(0.02 / current_volatility, 1.5)  # Lower size for high volatility
            position_score = min(current_momentum / self.momentum_threshold, 2.0) * volatility_factor
            
            # Wider stops for volatile stocks, but cap at reasonable levels
            stop_multiplier = max(2.0, min(5.0, current_volatility / 0.01))  # 2-5x volatility
            stop_distance = min(current_volatility * stop_multiplier, 0.10)  # Cap at 10%
            stop_loss = current_price * (1 - stop_distance)
            take_profit = current_price * (1 + stop_distance * 2)  # 2:1 risk/reward
            
            signal = Signal(
                action="BUY",
                symbol=symbol,
                quantity=0,  # Will be calculated by risk manager
                order_type="MARKET",
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'momentum': current_momentum,
                    'momentum_accel': momentum_accel,
                    'ma_ratio': current_price / current_ma,
                    'volatility': current_volatility,
                    'position_score': position_score,
                    'regime': 'bull' if self.spy_regime else 'neutral'
                }
            )
            signals.append(signal)
            
            # Initialize position tracking
            self.position_high_water_marks[symbol] = current_price
            self.trailing_stops[symbol] = stop_loss
        
        # Enhanced sell signal for shorts (optional)
        elif (current_price < current_ma and 
              current_momentum < -self.momentum_threshold and
              momentum_accel < 0 and  # Negative momentum accelerating
              volume_condition and
              self.config.metadata.get('allow_shorts', False)):
            
            logger.info(f"📉 ENHANCED SELL SIGNAL for {symbol}: "
                       f"Price=${current_price:.2f}, MA=${current_ma:.2f}, "
                       f"Momentum={current_momentum:.4f}, Accel={momentum_accel:.4f}")
            
            # Dynamic position sizing
            volatility_factor = min(0.02 / current_volatility, 1.5)
            position_score = min(abs(current_momentum) / self.momentum_threshold, 2.0) * volatility_factor
            
            # Stops for short positions
            stop_multiplier = max(2.0, min(5.0, current_volatility / 0.01))  # 2-5x volatility
            stop_distance = min(current_volatility * stop_multiplier, 0.10)  # Cap at 10%
            stop_loss = current_price * (1 + stop_distance)
            take_profit = current_price * (1 - stop_distance * 2)
            
            signal = Signal(
                action="SELL",
                symbol=symbol,
                quantity=0,
                order_type="MARKET",
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'momentum': current_momentum,
                    'momentum_accel': momentum_accel,
                    'ma_ratio': current_price / current_ma,
                    'volatility': current_volatility,
                    'position_score': position_score,
                    'regime': 'bear' if not self.spy_regime else 'neutral'
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
        
        # Check trailing stop
        if symbol in self.trailing_stops:
            trailing_stop = self.trailing_stops[symbol]
            if position.signal.is_buy and tick.last <= trailing_stop:
                logger.info(f"Trailing stop hit for {symbol}: ${tick.last:.2f} <= ${trailing_stop:.2f}")
                return True, "trailing_stop_hit"
        
        # Check fixed stop loss (backup)
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
        
        # Exit if momentum decelerates significantly
        current_momentum = self.momentum_values.get(symbol, 0)
        momentum_accel = self.momentum_acceleration.get(symbol, 0)
        
        if position.signal.is_buy:
            # Exit long if momentum turns negative or decelerates sharply
            if current_momentum < 0 or (momentum_accel < -0.001 and current_momentum < self.momentum_threshold * 0.5):
                return True, "momentum_deceleration"
        
        # Exit if we're outside trading windows and position is profitable
        if not self._is_trading_window(tick.timestamp):
            entry_price = position.signal.metadata.get('entry_price', position.signal.price)
            if position.signal.is_buy and tick.last > entry_price * 1.005:  # 0.5% profit
                return True, "trading_window_exit"
        
        # Exit if market regime turns bearish
        if self.spy_regime is False and position.signal.is_buy:
            return True, "bear_market_regime"
        
        return False, None
    
    def _calculate_indicators(self, symbol: str):
        """Calculate technical indicators"""
        prices = list(self.price_history[symbol])
        
        # Calculate moving averages
        if len(prices) >= self.ma_period:
            self.ma_values[symbol] = np.mean(prices[-self.ma_period:])
        
        if len(prices) >= self.regime_ma_period:
            self.regime_ma_values[symbol] = np.mean(prices[-self.regime_ma_period:])
        
        # Calculate momentum (rate of change)
        if len(prices) >= self.lookback_period:
            old_price = prices[-self.lookback_period]
            current_price = prices[-1]
            self.momentum_values[symbol] = (current_price - old_price) / old_price
            
            # Calculate momentum acceleration
            if len(prices) >= self.lookback_period * 2:
                older_price = prices[-self.lookback_period * 2]
                old_momentum = (old_price - older_price) / older_price
                self.momentum_acceleration[symbol] = self.momentum_values[symbol] - old_momentum
        
        # Calculate volatility (simplified ATR)
        if len(prices) >= self.volatility_period:
            returns = np.diff(prices[-self.volatility_period:]) / prices[-self.volatility_period:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            # Ensure reasonable bounds for volatility
            self.volatility[symbol] = max(0.10, min(2.0, volatility))  # Between 10% and 200%
        else:
            # Default volatility if not enough data
            self.volatility[symbol] = 0.20  # 20% default
    
    def _update_market_regime(self):
        """Update market regime based on SPY"""
        if 'SPY' not in self.price_history:
            return
        
        prices = list(self.price_history['SPY'])
        if len(prices) >= self.regime_ma_period:
            spy_price = prices[-1]
            spy_ma = np.mean(prices[-self.regime_ma_period:])
            self.spy_regime = spy_price > spy_ma
            logger.debug(f"Market regime: {'BULL' if self.spy_regime else 'BEAR'} "
                        f"(SPY ${spy_price:.2f} vs MA ${spy_ma:.2f})")
    
    def _update_trailing_stop(self, symbol: str, current_price: float):
        """Update trailing stop loss"""
        if symbol not in self.position_high_water_marks:
            return
        
        # Update high water mark
        if current_price > self.position_high_water_marks[symbol]:
            self.position_high_water_marks[symbol] = current_price
            
            # Update trailing stop (2x volatility below high water mark)
            volatility = self.volatility.get(symbol, 0.02)
            stop_distance = max(volatility * 2, 0.02)  # At least 2%
            new_stop = current_price * (1 - stop_distance)
            
            # Only move stop up, never down
            if new_stop > self.trailing_stops.get(symbol, 0):
                self.trailing_stops[symbol] = new_stop
                logger.debug(f"{symbol}: Updated trailing stop to ${new_stop:.2f} "
                           f"(HWM: ${current_price:.2f})")
    
    async def _load_historical_data(self, symbol: str):
        """Load historical data for indicators"""
        # This would load historical data from market data manager
        # For now, we'll start fresh
        logger.debug(f"Historical data loading for {symbol} not implemented yet")


# Configuration helper
class EnhancedMomentumConfig(StrategyConfig):
    """Configuration for Enhanced Momentum Strategy"""
    def __init__(self, **kwargs):
        # Set defaults
        kwargs.setdefault('name', 'EnhancedMomentum')
        kwargs.setdefault('metadata', {}).update({
            'lookback_period': kwargs.get('lookback_period', 20),
            'momentum_threshold': kwargs.get('momentum_threshold', 0.015),
            'volume_multiplier': kwargs.get('volume_multiplier', 1.2),
            'ma_period': kwargs.get('ma_period', 50),
            'regime_ma_period': kwargs.get('regime_ma_period', 200),
            'volatility_period': kwargs.get('volatility_period', 20),
            'min_adr_pct': kwargs.get('min_adr_pct', 0.015),
            'allow_shorts': kwargs.get('allow_shorts', False)
        })
        super().__init__(**kwargs)