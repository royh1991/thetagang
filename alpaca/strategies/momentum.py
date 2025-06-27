import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional
from collections import deque
import numpy as np
from datetime import datetime

from base_strategy import BaseStrategy
from backtest_framework import Order, OrderSide, OrderType, Portfolio


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy
    
    Buys stocks showing strong upward momentum (price > MA and rising)
    Sells when momentum weakens (price crosses below MA)
    
    Parameters:
        fast_ma_period: Period for fast moving average
        slow_ma_period: Period for slow moving average
        momentum_threshold: Minimum price increase % to trigger buy
        volume_factor: Minimum volume increase factor vs average
        trailing_stop_pct: Trailing stop loss percentage
        position_size_pct: Percentage of portfolio per position
        max_positions: Maximum concurrent positions
    """
    
    def __init__(self, params: Dict = None):
        super().__init__(params)
        
        # Strategy parameters
        self.fast_ma_period = self.get_param('fast_ma_period', 10)
        self.slow_ma_period = self.get_param('slow_ma_period', 30)
        self.momentum_threshold = self.get_param('momentum_threshold', 0.02)  # 2%
        self.volume_factor = self.get_param('volume_factor', 1.5)  # 50% above average
        self.trailing_stop_pct = self.get_param('trailing_stop_pct', 0.05)  # 5%
        self.position_size_pct = self.get_param('position_size_pct', 0.15)  # 15%
        self.max_positions = self.get_param('max_positions', 5)
        
        # Track highest prices for trailing stops
        self.highest_prices = {}
        
    def get_required_history(self) -> int:
        """Return number of bars needed before we can trade"""
        return self.slow_ma_period + 1
    
    def calculate_sma(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate simple moving average"""
        if len(prices) < period:
            return None
        return np.mean(prices[-period:])
    
    def calculate_momentum(self, prices: List[float], period: int = 5) -> Optional[float]:
        """Calculate price momentum (rate of change)"""
        if len(prices) < period + 1:
            return None
        
        current = prices[-1]
        past = prices[-(period + 1)]
        
        if past == 0:
            return None
        
        return (current - past) / past
    
    def is_volume_surge(self, volume_history: List[float], current_volume: float) -> bool:
        """Check if current volume is significantly higher than average"""
        if len(volume_history) < 20:
            return False
        
        avg_volume = np.mean(volume_history[-20:-1])  # Exclude current
        return current_volume > avg_volume * self.volume_factor
    
    def process_bars(self, bars: Dict, price_history: Dict[str, deque], 
                     portfolio: Portfolio) -> List[Order]:
        """Process new bars and generate trading signals"""
        orders = []
        
        # First, update trailing stops for existing positions
        for symbol, position in portfolio.positions.items():
            if symbol in bars:
                current_price = bars[symbol].close
                
                # Update highest price
                if symbol not in self.highest_prices:
                    self.highest_prices[symbol] = current_price
                else:
                    self.highest_prices[symbol] = max(self.highest_prices[symbol], current_price)
                
                # Check trailing stop
                stop_price = self.highest_prices[symbol] * (1 - self.trailing_stop_pct)
                
                if current_price <= stop_price:
                    order = self.create_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=position.quantity,
                        order_type=OrderType.MARKET
                    )
                    orders.append(order)
                    self.log(f"Trailing stop triggered for {symbol} @ ${current_price:.2f} "
                            f"(high: ${self.highest_prices[symbol]:.2f})")
                    del self.highest_prices[symbol]
        
        # Check for new opportunities
        total_positions = len(portfolio.positions)
        
        for symbol, bar in bars.items():
            # Skip if we already have a position or max positions reached
            if symbol in portfolio.positions or total_positions >= self.max_positions:
                continue
            
            # Check if we have enough history
            if not self.should_trade(symbol, price_history[symbol]):
                continue
            
            # Get price and volume history
            bars_list = list(price_history[symbol])
            prices = [b.close for b in bars_list]
            volumes = [b.volume for b in bars_list]
            current_price = bar.close
            
            # Calculate indicators
            fast_ma = self.calculate_sma(prices, self.fast_ma_period)
            slow_ma = self.calculate_sma(prices, self.slow_ma_period)
            momentum = self.calculate_momentum(prices)
            
            if fast_ma is None or slow_ma is None or momentum is None:
                continue
            
            # Buy conditions:
            # 1. Fast MA > Slow MA (uptrend)
            # 2. Price > Fast MA (above trend)
            # 3. Strong momentum
            # 4. Volume surge (optional)
            
            volume_surge = self.is_volume_surge(volumes, bar.volume)
            
            if (fast_ma > slow_ma and 
                current_price > fast_ma and 
                momentum >= self.momentum_threshold):
                
                # Extra confirmation from volume
                if not volume_surge and momentum < self.momentum_threshold * 1.5:
                    continue
                
                # Calculate position size
                position_size = self.calculate_position_size(symbol, current_price, portfolio)
                
                if position_size > 0:
                    order = self.create_order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=position_size,
                        order_type=OrderType.LIMIT,
                        limit_price=round(current_price * 1.002, 2)  # 0.2% above
                    )
                    orders.append(order)
                    total_positions += 1
                    
                    self.log(f"Momentum buy: {symbol} @ ${current_price:.2f} "
                            f"(momentum: {momentum:.2%}, "
                            f"fast MA: ${fast_ma:.2f}, slow MA: ${slow_ma:.2f}, "
                            f"volume surge: {volume_surge})")
        
        # Sell existing positions if momentum is lost
        for symbol, position in portfolio.positions.items():
            if symbol not in bars or symbol not in price_history:
                continue
            
            prices = [b.close for b in price_history[symbol]]
            current_price = bars[symbol].close
            
            fast_ma = self.calculate_sma(prices, self.fast_ma_period)
            
            # Sell if price drops below fast MA (momentum lost)
            if fast_ma and current_price < fast_ma:
                order = self.create_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    order_type=OrderType.LIMIT,
                    limit_price=round(current_price * 0.998, 2)
                )
                orders.append(order)
                
                pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
                self.log(f"Momentum sell: {symbol} @ ${current_price:.2f} "
                        f"(below fast MA ${fast_ma:.2f}, P&L: {pnl_pct:+.2f}%)")
                
                if symbol in self.highest_prices:
                    del self.highest_prices[symbol]
        
        return orders
    
    def on_order_filled(self, order: Order, fill_price: float, timestamp):
        """Track highest price when position is opened"""
        if order.side == OrderSide.BUY:
            self.highest_prices[order.symbol] = fill_price
    
    @property
    def description(self) -> str:
        """Return strategy description"""
        return (f"Momentum Strategy - "
                f"Fast MA: {self.fast_ma_period}, Slow MA: {self.slow_ma_period}, "
                f"Momentum: {self.momentum_threshold*100:.1f}%, "
                f"Trailing Stop: {self.trailing_stop_pct*100:.1f}%")
    
    def validate_params(self) -> bool:
        """Validate strategy parameters"""
        if self.fast_ma_period >= self.slow_ma_period:
            self.log("Fast MA period must be less than slow MA period", "ERROR")
            return False
        
        if self.momentum_threshold <= 0:
            self.log("Momentum threshold must be positive", "ERROR")
            return False
        
        if self.trailing_stop_pct <= 0 or self.trailing_stop_pct > 0.5:
            self.log("Trailing stop should be between 0 and 50%", "ERROR")
            return False
        
        return True