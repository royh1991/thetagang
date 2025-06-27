import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional
from collections import deque
import numpy as np
from datetime import datetime, timedelta

from base_strategy import BaseStrategy
from backtest_framework import Order, OrderSide, OrderType, Portfolio


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy
    
    Buys when price drops below moving average by a threshold
    Sells when price rises above moving average by a threshold
    
    Parameters:
        lookback_period: Number of bars for moving average calculation
        buy_threshold: Percentage below MA to trigger buy (e.g., -0.02 for 2%)
        sell_threshold: Percentage above MA to trigger sell (e.g., 0.01 for 1%)
        stop_loss_pct: Stop loss percentage (e.g., 0.03 for 3%)
        position_size_pct: Percentage of portfolio per position
        max_positions_per_symbol: Maximum concurrent positions per symbol
        min_trade_interval: Minimum seconds between trades
    """
    
    def __init__(self, params: Dict = None):
        super().__init__(params)
        
        # Strategy parameters with defaults
        self.lookback_period = self.get_param('lookback_period', 20)
        self.buy_threshold = self.get_param('buy_threshold', -0.002)  # -0.2%
        self.sell_threshold = self.get_param('sell_threshold', 0.001)  # 0.1%
        self.stop_loss_pct = self.get_param('stop_loss_pct', 0.03)  # 3%
        self.position_size_pct = self.get_param('position_size_pct', 0.1)  # 10%
        self.max_positions_per_symbol = self.get_param('max_positions_per_symbol', 1)
        self.min_trade_interval = self.get_param('min_trade_interval', 300)  # 5 minutes
        
        # Track last trade times
        self.last_trade_time = {}
        
    def get_required_history(self) -> int:
        """Return number of bars needed before we can trade"""
        return self.lookback_period
    
    def calculate_moving_average(self, prices: List[float]) -> Optional[float]:
        """Calculate simple moving average"""
        if len(prices) < self.lookback_period:
            return None
        return np.mean(prices[-self.lookback_period:])
    
    def process_bars(self, bars: Dict, price_history: Dict[str, deque], 
                     portfolio: Portfolio) -> List[Order]:
        """Process new bars and generate trading signals"""
        orders = []
        
        for symbol, bar in bars.items():
            # Check if we have enough history
            if not self.should_trade(symbol, price_history[symbol]):
                continue
            
            # Get price history
            prices = [b.close for b in price_history[symbol]]
            current_price = bar.close
            
            # Calculate moving average
            ma = self.calculate_moving_average(prices)
            if ma is None:
                continue
            
            # Calculate deviation from MA
            deviation = (current_price - ma) / ma
            
            # Check if we have a position
            has_position = symbol in portfolio.positions
            
            # Check stop loss for existing positions
            if has_position:
                position = portfolio.positions[symbol]
                loss_pct = (current_price - position.entry_price) / position.entry_price
                
                if loss_pct <= -self.stop_loss_pct:
                    # Stop loss triggered
                    order = self.create_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=position.quantity,
                        order_type=OrderType.MARKET
                    )
                    orders.append(order)
                    self.log(f"Stop loss triggered for {symbol} at {loss_pct:.2%}")
                    continue
            
            # Check trading cooldown
            if symbol in self.last_trade_time:
                time_since_trade = (bar.timestamp - self.last_trade_time[symbol]).total_seconds()
                if time_since_trade < self.min_trade_interval:
                    continue
            
            # Generate signals
            if not has_position and deviation <= self.buy_threshold:
                # Buy signal
                position_size = self.calculate_position_size(symbol, current_price, portfolio)
                
                if position_size > 0:
                    order = self.create_order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=position_size,
                        order_type=OrderType.LIMIT,
                        limit_price=round(current_price * 1.001, 2)  # Slightly above ask
                    )
                    orders.append(order)
                    self.last_trade_time[symbol] = bar.timestamp
                    self.log(f"Buy signal: {symbol} @ ${current_price:.2f} "
                            f"(deviation: {deviation:.3%} from MA ${ma:.2f})")
            
            elif has_position and deviation >= self.sell_threshold:
                # Sell signal
                position = portfolio.positions[symbol]
                order = self.create_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    order_type=OrderType.LIMIT,
                    limit_price=round(current_price * 0.999, 2)  # Slightly below bid
                )
                orders.append(order)
                self.last_trade_time[symbol] = bar.timestamp
                
                pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
                self.log(f"Sell signal: {symbol} @ ${current_price:.2f} "
                        f"(deviation: {deviation:.3%} from MA ${ma:.2f}, "
                        f"P&L: {pnl_pct:+.2f}%)")
        
        return orders
    
    def calculate_position_size(self, symbol: str, price: float, 
                                portfolio: Portfolio) -> int:
        """Calculate position size with risk management"""
        # Use base class method but apply our position limit
        base_size = super().calculate_position_size(symbol, price, portfolio)
        
        # Check if we already have positions in this symbol
        current_positions = 1 if symbol in portfolio.positions else 0
        
        if current_positions >= self.max_positions_per_symbol:
            return 0
        
        return base_size
    
    @property
    def description(self) -> str:
        """Return strategy description"""
        return (f"Mean Reversion Strategy - "
                f"MA Period: {self.lookback_period}, "
                f"Buy: {self.buy_threshold*100:.2f}%, "
                f"Sell: {self.sell_threshold*100:.2f}%, "
                f"Stop Loss: {self.stop_loss_pct*100:.1f}%")
    
    def validate_params(self) -> bool:
        """Validate strategy parameters"""
        if self.lookback_period < 2:
            self.log("Lookback period must be at least 2", "ERROR")
            return False
        
        if self.buy_threshold >= 0:
            self.log("Buy threshold should be negative", "ERROR")
            return False
        
        if self.sell_threshold <= 0:
            self.log("Sell threshold should be positive", "ERROR")
            return False
        
        if self.stop_loss_pct <= 0 or self.stop_loss_pct > 1:
            self.log("Stop loss percentage should be between 0 and 1", "ERROR")
            return False
        
        return True