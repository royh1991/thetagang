"""
Simple momentum strategy using moving average crossover
"""
from typing import List
from loguru import logger
from .strategy_base import StrategyBase, Bar, Signal


class SimpleMomentumStrategy(StrategyBase):
    """
    Simple momentum strategy that:
    - Buys when price crosses above SMA
    - Sells when price crosses below SMA
    """
    
    def __init__(self, symbol: str, sma_period: int = 20):
        super().__init__(symbol)
        self.sma_period = sma_period
        self.sma_values: List[float] = []
        
    def calculate_signal(self, bar: Bar) -> str:
        """Calculate trading signal based on SMA crossover"""
        
        # Need at least sma_period bars to calculate SMA
        if len(self.bars) < self.sma_period:
            return Signal.HOLD
        
        # Calculate current SMA
        recent_closes = [b.close for b in self.bars[-self.sma_period:]]
        current_sma = sum(recent_closes) / len(recent_closes)
        self.sma_values.append(current_sma)
        
        # Store in metadata for analysis
        self.metadata['current_sma'] = current_sma
        self.metadata['current_price'] = bar.close
        
        # Need at least 2 SMA values to detect crossover
        if len(self.sma_values) < 2:
            return Signal.HOLD
        
        prev_sma = self.sma_values[-2]
        prev_bar = self.bars[-2]
        
        # Detect crossovers
        price_above_sma = bar.close > current_sma
        price_was_above_sma = prev_bar.close > prev_sma
        
        # Generate signals
        if not price_was_above_sma and price_above_sma and self.position == 0:
            # Price crossed above SMA - BUY signal
            logger.info(f"BUY signal: price {bar.close:.2f} crossed above SMA {current_sma:.2f}")
            return Signal.BUY
        
        elif price_was_above_sma and not price_above_sma and self.position > 0:
            # Price crossed below SMA - SELL signal
            logger.info(f"SELL signal: price {bar.close:.2f} crossed below SMA {current_sma:.2f}")
            return Signal.SELL
        
        return Signal.HOLD
    
    def get_required_history(self) -> int:
        """We need at least SMA period + 1 bars before generating signals"""
        return self.sma_period + 1
    
    def get_stats(self) -> dict:
        """Get strategy statistics"""
        stats = super().get_stats()
        stats.update({
            'sma_period': self.sma_period,
            'sma_values_calculated': len(self.sma_values)
        })
        return stats


class EnhancedMomentumStrategy(StrategyBase):
    """
    Enhanced momentum strategy with additional filters:
    - Uses both fast and slow SMA
    - Includes volume confirmation
    - Adds RSI filter to avoid overbought/oversold
    """
    
    def __init__(self, symbol: str, fast_sma: int = 10, slow_sma: int = 30, rsi_period: int = 14):
        super().__init__(symbol)
        self.fast_sma = fast_sma
        self.slow_sma = slow_sma
        self.rsi_period = rsi_period
        
    def calculate_signal(self, bar: Bar) -> str:
        """Calculate trading signal with multiple confirmations"""
        
        # Need enough bars for slow SMA
        if len(self.bars) < self.slow_sma:
            return Signal.HOLD
        
        # Calculate SMAs
        fast_sma_value = sum(b.close for b in self.bars[-self.fast_sma:]) / self.fast_sma
        slow_sma_value = sum(b.close for b in self.bars[-self.slow_sma:]) / self.slow_sma
        
        # Calculate RSI
        rsi = self._calculate_rsi()
        
        # Calculate average volume
        avg_volume = sum(b.volume for b in self.bars[-20:]) / 20
        volume_surge = bar.volume > avg_volume * 1.2  # 20% above average
        
        # Store in metadata
        self.metadata.update({
            'fast_sma': fast_sma_value,
            'slow_sma': slow_sma_value,
            'rsi': rsi,
            'volume_surge': volume_surge
        })
        
        # Generate signals with multiple confirmations
        if self.position == 0:
            # Buy conditions:
            # 1. Fast SMA > Slow SMA (uptrend)
            # 2. Price > Fast SMA (momentum)
            # 3. RSI < 70 (not overbought)
            # 4. Volume surge (optional but preferred)
            if (fast_sma_value > slow_sma_value and 
                bar.close > fast_sma_value and
                rsi < 70):
                logger.info(f"BUY signal: Fast SMA {fast_sma_value:.2f} > Slow SMA {slow_sma_value:.2f}, "
                          f"RSI {rsi:.1f}, Volume surge: {volume_surge}")
                return Signal.BUY
        
        elif self.position > 0:
            # Sell conditions:
            # 1. Fast SMA < Slow SMA (downtrend)
            # OR
            # 2. Price < Fast SMA (momentum loss)
            # OR
            # 3. RSI > 80 (overbought)
            if (fast_sma_value < slow_sma_value or
                bar.close < fast_sma_value or
                rsi > 80):
                logger.info(f"SELL signal: Fast SMA {fast_sma_value:.2f} vs Slow SMA {slow_sma_value:.2f}, "
                          f"Price {bar.close:.2f} vs Fast SMA, RSI {rsi:.1f}")
                return Signal.SELL
        
        return Signal.HOLD
    
    def _calculate_rsi(self) -> float:
        """Calculate RSI indicator"""
        if len(self.bars) < self.rsi_period + 1:
            return 50.0  # Neutral
        
        # Calculate price changes
        changes = []
        for i in range(1, self.rsi_period + 1):
            change = self.bars[-i].close - self.bars[-i-1].close
            changes.append(change)
        
        # Separate gains and losses
        gains = [c for c in changes if c > 0]
        losses = [-c for c in changes if c < 0]
        
        avg_gain = sum(gains) / self.rsi_period if gains else 0
        avg_loss = sum(losses) / self.rsi_period if losses else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_required_history(self) -> int:
        """Need enough bars for slow SMA + RSI calculation"""
        return max(self.slow_sma, self.rsi_period + 1)