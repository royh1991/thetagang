"""
Nick's Funnel Breakout Strategy
Based on Pine Script that combines breakout detection with trend/chop filtering
"""
from typing import List, Optional, Tuple, Dict
import numpy as np
from loguru import logger
from .strategy_base import StrategyBase, Bar, Signal
from .data_fetcher import DataFetcher
import pandas as pd


class NickStrategy(StrategyBase):
    """
    Funnel Breakout Strategy that:
    - Detects breakouts from price ranges
    - Filters using ADX for trend vs chop
    - Confirms with volume spikes and RSI
    - Uses SPY as market context
    - Implements ATR-based stops and targets
    """
    
    def __init__(self, symbol: str, 
                 lookback_period: int = 20,
                 rsi_threshold: int = 50,
                 volume_multiplier: float = 1.5,
                 take_profit_atr: float = 2.0,
                 stop_loss_atr: float = 1.0,
                 adx_trend_threshold: int = 20):
        super().__init__(symbol)
        
        # Strategy parameters
        self.lookback_period = lookback_period
        self.rsi_threshold = rsi_threshold
        self.volume_multiplier = volume_multiplier
        self.take_profit_atr = take_profit_atr
        self.stop_loss_atr = stop_loss_atr
        self.adx_trend_threshold = adx_trend_threshold
        
        # Technical indicators storage
        self.rsi_values = []
        self.atr_values = []
        self.adx_values = []
        self.plus_di = []
        self.minus_di = []
        
        # SPY data for market context
        self.spy_data = None
        self.spy_sma20 = []
        self.spy_sma50 = []
        
        # Entry tracking
        self.entered_long = False
        self.entered_short = False
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        
        # Initialize data fetcher for SPY
        self.data_fetcher = DataFetcher()
        
    def calculate_signal(self, bar: Bar) -> str:
        """Calculate trading signal based on funnel breakout strategy"""
        
        # Need enough history
        if len(self.bars) < max(self.lookback_period, 50):
            return Signal.HOLD
            
        # Calculate technical indicators
        self._update_indicators()
        
        # Get current values
        current_rsi = self.rsi_values[-1] if self.rsi_values else 50
        current_adx = self.adx_values[-1] if self.adx_values else 0
        current_atr = self.atr_values[-1] if self.atr_values else 0
        
        # Check market context (SPY trend)
        macro_bullish, macro_bearish = self._check_market_context()
        
        # Calculate range breakout
        range_high = max(b.high for b in self.bars[-self.lookback_period-1:-1])
        range_low = min(b.low for b in self.bars[-self.lookback_period-1:-1])
        
        bull_breakout = bar.close > range_high
        bear_breakdown = bar.close < range_low
        
        # RSI conditions
        rsi_bullish = current_rsi > self.rsi_threshold
        rsi_bearish = current_rsi < (100 - self.rsi_threshold)
        
        # Volume spike
        avg_volume = sum(b.volume for b in self.bars[-self.lookback_period:]) / self.lookback_period
        volume_spike = bar.volume > avg_volume * self.volume_multiplier
        
        # Trend vs chop
        is_trending = current_adx > self.adx_trend_threshold
        is_choppy = current_adx <= self.adx_trend_threshold
        
        # Entry signals
        long_signal = (is_trending and macro_bullish and bull_breakout and 
                      volume_spike and rsi_bullish)
        short_signal = (is_trending and macro_bearish and bear_breakdown and 
                       volume_spike and rsi_bearish)
        
        # Exit conditions
        sma_exit = sum(b.close for b in self.bars[-5:]) / 5  # 5-period SMA
        
        exit_long = (self.entered_long and not long_signal and 
                    (is_choppy or bar.close < sma_exit))
        exit_short = (self.entered_short and not short_signal and 
                     (is_choppy or bar.close > sma_exit))
        
        # Check stop loss and take profit
        if self.entered_long:
            if bar.close <= self.stop_loss:
                logger.info(f"Long stop loss hit at {bar.close:.2f}")
                self.entered_long = False
                return Signal.SELL
            elif bar.close >= self.take_profit:
                logger.info(f"Long take profit hit at {bar.close:.2f}")
                self.entered_long = False
                return Signal.SELL
        
        if self.entered_short and self.position < 0:
            if bar.close >= self.stop_loss:
                logger.info(f"Short stop loss hit at {bar.close:.2f}")
                self.entered_short = False
                return Signal.BUY  # Buy to cover short
            elif bar.close <= self.take_profit:
                logger.info(f"Short take profit hit at {bar.close:.2f}")
                self.entered_short = False
                return Signal.BUY  # Buy to cover short
        
        # Generate signals
        if exit_long and self.position > 0:
            logger.info(f"Exit long signal: choppy={is_choppy}, below_sma={bar.close < sma_exit}")
            self.entered_long = False
            return Signal.SELL
            
        if exit_short and self.position < 0:
            logger.info(f"Exit short signal: choppy={is_choppy}, above_sma={bar.close > sma_exit}")
            self.entered_short = False
            return Signal.BUY  # Buy to cover
            
        if long_signal and self.position == 0:
            logger.info(f"Long signal: ADX={current_adx:.1f}, RSI={current_rsi:.1f}, "
                       f"Volume spike={volume_spike}, Breakout above {range_high:.2f}")
            self.entered_long = True
            self.entry_price = bar.close
            self.stop_loss = bar.close - (self.stop_loss_atr * current_atr)
            self.take_profit = bar.close + (self.take_profit_atr * current_atr)
            self.metadata['entry_type'] = 'long'
            self.metadata['stop_loss'] = self.stop_loss
            self.metadata['take_profit'] = self.take_profit
            return Signal.BUY
            
        if short_signal and self.position == 0:
            logger.info(f"Short signal: ADX={current_adx:.1f}, RSI={current_rsi:.1f}, "
                       f"Volume spike={volume_spike}, Breakdown below {range_low:.2f}")
            self.entered_short = True
            self.entry_price = bar.close
            self.stop_loss = bar.close + (self.stop_loss_atr * current_atr)
            self.take_profit = bar.close - (self.take_profit_atr * current_atr)
            self.metadata['entry_type'] = 'short'
            self.metadata['stop_loss'] = self.stop_loss
            self.metadata['take_profit'] = self.take_profit
            # Note: Our simple backtest doesn't support shorting, so we'll skip short trades
            logger.warning("Short signal generated but backtester doesn't support shorting")
            
        return Signal.HOLD
    
    def _update_indicators(self):
        """Update technical indicators"""
        # Calculate RSI
        if len(self.bars) >= 15:
            self._calculate_rsi()
        
        # Calculate ATR
        if len(self.bars) >= 15:
            self._calculate_atr()
        
        # Calculate ADX
        if len(self.bars) >= 15:
            self._calculate_adx()
    
    def _calculate_rsi(self, period: int = 14):
        """Calculate RSI"""
        if len(self.bars) < period + 1:
            return
        
        # Get price changes
        changes = []
        for i in range(len(self.bars) - period, len(self.bars)):
            change = self.bars[i].close - self.bars[i-1].close
            changes.append(change)
        
        # Calculate average gains and losses
        gains = [c for c in changes if c > 0]
        losses = [-c for c in changes if c < 0]
        
        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0
        
        # Calculate RSI
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        self.rsi_values.append(rsi)
        
        # Keep only recent values
        if len(self.rsi_values) > 100:
            self.rsi_values.pop(0)
    
    def _calculate_atr(self, period: int = 14):
        """Calculate Average True Range"""
        if len(self.bars) < period + 1:
            return
        
        # Calculate true ranges
        true_ranges = []
        for i in range(len(self.bars) - period, len(self.bars)):
            high = self.bars[i].high
            low = self.bars[i].low
            prev_close = self.bars[i-1].close
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
        
        # Calculate ATR
        atr = sum(true_ranges) / len(true_ranges)
        self.atr_values.append(atr)
        
        # Keep only recent values
        if len(self.atr_values) > 100:
            self.atr_values.pop(0)
    
    def _calculate_adx(self, period: int = 14):
        """Calculate ADX (Average Directional Index)"""
        if len(self.bars) < period + 2:
            return
        
        # Calculate directional movement
        plus_dm_sum = 0
        minus_dm_sum = 0
        tr_sum = 0
        
        for i in range(len(self.bars) - period, len(self.bars)):
            high = self.bars[i].high
            low = self.bars[i].low
            prev_high = self.bars[i-1].high
            prev_low = self.bars[i-1].low
            prev_close = self.bars[i-1].close
            
            # Calculate +DM and -DM
            up_move = high - prev_high
            down_move = prev_low - low
            
            plus_dm = 0
            minus_dm = 0
            
            if up_move > down_move and up_move > 0:
                plus_dm = up_move
            if down_move > up_move and down_move > 0:
                minus_dm = down_move
            
            # Calculate true range
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            
            plus_dm_sum += plus_dm
            minus_dm_sum += minus_dm
            tr_sum += tr
        
        # Calculate +DI and -DI
        if tr_sum > 0:
            plus_di = 100 * plus_dm_sum / tr_sum
            minus_di = 100 * minus_dm_sum / tr_sum
        else:
            plus_di = 0
            minus_di = 0
        
        # Calculate DX
        di_sum = plus_di + minus_di
        if di_sum > 0:
            dx = 100 * abs(plus_di - minus_di) / di_sum
        else:
            dx = 0
        
        # Simple moving average of DX for ADX (simplified)
        if len(self.adx_values) == 0:
            adx = dx
        else:
            # Exponential smoothing
            adx = (self.adx_values[-1] * (period - 1) + dx) / period
        
        self.adx_values.append(adx)
        self.plus_di.append(plus_di)
        self.minus_di.append(minus_di)
        
        # Keep only recent values
        if len(self.adx_values) > 100:
            self.adx_values.pop(0)
            self.plus_di.pop(0)
            self.minus_di.pop(0)
    
    def _check_market_context(self) -> Tuple[bool, bool]:
        """Check SPY trend for market context"""
        # For backtesting, we'll use a simplified approach
        # In live trading, you would fetch real SPY data
        
        # Calculate SPY SMAs from current data (simplified)
        if len(self.bars) >= 50:
            # Use current symbol as proxy for market (simplified)
            sma20 = sum(b.close for b in self.bars[-20:]) / 20
            sma50 = sum(b.close for b in self.bars[-50:]) / 50
            
            macro_bullish = sma20 > sma50
            macro_bearish = sma20 < sma50
            
            return macro_bullish, macro_bearish
        
        # Default to neutral
        return False, False
    
    def get_required_history(self) -> int:
        """Need at least 50 bars for all indicators"""
        return 50
    
    def get_stats(self) -> dict:
        """Get strategy statistics"""
        stats = super().get_stats()
        stats.update({
            'lookback_period': self.lookback_period,
            'rsi_threshold': self.rsi_threshold,
            'adx_trend_threshold': self.adx_trend_threshold,
            'current_rsi': self.rsi_values[-1] if self.rsi_values else None,
            'current_adx': self.adx_values[-1] if self.adx_values else None,
            'current_atr': self.atr_values[-1] if self.atr_values else None,
            'entered_long': self.entered_long,
            'entered_short': self.entered_short
        })
        return stats
    
    def get_indicators(self) -> Dict[str, List[float]]:
        """Get indicator values for plotting"""
        indicators = {}
        
        # Pad indicators to match data length
        data_length = len(self.bars)
        
        # RSI
        if self.rsi_values:
            padding = data_length - len(self.rsi_values)
            indicators['RSI'] = [None] * padding + self.rsi_values
        
        # ADX
        if self.adx_values:
            padding = data_length - len(self.adx_values)
            indicators['ADX'] = [None] * padding + self.adx_values
        
        # Moving averages for context
        if data_length >= 20:
            sma20 = []
            for i in range(data_length):
                if i < 19:
                    sma20.append(None)
                else:
                    sma20.append(sum(b.close for b in self.bars[i-19:i+1]) / 20)
            indicators['SMA20'] = sma20
        
        if data_length >= 50:
            sma50 = []
            for i in range(data_length):
                if i < 49:
                    sma50.append(None)
                else:
                    sma50.append(sum(b.close for b in self.bars[i-49:i+1]) / 50)
            indicators['SMA50'] = sma50
        
        return indicators