"""
Nick's Funnel Breakout Strategy
Based on Pine Script that combines breakout detection with trend/chop filtering
"""
from typing import List, Optional, Tuple, Dict, Union, Any
import numpy as np
from loguru import logger
from .strategy_base import StrategyBase, Bar, Signal, SignalInfo
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
                 rsi_threshold: int = 55,
                 volume_multiplier: float = 2.0,
                 take_profit_atr: float = 2.5,
                 stop_loss_atr: float = 1.0,
                 adx_trend_threshold: int = 30,
                 exit_sma_length: int = 5,
                 debug: bool = False):
        super().__init__(symbol)
        
        # Strategy parameters
        self.lookback_period = lookback_period
        self.rsi_threshold = rsi_threshold
        self.volume_multiplier = volume_multiplier
        self.take_profit_atr = take_profit_atr
        self.stop_loss_atr = stop_loss_atr
        self.adx_trend_threshold = adx_trend_threshold
        self.exit_sma_length = exit_sma_length
        self.debug = debug
        
        # Technical indicators storage
        self.rsi_values = []
        self.atr_values = []
        self.adx_values = []
        self.plus_di = []
        self.minus_di = []
        
        # SPY data for market context
        self.spy_bars = []  # Will store aligned SPY bars
        self.spy_sma20 = []
        self.spy_sma50 = []
        
        # Entry tracking (long only)
        self.entered_long = False
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        
        # Debug data storage
        self.debug_data = []
    
    def _return_signal(self, signal_info: SignalInfo) -> SignalInfo:
        """Helper method to store final signal and return it"""
        if self.debug and self.debug_data:
            # Add final signal to the last debug row
            self.debug_data[-1]['final_signal'] = signal_info.signal
            self.debug_data[-1]['signal_reason'] = signal_info.reason
        return signal_info
        
    def needs_market_data(self) -> bool:
        """This strategy needs SPY data for market context"""
        return True
    
    def set_market_data(self, symbol: str, data: Any):
        """Override to log when market data is set"""
        super().set_market_data(symbol, data)
        if symbol == 'SPY' and not data.empty:
            logger.info(f"SPY data set with {len(data)} bars from {data['timestamp'].min()} to {data['timestamp'].max()}")
    
    def calculate_signal(self, bar: Bar) -> Union[str, SignalInfo]:
        """Calculate trading signal based on funnel breakout strategy"""
        
        # Need enough history
        if len(self.bars) < max(self.lookback_period, 50):
            return self._return_signal(SignalInfo.hold("Insufficient history"))
            
        # Calculate technical indicators
        self._update_indicators()
        
        # Get current values
        current_rsi = self.rsi_values[-1] if self.rsi_values else 50
        current_adx = self.adx_values[-1] if self.adx_values else 0
        current_atr = self.atr_values[-1] if self.atr_values else 0
        
        # Check market context (SPY trend)
        macro_bullish, macro_bearish = self._check_market_context()
        
        # Calculate range breakout (excluding current bar, matching Pine's high[1] notation)
        # Pine uses ta.highest(high[1], lengthRange) which looks at previous bars only
        if len(self.bars) > self.lookback_period:
            # Get the previous lookback_period bars, excluding the current one
            # self.bars[-self.lookback_period-1:-1] gets previous lookback_period bars
            range_high = max(b.high for b in self.bars[-self.lookback_period-1:-1])
            range_low = min(b.low for b in self.bars[-self.lookback_period-1:-1])
        else:
            # Not enough history
            range_high = bar.high
            range_low = bar.low
        
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
        
        # Entry signals (Pine Script only trades long)
        long_signal = (is_trending and macro_bullish and bull_breakout and 
                      volume_spike and rsi_bullish)
        # Pine Script doesn't implement short trading
        short_signal = False
        
        # Exit conditions - Calculate SMA for exit
        sma_exit = None
        if len(self.bars) >= self.exit_sma_length:
            sma_exit = sum(b.close for b in self.bars[-self.exit_sma_length:]) / self.exit_sma_length
        
        # Check for two consecutive bars below SMA for exit (long only)
        smooth_exit_long = False
        
        if sma_exit and len(self.bars) >= self.exit_sma_length + 1:
            # Current bar and previous bar
            bar0_below = bar.close < sma_exit
            bar1_below = self.bars[-2].close < sma_exit
            smooth_exit_long = bar0_below and bar1_below
        
        # Exit signal based on 2 consecutive bars below SMA
        exit_long = self.entered_long and smooth_exit_long
        exit_short = False  # No short trading
        
        # Debug data collection
        if self.debug:
            # Get current SPY data if available
            spy_price = None
            spy_sma20_val = None
            spy_sma50_val = None
            
            if self.spy_bars:
                spy_price = self.spy_bars[-1].close
            if self.spy_sma20:
                spy_sma20_val = self.spy_sma20[-1]
            if self.spy_sma50:
                spy_sma50_val = self.spy_sma50[-1]
            
            debug_row = {
                'timestamp': bar.timestamp,
                'price': bar.close,
                'volume': bar.volume,
                'rsi': current_rsi,
                'adx': current_adx,
                'atr': current_atr,
                'range_low': range_low,
                'range_high': range_high,
                'exit_sma': sma_exit if sma_exit else None,
                'exit_sma_length': self.exit_sma_length,
                'bar0_below_sma': bar.close < sma_exit if sma_exit else None,
                'bar1_below_sma': self.bars[-2].close < sma_exit if sma_exit and len(self.bars) >= 2 else None,
                'smooth_exit_long': smooth_exit_long,
                'avg_volume': avg_volume,
                'volume_spike_threshold': avg_volume * self.volume_multiplier,
                'is_trending': is_trending,
                'is_choppy': is_choppy,
                'bull_breakout': bull_breakout,
                'bear_breakdown': bear_breakdown,
                'rsi_bullish': rsi_bullish,
                'rsi_bearish': rsi_bearish,
                'volume_spike': volume_spike,
                'macro_bullish': macro_bullish,
                'macro_bearish': macro_bearish,
                'spy_price': spy_price,
                'spy_sma20': spy_sma20_val,
                'spy_sma50': spy_sma50_val,
                'long_signal': long_signal,
                'short_signal': short_signal,
                'exit_long': exit_long,
                'exit_short': exit_short,
                'currently_long': self.entered_long,
                'currently_short': False,  # No short trading
                'entry_price': self.entry_price if self.entered_long else None,
                'stop_loss': self.stop_loss if self.entered_long else None,
                'take_profit': self.take_profit if self.entered_long else None,
                'position': self.position
            }
            self.debug_data.append(debug_row)
        
        # Check stop loss and take profit
        if self.entered_long:
            if bar.close <= self.stop_loss:
                logger.info(f"Long stop loss hit at {bar.close:.2f}")
                self.entered_long = False
                return self._return_signal(SignalInfo.sell(f"Stop loss hit at ${bar.close:.2f} (SL: ${self.stop_loss:.2f})"))
            elif bar.close >= self.take_profit:
                logger.info(f"Long take profit hit at {bar.close:.2f}")
                self.entered_long = False
                return self._return_signal(SignalInfo.sell(f"Take profit hit at ${bar.close:.2f} (TP: ${self.take_profit:.2f})"))
        
        # No short trading in Pine Script strategy
        
        # Generate signals
        if exit_long and self.position > 0:
            logger.info(f"Exit long signal: 2 consecutive bars below {self.exit_sma_length}-SMA")
            self.entered_long = False
            exit_reason = f"2 consecutive bars closed below {self.exit_sma_length}-SMA (${sma_exit:.2f})"
            return self._return_signal(SignalInfo.sell(exit_reason))
            
        # No short exit logic (Pine Script only trades long)
            
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
            
            # Build detailed reason
            reasons = []
            reasons.append(f"Breakout > ${range_high:.2f}")
            reasons.append(f"ADX={current_adx:.1f}")
            reasons.append(f"RSI={current_rsi:.1f}")
            if volume_spike:
                reasons.append(f"Vol spike {self.volume_multiplier:.1f}x")
            if macro_bullish:
                reasons.append("SPY bullish")
            reasons.append(f"SL=${self.stop_loss:.2f}")
            reasons.append(f"TP=${self.take_profit:.2f}")
            
            return self._return_signal(SignalInfo.buy(" | ".join(reasons)))
            
        # No short trading in Pine Script strategy
            
        return self._return_signal(SignalInfo.hold())
    
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
        """Check SPY trend for market context using actual SPY data"""
        # Process SPY bar for this timestamp if available
        if 'SPY' in self.market_data and self.current_bar:
            spy_df = self.market_data['SPY']
            current_timestamp = self.current_bar.timestamp
            
            # Debug: Log first time we process SPY data
            if not hasattr(self, '_logged_spy_data'):
                logger.debug(f"SPY data type: {type(spy_df)}, shape: {spy_df.shape if hasattr(spy_df, 'shape') else 'N/A'}")
                logger.debug(f"Current timestamp: {current_timestamp}")
                if isinstance(spy_df, pd.DataFrame) and not spy_df.empty:
                    logger.debug(f"SPY data range: {spy_df['timestamp'].min()} to {spy_df['timestamp'].max()}")
                    logger.debug(f"SPY sample price: {spy_df.iloc[0]['close']}")
                self._logged_spy_data = True
            
            # Find the SPY bar closest to current timestamp
            # First try exact match
            spy_row = spy_df[spy_df['timestamp'] == current_timestamp]
            
            if spy_row.empty:
                # Find the most recent SPY bar before current timestamp
                earlier_bars = spy_df[spy_df['timestamp'] <= current_timestamp]
                if not earlier_bars.empty:
                    spy_row = earlier_bars.iloc[-1:]
            
            # Add SPY bar to our tracking
            if not spy_row.empty:
                spy_bar = Bar(
                    timestamp=spy_row['timestamp'].iloc[0],
                    open=spy_row['open'].iloc[0],
                    high=spy_row['high'].iloc[0],
                    low=spy_row['low'].iloc[0],
                    close=spy_row['close'].iloc[0],
                    volume=int(spy_row['volume'].iloc[0])
                )
                
                # Maintain aligned SPY bars
                if not self.spy_bars or spy_bar.timestamp > self.spy_bars[-1].timestamp:
                    self.spy_bars.append(spy_bar)
                
                # Calculate SPY SMAs if we have enough data
                if len(self.spy_bars) >= 50:
                    sma20 = sum(b.close for b in self.spy_bars[-20:]) / 20
                    sma50 = sum(b.close for b in self.spy_bars[-50:]) / 50
                    
                    self.spy_sma20.append(sma20)
                    self.spy_sma50.append(sma50)
                    
                    # Keep only recent values
                    if len(self.spy_sma20) > 100:
                        self.spy_sma20.pop(0)
                        self.spy_sma50.pop(0)
                    
                    macro_bullish = sma20 > sma50
                    macro_bearish = sma20 < sma50
                    
                    return macro_bullish, macro_bearish
        
        # Fallback to using current symbol as proxy if no SPY data
        if len(self.bars) >= 50:
            sma20 = sum(b.close for b in self.bars[-20:]) / 20
            sma50 = sum(b.close for b in self.bars[-50:]) / 50
            
            # Store these as well for debugging
            self.spy_sma20.append(sma20)
            self.spy_sma50.append(sma50)
            
            if len(self.spy_sma20) > 100:
                self.spy_sma20.pop(0)
                self.spy_sma50.pop(0)
            
            macro_bullish = sma20 > sma50
            macro_bearish = sma20 < sma50
            
            return macro_bullish, macro_bearish
        
        # Default to neutral
        return False, False
    
    def reset(self):
        """Reset strategy state"""
        super().reset()
        # Clear SPY-specific data
        self.spy_bars.clear()
        self.spy_sma20.clear()
        self.spy_sma50.clear()
        # Clear other strategy-specific data
        self.rsi_values.clear()
        self.atr_values.clear()
        self.adx_values.clear()
        self.plus_di.clear()
        self.minus_di.clear()
        self.entered_long = False
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.debug_data.clear()
    
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
            'entered_long': self.entered_long
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
    
    def save_debug_csv(self, filename: str):
        """Save debug data to CSV file"""
        if not self.debug or not self.debug_data:
            logger.warning("No debug data to save")
            return
        
        # Convert to DataFrame and save
        df = pd.DataFrame(self.debug_data)
        df.to_csv(filename, index=False)
        logger.info(f"Debug data saved to {filename} ({len(df)} rows)")