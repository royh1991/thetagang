"""
Simple base class for trading strategies
"""
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union, Tuple
from loguru import logger


class Bar:
    """Represents a single price bar"""
    def __init__(self, timestamp: datetime, open: float, high: float, low: float, close: float, volume: int):
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
    
    def __repr__(self):
        return f"Bar({self.timestamp}, O:{self.open:.2f}, H:{self.high:.2f}, L:{self.low:.2f}, C:{self.close:.2f}, V:{self.volume})"


class Signal:
    """Trading signal constants"""
    BUY = 'BUY'
    SELL = 'SELL'
    HOLD = 'HOLD'


@dataclass
class SignalInfo:
    """Trading signal with detailed reason"""
    signal: str  # BUY, SELL, or HOLD
    reason: str = ""  # Detailed reason for the signal
    
    @classmethod
    def buy(cls, reason: str) -> 'SignalInfo':
        """Create a BUY signal with reason"""
        return cls(Signal.BUY, reason)
    
    @classmethod
    def sell(cls, reason: str) -> 'SignalInfo':
        """Create a SELL signal with reason"""
        return cls(Signal.SELL, reason)
    
    @classmethod
    def hold(cls, reason: str = "") -> 'SignalInfo':
        """Create a HOLD signal with optional reason"""
        return cls(Signal.HOLD, reason)


class StrategyBase(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.position = 0  # Current position (positive = long, negative = short, 0 = flat)
        self.bars: List[Bar] = []  # Historical bars
        self.current_bar: Optional[Bar] = None
        self.metadata: Dict[str, Any] = {}  # Strategy-specific data
        self.market_data: Dict[str, Any] = {}  # Market data (e.g., SPY for context)
        
    def on_bar(self, bar: Bar) -> Union[str, SignalInfo]:
        """
        Called when a new bar is received
        
        Args:
            bar: The new price bar
            
        Returns:
            Signal string or SignalInfo object
        """
        # Store the bar
        self.current_bar = bar
        self.bars.append(bar)
        
        # Call strategy-specific logic
        result = self.calculate_signal(bar)
        
        # Handle both old string format and new SignalInfo format
        if isinstance(result, SignalInfo):
            signal = result.signal
        else:
            signal = result
        
        # Update position tracking
        if signal == Signal.BUY and self.position == 0:
            self.position = 1
        elif signal == Signal.SELL and self.position > 0:
            self.position = 0
        
        return result
    
    @abstractmethod
    def calculate_signal(self, bar: Bar) -> Union[str, SignalInfo]:
        """
        Strategy-specific signal calculation
        
        Args:
            bar: Current price bar
            
        Returns:
            Signal string or SignalInfo object with signal and reason
        """
        pass
    
    def get_required_history(self) -> int:
        """
        Number of historical bars required before generating signals
        Override in subclass if needed
        """
        return 0
    
    def reset(self):
        """Reset strategy state"""
        self.position = 0
        self.bars.clear()
        self.current_bar = None
        self.metadata.clear()
        self.market_data.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        return {
            'symbol': self.symbol,
            'position': self.position,
            'bars_processed': len(self.bars),
            'metadata': self.metadata.copy()
        }
    
    def needs_market_data(self) -> bool:
        """
        Override this method to indicate if strategy needs market data (e.g., SPY)
        """
        return False
    
    def set_market_data(self, symbol: str, data: Any):
        """
        Set market data for the strategy
        
        Args:
            symbol: Market symbol (e.g., 'SPY')
            data: Historical data for the market symbol
        """
        self.market_data[symbol] = data