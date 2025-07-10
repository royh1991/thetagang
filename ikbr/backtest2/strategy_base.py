"""
Simple base class for trading strategies
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Dict, Any
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
    """Trading signal"""
    BUY = 'BUY'
    SELL = 'SELL'
    HOLD = 'HOLD'


class StrategyBase(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.position = 0  # Current position (positive = long, negative = short, 0 = flat)
        self.bars: List[Bar] = []  # Historical bars
        self.current_bar: Optional[Bar] = None
        self.metadata: Dict[str, Any] = {}  # Strategy-specific data
        
    def on_bar(self, bar: Bar) -> str:
        """
        Called when a new bar is received
        
        Args:
            bar: The new price bar
            
        Returns:
            Signal: BUY, SELL, or HOLD
        """
        # Store the bar
        self.current_bar = bar
        self.bars.append(bar)
        
        # Call strategy-specific logic
        signal = self.calculate_signal(bar)
        
        # Update position tracking
        if signal == Signal.BUY and self.position == 0:
            self.position = 1
        elif signal == Signal.SELL and self.position > 0:
            self.position = 0
        
        return signal
    
    @abstractmethod
    def calculate_signal(self, bar: Bar) -> str:
        """
        Strategy-specific signal calculation
        
        Args:
            bar: Current price bar
            
        Returns:
            Signal: BUY, SELL, or HOLD
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        return {
            'symbol': self.symbol,
            'position': self.position,
            'bars_processed': len(self.bars),
            'metadata': self.metadata.copy()
        }