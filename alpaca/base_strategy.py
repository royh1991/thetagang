from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from collections import deque
from backtest_framework import Order, OrderSide, OrderType, Portfolio


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize strategy with parameters
        
        Args:
            params: Dictionary of strategy-specific parameters
        """
        self.params = params or {}
        self.portfolio: Optional[Portfolio] = None
        self.name = self.__class__.__name__
        
    @abstractmethod
    def process_bars(self, bars: Dict, price_history: Dict[str, deque], 
                     portfolio: Portfolio) -> List[Order]:
        """
        Process new price bars and return list of orders
        
        Args:
            bars: Dictionary of symbol -> current bar
            price_history: Dictionary of symbol -> deque of historical bars
            portfolio: Current portfolio state
            
        Returns:
            List of Order objects to execute
        """
        pass
    
    @abstractmethod
    def get_required_history(self) -> int:
        """
        Return number of historical bars required before strategy can trade
        
        Returns:
            Number of bars needed for indicators/calculations
        """
        pass
    
    def on_order_filled(self, order: Order, fill_price: float, timestamp):
        """
        Called when an order is filled
        
        Args:
            order: The filled order
            fill_price: Price at which order was filled
            timestamp: Time of fill
        """
        pass
    
    def on_position_closed(self, symbol: str, pnl: float, pnl_pct: float):
        """
        Called when a position is closed
        
        Args:
            symbol: Symbol of closed position
            pnl: Realized profit/loss
            pnl_pct: Realized profit/loss percentage
        """
        pass
    
    def log(self, message: str, level: str = "INFO"):
        """
        Log a message with strategy context
        
        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        print(f"[{self.name}] {level}: {message}")
    
    @property
    def description(self) -> str:
        """
        Return a description of the strategy
        """
        return f"{self.name} strategy"
    
    def validate_params(self) -> bool:
        """
        Validate strategy parameters
        
        Returns:
            True if parameters are valid
        """
        return True
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Get a parameter value with optional default
        
        Args:
            key: Parameter key
            default: Default value if key not found
            
        Returns:
            Parameter value or default
        """
        return self.params.get(key, default)
    
    def create_order(self, symbol: str, side: OrderSide, quantity: int, 
                     order_type: OrderType = OrderType.MARKET, 
                     limit_price: Optional[float] = None,
                     stop_price: Optional[float] = None) -> Order:
        """
        Helper method to create an order
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Number of shares
            order_type: Type of order
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            
        Returns:
            Order object
        """
        return Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price
        )
    
    def should_trade(self, symbol: str, price_history: deque) -> bool:
        """
        Check if we have enough history to trade
        
        Args:
            symbol: Trading symbol
            price_history: Historical prices for symbol
            
        Returns:
            True if we can trade
        """
        return len(price_history) >= self.get_required_history()
    
    def calculate_position_size(self, symbol: str, price: float, 
                                portfolio: Portfolio) -> int:
        """
        Calculate position size based on strategy rules
        
        Args:
            symbol: Trading symbol
            price: Current price
            portfolio: Current portfolio state
            
        Returns:
            Number of shares to trade
        """
        # Default implementation: fixed position size or percentage of portfolio
        position_size_pct = self.get_param('position_size_pct', 0.1)  # 10% default
        max_position_value = portfolio.total_equity * position_size_pct
        shares = int(max_position_value / price)
        
        # Ensure we don't exceed available cash
        max_affordable = int(portfolio.cash / price)
        return min(shares, max_affordable)