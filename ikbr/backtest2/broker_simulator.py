"""
Simple broker simulator for backtesting
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from loguru import logger


@dataclass
class Trade:
    """Represents a single trade"""
    timestamp: datetime
    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    commission: float
    cash_before: float
    cash_after: float
    position_before: int
    position_after: int
    pnl: Optional[float] = None  # P&L for closing trades
    return_pct: Optional[float] = None  # Return percentage for closing trades
    trade_number: Optional[int] = None  # Sequential trade number
    signal: Optional[str] = None  # The signal that triggered this trade


class BrokerSimulator:
    """Simulates a broker for backtesting"""
    
    def __init__(self, initial_capital: float = 1000000, commission: float = 1.0, position_size_pct: float = 0.10):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.position = 0  # Number of shares held
        self.avg_price = 0.0  # Average entry price
        self.trades: List[Trade] = []
        self.symbol = None
        self.trade_counter = 0  # Track trade numbers
        self.position_size_pct = position_size_pct  # Percentage of equity to use per trade
        
    def buy(self, symbol: str, price: float, quantity: Optional[int] = None, timestamp: Optional[datetime] = None) -> bool:
        """
        Execute a buy order
        
        Args:
            symbol: Stock symbol
            price: Price per share
            quantity: Number of shares (if None, use all available cash)
            timestamp: Trade timestamp
            
        Returns:
            True if order executed, False otherwise
        """
        # Calculate quantity if not specified
        if quantity is None:
            # Calculate total equity (cash + position value)
            total_equity = self.cash + (self.position * price if self.position > 0 else 0)
            # Use position_size_pct of total equity
            position_value = total_equity * self.position_size_pct
            # Calculate shares, accounting for commission
            max_shares = int((position_value - self.commission) / price)
            quantity = max_shares
        
        if quantity <= 0:
            logger.warning(f"Cannot buy {symbol}: quantity is {quantity}")
            return False
        
        total_cost = (quantity * price) + self.commission
        
        if total_cost > self.cash:
            logger.warning(f"Cannot buy {quantity} shares of {symbol}: cost ${total_cost:.2f} > cash ${self.cash:.2f}")
            return False
        
        # Execute trade
        cash_before = self.cash
        position_before = self.position
        
        self.cash -= total_cost
        
        # Update average price
        if self.position == 0:
            self.avg_price = price
        else:
            # Calculate weighted average
            total_value = (self.position * self.avg_price) + (quantity * price)
            self.position += quantity
            self.avg_price = total_value / self.position
            quantity = quantity  # We already added to position above, so reset for trade record
            self.position -= quantity  # Reset for trade record
        
        self.position += quantity
        self.symbol = symbol
        
        # Record trade
        self.trade_counter += 1
        trade = Trade(
            timestamp=timestamp or datetime.now(),
            symbol=symbol,
            action='BUY',
            quantity=quantity,
            price=price,
            commission=self.commission,
            cash_before=cash_before,
            cash_after=self.cash,
            position_before=position_before,
            position_after=self.position,
            trade_number=self.trade_counter,
            signal='BUY'  # Will be updated by backtest engine if needed
        )
        self.trades.append(trade)
        
        logger.info(f"BUY {quantity} {symbol} @ ${price:.2f}, cost=${total_cost:.2f}, cash=${self.cash:.2f}")
        return True
    
    def sell(self, symbol: str, price: float, quantity: Optional[int] = None, timestamp: Optional[datetime] = None) -> bool:
        """
        Execute a sell order
        
        Args:
            symbol: Stock symbol
            price: Price per share
            quantity: Number of shares (if None, sell all)
            timestamp: Trade timestamp
            
        Returns:
            True if order executed, False otherwise
        """
        if self.position <= 0:
            logger.warning(f"Cannot sell {symbol}: no position")
            return False
        
        if self.symbol and self.symbol != symbol:
            logger.warning(f"Cannot sell {symbol}: holding {self.symbol}")
            return False
        
        # Use all shares if quantity not specified
        if quantity is None:
            quantity = self.position
        
        if quantity > self.position:
            logger.warning(f"Cannot sell {quantity} shares: only have {self.position}")
            quantity = self.position
        
        # Execute trade
        cash_before = self.cash
        position_before = self.position
        
        proceeds = (quantity * price) - self.commission
        self.cash += proceeds
        self.position -= quantity
        
        # Calculate P&L and return percentage
        pnl = (price - self.avg_price) * quantity - self.commission
        # Calculate return percentage based on the cost basis
        cost_basis = self.avg_price * quantity + self.commission
        return_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
        
        # Reset avg price if position closed
        if self.position == 0:
            self.avg_price = 0.0
            self.symbol = None
        
        # Record trade
        self.trade_counter += 1
        trade = Trade(
            timestamp=timestamp or datetime.now(),
            symbol=symbol,
            action='SELL',
            quantity=quantity,
            price=price,
            commission=self.commission,
            cash_before=cash_before,
            cash_after=self.cash,
            position_before=position_before,
            position_after=self.position,
            pnl=pnl,
            return_pct=return_pct,
            trade_number=self.trade_counter,
            signal='SELL'  # Will be updated by backtest engine if needed
        )
        self.trades.append(trade)
        
        logger.info(f"SELL {quantity} {symbol} @ ${price:.2f}, proceeds=${proceeds:.2f}, P&L=${pnl:.2f}, cash=${self.cash:.2f}")
        return True
    
    def get_position_value(self, current_price: float) -> float:
        """Get current value of position"""
        return self.position * current_price
    
    def get_total_equity(self, current_price: float) -> float:
        """Get total equity (cash + position value)"""
        return self.cash + self.get_position_value(current_price)
    
    def get_total_value(self, current_price: float) -> float:
        """Get total portfolio value (cash + position)"""
        return self.cash + self.get_position_value(current_price)
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Get unrealized P&L"""
        if self.position == 0:
            return 0.0
        return (current_price - self.avg_price) * self.position
    
    def get_trades(self) -> List[Trade]:
        """Get list of all trades"""
        return self.trades.copy()
    
    def get_stats(self, current_price: float) -> dict:
        """Get broker statistics"""
        total_value = self.get_total_value(current_price)
        total_return = (total_value - self.initial_capital) / self.initial_capital * 100
        
        # Calculate trade statistics
        winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl and t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in self.trades if t.pnl)
        
        return {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'position': self.position,
            'position_value': self.get_position_value(current_price),
            'total_value': total_value,
            'total_return_pct': total_return,
            'unrealized_pnl': self.get_unrealized_pnl(current_price),
            'realized_pnl': total_pnl,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len([t for t in self.trades if t.pnl]) if any(t.pnl for t in self.trades) else 0
        }


if __name__ == "__main__":
    # Test the broker simulator
    broker = BrokerSimulator(initial_capital=10000, commission=1.0)
    
    # Test trades
    broker.buy('SPY', 400.0, timestamp=datetime(2024, 1, 1, 9, 30))
    broker.sell('SPY', 405.0, 10, timestamp=datetime(2024, 1, 1, 10, 30))
    broker.buy('SPY', 403.0, 5, timestamp=datetime(2024, 1, 1, 11, 30))
    broker.sell('SPY', 407.0, timestamp=datetime(2024, 1, 1, 15, 30))
    
    # Print stats
    stats = broker.get_stats(current_price=407.0)
    print("\nBroker Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTrades:")
    for trade in broker.get_trades():
        print(f"  {trade.timestamp.strftime('%H:%M')} {trade.action} {trade.quantity} @ ${trade.price:.2f}"
              f" (P&L: ${trade.pnl:.2f})" if trade.pnl else "")