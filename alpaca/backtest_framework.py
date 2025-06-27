import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order"""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_timestamp: Optional[datetime] = None
    order_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.order_id is None:
            self.order_id = f"{self.symbol}_{self.timestamp.timestamp()}"


@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100
    
    @property
    def realized_pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.quantity


@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    quantity: int
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    
    @property
    def duration(self):
        return self.exit_time - self.entry_time


class Portfolio:
    """Manages positions, cash, and tracks performance"""
    
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.pending_orders: List[Order] = []
        self.order_history: List[Order] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
    @property
    def market_value(self) -> float:
        """Total market value of all positions"""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_equity(self) -> float:
        """Total portfolio value (cash + positions)"""
        return self.cash + self.market_value
    
    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def realized_pnl(self) -> float:
        """Total realized P&L from closed trades"""
        return sum(trade.pnl for trade in self.trades)
    
    def update_prices(self, prices: Dict[str, float], timestamp: datetime):
        """Update current prices for all positions"""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
        
        # Record equity curve
        self.equity_curve.append((timestamp, self.total_equity))
    
    def can_afford(self, symbol: str, quantity: int, price: float) -> bool:
        """Check if we have enough cash for the order"""
        required_cash = quantity * price
        return self.cash >= required_cash
    
    def execute_order(self, order: Order, fill_price: float, timestamp: datetime) -> bool:
        """Execute an order at the given price"""
        if order.side == OrderSide.BUY:
            # Check if we can afford it
            cost = order.quantity * fill_price
            if cost > self.cash:
                order.status = OrderStatus.REJECTED
                return False
            
            # Deduct cash
            self.cash -= cost
            
            # Create or add to position
            if order.symbol in self.positions:
                # Average into existing position
                pos = self.positions[order.symbol]
                total_value = (pos.quantity * pos.entry_price) + cost
                pos.quantity += order.quantity
                pos.entry_price = total_value / pos.quantity
            else:
                # New position
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    entry_price=fill_price,
                    entry_time=timestamp,
                    current_price=fill_price
                )
        
        else:  # SELL
            if order.symbol not in self.positions:
                order.status = OrderStatus.REJECTED
                return False
            
            pos = self.positions[order.symbol]
            if pos.quantity < order.quantity:
                order.status = OrderStatus.REJECTED
                return False
            
            # Calculate proceeds
            proceeds = order.quantity * fill_price
            self.cash += proceeds
            
            # Update or close position
            if pos.quantity == order.quantity:
                # Close entire position
                pos.exit_price = fill_price
                pos.exit_time = timestamp
                
                # Record trade
                trade = Trade(
                    symbol=order.symbol,
                    entry_time=pos.entry_time,
                    exit_time=timestamp,
                    quantity=order.quantity,
                    entry_price=pos.entry_price,
                    exit_price=fill_price,
                    pnl=(fill_price - pos.entry_price) * order.quantity,
                    pnl_pct=((fill_price - pos.entry_price) / pos.entry_price) * 100,
                    exit_reason="Strategy Signal"
                )
                self.trades.append(trade)
                
                # Remove position
                del self.positions[order.symbol]
            else:
                # Partial close
                pos.quantity -= order.quantity
        
        # Mark order as filled
        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_timestamp = timestamp
        self.order_history.append(order)
        
        return True
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0
            }
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0
        
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Calculate returns for Sharpe ratio
        if len(self.equity_curve) > 1:
            equity_values = [eq[1] for eq in self.equity_curve]
            returns = np.diff(equity_values) / equity_values[:-1]
            sharpe_ratio = np.sqrt(252) * (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0
            
            # Max drawdown
            peak = equity_values[0]
            max_drawdown = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        total_return = ((self.total_equity - self.initial_cash) / self.initial_cash) * 100
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'total_pnl': self.realized_pnl
        }


class BacktestEngine:
    """Main backtesting engine that runs strategies on historical data"""
    
    def __init__(self, data_client, initial_cash: float = 100000.0):
        self.data_client = data_client
        self.portfolio = Portfolio(initial_cash)
        self.strategy = None
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        
    def run(self, strategy, symbols: List[str], start_date: datetime, end_date: datetime, 
            timeframe=None, progress_callback=None):
        """Run backtest for given strategy and parameters"""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        self.strategy = strategy
        self.strategy.portfolio = self.portfolio
        
        # Fetch historical data for all symbols
        print(f"\n📊 Loading historical data for {', '.join(symbols)}...")
        
        all_bars = {}
        for symbol in symbols:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe or TimeFrame.Minute,
                start=start_date,
                end=end_date
            )
            
            bars_response = self.data_client.get_stock_bars(request)
            if symbol in bars_response.data:
                all_bars[symbol] = bars_response.data[symbol]
                print(f"   {symbol}: {len(all_bars[symbol])} bars loaded")
        
        # Create unified timeline
        timestamps = set()
        for bars in all_bars.values():
            timestamps.update(bar.timestamp for bar in bars)
        
        timestamps = sorted(timestamps)
        print(f"\n⏱️  Simulating {len(timestamps)} time periods...")
        
        # Process each timestamp
        for i, timestamp in enumerate(timestamps):
            # Get current prices
            current_prices = {}
            current_bars = {}
            
            for symbol, bars in all_bars.items():
                # Find bar for this timestamp
                for bar in bars:
                    if bar.timestamp == timestamp:
                        current_prices[symbol] = bar.close
                        current_bars[symbol] = bar
                        self.price_history[symbol].append(bar)
                        break
            
            # Update portfolio prices
            self.portfolio.update_prices(current_prices, timestamp)
            
            # Process pending orders
            self._process_pending_orders(current_bars, timestamp)
            
            # Call strategy
            if current_bars:
                signals = self.strategy.process_bars(
                    current_bars, 
                    dict(self.price_history),
                    self.portfolio
                )
                
                # Process any new orders from strategy
                if signals:
                    for order in signals:
                        self.portfolio.pending_orders.append(order)
            
            # Progress callback
            if progress_callback and i % 100 == 0:
                progress = (i / len(timestamps)) * 100
                progress_callback(progress, timestamp)
        
        # Close any remaining positions
        self._close_all_positions(timestamps[-1] if timestamps else end_date)
        
        return self.portfolio.get_performance_metrics()
    
    def _process_pending_orders(self, current_bars: Dict, timestamp: datetime):
        """Process pending orders against current market data"""
        filled_orders = []
        
        for order in self.portfolio.pending_orders:
            if order.symbol not in current_bars:
                continue
            
            bar = current_bars[order.symbol]
            fill_price = None
            
            # Determine fill price based on order type
            if order.order_type == OrderType.MARKET:
                # Fill at current close (simplified)
                fill_price = bar.close
            
            elif order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and bar.low <= order.limit_price:
                    fill_price = min(order.limit_price, bar.close)
                elif order.side == OrderSide.SELL and bar.high >= order.limit_price:
                    fill_price = max(order.limit_price, bar.close)
            
            # Execute order if we have a fill price
            if fill_price is not None:
                if self.portfolio.execute_order(order, fill_price, timestamp):
                    filled_orders.append(order)
                    
                    # Notify strategy
                    if hasattr(self.strategy, 'on_order_filled'):
                        self.strategy.on_order_filled(order, fill_price, timestamp)
        
        # Remove filled orders from pending
        for order in filled_orders:
            self.portfolio.pending_orders.remove(order)
    
    def _close_all_positions(self, timestamp: datetime):
        """Close all remaining positions at end of backtest"""
        for symbol, position in list(self.portfolio.positions.items()):
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=position.quantity,
                order_type=OrderType.MARKET,
                timestamp=timestamp
            )
            
            # Use last known price
            fill_price = position.current_price
            self.portfolio.execute_order(order, fill_price, timestamp)