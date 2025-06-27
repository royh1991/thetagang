"""
Backtesting Engine

Event-driven backtesting engine that simulates live trading
with historical data.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type
import numpy as np
import pandas as pd
from loguru import logger

from core.event_bus import EventBus, Event, EventTypes
from core.market_data import MarketDataManager, TickData
from core.order_manager import OrderManager, Signal
from core.risk_manager import RiskManager, PortfolioRisk
from strategies.base_strategy import BaseStrategy, StrategyConfig
from .data_provider import BacktestDataProvider
from .mock_broker import MockBroker


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000
    commission_per_share: float = 0.01
    slippage_pct: float = 0.001  # 0.1% slippage
    data_frequency: str = "1min"  # 1min, 5min, etc
    enable_shorting: bool = True
    use_adjusted_close: bool = True
    fill_at_next_bar: bool = True  # More realistic fills
    random_seed: Optional[int] = None


@dataclass
class BacktestResult:
    """Backtesting results and analytics"""
    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Time series data
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Additional metrics
    var_95: float = 0.0
    cvar_95: float = 0.0
    kelly_criterion: float = 0.0
    recovery_factor: float = 0.0
    
    def calculate_metrics(self):
        """Calculate all performance metrics"""
        if self.equity_curve.empty:
            return
        
        # Calculate returns
        returns = self.equity_curve['value'].pct_change().dropna()
        
        # Total and annualized return
        self.total_return = (self.equity_curve['value'].iloc[-1] / 
                           self.equity_curve['value'].iloc[0] - 1)
        
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        self.annualized_return = (1 + self.total_return) ** (365 / days) - 1
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if returns.std() > 0:
            self.sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            self.sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252)
        
        # Maximum drawdown
        rolling_max = self.equity_curve['value'].expanding().max()
        drawdown = (self.equity_curve['value'] - rolling_max) / rolling_max
        self.max_drawdown = drawdown.min()
        
        # Calmar ratio
        if self.max_drawdown < 0:
            self.calmar_ratio = self.annualized_return / abs(self.max_drawdown)
        
        # Value at Risk (95% confidence)
        if len(returns) > 20:
            self.var_95 = np.percentile(returns, 5)
            # Conditional VaR (expected shortfall)
            self.cvar_95 = returns[returns <= self.var_95].mean()
        
        # Trade statistics
        if not self.trades.empty:
            self.total_trades = len(self.trades)
            self.winning_trades = len(self.trades[self.trades['pnl'] > 0])
            self.losing_trades = len(self.trades[self.trades['pnl'] <= 0])
            
            if self.total_trades > 0:
                self.win_rate = self.winning_trades / self.total_trades
            
            if self.winning_trades > 0:
                self.avg_win = self.trades[self.trades['pnl'] > 0]['pnl'].mean()
            
            if self.losing_trades > 0:
                self.avg_loss = abs(self.trades[self.trades['pnl'] <= 0]['pnl'].mean())
            
            if self.avg_loss > 0:
                self.profit_factor = (self.avg_win * self.winning_trades) / \
                                   (self.avg_loss * self.losing_trades)
            
            # Kelly Criterion
            if self.win_rate > 0 and self.avg_win > 0 and self.avg_loss > 0:
                self.kelly_criterion = (self.win_rate * self.avg_win - 
                                      (1 - self.win_rate) * self.avg_loss) / self.avg_win
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary"""
        return {
            'performance': {
                'total_return': f"{self.total_return:.2%}",
                'annualized_return': f"{self.annualized_return:.2%}",
                'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
                'sortino_ratio': f"{self.sortino_ratio:.2f}",
                'max_drawdown': f"{self.max_drawdown:.2%}",
                'calmar_ratio': f"{self.calmar_ratio:.2f}",
                'var_95': f"{self.var_95:.2%}",
                'cvar_95': f"{self.cvar_95:.2%}"
            },
            'trades': {
                'total_trades': self.total_trades,
                'win_rate': f"{self.win_rate:.2%}",
                'avg_win': f"${self.avg_win:.2f}",
                'avg_loss': f"${self.avg_loss:.2f}",
                'profit_factor': f"{self.profit_factor:.2f}"
            }
        }


class BacktestEngine:
    """
    Event-driven backtesting engine
    
    Features:
    - Realistic order execution with slippage
    - Commission modeling
    - Multiple strategy support
    - Performance analytics
    - Risk management integration
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
        # Initialize components
        self.event_bus = EventBus("backtest")
        self.data_provider = BacktestDataProvider(config)
        self.mock_broker = MockBroker(config)
        
        # Market data manager will be initialized per backtest
        self.market_data: Optional[MarketDataManager] = None
        self.order_manager: Optional[OrderManager] = None
        self.risk_manager: Optional[RiskManager] = None
        
        # Strategies
        self.strategies: List[BaseStrategy] = []
        
        # Results tracking
        self.equity_curve: List[Dict] = []
        self.trades: List[Dict] = []
        self.positions_history: List[Dict] = []
        
        # Set random seed if specified
        if config.random_seed:
            np.random.seed(config.random_seed)
    
    def add_strategy(self, strategy_class: Type[BaseStrategy], 
                    config: StrategyConfig):
        """Add a strategy to backtest"""
        # Create strategy instance
        strategy = strategy_class(
            config=config,
            market_data=self.market_data,
            order_manager=self.order_manager,
            risk_manager=self.risk_manager
        )
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {config.name}")
    
    async def run(self) -> BacktestResult:
        """Run the backtest"""
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        try:
            # Initialize components
            await self._initialize()
            
            # Load historical data
            await self.data_provider.load_data(
                symbols=[s for strategy in self.strategies for s in strategy.config.symbols],
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )
            
            # Start all strategies
            for strategy in self.strategies:
                await strategy.start()
            
            # Run simulation
            await self._run_simulation()
            
            # Stop strategies
            for strategy in self.strategies:
                await strategy.stop()
            
            # Calculate and return results
            return self._calculate_results()
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
        finally:
            await self._cleanup()
    
    async def _initialize(self):
        """Initialize backtest components"""
        # Start event bus
        await self.event_bus.start()
        
        # Initialize mock broker
        self.mock_broker.initialize(self.config.initial_capital)
        
        # Create managers with mock IB
        mock_ib = self.mock_broker.get_mock_ib()
        self.market_data = MarketDataManager(mock_ib)
        self.order_manager = OrderManager(mock_ib)
        self.risk_manager = RiskManager(mock_ib)
        
        # Link components
        self.order_manager.set_risk_manager(self.risk_manager)
        
        # Start managers
        await self.market_data.start()
        await self.risk_manager.start()
        
        # Subscribe to events
        self.event_bus.subscribe(EventTypes.ORDER_FILLED, self._on_order_filled)
        
        logger.info("Backtest engine initialized")
    
    async def _cleanup(self):
        """Cleanup after backtest"""
        if self.market_data:
            await self.market_data.stop()
        if self.risk_manager:
            await self.risk_manager.stop()
        await self.event_bus.stop()
    
    async def _run_simulation(self):
        """Run the main simulation loop"""
        current_date = self.config.start_date
        
        while current_date <= self.config.end_date:
            # Get data for current timestamp
            tick_data = self.data_provider.get_tick_data(current_date)
            
            if tick_data:
                # Update mock broker with current prices
                self.mock_broker.update_prices(tick_data)
                
                # Process each tick
                for symbol, tick in tick_data.items():
                    # Emit tick event
                    await self.event_bus.emit(Event(
                        EventTypes.TICK,
                        tick,
                        source="BacktestEngine"
                    ))
                
                # Process pending orders
                await self.mock_broker.process_orders()
                
                # Record equity curve
                portfolio_value = self.mock_broker.get_portfolio_value()
                self.equity_curve.append({
                    'timestamp': current_date,
                    'value': portfolio_value,
                    'cash': self.mock_broker.cash,
                    'positions_value': portfolio_value - self.mock_broker.cash
                })
                
                # Record positions
                positions = self.mock_broker.get_positions()
                if positions:
                    self.positions_history.append({
                        'timestamp': current_date,
                        'positions': positions.copy()
                    })
            
            # Advance time
            if self.config.data_frequency == "1min":
                current_date += timedelta(minutes=1)
            elif self.config.data_frequency == "5min":
                current_date += timedelta(minutes=5)
            elif self.config.data_frequency == "1hour":
                current_date += timedelta(hours=1)
            else:
                current_date += timedelta(days=1)
            
            # Skip weekends and after hours
            if current_date.weekday() >= 5:  # Weekend
                current_date = current_date + timedelta(days=7-current_date.weekday())
                current_date = current_date.replace(hour=9, minute=30)
            elif current_date.hour < 9 or (current_date.hour == 9 and current_date.minute < 30):
                current_date = current_date.replace(hour=9, minute=30)
            elif current_date.hour >= 16:
                current_date = current_date + timedelta(days=1)
                current_date = current_date.replace(hour=9, minute=30)
        
        logger.info("Simulation completed")
    
    async def _on_order_filled(self, event: Event):
        """Record filled orders"""
        order_info = event.data.get('order_info')
        if order_info:
            self.trades.append({
                'timestamp': datetime.now(),
                'symbol': order_info.signal.symbol,
                'action': order_info.signal.action,
                'quantity': order_info.signal.quantity,
                'price': order_info.fill_price,
                'commission': order_info.commission,
                'strategy': order_info.signal.strategy_id
            })
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results"""
        # Convert to DataFrames
        equity_df = pd.DataFrame(self.equity_curve)
        if not equity_df.empty:
            equity_df.set_index('timestamp', inplace=True)
        
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate trade P&L
        if not trades_df.empty:
            # Group by symbol to match buys and sells
            for symbol in trades_df['symbol'].unique():
                symbol_trades = trades_df[trades_df['symbol'] == symbol].copy()
                
                # Simple FIFO matching for P&L calculation
                position = 0
                cost_basis = 0
                
                for idx, trade in symbol_trades.iterrows():
                    if trade['action'] == 'BUY':
                        position += trade['quantity']
                        cost_basis += trade['quantity'] * trade['price']
                    else:  # SELL
                        if position > 0:
                            avg_cost = cost_basis / position
                            pnl = (trade['price'] - avg_cost) * trade['quantity']
                            pnl -= trade['commission']
                            trades_df.loc[idx, 'pnl'] = pnl
                            
                            position -= trade['quantity']
                            cost_basis -= trade['quantity'] * avg_cost
        
        # Create result object
        result = BacktestResult(
            equity_curve=equity_df,
            trades=trades_df
        )
        
        # Calculate all metrics
        result.calculate_metrics()
        
        # Log summary
        logger.info(f"Backtest complete: Total return={result.total_return:.2%}, "
                   f"Sharpe={result.sharpe_ratio:.2f}, "
                   f"Max DD={result.max_drawdown:.2%}")
        
        return result
    
    def plot_results(self, result: BacktestResult):
        """Plot backtest results (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # Equity curve
            ax1 = axes[0]
            result.equity_curve['value'].plot(ax=ax1, label='Portfolio Value')
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True)
            
            # Drawdown
            ax2 = axes[1]
            rolling_max = result.equity_curve['value'].expanding().max()
            drawdown = (result.equity_curve['value'] - rolling_max) / rolling_max * 100
            drawdown.plot(ax=ax2, label='Drawdown %', color='red')
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.legend()
            ax2.grid(True)
            
            # Returns distribution
            ax3 = axes[2]
            returns = result.equity_curve['value'].pct_change().dropna()
            returns.hist(ax=ax3, bins=50, alpha=0.7)
            ax3.set_title('Returns Distribution')
            ax3.set_xlabel('Daily Returns')
            ax3.set_ylabel('Frequency')
            ax3.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")