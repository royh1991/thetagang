"""
Simple backtesting engine
"""
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
from loguru import logger

from .data_fetcher import DataFetcher
from .broker_simulator import BrokerSimulator
from .strategy_base import StrategyBase, Bar, Signal, SignalInfo


class BacktestEngine:
    """Simple backtesting engine that connects data, strategy, and broker"""
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.broker: Optional[BrokerSimulator] = None
        self.strategy: Optional[StrategyBase] = None
        self.data: Optional[pd.DataFrame] = None
        self.results: Dict[str, Any] = {}
        
    def run(self, 
            strategy: StrategyBase,
            symbol: str,
            days: int,
            initial_capital: float = 100000,
            commission: float = 1.0,
            bar_size: str = '5 mins') -> Dict[str, Any]:
        """
        Run a backtest
        
        Args:
            strategy: Trading strategy instance
            symbol: Stock symbol
            days: Number of days to backtest
            initial_capital: Starting capital
            commission: Commission per trade
            bar_size: Bar size for historical data
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest for {symbol} over {days} days")
        
        # Initialize components
        self.strategy = strategy
        self.broker = BrokerSimulator(initial_capital, commission)
        
        # Fetch historical data
        logger.info("Fetching historical data...")
        self.data = self.data_fetcher.fetch_historical_data(symbol, days, bar_size)
        
        if self.data.empty:
            logger.error("No data fetched")
            return {'error': 'No data available'}
        
        logger.info(f"Fetched {len(self.data)} bars from {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        
        # Reset strategy state
        self.strategy.reset()
        
        # Process each bar
        bar_count = 0
        signal_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for idx, row in self.data.iterrows():
            # Create bar object
            bar = Bar(
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=int(row['volume'])
            )
            
            # Process bar through strategy
            result = self.strategy.on_bar(bar)
            
            # Handle both old string format and new SignalInfo format
            if isinstance(result, SignalInfo):
                signal = result.signal
                reason = result.reason
            else:
                signal = result
                reason = ""
            
            signal_count[signal] += 1
            
            # Execute trades based on signal
            if signal == Signal.BUY:
                self.broker.buy(symbol, bar.close, timestamp=bar.timestamp, reason=reason)
            elif signal == Signal.SELL:
                self.broker.sell(symbol, bar.close, timestamp=bar.timestamp, reason=reason)
            
            bar_count += 1
            
            # Log progress every 1000 bars
            if bar_count % 1000 == 0:
                logger.info(f"Processed {bar_count} bars...")
        
        # Close any open positions at the end
        if self.broker.position > 0:
            last_bar = self.data.iloc[-1]
            logger.info(f"Closing open position at end of backtest")
            self.broker.sell(symbol, last_bar['close'], timestamp=last_bar['timestamp'], 
                           reason="End of backtest - closing open position")
        
        # Compile results
        final_price = self.data.iloc[-1]['close']
        broker_stats = self.broker.get_stats(final_price)
        strategy_stats = self.strategy.get_stats()
        
        # Get indicators if available
        indicators = {}
        if hasattr(self.strategy, 'get_indicators'):
            indicators = self.strategy.get_indicators()
        
        self.results = {
            'symbol': symbol,
            'start_date': self.data['timestamp'].min(),
            'end_date': self.data['timestamp'].max(),
            'days': days,
            'bars_processed': bar_count,
            'initial_capital': initial_capital,
            'final_value': broker_stats['total_value'],
            'total_return_pct': broker_stats['total_return_pct'],
            'total_trades': broker_stats['total_trades'],
            'winning_trades': broker_stats['winning_trades'],
            'losing_trades': broker_stats['losing_trades'],
            'win_rate': broker_stats['win_rate'],
            'realized_pnl': broker_stats['realized_pnl'],
            'unrealized_pnl': broker_stats['unrealized_pnl'],
            'signals': signal_count,
            'broker_stats': broker_stats,
            'strategy_stats': strategy_stats,
            'trades': [self._trade_to_dict(t) for t in self.broker.get_trades()],
            'indicators': indicators
        }
        
        logger.info(f"Backtest complete: {broker_stats['total_trades']} trades, "
                   f"{broker_stats['total_return_pct']:.2f}% return")
        
        return self.results
    
    def _trade_to_dict(self, trade) -> dict:
        """Convert Trade object to dictionary for serialization"""
        return {
            'trade_number': trade.trade_number,
            'timestamp': trade.timestamp.isoformat(),
            'symbol': trade.symbol,
            'action': trade.action,
            'signal': trade.signal,
            'quantity': trade.quantity,
            'price': trade.price,
            'commission': trade.commission,
            'pnl': trade.pnl,
            'return_pct': trade.return_pct
        }
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as a DataFrame"""
        if not self.broker:
            return pd.DataFrame()
        
        trades = self.broker.get_trades()
        if not trades:
            return pd.DataFrame()
        
        return pd.DataFrame([self._trade_to_dict(t) for t in trades])
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Calculate equity curve from trades"""
        if not self.broker or self.data is None or self.data.empty:
            return pd.DataFrame()
        
        # Create equity curve
        equity_data = []
        
        # Track portfolio value at each bar
        cash = self.broker.initial_capital
        position = 0
        avg_price = 0
        trade_idx = 0
        trades = self.broker.get_trades()
        
        for idx, row in self.data.iterrows():
            timestamp = row['timestamp']
            price = row['close']
            
            # Check if there's a trade at this timestamp
            while trade_idx < len(trades) and trades[trade_idx].timestamp <= timestamp:
                trade = trades[trade_idx]
                if trade.action == 'BUY':
                    # Update position and average price
                    if position == 0:
                        avg_price = trade.price
                    else:
                        total_value = (position * avg_price) + (trade.quantity * trade.price)
                        position += trade.quantity
                        avg_price = total_value / position
                        position -= trade.quantity  # Reset for next calculation
                    
                    position += trade.quantity
                    cash = trade.cash_after
                else:  # SELL
                    position = trade.position_after
                    cash = trade.cash_after
                    if position == 0:
                        avg_price = 0
                
                trade_idx += 1
            
            # Calculate portfolio value
            portfolio_value = cash + (position * price)
            
            equity_data.append({
                'timestamp': timestamp,
                'cash': cash,
                'position_value': position * price,
                'total_value': portfolio_value,
                'return_pct': ((portfolio_value - self.broker.initial_capital) / 
                              self.broker.initial_capital * 100)
            })
        
        return pd.DataFrame(equity_data)
    
    def cleanup(self):
        """Cleanup resources"""
        self.data_fetcher.disconnect()


if __name__ == "__main__":
    from .simple_momentum import SimpleMomentumStrategy
    
    # Test the backtest engine
    engine = BacktestEngine()
    strategy = SimpleMomentumStrategy('SPY', sma_period=20)
    
    try:
        results = engine.run(
            strategy=strategy,
            symbol='SPY',
            days=30,
            initial_capital=100000,
            commission=1.0
        )
        
        print("\nBacktest Results:")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Final Value: ${results['final_value']:,.2f}")
        
        # Show trades
        trades_df = engine.get_trades_df()
        if not trades_df.empty:
            print("\nFirst 5 trades:")
            print(trades_df.head())
            
    finally:
        engine.cleanup()