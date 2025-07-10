#!/usr/bin/env python3
"""
Example usage of the backtesting system
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backtest2.backtest_engine import BacktestEngine
from backtest2.simple_momentum import SimpleMomentumStrategy
from backtest2.metrics import PerformanceMetrics


def run_simple_backtest():
    """Run a simple backtest example"""
    print("Running Simple Momentum Strategy Backtest")
    print("-" * 50)
    
    # Create strategy
    strategy = SimpleMomentumStrategy('SPY', sma_period=20)
    
    # Create backtest engine
    engine = BacktestEngine()
    
    try:
        # Run backtest
        results = engine.run(
            strategy=strategy,
            symbol='SPY',
            days=30,  # 30 days of data
            initial_capital=100000,
            commission=1.0,
            bar_size='5 mins'
        )
        
        # Get equity curve
        equity_curve = engine.get_equity_curve()
        
        # Calculate performance metrics
        metrics = PerformanceMetrics.calculate_all_metrics(results, equity_curve)
        
        # Print results
        print(f"\nBacktest Period: {results['start_date']} to {results['end_date']}")
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"\nTotal Trades: {results['total_trades']}")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"Win Rate: {results['win_rate']:.1%}")
        
        print(f"\nSharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        
        # Show first few trades
        trades_df = engine.get_trades_df()
        if not trades_df.empty:
            print("\nFirst 5 Trades:")
            print(trades_df.head())
            
        return results
        
    finally:
        engine.cleanup()


def run_comparison():
    """Compare different strategies/parameters"""
    print("\n\nComparing Different SMA Periods")
    print("-" * 50)
    
    periods = [10, 20, 50]
    results_summary = []
    
    for period in periods:
        print(f"\nTesting SMA period: {period}")
        
        # Create strategy
        strategy = SimpleMomentumStrategy('SPY', sma_period=period)
        
        # Create backtest engine
        engine = BacktestEngine()
        
        try:
            # Run backtest
            results = engine.run(
                strategy=strategy,
                symbol='SPY',
                days=60,  # 60 days for comparison
                initial_capital=100000,
                commission=1.0
            )
            
            results_summary.append({
                'period': period,
                'return': results['total_return_pct'],
                'trades': results['total_trades'],
                'win_rate': results['win_rate']
            })
            
            print(f"  Return: {results['total_return_pct']:.2f}%")
            print(f"  Trades: {results['total_trades']}")
            print(f"  Win Rate: {results['win_rate']:.1%}")
            
        finally:
            engine.cleanup()
    
    # Print comparison summary
    print("\n\nComparison Summary:")
    print("-" * 50)
    print(f"{'SMA Period':<12} {'Return %':<10} {'Trades':<8} {'Win Rate':<10}")
    print("-" * 50)
    for r in results_summary:
        print(f"{r['period']:<12} {r['return']:<10.2f} {r['trades']:<8} {r['win_rate']:<10.1%}")


if __name__ == "__main__":
    # Run simple backtest
    run_simple_backtest()
    
    # Run comparison
    run_comparison()
    
    print("\n\nExample complete! You can now run backtests using:")
    print("  python backtest2/run_backtest.py --symbol SPY --days 30")
    print("\nOr with more options:")
    print("  python backtest2/run_backtest.py --symbol AAPL --days 60 --strategy enhanced --capital 50000 --output results.json")