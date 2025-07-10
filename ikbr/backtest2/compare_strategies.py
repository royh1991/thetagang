#!/usr/bin/env python3
"""
Compare different strategies on the same data
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backtest2.backtest_engine import BacktestEngine
from backtest2.simple_momentum import SimpleMomentumStrategy, EnhancedMomentumStrategy
from backtest2.nick_strategy import NickStrategy
from backtest2.metrics import PerformanceMetrics


def compare_strategies(symbol: str = 'SPY', days: int = 30, capital: float = 100000):
    """Run all strategies and compare results"""
    
    strategies = [
        ('Simple Momentum (SMA 20)', SimpleMomentumStrategy(symbol, sma_period=20)),
        ('Enhanced Momentum', EnhancedMomentumStrategy(symbol, fast_sma=10, slow_sma=30)),
        ('Nick Funnel Breakout', NickStrategy(symbol, lookback_period=20, rsi_threshold=50))
    ]
    
    results_summary = []
    
    for name, strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        engine = BacktestEngine()
        
        try:
            # Run backtest
            results = engine.run(
                strategy=strategy,
                symbol=symbol,
                days=days,
                initial_capital=capital,
                commission=1.0,
                bar_size='5 mins'
            )
            
            # Get metrics
            equity_curve = engine.get_equity_curve()
            metrics = PerformanceMetrics.calculate_all_metrics(results, equity_curve)
            
            # Store summary
            results_summary.append({
                'strategy': name,
                'total_return': results['total_return_pct'],
                'trades': results['total_trades'],
                'win_rate': results['win_rate'],
                'sharpe': metrics['sharpe_ratio'],
                'max_dd': metrics['max_drawdown_pct'],
                'profit_factor': metrics.get('profit_factor', 0)
            })
            
            # Print brief summary
            print(f"Return: {results['total_return_pct']:.2f}%")
            print(f"Trades: {results['total_trades']}")
            print(f"Win Rate: {results['win_rate']:.1%}")
            print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
            print(f"Max DD: {metrics['max_drawdown_pct']:.2f}%")
            
        finally:
            engine.cleanup()
    
    # Print comparison table
    print(f"\n\n{'='*80}")
    print(f"STRATEGY COMPARISON - {symbol} ({days} days)")
    print('='*80)
    print(f"{'Strategy':<25} {'Return%':>10} {'Trades':>8} {'Win Rate':>10} {'Sharpe':>8} {'Max DD%':>10}")
    print('-'*80)
    
    for r in results_summary:
        print(f"{r['strategy']:<25} {r['total_return']:>10.2f} {r['trades']:>8} "
              f"{r['win_rate']:>10.1%} {r['sharpe']:>8.2f} {r['max_dd']:>10.2f}")
    
    # Find best strategy by different metrics
    print("\n" + "="*80)
    print("BEST STRATEGIES BY METRIC:")
    print("="*80)
    
    # Best return
    best_return = max(results_summary, key=lambda x: x['total_return'])
    print(f"Highest Return: {best_return['strategy']} ({best_return['total_return']:.2f}%)")
    
    # Best Sharpe
    best_sharpe = max(results_summary, key=lambda x: x['sharpe'])
    print(f"Best Risk-Adjusted (Sharpe): {best_sharpe['strategy']} ({best_sharpe['sharpe']:.2f})")
    
    # Lowest drawdown
    best_dd = min(results_summary, key=lambda x: x['max_dd'])
    print(f"Lowest Drawdown: {best_dd['strategy']} ({best_dd['max_dd']:.2f}%)")
    
    # Best win rate
    best_wr = max(results_summary, key=lambda x: x['win_rate'])
    print(f"Highest Win Rate: {best_wr['strategy']} ({best_wr['win_rate']:.1%})")


if __name__ == "__main__":
    # Run comparison
    compare_strategies('SPY', days=30, capital=100000)
    
    print("\n\nNote: Past performance does not guarantee future results!")
    print("Always paper trade strategies before using real money.")