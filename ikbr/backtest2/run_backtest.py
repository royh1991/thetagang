#!/usr/bin/env python3
"""
Command-line runner for backtesting
"""
import sys
import json
import argparse
import os
from datetime import datetime
from pathlib import Path
from loguru import logger

# Add parent directory to path so we can import backtest2
sys.path.append(str(Path(__file__).parent.parent))

from backtest2.backtest_engine import BacktestEngine
from backtest2.simple_momentum import SimpleMomentumStrategy, EnhancedMomentumStrategy
from backtest2.nick_strategy import NickStrategy
from backtest2.metrics import PerformanceMetrics
# from backtest2.plotter import BacktestPlotter  # Temporarily disabled due to missing plotly


def setup_logging(verbose: bool = False, debug: bool = False):
    """Setup logging configuration"""
    logger.remove()  # Remove default handler
    
    if debug:
        logger.add(sys.stderr, level="DEBUG", format="{time:HH:mm:ss} | {level} | {message}")
    elif verbose:
        logger.add(sys.stderr, level="DEBUG", format="{time:HH:mm:ss} | {level} | {message}")
    else:
        logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def main():
    parser = argparse.ArgumentParser(description='Run backtest on historical data')
    
    # Required arguments
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol (e.g., SPY)')
    parser.add_argument('--days', type=int, required=True, help='Number of days to backtest')
    
    # Optional arguments
    parser.add_argument('--strategy', type=str, default='simple', 
                       choices=['simple', 'enhanced', 'nick'], help='Strategy to use')
    parser.add_argument('--capital', type=float, default=1000000, help='Initial capital')
    parser.add_argument('--commission', type=float, default=1.0, help='Commission per trade')
    parser.add_argument('--bar-size', type=str, default='5 mins', 
                       choices=['1 min', '5 mins', '15 mins', '30 mins', '1 hour', '1 day'],
                       help='Bar size for historical data')
    
    # Strategy parameters
    parser.add_argument('--sma-period', type=int, default=20, help='SMA period for simple strategy')
    parser.add_argument('--fast-sma', type=int, default=10, help='Fast SMA for enhanced strategy')
    parser.add_argument('--slow-sma', type=int, default=30, help='Slow SMA for enhanced strategy')
    
    # Nick strategy parameters
    parser.add_argument('--lookback-period', type=int, default=20, help='Breakout lookback period for Nick strategy')
    parser.add_argument('--rsi-threshold', type=int, default=50, help='RSI threshold for Nick strategy')
    parser.add_argument('--volume-multiplier', type=float, default=1.5, help='Volume spike multiplier for Nick strategy')
    parser.add_argument('--adx-threshold', type=int, default=20, help='ADX trend threshold for Nick strategy')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging for Nick strategy')
    parser.add_argument('--debug-csv', type=str, help='Save debug data to CSV file (requires --debug)')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output file for results (JSON format)')
    parser.add_argument('--trades-csv', type=str, help='Output file for trades (CSV format)')
    parser.add_argument('--plot', type=str, help='Save plot to HTML file')
    parser.add_argument('--show-plot', action='store_true', help='Show plot in browser')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.debug)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f"backtest2/runs/{timestamp}_{args.symbol}_{args.strategy}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Create strategy
    if args.strategy == 'simple':
        strategy = SimpleMomentumStrategy(args.symbol, sma_period=args.sma_period)
        logger.info(f"Using Simple Momentum Strategy with {args.sma_period} period SMA")
    elif args.strategy == 'enhanced':
        strategy = EnhancedMomentumStrategy(args.symbol, fast_sma=args.fast_sma, slow_sma=args.slow_sma)
        logger.info(f"Using Enhanced Momentum Strategy with {args.fast_sma}/{args.slow_sma} SMAs")
    else:  # nick
        strategy = NickStrategy(
            args.symbol,
            lookback_period=args.lookback_period,
            rsi_threshold=args.rsi_threshold,
            volume_multiplier=args.volume_multiplier,
            adx_trend_threshold=args.adx_threshold,
            debug=args.debug
        )
        logger.info(f"Using Nick's Funnel Breakout Strategy with ADX threshold {args.adx_threshold}")
    
    # Create and run backtest
    engine = BacktestEngine()
    
    try:
        # Run backtest
        results = engine.run(
            strategy=strategy,
            symbol=args.symbol,
            days=args.days,
            initial_capital=args.capital,
            commission=args.commission,
            bar_size=args.bar_size
        )
        
        # Get equity curve and calculate metrics
        equity_curve = engine.get_equity_curve()
        metrics = PerformanceMetrics.calculate_all_metrics(results, equity_curve)
        
        # Add metrics to results
        results['metrics'] = metrics
        
        # Display results
        print("\n" + "="*60)
        print(f"BACKTEST RESULTS - {args.symbol}")
        print("="*60)
        print(f"Period: {results['start_date']} to {results['end_date']}")
        print(f"Days: {results['days']}")
        print(f"Bars Processed: {results['bars_processed']}")
        print("")
        print(PerformanceMetrics.format_metrics(metrics))
        print("")
        print("=== Signal Summary ===")
        for signal_type, count in results['signals'].items():
            print(f"{signal_type}: {count}")
        
        # Always save a summary file
        summary = {
            'symbol': args.symbol,
            'strategy': args.strategy,
            'days': args.days,
            'bar_size': args.bar_size,
            'timestamp': timestamp,
            'metrics': metrics,
            'total_return_pct': results['total_return_pct'],
            'total_trades': results['total_trades'],
            'initial_capital': args.capital
        }
        summary_path = output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved to: {summary_path}")
        
        # Save results if requested
        if args.output:
            # Convert datetime objects to strings for JSON serialization
            results_json = json.dumps(results, default=str, indent=2)
            output_path = output_dir / args.output
            with open(output_path, 'w') as f:
                f.write(results_json)
            print(f"\nResults saved to: {output_path}")
        
        # Save trades if requested
        if args.trades_csv:
            trades_df = engine.get_trades_df()
            if not trades_df.empty:
                trades_path = output_dir / args.trades_csv
                trades_df.to_csv(trades_path, index=False)
                print(f"Trades saved to: {trades_path}")
            else:
                print("No trades to save")
        
        # Save debug CSV if requested
        if args.debug and args.debug_csv and args.strategy == 'nick':
            debug_path = output_dir / args.debug_csv
            strategy.save_debug_csv(str(debug_path))
            print(f"Debug data saved to: {debug_path}")
        
        # Create plot if requested
        if args.plot or args.show_plot:
            print("\nSkipping plot generation (plotly not installed)")
            # plotter = BacktestPlotter()
            # 
            # # Get data and indicators
            # data = engine.data
            # trades = results.get('trades', [])
            # indicators = results.get('indicators', {})
            # 
            # # Create the plot
            # if args.plot:
            #     plot_path = output_dir / args.plot
            #     save_path = str(plot_path)
            # else:
            #     save_path = str(output_dir / f"{args.symbol}_{args.strategy}_backtest.html")
            # 
            # plotter.plot_backtest_results(
            #     data=data,
            #     trades=trades,
            #     strategy_name=f"{args.strategy.title()} Strategy - {args.symbol}",
            #     show_indicators=indicators,
            #     save_path=save_path
            # )
            # 
            # # Also create metrics dashboard if we have trades
            # if trades:
            #     trades_df = engine.get_trades_df()
            #     if args.plot:
            #         metrics_path = str(output_dir / args.plot.replace('.html', '_metrics.html'))
            #     else:
            #         metrics_path = str(output_dir / f"{args.symbol}_{args.strategy}_metrics.html")
            #         
            #     plotter.plot_performance_metrics(
            #         metrics=metrics,
            #         trades_df=trades_df,
            #         save_path=metrics_path
            #     )
        
        # Return success
        return 0
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return 1
        
    finally:
        engine.cleanup()


if __name__ == "__main__":
    sys.exit(main())