#!/usr/bin/env python3
"""
Main backtest CLI script for running trading strategies
"""

import os
import sys
import argparse
import json
import importlib
from datetime import datetime, timedelta, time as datetime_time
from typing import Dict, List, Type
from dotenv import load_dotenv
from tabulate import tabulate

from alpaca.data.historical import StockHistoricalDataClient
from backtest_framework import BacktestEngine
from base_strategy import BaseStrategy

# Load environment variables
load_dotenv()

API_KEY = os.environ.get('ALPACA_API_KEY')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')

# Initialize data client
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def load_strategy(strategy_name: str) -> Type[BaseStrategy]:
    """Dynamically load a strategy class"""
    try:
        # Try to import from strategies module
        module = importlib.import_module(f'strategies.{strategy_name}')
        
        # Find the strategy class (assumes it ends with 'Strategy')
        for name, obj in module.__dict__.items():
            if (isinstance(obj, type) and 
                issubclass(obj, BaseStrategy) and 
                obj != BaseStrategy):
                return obj
        
        raise ValueError(f"No strategy class found in {strategy_name}")
        
    except ImportError as e:
        print(f"Error loading strategy '{strategy_name}': {e}")
        sys.exit(1)


def parse_strategy_params(param_strings: List[str]) -> Dict:
    """Parse strategy parameters from command line"""
    params = {}
    
    for param_str in param_strings or []:
        if '=' not in param_str:
            print(f"Invalid parameter format: {param_str}")
            print("Use format: key=value")
            sys.exit(1)
        
        key, value = param_str.split('=', 1)
        
        # Try to parse as number
        try:
            if '.' in value:
                params[key] = float(value)
            else:
                params[key] = int(value)
        except ValueError:
            # Keep as string
            params[key] = value
    
    return params


def print_results(metrics: Dict, strategy_name: str, symbols: List[str]):
    """Print backtest results in a formatted way"""
    print("\n" + "="*60)
    print(f"📊 BACKTEST RESULTS - {strategy_name}")
    print("="*60)
    print(f"Symbols: {', '.join(symbols)}")
    print()
    
    # Performance metrics
    performance_data = [
        ["Total Trades", metrics.get('total_trades', 0)],
        ["Winning Trades", f"{metrics.get('winning_trades', 0)} ({metrics.get('win_rate', 0):.1f}%)"],
        ["Losing Trades", metrics.get('losing_trades', 0)],
        ["Avg Win", f"${metrics.get('avg_win', 0):.2f}"],
        ["Avg Loss", f"${metrics.get('avg_loss', 0):.2f}"],
        ["Profit Factor", f"{metrics.get('profit_factor', 0):.2f}"],
        ["Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}"],
        ["Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%"],
    ]
    
    print(tabulate(performance_data, headers=["Metric", "Value"], tablefmt="grid"))
    
    print(f"\n💰 Returns:")
    print(f"   Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
    print(f"   Return: {metrics.get('total_return', 0):.2f}%")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Backtest trading strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run mean reversion strategy
  python backtest.py --strategy mean_reversion --symbols SPY,QQQ
  
  # Run with custom parameters
  python backtest.py --strategy momentum --symbols AAPL,TSLA \\
    --param fast_ma_period=5 --param slow_ma_period=20
  
  # Run for specific date range
  python backtest.py --strategy mean_reversion --symbols SPY \\
    --start "2024-01-01 09:30" --end "2024-01-31 16:00"
  
  # List available strategies
  python backtest.py --list-strategies
        '''
    )
    
    parser.add_argument('--strategy', '-s', required=False,
                        help='Strategy to run (e.g., mean_reversion, momentum)')
    
    parser.add_argument('--symbols', required=False,
                        help='Comma-separated list of symbols to trade')
    
    parser.add_argument('--start', type=str, default=None,
                        help='Start datetime (format: "YYYY-MM-DD HH:MM")')
    
    parser.add_argument('--end', type=str, default=None,
                        help='End datetime (format: "YYYY-MM-DD HH:MM")')
    
    parser.add_argument('--param', '-p', action='append', dest='params',
                        help='Strategy parameter (format: key=value)')
    
    parser.add_argument('--cash', '-c', type=float, default=100000,
                        help='Starting cash (default: 100000)')
    
    parser.add_argument('--list-strategies', action='store_true',
                        help='List available strategies')
    
    parser.add_argument('--describe', action='store_true',
                        help='Describe the selected strategy')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output during backtest')
    
    return parser.parse_args()


def list_strategies():
    """List all available strategies"""
    strategies_dir = os.path.join(os.path.dirname(__file__), 'strategies')
    
    print("\n📋 Available Strategies:")
    print("="*40)
    
    for filename in os.listdir(strategies_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            strategy_name = filename[:-3]
            
            try:
                strategy_class = load_strategy(strategy_name)
                instance = strategy_class()
                print(f"\n{strategy_name}")
                print(f"  {instance.description}")
            except Exception as e:
                print(f"\n{strategy_name} (error loading: {e})")


def progress_callback(progress: float, timestamp: datetime):
    """Show progress during backtest"""
    bar_length = 40
    filled_length = int(bar_length * progress // 100)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    print(f'\rProgress: |{bar}| {progress:.1f}% - {timestamp.strftime("%Y-%m-%d %H:%M")}', 
          end='', flush=True)


def main():
    """Main entry point"""
    args = parse_args()
    
    # List strategies if requested
    if args.list_strategies:
        list_strategies()
        return
    
    # Check required arguments
    if not args.strategy:
        print("Error: --strategy is required")
        print("Use --list-strategies to see available strategies")
        sys.exit(1)
    
    # Load strategy
    strategy_class = load_strategy(args.strategy)
    
    # Parse parameters
    params = parse_strategy_params(args.params)
    
    # Create strategy instance
    strategy = strategy_class(params)
    
    # Describe strategy if requested
    if args.describe:
        print(f"\n📖 Strategy: {args.strategy}")
        print(f"Description: {strategy.description}")
        print("\nParameters:")
        for key, value in strategy.params.items():
            print(f"  {key}: {value}")
        return
    
    # Check symbols
    if not args.symbols:
        print("Error: --symbols is required")
        sys.exit(1)
    
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Validate strategy parameters
    if not strategy.validate_params():
        print("Invalid strategy parameters")
        sys.exit(1)
    
    # Parse dates
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
    else:
        # Default to today
        today = datetime.now().date()
        start_date = datetime.combine(today, datetime_time(9, 30))
    
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d %H:%M")
    else:
        # Default to end of start date
        end_date = start_date.replace(hour=16, minute=0)
    
    print(f"\n🚀 Starting Backtest")
    print(f"Strategy: {args.strategy}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Starting Cash: ${args.cash:,.2f}")
    print(f"\n{strategy.description}")
    print("-" * 60)
    
    # Create and run backtest engine
    engine = BacktestEngine(data_client, initial_cash=args.cash)
    
    # Run backtest
    try:
        callback = progress_callback if args.verbose else None
        metrics = engine.run(strategy, symbols, start_date, end_date, 
                            progress_callback=callback)
        
        if args.verbose:
            print()  # New line after progress bar
        
        # Print results
        print_results(metrics, args.strategy, symbols)
        
        # Print portfolio summary
        portfolio = engine.portfolio
        print(f"\n📊 Final Portfolio:")
        print(f"   Cash: ${portfolio.cash:,.2f}")
        print(f"   Positions Value: ${portfolio.market_value:,.2f}")
        print(f"   Total Equity: ${portfolio.total_equity:,.2f}")
        
        if portfolio.positions:
            print(f"\n📈 Open Positions:")
            for symbol, pos in portfolio.positions.items():
                print(f"   {symbol}: {pos.quantity} shares @ ${pos.entry_price:.2f} "
                      f"(current: ${pos.current_price:.2f}, "
                      f"P&L: ${pos.unrealized_pnl:+.2f})")
        
    except Exception as e:
        print(f"\n❌ Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()