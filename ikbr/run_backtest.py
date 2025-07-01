"""
Run Trading Bot - Backtest or Live Mode

Supports both backtesting with historical data and live trading (paper or production).

Usage:
    # Backtesting
    python run_backtest.py --mode backtest --strategy momentum --days 30
    
    # Live paper trading (default)
    python run_backtest.py --mode live --strategy momentum
    
    # Live production trading (requires confirmation)
    python run_backtest.py --mode live --trading-mode prod --strategy momentum
"""

import asyncio
import sys
import argparse
from datetime import datetime, timedelta
from typing import List
from loguru import logger

sys.path.insert(0, '.')

from backtest.engine import BacktestEngine, BacktestConfig
from backtest.report_generator import ReportGenerator
from live.engine import LiveEngine, LiveConfig
from strategies.examples.momentum_strategy import MomentumStrategy, MomentumConfig
from strategies.examples.mean_reversion_strategy import MeanReversionStrategy, MeanReversionConfig
from strategies.examples.enhanced_momentum_strategy import EnhancedMomentumStrategy, EnhancedMomentumConfig


async def run_momentum_backtest():
    """Run momentum strategy backtest"""
    logger.info("Running Momentum Strategy Backtest")
    
    # Configure backtest
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        initial_capital=100000,
        commission_per_share=0.01,
        slippage_pct=0.001,
        data_frequency="5min"
    )
    
    # Create engine
    engine = BacktestEngine(config)
    
    # Add momentum strategy
    strategy_config = MomentumConfig(
        symbols=["SPY", "AAPL", "TSLA"],
        max_positions=3,
        position_size_pct=0.3,
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        metadata={
            'lookback_period': 20,
            'momentum_threshold': 0.015
        }
    )
    
    # Note: We need to initialize managers first
    await engine._initialize()
    
    engine.add_strategy(MomentumStrategy, strategy_config)
    
    # Run backtest
    result = await engine.run()
    
    # Print results
    logger.info("\nMomentum Strategy Results:")
    logger.info(f"Total Return: {result.total_return:.2%}")
    logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {result.max_drawdown:.2%}")
    logger.info(f"Win Rate: {result.win_rate:.2%}")
    logger.info(f"Total Trades: {result.total_trades}")
    
    return result


async def run_mean_reversion_backtest():
    """Run mean reversion strategy backtest"""
    logger.info("\nRunning Mean Reversion Strategy Backtest")
    
    # Configure backtest
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        initial_capital=100000,
        commission_per_share=0.01,
        slippage_pct=0.001,
        data_frequency="5min"
    )
    
    # Create engine
    engine = BacktestEngine(config)
    
    # Add mean reversion strategy
    strategy_config = MeanReversionConfig(
        symbols=["SPY", "AAPL"],
        max_positions=2,
        position_size_pct=0.4,
        stop_loss_pct=0.03,
        take_profit_pct=0.02,
        metadata={
            'ma_period': 20,
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }
    )
    
    await engine._initialize()
    engine.add_strategy(MeanReversionStrategy, strategy_config)
    
    # Run backtest
    result = await engine.run()
    
    # Print results
    logger.info("\nMean Reversion Strategy Results:")
    logger.info(f"Total Return: {result.total_return:.2%}")
    logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {result.max_drawdown:.2%}")
    logger.info(f"Win Rate: {result.win_rate:.2%}")
    logger.info(f"Total Trades: {result.total_trades}")
    
    return result


async def run_live_trading(strategy_name: str, symbols: List[str], 
                          trading_mode: str = "paper", **kwargs):
    """Run live trading with specified strategy"""
    
    # Create live config
    config = LiveConfig(
        trading_mode=trading_mode,
        log_trades=True,
        enable_notifications=True,
        safety_check_prod=True
    )
    
    # Create engine
    engine = LiveEngine(config)
    
    # Connect to IB
    await engine.connect()
    await engine.initialize()
    
    # Configure strategy
    if strategy_name == "momentum":
        strategy_config = MomentumConfig(
            symbols=symbols,
            max_positions=len(symbols),
            position_size_pct=kwargs.get('position_size', 0.1),  # Smaller for live
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            cooldown_period=60.0,  # 1 minute cooldown for live
            metadata={
                'lookback_period': 20,
                'momentum_threshold': kwargs.get('momentum_threshold', 0.015)
            }
        )
        engine.add_strategy(MomentumStrategy, strategy_config)
        
    elif strategy_name == "mean_reversion":
        strategy_config = MeanReversionConfig(
            symbols=symbols,
            max_positions=len(symbols),
            position_size_pct=kwargs.get('position_size', 0.1),
            stop_loss_pct=0.03,
            take_profit_pct=0.02,
            cooldown_period=60.0,
            metadata={
                'ma_period': 20,
                'rsi_oversold': 30,
                'rsi_overbought': 70
            }
        )
        engine.add_strategy(MeanReversionStrategy, strategy_config)
    
    # Run live trading
    await engine.run()


async def main():
    """Run trading bot with CLI options"""
    parser = argparse.ArgumentParser(description="Run IBKR Trading Bot - Backtest or Live")
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='backtest',
                       choices=['backtest', 'live'],
                       help='Run mode: backtest historical data or live trading')
    
    # Trading mode (for live mode)
    parser.add_argument('--trading-mode', type=str, default='paper',
                       choices=['paper', 'prod'],
                       help='Trading mode: paper (simulated) or prod (real money)')
    
    # Strategy selection
    parser.add_argument('--strategy', type=str, default='momentum',
                       choices=['momentum', 'mean_reversion', 'enhanced_momentum', 'all'],
                       help='Strategy to run')
    
    # Date range
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to backtest')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD)')
    
    # Symbols
    parser.add_argument('--symbols', type=str, nargs='+',
                       default=['SPY', 'AAPL', 'TSLA'],
                       help='Symbols to trade')
    
    # Capital and risk
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital')
    parser.add_argument('--position-size', type=float, default=0.3,
                       help='Position size as fraction of capital')
    
    # Data options
    parser.add_argument('--use-ib-data', action='store_true', default=True,
                       help='Use real IB historical data')
    parser.add_argument('--no-ib-data', dest='use_ib_data', action='store_false',
                       help='Use synthetic data instead of IB')
    parser.add_argument('--frequency', type=str, default='5min',
                       choices=['1min', '5min', '15min', '1hour'],
                       help='Data frequency')
    
    # Reporting
    parser.add_argument('--report', action='store_true', default=True,
                       help='Generate HTML report')
    parser.add_argument('--no-report', dest='report', action='store_false',
                       help='Skip report generation')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stdout, level=log_level)
    
    if args.mode == "live":
        logger.add("logs/live_trading.log", level="DEBUG")
    else:
        logger.add("logs/backtest.log", level="DEBUG")
    
    # Handle live trading mode
    if args.mode == "live":
        # Validate live mode arguments
        if args.strategy == "all":
            logger.error("Cannot run 'all' strategies in live mode. Please select one strategy.")
            return
            
        if args.trading_mode == "prod":
            logger.warning("⚠️  PRODUCTION MODE SELECTED - REAL MONEY TRADING ⚠️")
        
        # Run live trading
        await run_live_trading(
            strategy_name=args.strategy,
            symbols=args.symbols,
            trading_mode=args.trading_mode,
            position_size=args.position_size,
            momentum_threshold=0.001 if args.strategy == "momentum" else None
        )
        return
    
    # Handle backtest mode
    # Determine date range
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
    
    logger.info("="*60)
    logger.info("IBKR STRATEGY BACKTEST")
    logger.info("="*60)
    logger.info(f"Date Range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Symbols: {', '.join(args.symbols)}")
    logger.info(f"Initial Capital: ${args.capital:,.2f}")
    logger.info(f"Data Source: {'IB Historical' if args.use_ib_data else 'Synthetic'}")
    logger.info("="*60)
    
    # Create backtest config
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        data_frequency=args.frequency,
        use_ib_data=args.use_ib_data
    )
    
    # Report generator
    report_gen = ReportGenerator() if args.report else None
    
    results = []
    
    # Run selected strategies
    if args.strategy in ['momentum', 'all']:
        logger.info("\nRunning Momentum Strategy Backtest...")
        
        engine = BacktestEngine(config)
        
        strategy_config = MomentumConfig(
            symbols=args.symbols,
            max_positions=min(3, len(args.symbols)),
            position_size_pct=args.position_size,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            cooldown_period=60.0,  # 1 minute cooldown
            metadata={
                'lookback_period': 20,  # Proper lookback period
                'momentum_threshold': 0.02,  # 2% threshold
                'ma_period': 50,  # Standard MA period
                'volume_multiplier': 1.2  # Above average volume
            }
        )
        
        await engine._initialize()
        engine.add_strategy(MomentumStrategy, strategy_config)
        
        result = await engine.run()
        results.append(('Momentum', result))
        
        if report_gen:
            report = report_gen.generate_report(
                result, 
                'Momentum_Strategy'
            )
            logger.info(f"Report generated successfully")
    
    if args.strategy in ['enhanced_momentum', 'all']:
        logger.info("\nRunning Enhanced Momentum Strategy Backtest...")
        
        engine = BacktestEngine(config)
        
        strategy_config = EnhancedMomentumConfig(
            symbols=args.symbols,  # Only trade the requested symbols
            max_positions=min(3, len(args.symbols)),
            position_size_pct=args.position_size * 0.8,  # Slightly smaller due to volatility sizing
            stop_loss_pct=0.03,  # Will be overridden by volatility-based stops
            take_profit_pct=0.06,  # Will be overridden by volatility-based targets
            cooldown_period=300.0,  # 5 minute cooldown
            metadata={
                'lookback_period': 20,
                'momentum_threshold': 0.005,  # 0.5% - lowered from 1.5%
                'volume_multiplier': 1.1,      # Slightly lower volume requirement
                'ma_period': 20,               # Shorter MA for faster signals
                'regime_ma_period': 50,        # Shorter regime MA for backtest
                'volatility_period': 20,
                'min_adr_pct': 0.01,          # Lower minimum volatility
                'allow_shorts': False,
                'use_trading_windows': False   # Disable trading window restrictions
            }
        )
        
        await engine._initialize()
        engine.add_strategy(EnhancedMomentumStrategy, strategy_config)
        
        result = await engine.run()
        results.append(('Enhanced Momentum', result))
        
        if report_gen:
            report = report_gen.generate_report(
                result, 
                'EnhancedMomentum_Strategy'
            )
            logger.info(f"Report generated successfully")
    
    if args.strategy in ['mean_reversion', 'all']:
        logger.info("\nRunning Mean Reversion Strategy Backtest...")
        
        engine = BacktestEngine(config)
        
        strategy_config = MeanReversionConfig(
            symbols=args.symbols[:2],  # Mean reversion works better with fewer symbols
            max_positions=2,
            position_size_pct=args.position_size * 1.5,  # Larger positions
            stop_loss_pct=0.03,
            take_profit_pct=0.02,
            metadata={
                'ma_period': 20,
                'rsi_oversold': 30,
                'rsi_overbought': 70
            }
        )
        
        await engine._initialize()
        engine.add_strategy(MeanReversionStrategy, strategy_config)
        
        result = await engine.run()
        results.append(('Mean Reversion', result))
        
        if report_gen:
            report = report_gen.generate_report(
                result, 
                'MeanReversion_Strategy'
            )
            logger.info(f"Report generated successfully")
    
    # Compare results if multiple strategies
    if len(results) > 1:
        logger.info("\n" + "="*60)
        logger.info("STRATEGY COMPARISON")
        logger.info("="*60)
        
        # Header
        header = f"{'Metric':<20}"
        for name, _ in results:
            header += f" {name:>15}"
        logger.info(header)
        logger.info("-" * len(header))
        
        # Metrics
        metrics = [
            ('Total Return', lambda r: f"{r.total_return:.2%}"),
            ('Annualized Return', lambda r: f"{r.annualized_return:.2%}"),
            ('Sharpe Ratio', lambda r: f"{r.sharpe_ratio:.2f}"),
            ('Max Drawdown', lambda r: f"{r.max_drawdown:.2%}"),
            ('Win Rate', lambda r: f"{r.win_rate:.2%}"),
            ('Total Trades', lambda r: f"{r.total_trades:d}"),
            ('Profit Factor', lambda r: f"{r.profit_factor:.2f}")
        ]
        
        for metric_name, formatter in metrics:
            line = f"{metric_name:<20}"
            for _, result in results:
                line += f" {formatter(result):>15}"
            logger.info(line)
    
    logger.info("\n" + "="*60)
    logger.info("Backtest Complete!")
    
    # Plot comparison if matplotlib available
    if len(results) > 1:
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for name, result in results:
                if not result.equity_curve.empty:
                    result.equity_curve['value'].plot(ax=ax, label=name, linewidth=2)
            
            ax.set_title('Strategy Comparison - Equity Curves')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('strategy_comparison.png', dpi=150)
            logger.info("Comparison chart saved to strategy_comparison.png")
            
        except ImportError:
            pass


if __name__ == "__main__":
    # Check if we're running in live mode (need to start IB event loop)
    if "--mode" in sys.argv and "live" in sys.argv:
        from ib_async import util
        util.startLoop()
    
    asyncio.run(main())