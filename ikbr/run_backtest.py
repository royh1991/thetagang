"""
Run Backtest Example

Demonstrates backtesting the momentum and mean reversion strategies.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, '.')

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.examples.momentum_strategy import MomentumStrategy, MomentumConfig
from strategies.examples.mean_reversion_strategy import MeanReversionStrategy, MeanReversionConfig


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


async def main():
    """Run all backtests"""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add("logs/backtest.log", level="DEBUG")
    
    logger.info("="*60)
    logger.info("STRATEGY BACKTEST DEMONSTRATION")
    logger.info("="*60)
    
    # Run backtests
    momentum_result = await run_momentum_backtest()
    mean_reversion_result = await run_mean_reversion_backtest()
    
    # Compare strategies
    logger.info("\n" + "="*60)
    logger.info("STRATEGY COMPARISON")
    logger.info("="*60)
    
    logger.info(f"{'Metric':<20} {'Momentum':>15} {'Mean Reversion':>15}")
    logger.info("-"*50)
    logger.info(f"{'Total Return':<20} {momentum_result.total_return:>14.2%} {mean_reversion_result.total_return:>15.2%}")
    logger.info(f"{'Sharpe Ratio':<20} {momentum_result.sharpe_ratio:>14.2f} {mean_reversion_result.sharpe_ratio:>15.2f}")
    logger.info(f"{'Max Drawdown':<20} {momentum_result.max_drawdown:>14.2%} {mean_reversion_result.max_drawdown:>15.2%}")
    logger.info(f"{'Win Rate':<20} {momentum_result.win_rate:>14.2%} {mean_reversion_result.win_rate:>15.2%}")
    logger.info(f"{'Total Trades':<20} {momentum_result.total_trades:>14d} {mean_reversion_result.total_trades:>15d}")
    
    # Plot results if matplotlib available
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Momentum strategy
        momentum_result.equity_curve['value'].plot(ax=ax1, label='Momentum', color='blue')
        ax1.set_title('Momentum Strategy Equity Curve')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        ax1.legend()
        
        # Mean reversion strategy
        mean_reversion_result.equity_curve['value'].plot(ax=ax2, label='Mean Reversion', color='green')
        ax2.set_title('Mean Reversion Strategy Equity Curve')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        logger.info("\nBacktest charts saved to backtest_results.png")
        
    except ImportError:
        logger.info("\nInstall matplotlib to see charts: pip install matplotlib")


if __name__ == "__main__":
    asyncio.run(main())