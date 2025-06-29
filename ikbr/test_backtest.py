"""
Quick test of the backtesting framework
"""

import asyncio
import sys
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, '.')

from backtest.engine import BacktestEngine, BacktestConfig
from backtest.report_generator import ReportGenerator
from strategies.examples.momentum_strategy import MomentumStrategy, MomentumConfig


async def test_backtest():
    """Test the backtesting framework with minimal data"""
    logger.info("Testing backtest framework...")
    
    # Use a short time period for quick testing
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=5),
        end_date=datetime.now(),
        initial_capital=100000,
        commission_per_share=0.01,
        slippage_pct=0.001,
        data_frequency="5min",
        use_ib_data=True
    )
    
    # Create engine
    engine = BacktestEngine(config)
    
    # Simple momentum strategy with just SPY
    strategy_config = MomentumConfig(
        symbols=["SPY"],
        max_positions=1,
        position_size_pct=0.5,
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        metadata={
            'lookback_period': 10,
            'momentum_threshold': 0.01
        }
    )
    
    try:
        # Initialize engine
        await engine._initialize()
        
        # Add strategy
        engine.add_strategy(MomentumStrategy, strategy_config)
        
        # Run backtest
        logger.info("Running backtest...")
        result = await engine.run()
        
        # Show results
        logger.info("\n" + "="*50)
        logger.info("BACKTEST RESULTS")
        logger.info("="*50)
        logger.info(f"Total Return: {result.total_return:.2%}")
        logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {result.max_drawdown:.2%}")
        logger.info(f"Total Trades: {result.total_trades}")
        logger.info(f"Win Rate: {result.win_rate:.2%}")
        
        # Generate report
        logger.info("\nGenerating report...")
        report_gen = ReportGenerator()
        report = report_gen.generate_report(result, "Test_Momentum")
        
        logger.info("\nBacktest completed successfully!")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    # Run test
    asyncio.run(test_backtest())