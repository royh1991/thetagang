#!/usr/bin/env python3
"""Test to verify data accumulation in backtests"""

import asyncio
import sys
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, '.')

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.examples.enhanced_momentum_strategy import EnhancedMomentumStrategy, EnhancedMomentumConfig

async def run_backtest_with_days(days: int):
    """Run a backtest for specified number of days"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {days}-day backtest")
    logger.info('='*60)
    
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=days),
        end_date=datetime.now(),
        initial_capital=100000,
        commission_per_share=0.01,
        data_frequency="5min",
        use_ib_data=True
    )
    
    engine = BacktestEngine(config)
    
    strategy_config = EnhancedMomentumConfig(
        symbols=["AAPL"],
        max_positions=1,
        position_size_pct=0.3,
        metadata={
            'lookback_period': 20,
            'momentum_threshold': 0.005,
            'ma_period': 20,
            'regime_ma_period': 50,
            'use_trading_windows': False
        }
    )
    
    engine.add_strategy(EnhancedMomentumStrategy, strategy_config)
    
    result = await engine.run()
    
    logger.info(f"\nResults for {days}-day backtest:")
    logger.info(f"Total trades: {result.total_trades}")
    logger.info(f"Total return: {result.total_return:.2%}")
    
    if not result.trades.empty:
        logger.info("\nTrade details:")
        for _, trade in result.trades.iterrows():
            logger.info(f"  {trade['timestamp']} - {trade['action']} {trade['quantity']} @ ${trade['price']:.2f}")
    
    return result

async def main():
    """Compare different backtest periods"""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    # Run both backtests
    result_10 = await run_backtest_with_days(10)
    result_100 = await run_backtest_with_days(100)
    
    # Compare
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON")
    logger.info('='*60)
    logger.info(f"10-day backtest: {result_10.total_trades} trades")
    logger.info(f"100-day backtest: {result_100.total_trades} trades")
    
    if result_10.total_trades == result_100.total_trades:
        logger.warning("⚠️  Same number of trades! There's still an issue.")
    else:
        logger.success(f"✅ Different trade counts! 100-day has {result_100.total_trades - result_10.total_trades} more trades")

if __name__ == "__main__":
    asyncio.run(main())