#!/usr/bin/env python3
"""Test the fixed momentum strategy"""

import asyncio
import sys
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, '.')

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.examples.fixed_momentum_strategy import FixedMomentumStrategy, FixedMomentumConfig

async def run_test(days: int):
    """Run backtest with fixed strategy"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {days}-day backtest with FixedMomentumStrategy")
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
    
    strategy_config = FixedMomentumConfig(
        symbols=["AAPL"],
        max_positions=1,
        position_size_pct=0.2,
        cooldown_period=300.0,  # 5 minute cooldown between trades
        metadata={
            'lookback_period': 20,
            'momentum_threshold': 0.005,  # 0.5%
            'ma_period': 20
        }
    )
    
    engine.add_strategy(FixedMomentumStrategy, strategy_config)
    
    result = await engine.run()
    
    logger.info(f"\nResults for {days}-day backtest:")
    logger.info(f"Total trades: {result.total_trades}")
    logger.info(f"Total return: {result.total_return:.2%}")
    logger.info(f"Win rate: {result.win_rate:.2%}")
    
    if not result.trades.empty:
        logger.info(f"\nFirst 5 trades:")
        for i, (_, trade) in enumerate(result.trades.head().iterrows()):
            logger.info(f"  {trade['timestamp']} - {trade['action']} {trade['quantity']} @ ${trade['price']:.2f}")
            if i >= 4:
                break
                
        if len(result.trades) > 5:
            logger.info(f"  ... and {len(result.trades) - 5} more trades")
    
    return result

async def main():
    """Compare different backtest periods"""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    # Run both backtests
    result_10 = await run_test(10)
    result_100 = await run_test(100)
    
    # Compare
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON")
    logger.info('='*60)
    logger.info(f"10-day backtest: {result_10.total_trades} trades, {result_10.total_return:.2%} return")
    logger.info(f"100-day backtest: {result_100.total_trades} trades, {result_100.total_return:.2%} return")
    
    if result_10.total_trades == result_100.total_trades:
        logger.error("❌ Still getting same number of trades!")
    else:
        logger.success(f"✅ Different trade counts! 100-day has {result_100.total_trades - result_10.total_trades} more trades")

if __name__ == "__main__":
    asyncio.run(main())