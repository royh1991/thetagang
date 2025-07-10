#!/usr/bin/env python3
"""Trace data flow to understand the issue"""

import asyncio
import sys
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, '.')

from backtest.engine import BacktestEngine, BacktestConfig

# Monkey patch to trace data loading
original_fetch = None

async def trace_fetch_multiple_symbols(self, symbols, start_date, end_date, bar_size="5 mins"):
    """Trace what data is being fetched"""
    logger.info(f"📊 Fetching data for {symbols}")
    logger.info(f"   Date range: {start_date} to {end_date}")
    logger.info(f"   Duration: {(end_date - start_date).days} days")
    
    result = await original_fetch(self, symbols, start_date, end_date, bar_size)
    
    for symbol, df in result.items():
        logger.info(f"   {symbol}: {len(df)} bars fetched")
        if not df.empty:
            logger.info(f"     First: {df.index[0]}")
            logger.info(f"     Last: {df.index[-1]}")
            logger.info(f"     Actual range: {(df.index[-1] - df.index[0]).days} days")
    
    return result

async def trace_backtest(days):
    """Run traced backtest"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Tracing {days}-day backtest")
    logger.info('='*60)
    
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=days),
        end_date=datetime.now(),
        initial_capital=100000,
        data_frequency="5min",
        use_ib_data=True
    )
    
    # Patch the fetch method
    from backtest.ib_data_provider import IBDataProvider
    global original_fetch
    original_fetch = IBDataProvider.fetch_multiple_symbols
    IBDataProvider.fetch_multiple_symbols = trace_fetch_multiple_symbols
    
    engine = BacktestEngine(config)
    
    # Import after patching
    from strategies.examples.enhanced_momentum_strategy import EnhancedMomentumStrategy, EnhancedMomentumConfig
    
    strategy_config = EnhancedMomentumConfig(
        symbols=["AAPL"],
        max_positions=1,
        metadata={
            'momentum_threshold': 0.001,
            'use_trading_windows': False
        }
    )
    
    engine.add_strategy(EnhancedMomentumStrategy, strategy_config)
    
    result = await engine.run()
    
    # Restore
    IBDataProvider.fetch_multiple_symbols = original_fetch
    
    logger.info(f"\nResult: {result.total_trades} trades")
    return result

async def main():
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    await trace_backtest(10)
    await trace_backtest(100)

if __name__ == "__main__":
    asyncio.run(main())