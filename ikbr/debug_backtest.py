#!/usr/bin/env python3
"""Debug why backtests are only generating 4 trades"""

import asyncio
import sys
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, '.')

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.examples.enhanced_momentum_strategy import EnhancedMomentumStrategy, EnhancedMomentumConfig

# Track signals
signals_generated = []
ticks_processed = 0

# Monkey patch to track signals
original_calculate_signals = EnhancedMomentumStrategy.calculate_signals

async def patched_calculate_signals(self, tick):
    global ticks_processed, signals_generated
    ticks_processed += 1
    
    # Log every 1000th tick
    if ticks_processed % 1000 == 0:
        logger.info(f"Processed {ticks_processed} ticks so far...")
        if hasattr(self, 'price_history') and tick.symbol in self.price_history:
            logger.info(f"  {tick.symbol}: {len(self.price_history[tick.symbol])} price points")
    
    signals = await original_calculate_signals(self, tick)
    if signals:
        for signal in signals:
            signals_generated.append({
                'time': datetime.fromtimestamp(tick.timestamp),
                'symbol': signal.symbol,
                'action': signal.action,
                'price': tick.last
            })
            logger.info(f"📍 Signal generated: {signal.action} {signal.symbol} at ${tick.last:.2f}")
    
    return signals

async def debug_backtest(days: int):
    """Run a debug backtest"""
    global ticks_processed, signals_generated
    ticks_processed = 0
    signals_generated = []
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {days}-day debug backtest")
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
    
    # Very relaxed parameters to ensure signals
    strategy_config = EnhancedMomentumConfig(
        symbols=["AAPL"],
        max_positions=1,
        position_size_pct=0.3,
        metadata={
            'lookback_period': 10,      # Very short
            'momentum_threshold': 0.001, # Very low (0.1%)
            'ma_period': 10,            # Very short
            'regime_ma_period': 20,     # Very short
            'volume_multiplier': 0.5,   # Very low
            'use_trading_windows': False,
            'min_adr_pct': 0.001       # Very low
        }
    )
    
    # Patch the strategy
    EnhancedMomentumStrategy.calculate_signals = patched_calculate_signals
    
    engine.add_strategy(EnhancedMomentumStrategy, strategy_config)
    
    result = await engine.run()
    
    # Restore original method
    EnhancedMomentumStrategy.calculate_signals = original_calculate_signals
    
    logger.info(f"\n📊 Debug Summary for {days}-day backtest:")
    logger.info(f"Total ticks processed: {ticks_processed}")
    logger.info(f"Total signals generated: {len(signals_generated)}")
    logger.info(f"Total trades executed: {result.total_trades}")
    
    if signals_generated:
        logger.info("\nSignals generated:")
        for sig in signals_generated:
            logger.info(f"  {sig['time']} - {sig['action']} {sig['symbol']} @ ${sig['price']:.2f}")
    
    if not result.trades.empty:
        logger.info("\nTrades executed:")
        for _, trade in result.trades.iterrows():
            logger.info(f"  {trade['timestamp']} - {trade['action']} {trade['quantity']} @ ${trade['price']:.2f}")
    
    return result, signals_generated

async def main():
    """Run debug tests"""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    # Run both backtests
    result_10, signals_10 = await debug_backtest(10)
    result_100, signals_100 = await debug_backtest(100)
    
    # Compare
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON")
    logger.info('='*60)
    logger.info(f"10-day: {ticks_processed} ticks, {len(signals_10)} signals, {result_10.total_trades} trades")
    logger.info(f"100-day: {ticks_processed} ticks, {len(signals_100)} signals, {result_100.total_trades} trades")

if __name__ == "__main__":
    asyncio.run(main())