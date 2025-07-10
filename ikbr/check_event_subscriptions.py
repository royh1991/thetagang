#!/usr/bin/env python3
"""Check event subscriptions to debug duplicates"""

import asyncio
import sys
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, '.')

from core.event_bus import get_event_bus, EventTypes

# Track subscriptions
original_subscribe = None
subscription_counts = {}

def track_subscribe(self, event_type, handler, priority=0):
    """Track subscriptions"""
    key = f"{event_type}:{handler.__name__ if hasattr(handler, '__name__') else str(handler)}"
    subscription_counts[key] = subscription_counts.get(key, 0) + 1
    logger.info(f"📌 Subscribe #{subscription_counts[key]}: {key}")
    
    # Call original
    return original_subscribe(self, event_type, handler, priority)

async def test_subscriptions():
    """Test subscription tracking"""
    from core.event_bus import EventBus
    global original_subscribe
    original_subscribe = EventBus.subscribe
    EventBus.subscribe = track_subscribe
    
    from backtest.engine import BacktestEngine, BacktestConfig
    from strategies.examples.fixed_momentum_strategy import FixedMomentumStrategy, FixedMomentumConfig
    
    logger.info("Running first backtest...")
    
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=5),
        end_date=datetime.now(),
        initial_capital=100000,
        data_frequency="5min",
        use_ib_data=True
    )
    
    engine = BacktestEngine(config)
    
    strategy_config = FixedMomentumConfig(
        symbols=["AAPL"],
        max_positions=1
    )
    
    engine.add_strategy(FixedMomentumStrategy, strategy_config)
    result1 = await engine.run()
    
    logger.info(f"\nFirst backtest: {result1.total_trades} trades")
    
    # Check ORDER_FILLED subscriptions
    order_filled_subs = [k for k in subscription_counts.keys() if 'ORDER_FILLED' in k]
    logger.info(f"\nORDER_FILLED subscriptions: {len(order_filled_subs)}")
    for sub in order_filled_subs:
        logger.info(f"  {sub}: {subscription_counts[sub]} times")
    
    # Run second backtest
    logger.info("\n\nRunning second backtest...")
    
    engine2 = BacktestEngine(config)
    engine2.add_strategy(FixedMomentumStrategy, strategy_config)
    result2 = await engine2.run()
    
    logger.info(f"\nSecond backtest: {result2.total_trades} trades")
    
    # Check subscriptions again
    logger.info(f"\nFinal ORDER_FILLED subscriptions:")
    for sub in order_filled_subs:
        logger.info(f"  {sub}: {subscription_counts[sub]} times")
    
    # Restore
    EventBus.subscribe = original_subscribe

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    asyncio.run(test_subscriptions())