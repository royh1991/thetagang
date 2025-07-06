#!/usr/bin/env python3
"""
Minimal test to trace single order flow
"""

import asyncio
import sys
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, '.')

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.examples.enhanced_momentum_strategy import EnhancedMomentumStrategy, EnhancedMomentumConfig
from core.event_bus import EventTypes, get_event_bus

# Track ORDER_FILLED events
order_filled_count = 0

async def count_order_filled(event):
    """Count ORDER_FILLED events"""
    global order_filled_count
    order_filled_count += 1
    logger.info(f"ORDER_FILLED event #{order_filled_count} from {event.source}")

async def test_single_order():
    """Test with minimal setup to track order flow"""
    logger.info("Starting single order test...")
    
    # Subscribe to ORDER_FILLED events
    event_bus = get_event_bus()
    event_bus.subscribe(EventTypes.ORDER_FILLED, count_order_filled)
    
    # Configure a very short backtest
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=1),
        end_date=datetime.now(),
        initial_capital=100000,
        commission_per_share=0.01,
        data_frequency="5min",
        use_ib_data=True
    )
    
    # Create engine
    engine = BacktestEngine(config)
    
    # Add strategy
    strategy_config = EnhancedMomentumConfig(
        symbols=["AAPL"],
        max_positions=1,
        position_size_pct=0.3,
        metadata={
            'lookback_period': 20,
            'momentum_threshold': 0.001,  # Very low to ensure trades
            'use_trading_windows': False
        }
    )
    
    # DO NOT call _initialize()
    engine.add_strategy(EnhancedMomentumStrategy, strategy_config)
    
    # Run backtest
    result = await engine.run()
    
    # Check results
    logger.info(f"\nTotal ORDER_FILLED events: {order_filled_count}")
    logger.info(f"Total trades recorded: {len(result.trades)}")
    
    if not result.trades.empty:
        logger.info("\nTrades:")
        for idx, trade in result.trades.iterrows():
            logger.info(f"  {trade['timestamp']} - {trade['action']} {trade['quantity']} {trade['symbol']} @ ${trade['price']}")
    
    # Check for mismatch
    if order_filled_count != len(result.trades):
        logger.error(f"❌ Mismatch! {order_filled_count} events but {len(result.trades)} trades recorded")
    else:
        logger.success(f"✅ Match! {order_filled_count} events and {len(result.trades)} trades")

if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    asyncio.run(test_single_order())