#!/usr/bin/env python3
"""
Verify the duplicate trades fix
"""

import asyncio
import sys
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, '.')

from backtest.engine import BacktestEngine, BacktestConfig
from backtest.report_generator import ReportGenerator
from strategies.examples.simple_momentum_strategy import SimpleMomentumStrategy, SimpleMomentumConfig
from core.event_bus import EventTypes, Event

# Track ORDER_FILLED events
order_filled_events = []

async def track_order_filled(event: Event):
    order_info = event.data.get('order_info')
    if order_info:
        order_filled_events.append({
            'source': event.source,
            'order_id': order_info.order_id,
            'symbol': order_info.signal.symbol,
            'action': order_info.signal.action,
            'quantity': order_info.signal.quantity,
            'fill_price': order_info.fill_price
        })

async def main():
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    print("=== Testing duplicate trades fix ===")
    
    # Backtest config
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        initial_capital=100000,
        data_frequency="5min",
        use_ib_data=True
    )
    
    # Create engine
    engine = BacktestEngine(config)
    
    # Strategy config
    strategy_config = SimpleMomentumConfig(
        symbols=['AAPL', 'NVDA'],
        max_positions=2,
        position_size_pct=0.4,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        cooldown_period=60.0,
        metadata={
            'lookback_period': 10,
            'momentum_threshold': 0.001
        }
    )
    
    # Add strategy
    engine.add_strategy(SimpleMomentumStrategy, strategy_config)
    
    # DO NOT call _initialize() here - let run() handle it
    # Subscribe to ORDER_FILLED events using the event bus directly
    engine.event_bus.subscribe(EventTypes.ORDER_FILLED, track_order_filled, priority=100)
    
    # Run backtest
    result = await engine.run()
    
    # Print analysis
    print(f"\n=== Results ===")
    print(f"ORDER_FILLED events: {len(order_filled_events)}")
    print(f"Trades recorded in engine: {len(engine.trades)}")
    print(f"Trades in result DataFrame: {len(result.trades)}")
    
    # Check for duplicates
    from collections import defaultdict
    events_by_order = defaultdict(list)
    for event in order_filled_events:
        events_by_order[event['order_id']].append(event)
    
    duplicates = {oid: events for oid, events in events_by_order.items() if len(events) > 1}
    
    if duplicates:
        print(f"\n⚠️  ISSUE: Found duplicate ORDER_FILLED events:")
        for order_id, events in duplicates.items():
            print(f"  Order {order_id[:8]}... has {len(events)} events")
    else:
        print("\n✅ SUCCESS: No duplicate ORDER_FILLED events!")
    
    # Check trade recording
    if len(order_filled_events) != len(result.trades):
        print(f"\n⚠️  ISSUE: Mismatch between events ({len(order_filled_events)}) and trades ({len(result.trades)})")
    else:
        print(f"✅ SUCCESS: Trade count matches event count!")
    
    # Save results
    if len(result.trades) > 0:
        result.trades.to_csv("fixed_trades.csv", index=False)
        print(f"\nTrades saved to fixed_trades.csv")
        print("\nFirst few trades:")
        print(result.trades[['timestamp', 'symbol', 'action', 'quantity', 'price']].head())

if __name__ == "__main__":
    asyncio.run(main())