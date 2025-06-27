"""
Component Demonstration

Shows the core components working together without requiring IB Gateway.
"""

import asyncio
import time
from loguru import logger

# Core components
from core.event_bus import EventBus, Event, EventTypes
from core.market_data import TickData, TickBuffer
from core.order_manager import Signal, validate_signal
from core.risk_manager import RiskLimits, PositionRisk


async def demo_event_bus():
    """Demonstrate event bus functionality"""
    logger.info("\n=== EVENT BUS DEMO ===")
    
    # Create event bus
    bus = EventBus("demo")
    await bus.start()
    
    # Track received events
    received_events = []
    
    async def event_handler(event: Event):
        received_events.append(event)
        logger.info(f"Received event: {event.event_type} with data: {event.data}")
    
    # Subscribe to events
    bus.subscribe(EventTypes.TICK, event_handler)
    bus.subscribe(EventTypes.ORDER_SUBMITTED, event_handler)
    
    # Emit some events
    tick_data = TickData(
        symbol="SPY",
        timestamp=time.time(),
        last=450.25,
        bid=450.20,
        ask=450.30,
        volume=1000000
    )
    
    await bus.emit(Event(EventTypes.TICK, tick_data))
    await bus.emit(Event(EventTypes.ORDER_SUBMITTED, {"order_id": "123", "symbol": "SPY"}))
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Check metrics
    metrics = bus.get_metrics()
    logger.info(f"Event bus metrics: {metrics}")
    
    await bus.stop()
    
    return len(received_events) == 2


def demo_tick_buffer():
    """Demonstrate tick buffer functionality"""
    logger.info("\n=== TICK BUFFER DEMO ===")
    
    # Create buffer
    buffer = TickBuffer(size=100)
    
    # Add some ticks
    base_price = 100.0
    for i in range(10):
        timestamp = time.time() + i
        price = base_price + i * 0.1
        volume = 1000 + i * 100
        buffer.add(timestamp, price, volume)
    
    # Get recent ticks
    timestamps, prices, volumes = buffer.get_recent(5)
    
    logger.info(f"Added {buffer.count} ticks to buffer")
    logger.info(f"Last 5 prices: {prices}")
    logger.info(f"Last 5 volumes: {volumes}")
    
    return buffer.count == 10


def demo_signal_validation():
    """Demonstrate signal creation and validation"""
    logger.info("\n=== SIGNAL VALIDATION DEMO ===")
    
    # Valid market order
    signal1 = Signal("BUY", "SPY", 100, "MARKET")
    valid1 = validate_signal(signal1)
    logger.info(f"Market order valid: {valid1}")
    
    # Valid limit order
    signal2 = Signal("SELL", "AAPL", 50, "LIMIT", limit_price=175.50)
    valid2 = validate_signal(signal2)
    logger.info(f"Limit order valid: {valid2}")
    
    # Invalid order (no limit price)
    signal3 = Signal("BUY", "TSLA", 10, "LIMIT")
    valid3 = validate_signal(signal3)
    logger.info(f"Limit order without price valid: {valid3}")
    
    # Invalid quantity
    signal4 = Signal("BUY", "SPY", 0, "MARKET")
    valid4 = validate_signal(signal4)
    logger.info(f"Zero quantity order valid: {valid4}")
    
    return valid1 and valid2 and not valid3 and not valid4


def demo_risk_calculations():
    """Demonstrate risk calculations"""
    logger.info("\n=== RISK CALCULATIONS DEMO ===")
    
    # Create position
    position = PositionRisk(
        symbol="SPY",
        quantity=100,
        market_value=45000,
        unrealized_pnl=500,
        realized_pnl=200,
        cost_basis=44300,
        current_price=450,
        position_pct=0.45
    )
    
    logger.info(f"Position: {position.symbol}")
    logger.info(f"Market Value: ${position.market_value:,.2f}")
    logger.info(f"Total P&L: ${position.total_pnl:,.2f}")
    logger.info(f"P&L %: {position.pnl_pct:.2%}")
    
    # Risk limits
    limits = RiskLimits(
        max_position_size=10000,
        max_positions=5,
        max_drawdown_pct=0.10,
        risk_per_trade_pct=0.01
    )
    
    logger.info(f"\nRisk Limits:")
    logger.info(f"Max position size: ${limits.max_position_size:,}")
    logger.info(f"Max positions: {limits.max_positions}")
    logger.info(f"Max drawdown: {limits.max_drawdown_pct:.1%}")
    
    return True


async def demo_integration():
    """Demonstrate components working together"""
    logger.info("\n=== INTEGRATION DEMO ===")
    
    # Create event bus
    bus = EventBus("integration")
    await bus.start()
    
    # Track order flow
    order_events = []
    
    async def track_orders(event: Event):
        order_events.append(event)
        logger.info(f"Order event: {event.event_type}")
    
    bus.subscribe(EventTypes.ORDER_SUBMITTED, track_orders)
    bus.subscribe(EventTypes.SIGNAL_GENERATED, track_orders)
    
    # Simulate strategy signal
    signal = Signal(
        action="BUY",
        symbol="SPY",
        quantity=100,
        order_type="LIMIT",
        limit_price=450.00,
        stop_loss=445.00,
        take_profit=455.00
    )
    
    # Emit signal generated event
    await bus.emit(Event(
        EventTypes.SIGNAL_GENERATED,
        {"strategy": "Demo", "signal": signal}
    ))
    
    # Simulate order submission
    await bus.emit(Event(
        EventTypes.ORDER_SUBMITTED,
        {
            "order_id": "demo-001",
            "signal": signal,
            "timestamp": time.time()
        }
    ))
    
    await asyncio.sleep(0.1)
    
    logger.info(f"Captured {len(order_events)} order events")
    
    await bus.stop()
    
    return len(order_events) == 2


async def main():
    """Run all demonstrations"""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    logger.info("="*60)
    logger.info("TRADING SYSTEM COMPONENT DEMONSTRATION")
    logger.info("="*60)
    
    results = {}
    
    # Run demos
    results['event_bus'] = await demo_event_bus()
    results['tick_buffer'] = demo_tick_buffer()
    results['signal_validation'] = demo_signal_validation()
    results['risk_calculations'] = demo_risk_calculations()
    results['integration'] = await demo_integration()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DEMONSTRATION RESULTS")
    logger.info("="*60)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for component, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{component}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} components working correctly")
    
    if passed == total:
        logger.success("\n✓ All components demonstrated successfully!")
    else:
        logger.warning(f"\n⚠ {total - passed} components had issues")


if __name__ == "__main__":
    import sys
    asyncio.run(main())