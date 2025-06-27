#!/usr/bin/env python3
"""
Bot Components Explorer - Fixed Style
=====================================

This explores the actual bot components we've built.
Shows how EventBus, MarketDataManager, OrderManager work.

Usage:
    python explore_components.py

Then use the commands directly (no await needed!)
"""

from ib_async import *
import time
import os
from datetime import datetime
from dotenv import load_dotenv

# Import our components
from core.event_bus import EventBus, Event, EventTypes, get_event_bus
from core.market_data import MarketDataManager, TickData
from core.order_manager import OrderManager, Signal
from core.risk_manager import RiskManager, RiskLimits

# Load environment variables
load_dotenv()

# Global objects - initialized by setup()
ib = None
event_bus = None
market_data = None
order_manager = None
risk_manager = None

# ============================================
# SETUP AND CONNECTION
# ============================================

def setup():
    """Initialize all components - run this first!"""
    global ib, event_bus, market_data, order_manager, risk_manager
    
    print("Setting up bot components...")
    
    # 1. IB Connection
    ib = IB()
    port = 4002 if os.getenv('TRADING_MODE', 'paper') == 'paper' else 4001
    print(f"Connecting to IB Gateway on port {port}...")
    
    try:
        ib.connect('localhost', port, clientId=1)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    if not ib.isConnected():
        print("❌ Failed to connect!")
        return False
    
    print("✅ Connected to IB Gateway")
    
    # 2. Event Bus - the message system
    event_bus = get_event_bus()
    event_bus.start()  # This is synchronous in our implementation
    print("✅ Event Bus started")
    
    # 3. Market Data Manager
    market_data = MarketDataManager(ib)
    market_data.start()  # This is synchronous too
    print("✅ Market Data Manager started")
    
    # 4. Risk Manager
    risk_limits = RiskLimits(
        max_position_size=5000,
        max_total_exposure=20000,
        max_daily_loss=1000
    )
    risk_manager = RiskManager(ib, risk_limits)
    print("✅ Risk Manager created")
    
    # 5. Order Manager
    order_manager = OrderManager(ib)
    order_manager.risk_manager = risk_manager
    print("✅ Order Manager created")
    
    print("\nAll components ready! Try the test functions below.")
    return True

def teardown():
    """Cleanup all components"""
    global ib, event_bus, market_data
    
    print("Cleaning up...")
    
    if market_data:
        market_data.stop()
    if event_bus:
        event_bus.stop()
    if ib and ib.isConnected():
        ib.disconnect()
    
    print("✅ Cleanup complete")

# ============================================
# EVENT BUS DEMOS
# ============================================

def test_events():
    """See how events flow through the system - simple demo"""
    
    print("\n" + "="*50)
    print("EVENT BUS DEMO")
    print("="*50)
    
    # Track events we receive
    events_received = []
    
    def my_handler(event: Event):
        """Simple handler that prints events"""
        events_received.append(event)
        print(f"\n📨 Event received!")
        print(f"  Type: {event.event_type}")
        print(f"  Source: {event.source}")
        print(f"  Age: {event.age_ms:.1f}ms")
        if isinstance(event.data, TickData):
            print(f"  Data: {event.data.symbol} @ ${event.data.last}")
        else:
            print(f"  Data: {event.data}")
    
    # Subscribe to tick events
    event_bus.subscribe(EventTypes.TICK, my_handler)
    print("✅ Subscribed to TICK events")
    
    # Emit a test tick event
    print("\nEmitting test tick event...")
    event_bus.emit(Event(
        EventTypes.TICK,
        TickData(
            symbol='TEST',
            timestamp=time.time(),
            last=100.50,
            bid=100.49,
            ask=100.51,
            volume=1000
        ),
        source="TestScript"
    ))
    
    # Wait a moment for processing
    ib.sleep(0.5)
    
    print(f"\n📊 Received {len(events_received)} events")
    
    # Show metrics
    metrics = event_bus.get_metrics()
    print("\n📈 Event Bus Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    event_bus.unsubscribe(EventTypes.TICK, my_handler)
    print("\n✅ Unsubscribed from events")

def monitor_events(duration=30):
    """
    Monitor ALL events in the system for a period
    
    Example:
        monitor_events(60)  # Monitor for 60 seconds
    """
    
    print(f"\nMonitoring all events for {duration} seconds...")
    print("This shows everything happening in the bot:\n")
    
    event_counts = {}
    
    def universal_handler(event: Event):
        """Handler that counts and displays events"""
        # Count events by type
        if event.event_type not in event_counts:
            event_counts[event.event_type] = 0
        event_counts[event.event_type] += 1
        
        # Display based on type
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if event.event_type == EventTypes.TICK:
            tick = event.data
            if isinstance(tick, TickData) and tick.last:
                print(f"[{timestamp}] TICK: {tick.symbol} @ ${tick.last:.2f}")
        else:
            print(f"[{timestamp}] {event.event_type} from {event.source}")
    
    # Subscribe to all event types
    event_types = [EventTypes.TICK, EventTypes.BAR, EventTypes.ORDER_SUBMITTED,
                   EventTypes.ORDER_FILLED, EventTypes.SIGNAL_GENERATED]
    
    for event_type in event_types:
        event_bus.subscribe(event_type, universal_handler)
    
    print("Monitoring started. Press Ctrl+C to stop early.\n")
    
    try:
        # Monitor for duration
        for i in range(duration):
            ib.sleep(1)
            if i > 0 and i % 10 == 0:
                print(f"\n--- {i}s elapsed, event counts: {event_counts} ---\n")
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    # Cleanup
    for event_type in event_types:
        event_bus.unsubscribe(event_type, universal_handler)
    
    print(f"\nMonitoring complete. Total events: {event_counts}")

# ============================================
# MARKET DATA DEMOS
# ============================================

def test_market_data(symbol='AAPL'):
    """
    Test MarketDataManager with a single symbol
    
    Example:
        test_market_data('SPY')
    """
    
    print("\n" + "="*50)
    print(f"MARKET DATA DEMO - {symbol}")
    print("="*50)
    
    # Subscribe to the symbol
    print(f"\nSubscribing to {symbol}...")
    success = market_data.subscribe_ticker(symbol)
    
    if not success:
        print(f"❌ Failed to subscribe to {symbol}")
        return
    
    print(f"✅ Subscribed to {symbol}")
    
    # Track ticks we receive
    tick_count = 0
    
    def on_tick(event: Event):
        nonlocal tick_count
        tick: TickData = event.data
        
        if tick.symbol == symbol:
            tick_count += 1
            print(f"\n📊 Tick #{tick_count} for {symbol}:")
            print(f"  Last: ${tick.last}")
            print(f"  Bid: ${tick.bid} x {tick.bid_size}")
            print(f"  Ask: ${tick.ask} x {tick.ask_size}")
            if tick.spread:
                print(f"  Spread: ${tick.spread:.3f}")
            if tick.mid:
                print(f"  Mid: ${tick.mid:.2f}")
    
    # Subscribe to tick events
    event_bus.subscribe(EventTypes.TICK, on_tick)
    
    # Wait for ticks
    print("\nWaiting for market data (10 seconds)...")
    ib.sleep(10)
    
    # Get latest from cache
    latest = market_data.get_latest_tick(symbol)
    if latest:
        print(f"\n📍 Latest tick from cache:")
        print(f"  Symbol: {latest.symbol}")
        print(f"  Last: ${latest.last}")
        print(f"  Time: {datetime.fromtimestamp(latest.timestamp).strftime('%H:%M:%S')}")
    
    # Get tick buffer stats
    buffer = market_data.get_tick_buffer(symbol)
    if buffer and buffer.count > 0:
        print(f"\n📈 Tick buffer has {buffer.count} ticks")
    
    # Cleanup
    event_bus.unsubscribe(EventTypes.TICK, on_tick)
    market_data.unsubscribe_ticker(symbol)
    print(f"\n✅ Unsubscribed from {symbol}")

def stream_multiple(symbols=['AAPL', 'MSFT', 'GOOGL'], duration=30):
    """
    Stream multiple symbols simultaneously
    
    Example:
        stream_multiple(['SPY', 'QQQ', 'IWM'], 60)
    """
    
    print(f"\nStreaming {len(symbols)} symbols for {duration} seconds...")
    
    # Subscribe to all symbols
    subscribed = []
    for symbol in symbols:
        if market_data.subscribe_ticker(symbol):
            print(f"✅ Subscribed to {symbol}")
            subscribed.append(symbol)
        else:
            print(f"❌ Failed to subscribe to {symbol}")
    
    if not subscribed:
        print("No symbols subscribed!")
        return
    
    # Track updates
    update_counts = {symbol: 0 for symbol in subscribed}
    
    def on_tick(event: Event):
        tick: TickData = event.data
        if tick.symbol in update_counts:
            update_counts[tick.symbol] += 1
    
    event_bus.subscribe(EventTypes.TICK, on_tick)
    
    print("\nStreaming... Press Ctrl+C to stop early.\n")
    
    try:
        # Show updates periodically
        for i in range(0, duration, 5):
            ib.sleep(5)
            
            print(f"[{i+5}s] Update counts:")
            for symbol in subscribed:
                latest = market_data.get_latest_tick(symbol)
                if latest and latest.last:
                    print(f"  {symbol}: {update_counts[symbol]} updates, last=${latest.last:.2f}")
                else:
                    print(f"  {symbol}: {update_counts[symbol]} updates, no price yet")
            print()
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    # Cleanup
    event_bus.unsubscribe(EventTypes.TICK, on_tick)
    for symbol in subscribed:
        market_data.unsubscribe_ticker(symbol)
    
    print("✅ Streaming complete")

def test_historical_data(symbol='SPY'):
    """Test getting historical data through MarketDataManager"""
    
    print(f"\n📊 Getting historical bars for {symbol}...")
    
    # Get 1 day of 5-minute bars
    bars = market_data.get_historical_bars(
        symbol,
        duration='1 D',
        bar_size='5 mins'
    )
    
    if bars:
        print(f"Got {len(bars)} bars")
        print(f"First bar: {bars[0].date} - Open=${bars[0].open:.2f}, Close=${bars[0].close:.2f}")
        print(f"Last bar: {bars[-1].date} - Open=${bars[-1].open:.2f}, Close=${bars[-1].close:.2f}")
        
        # Calculate day stats
        day_open = bars[0].open
        day_close = bars[-1].close
        day_high = max(bar.high for bar in bars)
        day_low = min(bar.low for bar in bars)
        
        print(f"\nDay Summary:")
        print(f"  Open: ${day_open:.2f}")
        print(f"  High: ${day_high:.2f}")
        print(f"  Low: ${day_low:.2f}")
        print(f"  Close: ${day_close:.2f}")
        print(f"  Change: ${day_close - day_open:.2f} ({(day_close - day_open) / day_open * 100:+.2f}%)")
    else:
        print("No historical data received")

# ============================================
# SIGNAL AND ORDER DEMOS
# ============================================

def test_signals():
    """Demo different types of trading signals"""
    
    print("\n" + "="*50)
    print("SIGNAL EXAMPLES")
    print("="*50)
    
    # 1. Market order
    market_signal = Signal(
        action="BUY",
        symbol="AAPL",
        quantity=100,
        order_type="MARKET"
    )
    print("\n1. Market Order Signal:")
    print(f"  {market_signal.action} {market_signal.quantity} {market_signal.symbol} @ MARKET")
    print(f"  Is Buy: {market_signal.is_buy}")
    
    # 2. Limit order
    limit_signal = Signal(
        action="BUY",
        symbol="SPY",
        quantity=10,
        order_type="LIMIT",
        limit_price=440.50,
        time_in_force="DAY"
    )
    print("\n2. Limit Order Signal:")
    print(f"  {limit_signal.action} {limit_signal.quantity} {limit_signal.symbol} @ ${limit_signal.limit_price}")
    print(f"  Time in Force: {limit_signal.time_in_force}")
    
    # 3. Bracket order (with stop loss and take profit)
    bracket_signal = Signal(
        action="BUY",
        symbol="TSLA",
        quantity=5,
        order_type="LIMIT",
        limit_price=200.00,
        stop_loss=195.00,
        take_profit=210.00,
        strategy_id="momentum_strategy"
    )
    print("\n3. Bracket Order Signal:")
    print(f"  {bracket_signal.action} {bracket_signal.quantity} {bracket_signal.symbol}")
    print(f"  Entry: ${bracket_signal.limit_price}")
    print(f"  Stop Loss: ${bracket_signal.stop_loss} (-${bracket_signal.limit_price - bracket_signal.stop_loss:.2f})")
    print(f"  Take Profit: ${bracket_signal.take_profit} (+${bracket_signal.take_profit - bracket_signal.limit_price:.2f})")
    print(f"  Strategy: {bracket_signal.strategy_id}")

def test_risk_checks():
    """Demo risk manager functionality"""
    
    print("\n" + "="*50)
    print("RISK MANAGER DEMO")
    print("="*50)
    
    print("\nRisk Limits:")
    print(f"  Max position size: ${risk_manager.limits.max_position_size:,}")
    print(f"  Max total exposure: ${risk_manager.limits.max_total_exposure:,}")
    print(f"  Max daily loss: ${risk_manager.limits.max_daily_loss:,}")
    
    # Test signals
    test_signals = [
        Signal("BUY", "AAPL", 100, "MARKET"),   # Normal size
        Signal("BUY", "AAPL", 10000, "MARKET"), # Too large
        Signal("BUY", "TSLA", 50, "MARKET"),    # Different symbol
    ]
    
    print("\nChecking signals against risk limits...")
    
    for i, signal in enumerate(test_signals, 1):
        print(f"\nSignal {i}: {signal.action} {signal.quantity} {signal.symbol}")
        
        # Get current price to calculate position value
        latest = market_data.get_latest_tick(signal.symbol)
        if latest and latest.last:
            position_value = signal.quantity * latest.last
            print(f"  Position value: ${position_value:,.2f}")
            
            # Simple risk check
            if position_value > risk_manager.limits.max_position_size:
                print(f"  ❌ Exceeds max position size (${risk_manager.limits.max_position_size:,})")
            else:
                print(f"  ✅ Within position size limit")
        else:
            print(f"  ⚠️  No price data for {signal.symbol}")
    
    # Show current risk metrics
    metrics = risk_manager.get_metrics()
    print("\n📊 Current Risk Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

# ============================================
# COMPLETE FLOW DEMO
# ============================================

def demo_flow(symbol='AAPL'):
    """
    Demo complete flow: Market Data → Strategy → Signal → Risk Check
    
    This is a SIMULATION - no real orders placed
    """
    
    print("\n" + "="*50)
    print(f"COMPLETE FLOW DEMO - {symbol}")
    print("="*50)
    
    print("\nFlow: Market Data → Event → Strategy → Signal → Risk Check")
    
    # Step 1: Subscribe to market data
    print(f"\n1️⃣  Subscribing to {symbol}...")
    if not market_data.subscribe_ticker(symbol):
        print("Failed to subscribe!")
        return
    
    # Step 2: Strategy logic
    print("\n2️⃣  Setting up simple momentum strategy...")
    
    price_history = []
    signal_sent = False
    
    def momentum_strategy(event: Event):
        nonlocal signal_sent
        
        tick: TickData = event.data
        if tick.symbol != symbol or not tick.last or signal_sent:
            return
        
        price_history.append(tick.last)
        print(f"\n  📊 Price update: ${tick.last:.2f}")
        
        # Simple momentum: buy if last 3 prices are increasing
        if len(price_history) >= 3:
            if price_history[-1] > price_history[-2] > price_history[-3]:
                print(f"  💡 Momentum detected! Prices: {price_history[-3:]} → RISING")
                print(f"  🎯 Generating BUY signal")
                
                # Create signal
                signal = Signal(
                    action="BUY",
                    symbol=symbol,
                    quantity=10,
                    order_type="LIMIT",
                    limit_price=tick.ask,
                    metadata={'strategy': 'momentum', 'trigger': 'price_rising'}
                )
                
                # Emit signal
                event_bus.emit(Event(
                    EventTypes.SIGNAL_GENERATED,
                    {'signal': signal},
                    source="MomentumStrategy"
                ))
                
                signal_sent = True
    
    # Step 3: Signal handler
    print("\n3️⃣  Setting up signal handler...")
    
    def handle_signal(event: Event):
        signal = event.data['signal']
        print(f"\n  📨 Signal received: {signal.action} {signal.quantity} {signal.symbol} @ ${signal.limit_price}")
        print(f"     Strategy metadata: {signal.metadata}")
        print(f"\n  🏦 In production, OrderManager would:")
        print(f"     1. Run risk checks")
        print(f"     2. Submit order to broker")
        print(f"     3. Track order status")
    
    # Connect handlers
    event_bus.subscribe(EventTypes.TICK, momentum_strategy)
    event_bus.subscribe(EventTypes.SIGNAL_GENERATED, handle_signal)
    
    # Step 4: Run for a bit
    print("\n4️⃣  Waiting for momentum (need 3 rising prices)...")
    
    try:
        for i in range(30):
            ib.sleep(1)
            if signal_sent:
                print("\n✅ Signal generated!")
                break
            if i > 0 and i % 5 == 0:
                print(f"  Waiting... {i}s (got {len(price_history)} prices so far)")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    if not signal_sent:
        print("\n⏰ No momentum detected in 30 seconds")
        if price_history:
            print(f"   Price history: {price_history}")
    
    # Cleanup
    event_bus.unsubscribe(EventTypes.TICK, momentum_strategy)
    event_bus.unsubscribe(EventTypes.SIGNAL_GENERATED, handle_signal)
    market_data.unsubscribe_ticker(symbol)
    
    print("\n✅ Demo complete!")

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("""
Bot Components Explorer
======================

This tool explores the bot's core components:
- EventBus: Message passing system
- MarketDataManager: Real-time and historical data
- OrderManager: Signal generation and orders
- RiskManager: Position and risk limits

Quick Start:
    setup()                         # Initialize all components
    
📊 Market Data:
    test_market_data('AAPL')       # Watch single symbol
    stream_multiple(['SPY','QQQ'])  # Stream multiple symbols
    test_historical_data('SPY')     # Get historical bars
    
📨 Event System:
    test_events()                   # See how events work
    monitor_events(30)              # Monitor all events for 30s
    
💡 Signals & Risk:
    test_signals()                  # Different signal types
    test_risk_checks()              # Risk limit checks
    
🔄 Complete Flow:
    demo_flow('AAPL')              # See full data → signal flow
    
🧹 Cleanup:
    teardown()                      # Disconnect and cleanup

Tips:
    - Run setup() first!
    - All functions work without 'await'
    - Watch console for real-time updates
    - Use Ctrl+C to stop monitoring functions
    """)