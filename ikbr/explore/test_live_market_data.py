#!/usr/bin/env python3
"""
Simple SPY Price Streamer
========================

Streams live price data for SPY to test IB connection.

Usage:
    python explore_fixed.py
"""

from ib_async import *
import os
from datetime import datetime
from dotenv import load_dotenv
import signal
import sys

# Load environment variables
load_dotenv()

# Global IB connection
ib = IB()

# Flag to control streaming
streaming = True

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global streaming
    print("\n\nStopping stream...")
    streaming = False

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

def connect():
    """Connect to IB Gateway"""
    port = 4102 if os.getenv('TRADING_MODE', 'paper') == 'paper' else 4101
    
    print(f"Connecting to IB Gateway on port {port}...")
    
    try:
        ib.connect('localhost', port, clientId=1)
        
        if ib.isConnected():
            print("✅ Connected to IB Gateway!")
            
            # Set market data type
            # 1 = Live, 2 = Frozen, 3 = Delayed, 4 = Delayed Frozen
            ib.reqMarketDataType(3)  # Use delayed data by default
            print("📊 Using delayed market data (15 min delay)")
            
            return True
        else:
            print("❌ Connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Docker is running: docker ps")
        print("2. Check IB Gateway logs: docker-compose logs ib-gateway")
        print("3. Wait 30-60 seconds after starting the container")
        return False

def stream_spy():
    """Stream SPY price data"""
    global streaming
    
    # Create SPY contract
    spy = Stock('SPY', 'SMART', 'USD')
    
    try:
        # Qualify the contract
        ib.qualifyContracts(spy)
        print(f"✅ Qualified SPY contract (conId: {spy.conId})")
    except Exception as e:
        print(f"❌ Failed to qualify contract: {e}")
        return
    
    # Request market data
    ticker = ib.reqMktData(spy, '', False, False)
    print("📈 Subscribed to SPY market data")
    print("\nStreaming SPY prices (Press Ctrl+C to stop)...")
    print("-" * 60)
    
    # Header
    print(f"{'Time':<12} {'Last':<8} {'Bid':<8} {'Ask':<8} {'Volume':<12} {'Spread':<8}")
    print("-" * 60)
    
    last_update = None
    update_count = 0
    
    while streaming:
        # Check if we have data
        if ticker.last and ticker.last == ticker.last:  # NaN check
            # Only print if data has changed
            current_data = (ticker.last, ticker.bid, ticker.ask, ticker.volume)
            
            if current_data != last_update:
                update_count += 1
                last_update = current_data
                
                # Calculate spread
                spread = ticker.ask - ticker.bid if ticker.bid and ticker.ask else 0
                
                # Format and print
                print(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]} "
                      f"${ticker.last:<7.2f} "
                      f"${ticker.bid:<7.2f} "
                      f"${ticker.ask:<7.2f} "
                      f"{int(ticker.volume):<11,d} "
                      f"${spread:<7.3f}")
        
        # Small sleep to prevent CPU spinning
        ib.sleep(0.1)
    
    # Cleanup
    print("\n" + "-" * 60)
    print(f"✅ Stream stopped. Received {update_count} price updates.")
    ib.cancelMktData(spy)
    print("📊 Unsubscribed from market data")

def main():
    """Main function"""
    print("""
SPY Price Streamer
==================

This tool streams live SPY price data to test your IB connection.
""")
    
    # Connect to IB
    if not connect():
        print("\n❌ Failed to connect. Exiting.")
        return
    
    print()
    
    # Check if market is open
    now = datetime.now()
    weekday = now.weekday()
    hour = now.hour
    minute = now.minute
    
    if weekday >= 5:  # Weekend
        print("⚠️  Market is closed (weekend)")
        print("   You'll see the last traded prices (frozen data)")
    elif hour < 9 or (hour == 9 and minute < 30) or hour >= 16:
        print("⚠️  Market is closed (outside trading hours)")
        print("   You'll see the last traded prices (frozen data)")
    else:
        print("✅ Market is open!")
    
    print()
    
    # Stream prices
    try:
        stream_spy()
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Disconnect
        if ib.isConnected():
            ib.disconnect()
            print("\n✅ Disconnected from IB Gateway")

if __name__ == "__main__":
    main()