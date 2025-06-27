#!/usr/bin/env python3
"""
Step-by-Step IBKR Explorer
==========================

Run this script and it will pause at each step so you can see what's happening.
Just run: python step_by_step.py
"""

import asyncio
import time
from ib_async import *
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================
# STEP-BY-STEP EXECUTION
# ============================================

def step(number, description):
    """Print step header and wait for user"""
    print(f"\n{'='*60}")
    print(f"STEP {number}: {description}")
    print(f"{'='*60}")
    input("Press Enter to continue...")

async def main():
    """Main function that runs through everything step by step"""
    
    print("""
    IBKR Step-by-Step Explorer
    ==========================
    
    This script will walk through each operation one at a time.
    You'll see exactly what happens at each step.
    
    Press Enter to continue through each step...
    """)
    
    input("Press Enter to begin...")
    
    # ----------------------------------------
    step(1, "Create IB Connection Object")
    
    print("Creating IB() object...")
    ib = IB()
    print(f"✅ IB object created: {ib}")
    print(f"   Connected? {ib.isConnected()}")
    
    # ----------------------------------------
    step(2, "Connect to IB Gateway")
    
    port = 4002 if os.getenv('TRADING_MODE', 'paper') == 'paper' else 4001
    print(f"Connecting to localhost:{port}")
    print("(Make sure Docker is running with IB Gateway!)")
    
    try:
        await ib.connectAsync('localhost', port, clientId=1, timeout=10)
        print(f"✅ Connected: {ib.isConnected()}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # ----------------------------------------
    step(3, "Request Market Data Type")
    
    print("Setting market data type to DELAYED (free)...")
    ib.reqMarketDataType(3)
    print("✅ Market data type set")
    print("   Note: Delayed data is 15 minutes behind")
    
    # ----------------------------------------
    step(4, "Create a Stock Contract")
    
    print("Creating AAPL stock contract...")
    aapl = Stock('AAPL', 'SMART', 'USD')
    print(f"✅ Contract created: {aapl}")
    print(f"   Symbol: {aapl.symbol}")
    print(f"   Exchange: {aapl.exchange}")
    print(f"   Currency: {aapl.currency}")
    
    # ----------------------------------------
    step(5, "Qualify the Contract")
    
    print("Qualifying contract (IB fills in missing details)...")
    ib.qualifyContracts(aapl)
    print(f"✅ Contract qualified:")
    print(f"   ConId: {aapl.conId}")
    print(f"   Primary Exchange: {aapl.primaryExchange}")
    
    # ----------------------------------------
    step(6, "Request Market Data")
    
    print("Requesting market data stream...")
    ticker = ib.reqMktData(aapl, '', False, False)
    print(f"✅ Ticker object created: {ticker}")
    print("   Waiting for data to arrive...")
    
    # ----------------------------------------
    step(7, "Wait for First Tick")
    
    print("Waiting 3 seconds for market data...")
    for i in range(3):
        ib.sleep(1)
        print(f"   {i+1} second...")
        if ticker.last:
            print(f"   ✅ Got price! ${ticker.last}")
            break
    
    # ----------------------------------------
    step(8, "Display All Market Data")
    
    print("Current market data for AAPL:")
    print(f"  Last: ${ticker.last}")
    print(f"  Bid: ${ticker.bid} (size: {ticker.bidSize})")
    print(f"  Ask: ${ticker.ask} (size: {ticker.askSize})")
    print(f"  Volume: {ticker.volume:,}")
    print(f"  High: ${ticker.high}")
    print(f"  Low: ${ticker.low}")
    print(f"  Close: ${ticker.close}")
    
    # ----------------------------------------
    step(9, "Watch Price Updates")
    
    print("Watching price updates for 5 seconds...")
    print("(Prices update as new data arrives)")
    
    last_price = ticker.last
    for i in range(5):
        ib.sleep(1)
        if ticker.last != last_price:
            print(f"  Price changed: ${last_price} → ${ticker.last}")
            last_price = ticker.last
        else:
            print(f"  {i+1}s: ${ticker.last} (no change)")
    
    # ----------------------------------------
    step(10, "Cancel Market Data")
    
    print("Canceling market data subscription...")
    ib.cancelMktData(aapl)
    print("✅ Market data canceled")
    
    # ----------------------------------------
    step(11, "Get Account Summary")
    
    print("Fetching account summary...")
    summary = await ib.accountSummaryAsync()
    
    print("\nKey account values:")
    for item in summary[:10]:  # Show first 10 items
        if item.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower']:
            print(f"  {item.tag}: ${float(item.value):,.2f}")
    
    # ----------------------------------------
    step(12, "Get Positions")
    
    print("Fetching current positions...")
    positions = ib.positions()
    
    if positions:
        print(f"\nYou have {len(positions)} positions:")
        for pos in positions[:5]:  # Show first 5
            print(f"  {pos.contract.symbol}: {pos.position} shares @ ${pos.avgCost:.2f}")
    else:
        print("No open positions")
    
    # ----------------------------------------
    step(13, "Get Historical Data")
    
    print("Fetching historical data for SPY...")
    print("Duration: 1 day, Bar size: 1 hour")
    
    spy = Stock('SPY', 'SMART', 'USD')
    ib.qualifyContracts(spy)
    
    bars = await ib.reqHistoricalDataAsync(
        spy,
        endDateTime='',
        durationStr='1 D',
        barSizeSetting='1 hour',
        whatToShow='TRADES',
        useRTH=True
    )
    
    print(f"\n✅ Got {len(bars)} bars")
    print("\nLast 3 bars:")
    for bar in bars[-3:]:
        print(f"  {bar.date}: Open=${bar.open:.2f}, Close=${bar.close:.2f}, Volume={bar.volume:,}")
    
    # ----------------------------------------
    step(14, "Monitor Multiple Symbols")
    
    print("Let's watch 3 symbols at once...")
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    tickers = {}
    
    # Subscribe to all
    for symbol in symbols:
        contract = Stock(symbol, 'SMART', 'USD')
        ib.qualifyContracts(contract)
        ticker = ib.reqMktData(contract, '', False, False)
        tickers[symbol] = ticker
        print(f"  Subscribed to {symbol}")
    
    # Wait for data
    print("\nWaiting for data...")
    ib.sleep(2)
    
    # Display
    print("\nCurrent prices:")
    for symbol, ticker in tickers.items():
        print(f"  {symbol}: ${ticker.last}")
    
    # Cleanup
    for ticker in tickers.values():
        ib.cancelMktData(ticker.contract)
    
    # ----------------------------------------
    step(15, "Disconnect")
    
    print("Disconnecting from IB Gateway...")
    ib.disconnect()
    print("✅ Disconnected")
    
    print("\n" + "="*60)
    print("TUTORIAL COMPLETE!")
    print("="*60)
    print("""
    You've learned how to:
    - Connect to IB Gateway
    - Create and qualify contracts
    - Get real-time market data
    - Access account information
    - Fetch historical data
    - Monitor multiple symbols
    
    Next steps:
    - Try the explore.py script for interactive exploration
    - Modify this script to test different symbols
    - Explore the bot components with explore_components.py
    """)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())