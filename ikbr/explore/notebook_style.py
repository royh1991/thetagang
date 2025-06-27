#!/usr/bin/env python3
"""
IBKR Notebook-Style Commands
============================

Copy and paste these commands one by one into a Python REPL.
Start with: python

Then paste each section.
"""

# ============================================
# SECTION 1: Import and Setup
# ============================================

from ib_async import *
import os
from dotenv import load_dotenv
load_dotenv()

# Create IB connection object
ib = IB()

# ============================================
# SECTION 2: Connect
# ============================================

# Connect to paper trading
ib.connect('localhost', 4002, clientId=1)

# Check if connected
print(f"Connected: {ib.isConnected()}")

# Request delayed data (free)
ib.reqMarketDataType(3)

# ============================================
# SECTION 3: Get a Stock Price
# ============================================

# Create Apple stock contract
aapl = Stock('AAPL', 'SMART', 'USD')

# Qualify it (get full details)
ib.qualifyContracts(aapl)
print(f"AAPL conId: {aapl.conId}")

# Request market data
ticker = ib.reqMktData(aapl, '', False, False)

# Wait for data
ib.sleep(2)

# Print the price
print(f"AAPL price: ${ticker.last}")
print(f"Bid: ${ticker.bid}, Ask: ${ticker.ask}")

# Cancel market data
ib.cancelMktData(aapl)

# ============================================
# SECTION 4: Get Historical Data
# ============================================

# Get 1 day of 5-minute bars for SPY
spy = Stock('SPY', 'SMART', 'USD')
ib.qualifyContracts(spy)

bars = ib.reqHistoricalData(
    spy,
    endDateTime='',
    durationStr='1 D',
    barSizeSetting='5 mins',
    whatToShow='TRADES',
    useRTH=True
)

# Show last 5 bars
print(f"Got {len(bars)} bars")
for bar in bars[-5:]:
    print(f"{bar.date}: ${bar.close}")

# ============================================
# SECTION 5: Get Account Info
# ============================================

# Get account summary
summary = ib.accountSummary()

# Show key values
for item in summary:
    if item.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower']:
        print(f"{item.tag}: ${float(item.value):,.2f}")

# ============================================
# SECTION 6: Watch Multiple Stocks
# ============================================

# Subscribe to multiple symbols
symbols = ['AAPL', 'MSFT', 'GOOGL']
tickers = {}

for symbol in symbols:
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    ticker = ib.reqMktData(contract, '', False, False)
    tickers[symbol] = ticker

# Wait and then display
ib.sleep(2)

for symbol, ticker in tickers.items():
    print(f"{symbol}: ${ticker.last}")

# Cleanup
for ticker in tickers.values():
    ib.cancelMktData(ticker.contract)

# ============================================
# SECTION 7: Using the Event Bus
# ============================================

from core.event_bus import get_event_bus, Event, EventTypes

# Get event bus
event_bus = get_event_bus()

# Start it (run in another terminal: python -m asyncio)
import asyncio
asyncio.run(event_bus.start())

# Define a handler
def my_handler(event):
    print(f"Got event: {event.event_type}")

# Subscribe
event_bus.subscribe(EventTypes.TICK, my_handler)

# Emit an event
asyncio.run(event_bus.emit(Event(EventTypes.TICK, {'test': 'data'})))

# ============================================
# SECTION 8: Disconnect
# ============================================

# Always disconnect when done
ib.disconnect()
print("Disconnected")

# ============================================
# USEFUL ONE-LINERS
# ============================================

# Quick price check (assuming connected)
ib.reqMktData(Stock('SPY', 'SMART', 'USD'), '', False, False); ib.sleep(2); print(f"SPY: ${_.last}"); ib.cancelMktData(_.contract)

# Get all positions
[print(f"{p.contract.symbol}: {p.position} @ ${p.avgCost:.2f}") for p in ib.positions()]

# Check connection
print(f"Connected: {ib.isConnected()}, Client ID: {ib.client.clientId if ib.isConnected() else 'N/A'}")

# Get option chain
chains = ib.reqSecDefOptParams('SPY', '', 'STK', 106688); print(f"SPY expirations: {chains[0].expirations[:5]}")

# ============================================
# TIPS FOR INTERACTIVE USE
# ============================================

"""
1. Use IPython for better experience:
   pip install ipython
   ipython
   
2. In IPython, you can use ? for help:
   ib.reqMktData?
   Stock?
   
3. Use tab completion:
   ib.<TAB>
   ticker.<TAB>
   
4. Store results in variables:
   aapl_ticker = ib.reqMktData(Stock('AAPL', 'SMART', 'USD'), '', False, False)
   
5. ib_async uses ib.sleep() not time.sleep():
   ib.sleep(2)  # Correct
   time.sleep(2)  # Wrong - will break event loop
   
6. Always qualify contracts before using them:
   contract = Stock('TSLA', 'SMART', 'USD')
   ib.qualifyContracts(contract)  # Do this!
   
7. Check for NaN values:
   if ticker.last and ticker.last == ticker.last:  # NaN != NaN
       print(ticker.last)
"""