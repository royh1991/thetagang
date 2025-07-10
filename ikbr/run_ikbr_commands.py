#!/usr/bin/env python3
"""
Interactive IB API playground - paste these commands line by line in a Python shell
Run: python -i run_ikbr_commands.py
Or just copy/paste sections into ipython/jupyter
"""

# ============================================
# IMPORTS - Run these first
# ============================================
import asyncio
from datetime import datetime, timedelta
import pandas as pd
from ib_async import IB, Stock, Option, Contract, Order, MarketOrder, LimitOrder, StopOrder, util
import numpy as np
from loguru import logger

# Setup logging
logger.remove()
logger.add(lambda msg: print(msg), level="INFO")

# ============================================
# CONNECT TO IB GATEWAY
# ============================================

# Create IB instance
ib = IB()

# Connect to IB Gateway (paper trading port)
ib.connect('localhost', 4102, clientId=999)

# Check connection
print(f"Connected: {ib.isConnected()}")
print(f"Server version: {ib.serverVersion()}")

# ============================================
# BASIC ACCOUNT INFO
# ============================================

# Get account summary
account_summary = ib.accountSummary()
for item in account_summary[:5]:  # Show first 5 items
    print(f"{item.tag}: {item.value}")

# Get account values
account_values = ib.accountValues()
for val in account_values[:10]:  # Show first 10
    print(f"{val.tag}: {val.value} {val.currency}")

# Get positions
positions = ib.positions()
print(f"\nPositions ({len(positions)} total):")
for pos in positions:
    print(f"  {pos.contract.symbol}: {pos.position} shares @ avg ${pos.avgCost:.2f}")

# ============================================
# CREATE CONTRACTS
# ============================================

# Stock contract
aapl = Stock('AAPL', 'SMART', 'USD')
ib.qualifyContracts(aapl)
print(f"AAPL contract: {aapl}")

# Multiple stocks
symbols = ['TSLA', 'NVDA', 'SPY', 'QQQ']
stocks = [Stock(symbol, 'SMART', 'USD') for symbol in symbols]
ib.qualifyContracts(*stocks)

# Option contract
spy_call = Option('SPY', '20250717', 450, 'C', 'SMART')
ib.qualifyContracts(spy_call)

# ============================================
# GET CURRENT MARKET DATA
# ============================================

# Get snapshot data (no subscription needed)
ticker = ib.reqMktData(aapl, '', False, False)
ib.sleep(2)  # Wait for data
print(f"\nAAPL snapshot:")
print(f"  Last: ${ticker.last}")
print(f"  Bid: ${ticker.bid}")
print(f"  Ask: ${ticker.ask}")
print(f"  Volume: {ticker.volume:,}")

# Cancel market data
ib.cancelMktData(aapl)

# Get multiple snapshots
tickers = []
for stock in stocks[:3]:  # First 3 stocks
    ticker = ib.reqMktData(stock, '', False, False)
    tickers.append(ticker)

ib.sleep(2)
print("\nMultiple snapshots:")
for ticker in tickers:
    if ticker.last:
        print(f"  {ticker.contract.symbol}: ${ticker.last} (vol: {ticker.volume:,})")

# Clean up
for ticker in tickers:
    ib.cancelMktData(ticker.contract)

# ============================================
# LIVE STREAMING DATA
# ============================================

# Stream live data for one symbol
ticker = ib.reqMktData(aapl)

# Print updates for 10 seconds
def onPendingTickers(tickers):
    for t in tickers:
        print(f"{datetime.now().strftime('%H:%M:%S')} - {t.contract.symbol}: ${t.last} "
              f"(bid: ${t.bid}, ask: ${t.ask})")

ib.pendingTickersEvent += onPendingTickers
ib.sleep(10)  # Stream for 10 seconds
ib.pendingTickersEvent -= onPendingTickers

# Cancel streaming
ib.cancelMktData(aapl)

# ============================================
# HISTORICAL DATA
# ============================================

# Get 1 day of 5-min bars
bars = ib.reqHistoricalData(
    aapl,
    endDateTime='',  # Use current time
    durationStr='1 D',
    barSizeSetting='5 mins',
    whatToShow='TRADES',
    useRTH=True,  # Regular trading hours only
    formatDate=1
)

# Convert to DataFrame
df = util.df(bars)
print(f"\nHistorical data ({len(df)} bars):")
print(df.head())
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Get different timeframes
timeframes = [
    ('1 D', '1 min'),    # 1 day of 1-minute bars
    ('1 W', '5 mins'),   # 1 week of 5-minute bars  
    ('1 M', '1 hour'),   # 1 month of hourly bars
    ('1 Y', '1 day')     # 1 year of daily bars
]

for duration, bar_size in timeframes:
    bars = ib.reqHistoricalData(
        aapl, '', duration, bar_size, 'TRADES', True, 1
    )
    print(f"\n{duration} of {bar_size} bars: {len(bars)} bars")
    if bars:
        print(f"  First: {bars[0].date}, Last: {bars[-1].date}")

# ============================================
# HISTORICAL DATA WITH SPECIFIC DATES
# ============================================

# Get data for specific date range
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

bars = ib.reqHistoricalData(
    aapl,
    endDateTime=end_date.strftime('%Y%m%d %H:%M:%S'),
    durationStr='30 D',
    barSizeSetting='1 hour',
    whatToShow='TRADES',
    useRTH=True,
    formatDate=1
)

df = util.df(bars)
print(f"\n30 days of hourly data: {len(df)} bars")
print(f"Average volume: {df['volume'].mean():,.0f}")
print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")

# ============================================
# TECHNICAL INDICATORS FROM HISTORICAL DATA
# ============================================

# Calculate simple indicators
df['sma_20'] = df['close'].rolling(20).mean()
df['rsi'] = calculate_rsi(df['close'])  # Need to define calculate_rsi
df['returns'] = df['close'].pct_change()

# Basic momentum signal
df['signal'] = ((df['close'] > df['sma_20']) & 
                (df['returns'] > 0.001)).astype(int)

print(f"\nSignals generated: {df['signal'].sum()}")

# ============================================
# ORDER PLACEMENT (PAPER TRADING)
# ============================================

# Get current price first
ticker = ib.reqMktData(aapl, '', False, False)
ib.sleep(2)
current_price = ticker.last
print(f"\nCurrent AAPL price: ${current_price}")

# Create market order
market_order = MarketOrder('BUY', 100)
market_order.transmit = False  # Don't transmit yet

# Create limit order
limit_order = LimitOrder('BUY', 100, current_price - 1)

# Create stop loss order
stop_order = StopOrder('SELL', 100, current_price - 5)

# Create bracket order (entry + profit target + stop loss)
parent = MarketOrder('BUY', 100)
parent.orderId = ib.client.getReqId()
parent.transmit = False

profit_target = LimitOrder('SELL', 100, current_price + 10)
profit_target.parentId = parent.orderId
profit_target.transmit = False

stop_loss = StopOrder('SELL', 100, current_price - 5)
stop_loss.parentId = parent.orderId
stop_loss.transmit = True  # This transmits all orders

# Place orders (uncomment to actually place)
# trade = ib.placeOrder(aapl, market_order)
# print(f"Order placed: {trade.order.orderId}")

# ============================================
# ORDER MANAGEMENT
# ============================================

# Get open orders
open_orders = ib.openOrders()
print(f"\nOpen orders: {len(open_orders)}")
for order in open_orders:
    print(f"  Order {order.orderId}: {order.action} {order.totalQuantity} "
          f"{order.orderType} @ {order.lmtPrice if hasattr(order, 'lmtPrice') else 'MKT'}")

# Get trades (includes executed trades)
trades = ib.trades()
print(f"\nActive trades: {len(trades)}")
for trade in trades:
    print(f"  {trade.contract.symbol}: {trade.order.action} "
          f"{trade.orderStatus.filled}/{trade.order.totalQuantity} "
          f"@ {trade.orderStatus.avgFillPrice or 'pending'}")

# Cancel all orders
# ib.reqGlobalCancel()

# ============================================
# OPTIONS CHAIN
# ============================================

# Get option chain for SPY
spy = Stock('SPY', 'SMART', 'USD')
ib.qualifyContracts(spy)

chains = ib.reqSecDefOptParams(spy.symbol, '', spy.secType, spy.conId)
chain = next(c for c in chains if c.exchange == 'SMART')

print(f"\nSPY option chain:")
print(f"  Expirations: {chain.expirations[:5]}...")  # First 5
print(f"  Strikes: {chain.strikes[:10]}...")  # First 10

# Get specific option contracts
expiration = chain.expirations[0]  # Next expiration
strikes = sorted([s for s in chain.strikes if abs(s - current_price) < 10])[:5]

calls = [Option('SPY', expiration, strike, 'C', 'SMART') for strike in strikes]
ib.qualifyContracts(*calls)

# Get option prices
for call in calls:
    ticker = ib.reqMktData(call, '', False, False)
    
ib.sleep(2)
print(f"\nSPY calls for {expiration}:")
for i, call in enumerate(calls):
    ticker = ib.reqTickers(call)[0] if ib.reqTickers(call) else None
    if ticker and ticker.last:
        print(f"  Strike ${call.strike}: ${ticker.last} "
              f"(bid: ${ticker.bid}, ask: ${ticker.ask})")

# ============================================
# SCANNER - FIND STOCKS
# ============================================

# Most active stocks
from ib_async import ScannerSubscription

sub = ScannerSubscription(
    instrument='STK',
    locationCode='STK.US.MAJOR',
    scanCode='TOP_PERC_GAIN'  # Top gainers
)

scanData = ib.reqScannerData(sub)
print(f"\nTop gainers:")
for data in scanData[:10]:
    print(f"  {data.contractDetails.contract.symbol}: "
          f"Rank {data.rank}")

# ============================================
# FUNDAMENTAL DATA
# ============================================

# Get fundamental data (requires subscription)
# fundamental_data = ib.reqFundamentalData(aapl, 'ReportSnapshot')
# print(fundamental_data)

# ============================================
# REAL-TIME BARS (5-second bars)
# ============================================

bars = []
def onBarUpdate(bar, hasNewBar):
    if hasNewBar:
        bars.append(bar)
        print(f"{bar.time}: ${bar.close} (vol: {bar.volume})")

# Request real-time bars
req = ib.reqRealTimeBars(aapl, 5, 'TRADES', False)
req.updateEvent += onBarUpdate

# Let it run for 30 seconds
ib.sleep(30)

# Cancel real-time bars
ib.cancelRealTimeBars(req)
req.updateEvent -= onBarUpdate

print(f"Collected {len(bars)} real-time bars")

# ============================================
# DISCONNECT
# ============================================

# Always disconnect when done
ib.disconnect()
print("\nDisconnected from IB Gateway")

# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_market_hours(contract):
    """Get market hours for a contract"""
    details = ib.reqContractDetails(contract)[0]
    print(f"Trading hours: {details.tradingHours}")
    print(f"Time zone: {details.timeZoneId}")

def find_contracts(pattern):
    """Search for contracts by pattern"""
    contracts = ib.reqMatchingSymbols(pattern)
    for c in contracts[:10]:
        print(f"{c.contract.symbol}: {c.contract.primaryExchange} "
              f"({c.contract.currency})")

# ============================================
# ASYNC OPERATIONS (for more complex scenarios)
# ============================================

async def stream_multiple_symbols(symbols, duration=10):
    """Stream multiple symbols concurrently"""
    contracts = [Stock(s, 'SMART', 'USD') for s in symbols]
    ib.qualifyContracts(*contracts)
    
    tickers = []
    for contract in contracts:
        ticker = ib.reqMktData(contract)
        tickers.append(ticker)
    
    async def print_updates():
        for _ in range(duration):
            await asyncio.sleep(1)
            for ticker in tickers:
                if ticker.last:
                    print(f"{ticker.contract.symbol}: ${ticker.last}")
            print("-" * 30)
    
    await print_updates()
    
    # Cleanup
    for ticker in tickers:
        ib.cancelMktData(ticker.contract)

# Run async function
# asyncio.run(stream_multiple_symbols(['AAPL', 'TSLA', 'NVDA'], 5))

# ============================================
# NOTES AND TIPS
# ============================================

"""
TIPS:
1. Always qualify contracts before using them
2. Use ib.sleep() instead of time.sleep() to keep event loop running
3. Always cancel market data subscriptions when done
4. Check ib.isConnected() before operations
5. Handle connection errors gracefully
6. Paper trading uses port 4102, live uses 4101

COMMON ISSUES:
- "No market data permissions": Check your IB account subscriptions
- "Invalid contract": Make sure to qualify contracts first
- "Order rejected": Check account balance and permissions
- Connection drops: IB Gateway may timeout after inactivity

USEFUL IB METHODS:
- ib.whatIfOrder(): Check margin impact before placing order
- ib.reqExecutions(): Get today's executions
- ib.reqAllOpenOrders(): Get all open orders (not just yours)
- ib.reqAccountUpdates(): Subscribe to account updates
- ib.reqPnL(): Get real-time PnL
"""

print("\nIB API Playground loaded. Run sections interactively!")
print("Start with connecting: ib.connect('localhost', 4102, clientId=999)")