#!/usr/bin/env python3
"""
IBKR Bot Explorer - Fixed for ib_async
======================================

This version properly handles ib_async's event loop requirements.

Usage:
    python explore_fixed.py

Then use the commands directly (no await needed!)
"""

from ib_async import *
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global IB connection - ib_async manages its own event loop
ib = IB()

# ============================================
# CONNECTION FUNCTIONS
# ============================================

def connect(market_data_type='frozen'):
    """
    Connect to IB Gateway
    
    market_data_type options:
    - 'live' (1): Live market data (requires subscription)
    - 'frozen' (2): Frozen market data (last available price when market closed)
    - 'delayed' (3): Delayed market data (15 min delay, free)
    - 'delayed_frozen' (4): Delayed frozen data
    """
    port = 4102 if os.getenv('TRADING_MODE', 'paper') == 'paper' else 4101
    
    print(f"Connecting to IB Gateway on port {port}...")
    
    # ib_async's connect method is synchronous by default
    ib.connect('localhost', port, clientId=1)
    
    if ib.isConnected():
        print("✅ Connected!")
        
        # Set market data type
        data_types = {
            'live': 1,
            'frozen': 2,
            'delayed': 3,
            'delayed_frozen': 4
        }
        
        data_type_num = data_types.get(market_data_type, 2)  # Default to frozen
        ib.reqMarketDataType(data_type_num)
        print(f"📊 Market data type: {market_data_type} ({data_type_num})")
        
        if market_data_type == 'frozen':
            print("   Using frozen data - shows last price when market closed")
        
        return True
    else:
        print("❌ Connection failed")
        print("Make sure Docker is running: docker ps")
        print("Check if IB Gateway is up: docker-compose logs ib-gateway")
        return False

def disconnect():
    """Disconnect from IB Gateway"""
    if ib.isConnected():
        ib.disconnect()
        print("✅ Disconnected")
    else:
        print("Not connected")

def check_connection():
    """Check connection status"""
    if ib.isConnected():
        print(f"✅ Connected to IB Gateway")
        print(f"   Client ID: {ib.client.clientId}")
        print(f"   Server Version: {ib.client.serverVersion()}")
    else:
        print("❌ Not connected")
    return ib.isConnected()

# ============================================
# MARKET DATA FUNCTIONS
# ============================================

def get_ticker(symbol, wait_time=2):
    """Get market data for a symbol"""
    if not ib.isConnected():
        print("Not connected! Run: connect()")
        return None
    
    # Create contract
    contract = Stock(symbol, 'SMART', 'USD')
    
    # Qualify it (get full details from IB)
    ib.qualifyContracts(contract)
    print(f"Qualified {symbol}: conId={contract.conId}")
    
    # Request market data
    ticker = ib.reqMktData(contract, '', False, False)
    
    # Wait for data using ib.sleep (not time.sleep!)
    print(f"Waiting {wait_time} seconds for data...")
    ib.sleep(wait_time)
    
    # Display data
    print(f"\n{symbol} Market Data:")
    print(f"  Last: ${ticker.last}")
    print(f"  Bid: ${ticker.bid} (size: {ticker.bidSize})")
    print(f"  Ask: ${ticker.ask} (size: {ticker.askSize})")
    print(f"  Volume: {ticker.volume:,}")
    
    # Calculate spread if we have bid/ask
    if ticker.bid and ticker.ask:
        spread = ticker.ask - ticker.bid
        print(f"  Spread: ${spread:.3f}")
    
    # Cancel subscription
    ib.cancelMktData(contract)
    
    return ticker

def watch_prices(symbols, duration=30):
    """Watch multiple symbols update"""
    if not ib.isConnected():
        print("Not connected! Run: connect()")
        return
    
    # Subscribe to all symbols
    tickers = {}
    for symbol in symbols:
        contract = Stock(symbol, 'SMART', 'USD')
        ib.qualifyContracts(contract)
        ticker = ib.reqMktData(contract, '', False, False)
        tickers[symbol] = ticker
        print(f"Subscribed to {symbol}")
    
    print(f"\nWatching {len(symbols)} symbols for {duration} seconds...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        # Watch prices
        for i in range(duration):
            print(f"\n--- Update {i+1}/{duration} at {datetime.now().strftime('%H:%M:%S')} ---")
            
            for symbol, ticker in tickers.items():
                if ticker.last and ticker.last == ticker.last:  # NaN check
                    print(f"{symbol}: ${ticker.last:.2f} "
                          f"(bid: ${ticker.bid:.2f}, ask: ${ticker.ask:.2f})")
                else:
                    print(f"{symbol}: Waiting for data...")
            
            ib.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        # Always cleanup subscriptions
        for symbol, ticker in tickers.items():
            ib.cancelMktData(ticker.contract)
        print("\nUnsubscribed from all symbols")

# ============================================
# ACCOUNT FUNCTIONS
# ============================================

def show_account():
    """Show account summary"""
    if not ib.isConnected():
        print("Not connected! Run: connect()")
        return
    
    print("Fetching account summary...")
    
    # accountSummary() is synchronous in ib_async
    summary = ib.accountSummary()
    
    print("\nAccount Summary:")
    print("-" * 40)
    
    # Group by tag for easier reading
    values = {}
    for item in summary:
        if item.tag in values:
            values[item.tag] += float(item.value)
        else:
            values[item.tag] = float(item.value)
    
    # Display key values
    important = ['NetLiquidation', 'TotalCashValue', 'BuyingPower', 
                 'UnrealizedPnL', 'RealizedPnL']
    
    for tag in important:
        if tag in values:
            print(f"{tag}: ${values[tag]:,.2f}")

def show_positions():
    """Show current positions"""
    if not ib.isConnected():
        print("Not connected! Run: connect()")
        return
    
    positions = ib.positions()
    
    if not positions:
        print("No open positions")
        return
    
    print(f"\nOpen Positions ({len(positions)} total):")
    print("-" * 60)
    
    total_value = 0
    for pos in positions:
        value = pos.position * pos.avgCost
        total_value += abs(value)
        
        print(f"\n{pos.contract.symbol}:")
        print(f"  Shares: {pos.position}")
        print(f"  Avg Cost: ${pos.avgCost:.2f}")
        print(f"  Value: ${value:,.2f}")
    
    print(f"\nTotal Position Value: ${total_value:,.2f}")

# ============================================
# HISTORICAL DATA
# ============================================

def get_today_bars(symbol, bar_size='5 mins'):
    """
    Get today's historical bars - perfect for when market is closed!
    This gets data from market open to current time (or market close).
    """
    if not ib.isConnected():
        print("Not connected! Run: connect()")
        return None
    
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    
    print(f"Getting today's {bar_size} bars for {symbol}...")
    
    # Get today's data
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',  # Empty = up to now
        durationStr='1 D',  # Today only
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=True,  # Regular trading hours only
        formatDate=1
    )
    
    if not bars:
        print("No data received!")
        return None
    
    print(f"Got {len(bars)} bars from today\n")
    
    # Show market open
    print(f"Market Open ({bars[0].date}):")
    print(f"  Open: ${bars[0].open:.2f}")
    print(f"  High: ${bars[0].high:.2f}")
    print(f"  Low: ${bars[0].low:.2f}")
    print(f"  Close: ${bars[0].close:.2f}")
    print(f"  Volume: {bars[0].volume:,}")
    
    # Show current/last bar
    print(f"\nLast Bar ({bars[-1].date}):")
    print(f"  Open: ${bars[-1].open:.2f}")
    print(f"  High: ${bars[-1].high:.2f}")
    print(f"  Low: ${bars[-1].low:.2f}")
    print(f"  Close: ${bars[-1].close:.2f}")
    print(f"  Volume: {bars[-1].volume:,}")
    
    # Daily summary
    daily_high = max(bar.high for bar in bars)
    daily_low = min(bar.low for bar in bars)
    daily_volume = sum(bar.volume for bar in bars)
    
    print(f"\nToday's Summary:")
    print(f"  Open: ${bars[0].open:.2f}")
    print(f"  High: ${daily_high:.2f}")
    print(f"  Low: ${daily_low:.2f}")
    print(f"  Last: ${bars[-1].close:.2f}")
    print(f"  Change: ${bars[-1].close - bars[0].open:.2f} ({(bars[-1].close - bars[0].open) / bars[0].open * 100:+.2f}%)")
    print(f"  Volume: {daily_volume:,}")
    
    return bars

def replay_market_day(symbol, date=None, speed=1):
    """
    Replay a market day bar by bar - see how the day unfolded!
    
    date: YYYYMMDD format (e.g., '20240627'). None = today
    speed: seconds between bars (1 = 1 second per bar)
    """
    if not ib.isConnected():
        print("Not connected! Run: connect()")
        return None
    
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    
    # Set end date/time
    if date:
        end_dt = date + ' 23:59:59'
    else:
        end_dt = ''  # Today
    
    print(f"Replaying {symbol} for {'today' if not date else date}...")
    
    # Get 1-minute bars for the day
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_dt,
        durationStr='1 D',
        barSizeSetting='1 min',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    
    if not bars:
        print("No data!")
        return
    
    print(f"Got {len(bars)} minute bars. Press Ctrl+C to stop.\n")
    
    try:
        # Replay the day
        for i, bar in enumerate(bars):
            # Handle both string and datetime formats
            if isinstance(bar.date, str):
                time_str = bar.date.split()[1] if ' ' in bar.date else bar.date
            else:
                time_str = bar.date.strftime('%H:%M:%S')
            
            # Show every 5th bar to avoid spam
            if i % 5 == 0:
                print(f"{time_str} - ${bar.close:.2f} "
                      f"(H: ${bar.high:.2f} L: ${bar.low:.2f}) "
                      f"Vol: {bar.volume:,}")
            
            # Highlight big moves
            if i > 0:
                change = bar.close - bars[i-1].close
                pct_change = abs(change / bars[i-1].close * 100)
                if pct_change > 0.2:  # 0.2% move
                    print(f"  >>> Big move! {change:+.2f} ({change / bars[i-1].close * 100:+.2f}%)")
            
            ib.sleep(speed)
            
    except KeyboardInterrupt:
        print("\nReplay stopped")
    
    # Summary
    print(f"\nDay Summary:")
    print(f"  Started: ${bars[0].open:.2f}")
    print(f"  Ended: ${bars[-1].close:.2f}")
    print(f"  Day Range: ${min(b.low for b in bars):.2f} - ${max(b.high for b in bars):.2f}")

def get_bars(symbol, duration='1 D', bar_size='5 mins', show_all=False):
    """
    Get historical bars
    
    Duration: '1 D', '1 W', '1 M', '1 Y'
    Bar size: '1 min', '5 mins', '15 mins', '1 hour', '1 day'
    """
    if not ib.isConnected():
        print("Not connected! Run: connect()")
        return None
    
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    
    print(f"Getting {duration} of {bar_size} bars for {symbol}...")
    
    # reqHistoricalData is synchronous in ib_async
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    
    if not bars:
        print("No data received!")
        return None
    
    print(f"Got {len(bars)} bars\n")
    
    if show_all:
        # Show all bars
        for bar in bars:
            print(f"{bar.date}: O={bar.open:.2f} H={bar.high:.2f} "
                  f"L={bar.low:.2f} C={bar.close:.2f} V={bar.volume}")
    else:
        # Show summary
        print("First 3 bars:")
        for bar in bars[:3]:
            print(f"  {bar.date}: ${bar.close:.2f} (vol: {bar.volume:,})")
        
        print("\nLast 3 bars:")
        for bar in bars[-3:]:
            print(f"  {bar.date}: ${bar.close:.2f} (vol: {bar.volume:,})")
        
        # Calculate stats
        closes = [bar.close for bar in bars]
        print(f"\nStats for {symbol}:")
        print(f"  Period High: ${max(closes):.2f}")
        print(f"  Period Low: ${min(closes):.2f}")
        print(f"  Current: ${closes[-1]:.2f}")
        print(f"  Change: ${closes[-1] - closes[0]:.2f} "
              f"({(closes[-1] - closes[0]) / closes[0] * 100:+.2f}%)")
    
    return bars

# ============================================
# OPTIONS
# ============================================

def get_options(symbol, expiry_index=0):
    """Get option chain for a symbol"""
    if not ib.isConnected():
        print("Not connected! Run: connect()")
        return
    
    # Get stock contract
    stock = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(stock)
    
    # Get current stock price
    ticker = ib.reqMktData(stock, '', False, False)
    ib.sleep(2)
    stock_price = ticker.last or ticker.close
    ib.cancelMktData(stock)
    
    print(f"\n{symbol} current price: ${stock_price:.2f}")
    
    # Get option chains
    chains = ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
    
    if not chains:
        print("No option chains found!")
        return
    
    chain = chains[0]
    print(f"\nExchange: {chain.exchange}")
    print(f"Expirations: {chain.expirations[:5]}...")  # Show first 5
    
    # Get specific expiry
    expiry = chain.expirations[expiry_index]
    print(f"\nShowing options for: {expiry}")
    
    # Get strikes near current price (within 5%)
    strikes = [s for s in chain.strikes 
               if stock_price * 0.95 <= s <= stock_price * 1.05]
    strikes = sorted(strikes)[:5]  # Just show 5
    
    print(f"Strikes near current price: {strikes}")
    
    # Get a few option contracts
    print("\nSample option prices:")
    for strike in strikes[:3]:
        # Call option
        call = Option(symbol, expiry, strike, 'C', 'SMART')
        ib.qualifyContracts(call)
        
        call_ticker = ib.reqMktData(call, '', False, False)
        ib.sleep(1)
        
        print(f"\n  ${strike} Call:")
        print(f"    Bid: ${call_ticker.bid}  Ask: ${call_ticker.ask}")
        
        ib.cancelMktData(call)

# ============================================
# HISTORICAL ANALYSIS FUNCTIONS
# ============================================

def analyze_day(symbol, date=None):
    """
    Analyze a specific trading day - great for understanding price action
    
    date: YYYYMMDD format (e.g., '20240627'). None = today
    """
    if not ib.isConnected():
        print("Not connected! Run: connect()")
        return None
    
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    
    # Get 5-minute bars for analysis
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=date + ' 23:59:59' if date else '',
        durationStr='1 D',
        barSizeSetting='5 mins',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    
    if not bars:
        print("No data!")
        return None
    
    print(f"\nAnalyzing {symbol} for {date or 'today'}...")
    print("=" * 50)
    
    # Basic stats
    opens = [b.open for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    closes = [b.close for b in bars]
    volumes = [b.volume for b in bars]
    
    # Price movement
    day_open = bars[0].open
    day_close = bars[-1].close
    day_high = max(highs)
    day_low = min(lows)
    
    # Helper to format time from bar
    def get_time(bar):
        if isinstance(bar.date, str):
            return bar.date.split()[1] if ' ' in bar.date else bar.date
        else:
            return bar.date.strftime('%H:%M:%S')
    
    print(f"\n📊 Price Action:")
    print(f"  Open:  ${day_open:.2f}")
    print(f"  High:  ${day_high:.2f} (reached at {get_time(bars[highs.index(day_high)])})")
    print(f"  Low:   ${day_low:.2f} (reached at {get_time(bars[lows.index(day_low)])})")
    print(f"  Close: ${day_close:.2f}")
    print(f"  Range: ${day_high - day_low:.2f} ({(day_high - day_low) / day_open * 100:.2f}%)")
    
    # Direction
    change = day_close - day_open
    print(f"\n📈 Day Performance:")
    print(f"  Change: ${change:.2f} ({change / day_open * 100:+.2f}%)")
    print(f"  Direction: {'🟢 UP' if change > 0 else '🔴 DOWN' if change < 0 else '⚪ FLAT'}")
    
    # Volatility
    bar_changes = [abs(bars[i].close - bars[i-1].close) for i in range(1, len(bars))]
    avg_move = sum(bar_changes) / len(bar_changes) if bar_changes else 0
    max_move = max(bar_changes) if bar_changes else 0
    
    print(f"\n💫 Volatility:")
    print(f"  Average 5-min move: ${avg_move:.3f} ({avg_move / day_open * 100:.3f}%)")
    print(f"  Largest 5-min move: ${max_move:.2f} ({max_move / day_open * 100:.2f}%)")
    
    # Volume analysis
    total_volume = sum(volumes)
    avg_volume = total_volume / len(volumes)
    
    print(f"\n📊 Volume:")
    print(f"  Total: {total_volume:,}")
    print(f"  Average per bar: {int(avg_volume):,}")
    print(f"  Highest volume bar: {max(volumes):,} at {get_time(bars[volumes.index(max(volumes))])}")
    
    # Trading session analysis
    def get_hour_min(bar):
        if isinstance(bar.date, str):
            return bar.date.split()[1] if ' ' in bar.date else bar.date
        else:
            return bar.date.strftime('%H:%M')
    
    morning_bars = [b for b in bars if '09:30' <= get_hour_min(b) <= '11:30']
    afternoon_bars = [b for b in bars if '14:00' <= get_hour_min(b) <= '16:00']
    
    if morning_bars and afternoon_bars:
        morning_move = morning_bars[-1].close - morning_bars[0].open
        afternoon_move = afternoon_bars[-1].close - afternoon_bars[0].open
        
        print(f"\n⏰ Session Analysis:")
        print(f"  Morning (9:30-11:30): {morning_move:+.2f} ({morning_move / morning_bars[0].open * 100:+.2f}%)")
        print(f"  Afternoon (2:00-4:00): {afternoon_move:+.2f} ({afternoon_move / afternoon_bars[0].open * 100:+.2f}%)")
    
    return bars

def compare_days(symbol, days_back=5):
    """
    Compare recent trading days - see patterns
    """
    if not ib.isConnected():
        print("Not connected! Run: connect()")
        return None
    
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    
    print(f"\nComparing last {days_back} days for {symbol}...")
    
    # Get daily bars
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=f'{days_back} D',
        barSizeSetting='1 day',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    
    if not bars:
        print("No data!")
        return None
    
    print("\nDate       | Open    | Close   | Change  | Range   | Volume")
    print("-" * 70)
    
    for bar in bars:
        change = bar.close - bar.open
        change_pct = change / bar.open * 100
        range_size = bar.high - bar.low
        
        # Emoji for direction
        emoji = '🟢' if change > 0 else '🔴' if change < 0 else '⚪'
        
        # Format date
        if isinstance(bar.date, str):
            date_str = bar.date.split()[0] if ' ' in bar.date else bar.date
        else:
            date_str = bar.date.strftime('%Y-%m-%d')
        
        print(f"{date_str} | "
              f"${bar.open:7.2f} | "
              f"${bar.close:7.2f} | "
              f"{change:+6.2f} {emoji} | "
              f"${range_size:6.2f} | "
              f"{bar.volume:>10,}")
    
    # Summary stats
    changes = [(b.close - b.open) / b.open * 100 for b in bars]
    up_days = sum(1 for c in changes if c > 0)
    down_days = sum(1 for c in changes if c < 0)
    
    print(f"\nSummary:")
    print(f"  Up days: {up_days} ({up_days/len(bars)*100:.0f}%)")
    print(f"  Down days: {down_days} ({down_days/len(bars)*100:.0f}%)")
    print(f"  Average daily move: {sum(changes)/len(changes):+.2f}%")
    print(f"  Biggest gain: {max(changes):+.2f}%")
    print(f"  Biggest loss: {min(changes):+.2f}%")

def get_intraday_stats(symbol):
    """
    Get detailed intraday statistics - useful for strategy development
    """
    if not ib.isConnected():
        print("Not connected! Run: connect()")
        return None
    
    # Get today's 1-minute bars for detailed analysis
    bars = get_bars(symbol, duration='1 D', bar_size='1 min', show_all=False)
    
    if not bars:
        return None
    
    print(f"\n📊 Intraday Statistics for {symbol}")
    print("=" * 50)
    
    # Time-based analysis
    hour_stats = {}
    for bar in bars:
        # Get hour from date
        if isinstance(bar.date, str):
            time_part = bar.date.split()[1] if ' ' in bar.date else bar.date
            hour = time_part.split(':')[0]
        else:
            hour = bar.date.strftime('%H')
            
        if hour not in hour_stats:
            hour_stats[hour] = {'moves': [], 'volumes': []}
        
        if len(hour_stats[hour]['moves']) > 0:
            move = abs(bar.close - hour_stats[hour]['moves'][-1])
            hour_stats[hour]['moves'].append(move)
        else:
            hour_stats[hour]['moves'].append(0)
        
        hour_stats[hour]['volumes'].append(bar.volume)
    
    print("\nHourly Activity:")
    print("Hour | Avg Move | Avg Volume | Activity")
    print("-" * 50)
    
    for hour in sorted(hour_stats.keys()):
        avg_move = sum(hour_stats[hour]['moves']) / len(hour_stats[hour]['moves'])
        avg_vol = sum(hour_stats[hour]['volumes']) / len(hour_stats[hour]['volumes'])
        
        # Activity indicator
        activity = '█' * int(avg_vol / 10000)
        
        print(f" {hour}  | ${avg_move:7.4f} | {int(avg_vol):>10,} | {activity}")
    
    return bars

# ============================================
# SIMPLE TEST FUNCTIONS
# ============================================

def test_connection():
    """Test basic connection"""
    print("Testing IB Gateway connection...\n")
    
    if connect():
        print("\n✅ Connection successful!")
        
        # Try to get some data
        print("\nTesting market data...")
        ticker = get_ticker('SPY', wait_time=3)
        
        if ticker and ticker.last:
            print(f"\n✅ Market data working! SPY = ${ticker.last}")
        else:
            print("\n⚠️  No market data received (market might be closed)")
        
        disconnect()
    else:
        print("\n❌ Connection failed!")
        print("\nTroubleshooting:")
        print("1. Check Docker: docker ps")
        print("2. Check logs: docker-compose logs ib-gateway")
        print("3. Wait 30-60 seconds after starting container")

def demo():
    """Run a quick demo of main features"""
    print("Running IBKR Demo...\n")
    
    # Connect
    if not connect():
        return
    
    print("\n1. Getting single ticker...")
    get_ticker('AAPL')
    
    print("\n2. Watching multiple tickers...")
    watch_prices(['SPY', 'QQQ', 'IWM'], duration=5)
    
    print("\n3. Getting account info...")
    show_account()
    
    print("\n4. Getting historical data...")
    get_bars('SPY', '1 D', '1 hour')
    
    print("\n✅ Demo complete!")
    disconnect()

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("""
IBKR Explorer Ready!
===================

This version is designed to work with ib_async's event loop.
No 'await' needed - all functions are synchronous!

Quick Start:
    connect()                       # Connect to IB Gateway (uses frozen data by default)
    connect('delayed')              # Use delayed data instead
    
🕐 HISTORICAL DATA (Works when market is closed!):
    get_today_bars('AAPL')         # Get today's 5-min bars with summary
    analyze_day('SPY')             # Detailed analysis of today
    analyze_day('SPY', '20240626') # Analyze a specific date
    replay_market_day('TSLA')      # Replay today bar by bar
    compare_days('QQQ', 10)        # Compare last 10 trading days
    get_intraday_stats('AAPL')     # Hourly activity patterns
    
📊 LIVE/FROZEN DATA (May show NaN when market closed):
    get_ticker('AAPL')             # Get current/last price
    watch_prices(['SPY','QQQ'], 20) # Watch prices for 20 seconds
    
📈 MORE HISTORICAL:
    get_bars('SPY')                # Get historical data (configurable)
    get_bars('AAPL', '1 W', '1 hour')  # Week of hourly bars
    get_bars('TSLA', '1 M', '1 day')   # Month of daily bars
    
💼 ACCOUNT INFO:
    show_account()                 # Show account summary
    show_positions()               # Show your positions
    get_options('SPY')             # Get option chain
    
🧪 TEST & UTILITIES:
    test_connection()              # Test if everything works
    demo()                         # Run a quick demo
    check_connection()             # Check connection status
    disconnect()                   # Disconnect when done

Examples for after hours:
    # Analyze today's trading
    connect()  # Uses frozen data by default
    get_today_bars('AAPL')
    analyze_day('SPY')
    
    # Compare recent days
    compare_days('TSLA', 5)
    
    # Replay the market day
    replay_market_day('QQQ', speed=0.5)  # 0.5 sec per bar

Tips:
    - Historical data works great when market is closed!
    - Frozen data shows last traded prices
    - Use dates in YYYYMMDD format (e.g., '20240627')
    - All times are in Eastern Time
    """)
    
    # If running interactively, Python will drop to REPL after this