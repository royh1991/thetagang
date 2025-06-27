"""
Simple Market Data Test

Tests market data without async complications.
"""

from ib_async import *
import os
import time
from dotenv import load_dotenv

load_dotenv()

def test_market_data():
    """Test market data synchronously"""
    ib = IB()
    
    try:
        # Connect
        port = int(os.getenv('IB_GATEWAY_PORT', 4002))
        print(f"Connecting to IB Gateway on port {port}...")
        ib.connect('localhost', port, clientId=1)
        
        print("✅ Connected successfully!")
        
        # Test SPY
        print("\nTesting SPY market data:")
        spy = Stock('SPY', 'SMART', 'USD')
        ib.qualifyContracts(spy)
        print(f"Qualified SPY: conId={spy.conId}")
        
        # Try regular market data
        print("\n1. Regular market data:")
        ticker = ib.reqMktData(spy, '', False, False)
        ib.sleep(3)  # Use ib.sleep instead of time.sleep
        
        print(f"   Last: ${ticker.last}")
        print(f"   Bid: ${ticker.bid}")
        print(f"   Ask: ${ticker.ask}")
        print(f"   High: ${ticker.high}")
        print(f"   Low: ${ticker.low}")
        print(f"   Volume: {ticker.volume}")
        
        # Try delayed data
        print("\n2. Delayed data (type 3):")
        ib.reqMarketDataType(3)  # Delayed data
        ticker2 = ib.reqMktData(spy, '', False, False)
        ib.sleep(3)
        
        print(f"   Delayed Last: ${ticker2.last}")
        print(f"   Delayed Bid: ${ticker2.bid}")
        print(f"   Delayed Ask: ${ticker2.ask}")
        
        # Try historical data
        print("\n3. Historical data:")
        bars = ib.reqHistoricalData(
            spy,
            endDateTime='',
            durationStr='1 D',
            barSizeSetting='5 mins',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        if bars:
            print(f"   Got {len(bars)} historical bars")
            print(f"   Latest: {bars[-1].date} Close=${bars[-1].close}")
        
        # Check account
        print("\n4. Account check:")
        positions = ib.positions()
        account_values = ib.accountValues()
        
        for av in account_values:
            if av.tag in ["AccountType", "NetLiquidation"]:
                print(f"   {av.tag}: {av.value}")
        
        # Summary
        print("\n" + "="*50)
        if not util.isNan(ticker.last):
            print("✅ Real-time market data working!")
        elif not util.isNan(ticker2.last):
            print("⚠️  Only delayed data working")
            print("   - This is normal for paper accounts")
            print("   - Use reqMarketDataType(3) for delayed data")
        else:
            print("❌ No market data received")
            print("   Possible issues:")
            print("   1. Market data subscriptions not enabled")
            print("   2. Outside market hours")
            print("   3. Need to enable delayed data in TWS/Gateway")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("\nDisconnected")


if __name__ == "__main__":
    # Run synchronously
    test_market_data()