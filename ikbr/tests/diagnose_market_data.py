"""
Diagnose Market Data Issues

Comprehensive test to identify why market data isn't working.
"""

import asyncio
from ib_async import *
import os
from dotenv import load_dotenv

load_dotenv()

async def test_market_data():
    """Test market data with various approaches"""
    ib = IB()
    
    try:
        # Connect
        port = int(os.getenv('IB_GATEWAY_PORT', 4002))
        print(f"Connecting to IB Gateway on port {port}...")
        ib.connect('localhost', port, clientId=1)
        await asyncio.sleep(2)
        
        if not ib.isConnected():
            print("❌ Failed to connect!")
            return
        
        print("✅ Connected successfully!")
        
        # Get account info
        account_values = ib.accountValues()
        for av in account_values:
            if av.tag == "AccountType":
                print(f"Account Type: {av.value}")
            elif av.tag == "NetLiquidation":
                print(f"Net Liquidation: ${av.value}")
        
        print("\n" + "="*50)
        print("TESTING MARKET DATA")
        print("="*50)
        
        # Test 1: Basic SPY contract
        print("\n1. Testing basic SPY contract:")
        spy = Stock('SPY', 'SMART', 'USD')
        ib.qualifyContracts(spy)
        print(f"   Qualified: conId={spy.conId}, exchange={spy.exchange}")
        
        # Request market data
        ticker = ib.reqMktData(spy, '', False, False)
        await asyncio.sleep(3)
        
        print(f"   Last: ${ticker.last}")
        print(f"   Bid: ${ticker.bid}")
        print(f"   Ask: ${ticker.ask}")
        print(f"   Volume: {ticker.volume}")
        
        # Test 2: Try with delayed data
        print("\n2. Testing with delayed data request:")
        ib.reqMarketDataType(4)  # Delayed frozen data
        ticker2 = ib.reqMktData(spy, '', False, False)
        await asyncio.sleep(3)
        
        print(f"   Delayed Last: ${ticker2.last}")
        print(f"   Delayed Bid: ${ticker2.bid}")
        print(f"   Delayed Ask: ${ticker2.ask}")
        
        # Test 3: Request specific ticks
        print("\n3. Testing with specific tick types:")
        ticker3 = ib.reqMktData(spy, '233', False, False)  # RTVolume
        await asyncio.sleep(3)
        
        print(f"   With tick types - Last: ${ticker3.last}")
        
        # Test 4: Try snapshot
        print("\n4. Testing snapshot mode:")
        ticker4 = ib.reqMktData(spy, '', True, False)  # Snapshot mode
        await asyncio.sleep(3)
        
        print(f"   Snapshot - Last: ${ticker4.last}")
        
        # Test 5: Check market hours
        print("\n5. Checking market hours:")
        contract_details = ib.reqContractDetails(spy)
        if contract_details:
            hours = contract_details[0].tradingHours
            print(f"   Trading hours: {hours}")
        
        # Test 6: Try different exchange
        print("\n6. Testing with ARCA exchange:")
        spy_arca = Stock('SPY', 'ARCA', 'USD')
        ib.qualifyContracts(spy_arca)
        ticker5 = ib.reqMktData(spy_arca, '', False, False)
        await asyncio.sleep(3)
        
        print(f"   ARCA - Last: ${ticker5.last}")
        
        # Test 7: Check for errors
        print("\n7. Checking for errors:")
        def onError(reqId, errorCode, errorString, contract):
            print(f"   Error {errorCode}: {errorString}")
            if contract:
                print(f"   Contract: {contract.symbol}")
        
        ib.errorEvent += onError
        
        # Test historical data
        print("\n8. Testing historical data:")
        bars = ib.reqHistoricalData(
            spy,
            endDateTime='',
            durationStr='1 D',
            barSizeSetting='1 hour',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        if bars:
            print(f"   Got {len(bars)} historical bars")
            print(f"   Latest bar: {bars[-1].date} Close=${bars[-1].close}")
        else:
            print("   No historical data received")
        
        # Summary
        print("\n" + "="*50)
        print("DIAGNOSIS SUMMARY")
        print("="*50)
        
        if not util.isNan(ticker.last) or not util.isNan(ticker2.last):
            print("✅ Market data is working!")
            print("   - Check if you're using the correct data during market hours")
        else:
            print("❌ Market data not working. Possible issues:")
            print("   1. No market data subscription for this account")
            print("   2. Paper account may need delayed data enabled in TWS")
            print("   3. Try logging into TWS/Gateway directly to check subscriptions")
            print("   4. Ensure API permissions include market data")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("\nDisconnected from IB")


if __name__ == "__main__":
    util.run(test_market_data())