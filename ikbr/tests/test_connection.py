#!/usr/bin/env python3
"""
Test connection to IB Gateway
"""

from ib_async import *
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_connection():
    """Test basic connection and data retrieval"""
    ib = IB()
    
    # Get port based on trading mode
    trading_mode = os.getenv('TRADING_MODE', 'paper')
    if trading_mode == 'paper':
        port = int(os.getenv('IB_GATEWAY_PORT', 4002))
    else:
        port = int(os.getenv('IB_GATEWAY_PORT_LIVE', 4001))
    
    print(f"Connecting to IB Gateway on port {port} ({trading_mode} mode)...")
    
    try:
        # Connect
        ib.connect('localhost', port, clientId=1)
        print("✅ Connected successfully!")
        
        # Get account info
        account = os.getenv('ACCOUNT_ID')
        print(f"\nAccount: {account}")
        
        summary = ib.accountSummary(account)
        for item in summary[:5]:  # Show first 5 items
            print(f"  {item.tag}: {item.value}")
        
        # Get positions
        positions = ib.positions()
        print(f"\nPositions: {len(positions)}")
        for pos in positions:
            print(f"  {pos.contract.symbol}: {pos.position} @ ${pos.avgCost:.2f}")
        
        # Test market data
        print("\nTesting market data for SPY...")
        contract = Stock('SPY', 'SMART', 'USD')
        ib.qualifyContracts(contract)
        
        # Request market data - paper accounts typically use delayed
        ib.reqMarketDataType(3)  # Request delayed data for paper accounts
        ticker = ib.reqMktData(contract, '', False, False)
        ib.sleep(2)  # Wait for data
        
        if not util.isNan(ticker.last):
            print(f"✅ SPY (Delayed): Last=${ticker.last:.2f}, Bid=${ticker.bid:.2f}, Ask=${ticker.ask:.2f}")
            print("   Note: Paper accounts use 15-20 min delayed data")
        else:
            print("❌ No market data available")
            print("   Check IB Gateway logs and ensure market is open")
        
        # Clean up
        ib.cancelMktData(contract)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if ib.isConnected():
            ib.disconnect()
            print("\n✅ Disconnected")

if __name__ == "__main__":
    test_connection()