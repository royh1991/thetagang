#!/usr/bin/env python3
"""
Simplest possible test of IBKR connection
"""

from ib_async import IB
import os
from dotenv import load_dotenv

load_dotenv()

# Create IB instance
ib = IB()

# Determine port
port = 4002 if os.getenv('TRADING_MODE', 'paper') == 'paper' else 4001

print(f"Attempting to connect to localhost:{port}")
print("Make sure Docker is running first!")
print("  docker-compose up -d")
print()

try:
    # Try to connect
    ib.connect('localhost', port, clientId=1, timeout=10)
    print("✅ Connected successfully!")
    
    # Test getting some data
    from ib_async import Stock
    
    # Create a stock contract
    aapl = Stock('AAPL', 'SMART', 'USD')
    
    # Qualify it
    ib.qualifyContracts(aapl)
    print(f"✅ Qualified AAPL: conId={aapl.conId}")
    
    # Request market data
    ticker = ib.reqMktData(aapl, '', False, False)
    
    # Wait a bit
    ib.sleep(2)
    
    # Show what we got
    print(f"✅ AAPL last price: ${ticker.last}")
    
    # Cleanup
    ib.cancelMktData(aapl)
    ib.disconnect()
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Is Docker running? Check with: docker ps")
    print("2. Start IB Gateway: docker-compose up -d")
    print("3. Wait 30-60 seconds for it to start")
    print("4. Check logs: docker-compose logs ib-gateway")