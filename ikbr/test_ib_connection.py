#!/usr/bin/env python3
"""
Simple script to test IB Gateway connection
"""

from ib_async import IB, util
import asyncio
import sys

async def test_connection():
    """Test connection to IB Gateway"""
    ib = IB()
    
    try:
        print("Attempting to connect to IB Gateway on localhost:4102...")
        
        # Try to connect with a timeout
        await ib.connectAsync(host='localhost', port=4102, clientId=999, timeout=10)
        
        print("✓ Successfully connected to IB Gateway!")
        print(f"  Server Version: {ib.serverVersion()}")
        print(f"  Connection Time: {ib.reqCurrentTime()}")
        
        # Get account info if available
        accounts = ib.managedAccounts()
        if accounts:
            print(f"  Connected Accounts: {', '.join(accounts)}")
        
        # Test basic functionality by requesting current time
        server_time = ib.reqCurrentTime()
        print(f"  Server Time: {server_time}")
        
        return True
        
    except asyncio.TimeoutError:
        print("✗ Connection timeout - IB Gateway may not be running or port 4102 is not accessible")
        return False
        
    except ConnectionRefusedError:
        print("✗ Connection refused - IB Gateway is not running on localhost:4102")
        return False
        
    except Exception as e:
        print(f"✗ Connection failed with error: {type(e).__name__}: {e}")
        return False
        
    finally:
        if ib.isConnected():
            print("\nDisconnecting from IB Gateway...")
            ib.disconnect()
            print("Disconnected.")

def main():
    """Main function"""
    print("IB Gateway Connection Test")
    print("=" * 50)
    
    # Run the async test
    util.startLoop()
    success = asyncio.run(test_connection())
    
    if success:
        print("\nConnection test PASSED")
        sys.exit(0)
    else:
        print("\nConnection test FAILED")
        print("\nTroubleshooting tips:")
        print("1. Ensure IB Gateway is running")
        print("2. Check that API connections are enabled in IB Gateway")
        print("3. Verify port 4102 is the correct API port")
        print("4. Check that 'localhost' is in the allowed hosts list")
        sys.exit(1)

if __name__ == "__main__":
    main()