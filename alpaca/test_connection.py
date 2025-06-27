import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

# Load environment variables from .env file
load_dotenv()

# Simple connection test
API_KEY = os.environ.get('ALPACA_API_KEY', 'YOUR_API_KEY_HERE')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', 'YOUR_SECRET_KEY_HERE')

if API_KEY == 'YOUR_API_KEY_HERE':
    print("Please set your API credentials first!")
    print("Either set environment variables or update the script with your keys")
    exit(1)

try:
    # Initialize client
    client = TradingClient(API_KEY, SECRET_KEY)
    
    # Get account info
    account = client.get_account()
    print("✅ Successfully connected to Alpaca!")
    print(f"Account Number: {account.account_number}")
    print(f"Status: {account.status}")
    print(f"Cash: ${account.cash}")
    print(f"Buying Power: ${account.buying_power}")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("Please check your API credentials")