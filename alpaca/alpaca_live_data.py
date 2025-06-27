import os
from dotenv import load_dotenv
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockTradesRequest, StockQuotesRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

# Load environment variables from .env file
load_dotenv()

# Set your API credentials
API_KEY = os.environ.get('ALPACA_API_KEY', 'YOUR_API_KEY_HERE')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', 'YOUR_SECRET_KEY_HERE')

# Initialize the trading client
trading_client = TradingClient(API_KEY, SECRET_KEY)

# Initialize the data stream for live market data
data_stream = StockDataStream(API_KEY, SECRET_KEY)


async def handle_trade(data):
    """Handle incoming trade data"""
    print(f"Trade: {data.symbol} @ ${data.price} x {data.size} shares")


async def handle_quote(data):
    """Handle incoming quote data"""
    print(f"Quote: {data.symbol} - Bid: ${data.bid_price} x {data.bid_size}, Ask: ${data.ask_price} x {data.ask_size}")


def get_account_info():
    """Get basic account information"""
    try:
        account = trading_client.get_account()
        print(f"Account Status: {account.status}")
        print(f"Buying Power: ${account.buying_power}")
        print(f"Portfolio Value: ${account.portfolio_value}")
        return account
    except Exception as e:
        print(f"Error getting account info: {e}")
        return None


def list_positions():
    """List current positions"""
    try:
        positions = trading_client.get_all_positions()
        if positions:
            print("\nCurrent Positions:")
            for position in positions:
                print(f"  {position.symbol}: {position.qty} shares @ avg ${position.avg_entry_price}")
        else:
            print("\nNo open positions")
    except Exception as e:
        print(f"Error listing positions: {e}")


async def stream_market_data(symbols):
    """Stream live market data for specified symbols"""
    # Subscribe to trades and quotes
    for symbol in symbols:
        data_stream.subscribe_trades(handle_trade, symbol)
        data_stream.subscribe_quotes(handle_quote, symbol)
    
    print(f"Streaming live data for: {', '.join(symbols)}")
    print("Press Ctrl+C to stop...\n")
    
    # Start streaming
    await data_stream.run()


def main():
    """Main function to run the data stream"""
    # First, check if we can connect and get account info
    print("Checking Alpaca connection...")
    account = get_account_info()
    
    if account:
        print("\nConnection successful!")
        list_positions()
        
        # Define symbols to stream
        symbols = ["SPY", "AAPL", "TSLA"]  # You can change these
        
        # Run the async stream
        print(f"\nStarting live data stream...")
        try:
            # Run the data stream synchronously
            for symbol in symbols:
                data_stream.subscribe_trades(handle_trade, symbol)
                data_stream.subscribe_quotes(handle_quote, symbol)
            
            print(f"Streaming live data for: {', '.join(symbols)}")
            print("Press Ctrl+C to stop...\n")
            
            data_stream.run()
        except KeyboardInterrupt:
            print("\nStream stopped by user")
        finally:
            print("Closing connection...")
            data_stream.stop()
    else:
        print("\nFailed to connect. Please check your API credentials.")
        print("Set them as environment variables:")
        print("  export ALPACA_API_KEY='your_key_here'")
        print("  export ALPACA_SECRET_KEY='your_secret_here'")


if __name__ == "__main__":
    main()