import os
import time
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest
from alpaca.trading.client import TradingClient

# Load environment variables from .env file
load_dotenv()

# Set your API credentials
API_KEY = os.environ.get('ALPACA_API_KEY')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')

# Initialize clients
trading_client = TradingClient(API_KEY, SECRET_KEY)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def get_latest_quotes(symbols):
    """Get latest quotes for symbols"""
    request_params = StockLatestQuoteRequest(symbol_or_symbols=symbols)
    quotes = data_client.get_stock_latest_quote(request_params)
    
    print("\n📊 Latest Quotes:")
    for symbol, quote in quotes.items():
        print(f"{symbol}: Bid ${quote.bid_price:.2f} x {quote.bid_size} | Ask ${quote.ask_price:.2f} x {quote.ask_size}")
        print(f"  Timestamp: {quote.timestamp}")


def get_latest_trades(symbols):
    """Get latest trades for symbols"""
    request_params = StockLatestTradeRequest(symbol_or_symbols=symbols)
    trades = data_client.get_stock_latest_trade(request_params)
    
    print("\n💰 Latest Trades:")
    for symbol, trade in trades.items():
        print(f"{symbol}: ${trade.price:.2f} x {trade.size} shares")
        print(f"  Timestamp: {trade.timestamp}")


def stream_quotes_simple(symbols, interval=1):
    """Simple quote streaming using REST API"""
    print(f"\n🔄 Streaming quotes for {', '.join(symbols)} every {interval} seconds")
    print("Press Ctrl+C to stop...\n")
    
    try:
        while True:
            get_latest_quotes(symbols)
            get_latest_trades(symbols)
            print("-" * 60)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n✅ Streaming stopped")


def main():
    # Check connection
    print("🔌 Checking Alpaca connection...")
    try:
        account = trading_client.get_account()
        print(f"✅ Connected! Account: {account.account_number}")
        print(f"💵 Buying Power: ${account.buying_power}")
        print(f"📈 Portfolio Value: ${account.portfolio_value}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Define symbols
    symbols = ["SPY", "AAPL", "TSLA", "QQQ"]
    
    # Get single quotes first
    get_latest_quotes(symbols)
    get_latest_trades(symbols)
    
    # Ask if user wants to stream
    print("\nWould you like to stream live quotes? (y/n): ", end="")
    if input().lower() == 'y':
        stream_quotes_simple(symbols, interval=2)


if __name__ == "__main__":
    main()