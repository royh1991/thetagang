import os
import asyncio
from dotenv import load_dotenv
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient

# Load environment variables
load_dotenv()

API_KEY = os.environ.get('ALPACA_API_KEY')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')

# Initialize clients
trading_client = TradingClient(API_KEY, SECRET_KEY)
wss_client = StockDataStream(API_KEY, SECRET_KEY)

# Track statistics
stats = {
    'trades': {},
    'quotes': {}
}


async def quote_handler(data):
    """Handle quote updates"""
    symbol = data.symbol
    if symbol not in stats['quotes']:
        stats['quotes'][symbol] = 0
    stats['quotes'][symbol] += 1
    
    print(f"📊 Quote: {symbol} - Bid: ${data.bid_price:.2f} x {data.bid_size} | Ask: ${data.ask_price:.2f} x {data.ask_size}")


async def trade_handler(data):
    """Handle trade updates"""
    symbol = data.symbol
    if symbol not in stats['trades']:
        stats['trades'][symbol] = 0
    stats['trades'][symbol] += 1
    
    print(f"💰 Trade: {symbol} @ ${data.price:.2f} x {data.size} shares | Conditions: {data.conditions}")


async def news_handler(data):
    """Handle news updates"""
    print(f"📰 News: {data.headline[:80]}...")


async def main():
    """Main async function"""
    # Check connection first
    print("🔌 Checking connection...")
    try:
        account = trading_client.get_account()
        print(f"✅ Connected to Alpaca!")
        print(f"📊 Account: {account.account_number}")
        print(f"💵 Cash: ${account.cash}")
        print(f"📈 Equity: ${account.equity}\n")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return
    
    # Subscribe to data streams
    symbols = ["SPY", "AAPL", "TSLA", "QQQ", "NVDA"]
    
    print(f"🎯 Subscribing to: {', '.join(symbols)}")
    print("Data types: Quotes, Trades")
    print("\nPress Ctrl+C to stop streaming...\n")
    
    # Subscribe to quotes and trades
    wss_client.subscribe_quotes(quote_handler, *symbols)
    wss_client.subscribe_trades(trade_handler, *symbols)
    
    # Optional: Subscribe to news for specific symbols
    # wss_client.subscribe_news(news_handler, *symbols)
    
    try:
        # Start the WebSocket client
        await wss_client._run_forever()
    except KeyboardInterrupt:
        print("\n\n📊 Stream Statistics:")
        print("Quotes received:", {k: v for k, v in stats['quotes'].items()})
        print("Trades received:", {k: v for k, v in stats['trades'].items()})
    finally:
        await wss_client.close()
        print("\n✅ Connection closed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")