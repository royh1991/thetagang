import os
from datetime import datetime
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, 
    LimitOrderRequest,
    GetOrdersRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

# Load environment variables
load_dotenv()

API_KEY = os.environ.get('ALPACA_API_KEY')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')

# Initialize clients
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)  # Use paper=True for paper trading
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


def get_current_price(symbol):
    """Get current market price for a symbol"""
    request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
    quotes = data_client.get_stock_latest_quote(request)
    quote = quotes[symbol]
    mid_price = (quote.bid_price + quote.ask_price) / 2
    return mid_price, quote


def place_market_order(symbol, qty, side):
    """Place a market order"""
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY
    )
    
    order = trading_client.submit_order(order_data)
    print(f"✅ Market order placed: {side} {qty} shares of {symbol}")
    print(f"   Order ID: {order.id}")
    print(f"   Status: {order.status}")
    return order


def place_limit_order(symbol, qty, side, limit_price):
    """Place a limit order"""
    order_data = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
        limit_price=limit_price
    )
    
    order = trading_client.submit_order(order_data)
    print(f"✅ Limit order placed: {side} {qty} shares of {symbol} @ ${limit_price}")
    print(f"   Order ID: {order.id}")
    print(f"   Status: {order.status}")
    return order


def get_positions():
    """Get all current positions"""
    positions = trading_client.get_all_positions()
    
    if not positions:
        print("📊 No open positions")
        return []
    
    print("\n📊 Current Positions:")
    for position in positions:
        pnl = float(position.unrealized_pl)
        pnl_pct = float(position.unrealized_plpc) * 100
        print(f"  {position.symbol}: {position.qty} shares")
        print(f"    Avg Entry: ${position.avg_entry_price}")
        print(f"    Current: ${position.current_price}")
        print(f"    P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
    
    return positions


def get_orders(status=None):
    """Get orders with optional status filter"""
    request = GetOrdersRequest(
        status=status,
        limit=10
    )
    orders = trading_client.get_orders(request)
    
    if not orders:
        print(f"📋 No {status or 'recent'} orders")
        return []
    
    print(f"\n📋 {status or 'Recent'} Orders:")
    for order in orders:
        print(f"  {order.symbol}: {order.side} {order.qty} @ {order.order_type}")
        print(f"    Status: {order.status}")
        print(f"    Submitted: {order.submitted_at}")
        if order.filled_at:
            print(f"    Filled: {order.filled_at} @ ${order.filled_avg_price}")
    
    return orders


def cancel_all_orders():
    """Cancel all open orders"""
    try:
        trading_client.cancel_orders()
        print("❌ All open orders cancelled")
    except Exception as e:
        print(f"Error cancelling orders: {e}")


def main():
    print("🔌 Alpaca Trading Example (IBKR Integration)")
    print("=" * 50)
    
    # Check account
    account = trading_client.get_account()
    print(f"💼 Account: {account.account_number}")
    print(f"💵 Buying Power: ${account.buying_power}")
    print(f"📊 Portfolio Value: ${account.portfolio_value}")
    print(f"⚠️  Paper Trading: {account.trading_blocked == False}")
    
    # Example symbol
    symbol = "AAPL"
    
    # Get current price
    print(f"\n📈 Getting quote for {symbol}...")
    price, quote = get_current_price(symbol)
    print(f"   Bid: ${quote.bid_price:.2f} x {quote.bid_size}")
    print(f"   Ask: ${quote.ask_price:.2f} x {quote.ask_size}")
    print(f"   Mid: ${price:.2f}")
    
    # Show current positions
    get_positions()
    
    # Show recent orders
    get_orders()
    
    # Example: Place a limit buy order (uncomment to execute)
    # print(f"\n🛒 Placing limit buy order...")
    # buy_price = round(price * 0.99, 2)  # 1% below current price
    # place_limit_order(symbol, 1, OrderSide.BUY, buy_price)
    
    # Example: Place a market sell order (uncomment to execute)
    # print(f"\n💰 Placing market sell order...")
    # place_market_order(symbol, 1, OrderSide.SELL)
    
    print("\n\n💡 Uncomment the order examples in the code to place actual orders")
    print("⚠️  Make sure you're in paper trading mode first!")


if __name__ == "__main__":
    main()