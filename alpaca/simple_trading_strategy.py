import os
import time
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

# Load environment variables
load_dotenv()

API_KEY = os.environ.get('ALPACA_API_KEY')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')

# Initialize clients (paper trading)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


class SimpleMeanReversionStrategy:
    """
    A simple mean reversion strategy that:
    1. Tracks 20-period moving average
    2. Buys when price drops 2% below MA (oversold)
    3. Sells when price rises 1% above MA (overbought)
    4. Uses position sizing and stop losses
    """
    
    def __init__(self, symbol, lookback_periods=20, buy_threshold=-0.02, sell_threshold=0.01):
        self.symbol = symbol
        self.lookback_periods = lookback_periods
        self.buy_threshold = buy_threshold  # -2% below MA
        self.sell_threshold = sell_threshold  # +1% above MA
        self.position_size = 100  # shares per trade
        self.max_positions = 1  # max positions per symbol
        self.stop_loss_pct = 0.03  # 3% stop loss
        
        # Price history
        self.price_history = deque(maxlen=lookback_periods)
        self.last_trade_time = None
        self.min_trade_interval = 300  # 5 minutes between trades
        
    def get_current_position(self):
        """Check if we have an open position"""
        try:
            positions = trading_client.get_all_positions()
            for pos in positions:
                if pos.symbol == self.symbol:
                    return pos
            return None
        except Exception as e:
            print(f"Error getting position: {e}")
            return None
    
    def calculate_moving_average(self):
        """Calculate simple moving average"""
        if len(self.price_history) < self.lookback_periods:
            return None
        return np.mean(self.price_history)
    
    def get_historical_data(self):
        """Initialize price history with recent data"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=self.symbol,
                timeframe=TimeFrame.Minute,
                start=datetime.now() - timedelta(minutes=self.lookback_periods + 10)
            )
            bars = data_client.get_stock_bars(request)
            
            if self.symbol in bars.data:
                for bar in bars.data[self.symbol][-self.lookback_periods:]:
                    self.price_history.append(bar.close)
                print(f"Loaded {len(self.price_history)} historical prices")
        except Exception as e:
            print(f"Error loading historical data: {e}")
    
    def should_buy(self, current_price, ma):
        """Check if we should buy"""
        if not ma or len(self.price_history) < self.lookback_periods:
            return False
        
        # Check if we already have a position
        position = self.get_current_position()
        if position:
            return False
        
        # Check if enough time has passed since last trade
        if self.last_trade_time:
            time_since_trade = time.time() - self.last_trade_time
            if time_since_trade < self.min_trade_interval:
                return False
        
        # Calculate deviation from MA
        deviation = (current_price - ma) / ma
        
        # Buy if price is 2% below MA
        return deviation <= self.buy_threshold
    
    def should_sell(self, current_price, ma, position):
        """Check if we should sell"""
        if not position or not ma:
            return False
        
        entry_price = float(position.avg_entry_price)
        
        # Check stop loss
        loss_pct = (current_price - entry_price) / entry_price
        if loss_pct <= -self.stop_loss_pct:
            print(f"Stop loss triggered: {loss_pct:.2%}")
            return True
        
        # Check if price is above MA threshold
        deviation = (current_price - ma) / ma
        return deviation >= self.sell_threshold
    
    def place_buy_order(self, current_price):
        """Place a buy order"""
        try:
            # Use limit order slightly above current ask
            limit_price = round(current_price * 1.001, 2)
            
            order_data = LimitOrderRequest(
                symbol=self.symbol,
                qty=self.position_size,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price
            )
            
            order = trading_client.submit_order(order_data)
            self.last_trade_time = time.time()
            
            print(f"🟢 BUY ORDER: {self.position_size} {self.symbol} @ ${limit_price}")
            print(f"   Order ID: {order.id}")
            print(f"   Reason: Price ${current_price:.2f} is {((current_price/np.mean(self.price_history))-1)*100:.1f}% below MA")
            
            return order
            
        except Exception as e:
            print(f"Error placing buy order: {e}")
            return None
    
    def place_sell_order(self, position, current_price):
        """Place a sell order"""
        try:
            # Use limit order slightly below current bid
            limit_price = round(current_price * 0.999, 2)
            
            order_data = LimitOrderRequest(
                symbol=self.symbol,
                qty=position.qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price
            )
            
            order = trading_client.submit_order(order_data)
            self.last_trade_time = time.time()
            
            # Calculate P&L
            entry = float(position.avg_entry_price)
            pnl = (limit_price - entry) * float(position.qty)
            pnl_pct = ((limit_price - entry) / entry) * 100
            
            print(f"🔴 SELL ORDER: {position.qty} {self.symbol} @ ${limit_price}")
            print(f"   Entry: ${entry:.2f}, Exit: ${limit_price:.2f}")
            print(f"   P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            
            return order
            
        except Exception as e:
            print(f"Error placing sell order: {e}")
            return None
    
    def run_strategy(self):
        """Main strategy loop"""
        print(f"\n🚀 Starting Mean Reversion Strategy for {self.symbol}")
        print(f"   MA Period: {self.lookback_periods}")
        print(f"   Buy Threshold: {self.buy_threshold*100:.1f}% below MA")
        print(f"   Sell Threshold: {self.sell_threshold*100:.1f}% above MA")
        print(f"   Stop Loss: {self.stop_loss_pct*100:.1f}%")
        
        # Load historical data
        self.get_historical_data()
        
        # Main loop
        while True:
            try:
                # Get current quote
                quote_request = StockLatestQuoteRequest(symbol_or_symbols=self.symbol)
                quote = data_client.get_stock_latest_quote(quote_request)[self.symbol]
                
                # Use mid price
                current_price = (quote.bid_price + quote.ask_price) / 2
                self.price_history.append(current_price)
                
                # Calculate MA
                ma = self.calculate_moving_average()
                
                if ma:
                    deviation = ((current_price - ma) / ma) * 100
                    
                    # Display current state
                    print(f"\r{datetime.now().strftime('%H:%M:%S')} | "
                          f"{self.symbol}: ${current_price:.2f} | "
                          f"MA: ${ma:.2f} | "
                          f"Deviation: {deviation:+.1f}% | "
                          f"Bid/Ask: ${quote.bid_price:.2f}/${quote.ask_price:.2f}", 
                          end='', flush=True)
                    
                    # Check for trades
                    position = self.get_current_position()
                    
                    if self.should_buy(current_price, ma):
                        print()  # New line for order message
                        self.place_buy_order(current_price)
                    
                    elif position and self.should_sell(current_price, ma, position):
                        print()  # New line for order message
                        self.place_sell_order(position, current_price)
                
                # Wait before next check
                time.sleep(5)  # Check every 5 seconds
                
            except KeyboardInterrupt:
                print("\n\n⏹️  Strategy stopped by user")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                time.sleep(10)


def main():
    """Run the strategy"""
    # Check account
    account = trading_client.get_account()
    print(f"💼 Account: {account.account_number}")
    print(f"💵 Buying Power: ${account.buying_power}")
    print(f"📊 Portfolio Value: ${account.portfolio_value}")
    
    # Choose symbol
    symbol = "SPY"  # Very liquid, good for mean reversion
    
    # Create and run strategy
    strategy = SimpleMeanReversionStrategy(
        symbol=symbol,
        lookback_periods=20,
        buy_threshold=-0.002,  # Buy when 0.2% below MA
        sell_threshold=0.001   # Sell when 0.1% above MA
    )
    
    strategy.run_strategy()


if __name__ == "__main__":
    main()