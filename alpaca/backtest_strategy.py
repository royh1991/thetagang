import os
import argparse
import numpy as np
from datetime import datetime, timedelta, time as datetime_time
from collections import deque
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Load environment variables
load_dotenv()

API_KEY = os.environ.get('ALPACA_API_KEY')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')

# Initialize data client
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)


class BacktestMeanReversion:
    """Replay historical data to test the strategy"""
    
    def __init__(self, symbol, lookback_periods=20, buy_threshold=-0.002, sell_threshold=0.001):
        self.symbol = symbol
        self.lookback_periods = lookback_periods
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.position_size = 100
        self.stop_loss_pct = 0.03
        
        # Track trades
        self.trades = []
        self.current_position = None
        self.cash = 100000  # Starting cash
        self.portfolio_value = self.cash
        
    def calculate_moving_average(self, prices):
        """Calculate simple moving average"""
        if len(prices) < self.lookback_periods:
            return None
        return np.mean(list(prices)[-self.lookback_periods:])
    
    def run_backtest(self, start_date=None, end_date=None):
        """Run backtest on historical data"""
        
        # Default to today's market hours
        if not start_date:
            today = datetime.now().date()
            # Market hours: 9:30 AM - 4:00 PM ET
            start_date = datetime.combine(today, datetime_time(9, 30))
            end_date = datetime.combine(today, datetime_time(16, 0))
        
        print(f"📊 Backtesting {self.symbol} from {start_date} to {end_date}")
        print(f"💵 Starting Cash: ${self.cash:,.2f}\n")
        
        # Fetch all minute bars for the period
        request = StockBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=TimeFrame.Minute,
            start=start_date - timedelta(minutes=self.lookback_periods),
            end=end_date
        )
        
        bars_response = data_client.get_stock_bars(request)
        
        if self.symbol not in bars_response.data:
            print("No data available for symbol")
            return
        
        bars = bars_response.data[self.symbol]
        print(f"Loaded {len(bars)} bars of data\n")
        
        # Process bars sequentially (simulating real-time)
        price_history = deque(maxlen=self.lookback_periods)
        
        for i, bar in enumerate(bars):
            # Add to price history
            price_history.append(bar.close)
            
            # Skip until we have enough history
            if len(price_history) < self.lookback_periods:
                continue
            
            # Calculate current metrics
            current_price = bar.close
            ma = self.calculate_moving_average(price_history)
            deviation = ((current_price - ma) / ma) * 100
            
            # Print current state every 10 bars
            if i % 10 == 0:
                print(f"{bar.timestamp.strftime('%H:%M')} | "
                      f"Price: ${current_price:.2f} | "
                      f"MA: ${ma:.2f} | "
                      f"Dev: {deviation:+.1f}%", end='')
                if self.current_position:
                    pnl = (current_price - self.current_position['entry']) * self.position_size
                    print(f" | Position P&L: ${pnl:+.2f}", end='')
                print()
            
            # Check for exit signals first
            if self.current_position:
                entry_price = self.current_position['entry']
                
                # Check stop loss
                loss_pct = (current_price - entry_price) / entry_price
                if loss_pct <= -self.stop_loss_pct:
                    self.close_position(bar, "STOP LOSS")
                
                # Check take profit
                elif deviation >= self.sell_threshold * 100:
                    self.close_position(bar, "TAKE PROFIT")
            
            # Check for entry signal
            elif deviation <= self.buy_threshold * 100:
                # Make sure we have enough cash
                cost = current_price * self.position_size
                if cost <= self.cash:
                    self.open_position(bar)
        
        # Close any remaining position at end
        if self.current_position:
            self.close_position(bars[-1], "END OF DAY")
        
        # Print results
        self.print_results(bars[-1].close)
    
    def open_position(self, bar):
        """Open a new position"""
        self.current_position = {
            'entry': bar.close,
            'entry_time': bar.timestamp,
            'size': self.position_size
        }
        
        cost = bar.close * self.position_size
        self.cash -= cost
        
        print(f"\n🟢 BUY: {self.position_size} shares @ ${bar.close:.2f}")
        print(f"   Time: {bar.timestamp.strftime('%H:%M:%S')}")
        print(f"   Cost: ${cost:,.2f}")
        print(f"   Cash remaining: ${self.cash:,.2f}\n")
    
    def close_position(self, bar, reason):
        """Close current position"""
        if not self.current_position:
            return
        
        entry = self.current_position['entry']
        exit_price = bar.close
        pnl = (exit_price - entry) * self.position_size
        pnl_pct = ((exit_price - entry) / entry) * 100
        hold_time = bar.timestamp - self.current_position['entry_time']
        
        # Record trade
        self.trades.append({
            'entry_time': self.current_position['entry_time'],
            'exit_time': bar.timestamp,
            'entry': entry,
            'exit': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason
        })
        
        # Update cash
        self.cash += exit_price * self.position_size
        
        print(f"\n🔴 SELL: {self.position_size} shares @ ${exit_price:.2f}")
        print(f"   Time: {bar.timestamp.strftime('%H:%M:%S')}")
        print(f"   Entry: ${entry:.2f} → Exit: ${exit_price:.2f}")
        print(f"   P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        print(f"   Hold time: {hold_time}")
        print(f"   Reason: {reason}\n")
        
        self.current_position = None
    
    def print_results(self, final_price):
        """Print backtest results"""
        print("\n" + "="*50)
        print("📈 BACKTEST RESULTS")
        print("="*50)
        
        if not self.trades:
            print("No trades executed")
            return
        
        # Calculate statistics
        total_pnl = sum(t['pnl'] for t in self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Portfolio value
        if self.current_position:
            self.portfolio_value = self.cash + (final_price * self.position_size)
        else:
            self.portfolio_value = self.cash
        
        print(f"Total Trades: {len(self.trades)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"\nAverage Win: ${avg_win:+.2f}")
        print(f"Average Loss: ${avg_loss:+.2f}")
        print(f"Total P&L: ${total_pnl:+.2f}")
        print(f"\nStarting Value: $100,000")
        print(f"Ending Value: ${self.portfolio_value:,.2f}")
        print(f"Return: {((self.portfolio_value - 100000) / 100000) * 100:+.2f}%")
        
        # Trade log
        print("\n📋 TRADE LOG:")
        print("-" * 80)
        for i, trade in enumerate(self.trades, 1):
            print(f"{i}. {trade['entry_time'].strftime('%H:%M')} → "
                  f"{trade['exit_time'].strftime('%H:%M')} | "
                  f"${trade['entry']:.2f} → ${trade['exit']:.2f} | "
                  f"P&L: ${trade['pnl']:+.2f} ({trade['pnl_pct']:+.2f}%) | "
                  f"{trade['reason']}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Backtest mean reversion strategy on historical data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run for today with default symbol (SPY)
  python backtest_strategy.py
  
  # Run for specific symbol
  python backtest_strategy.py --symbol AAPL
  
  # Run for specific date range
  python backtest_strategy.py --start "2024-01-24 09:30" --end "2024-01-24 16:00"
  
  # Run with custom parameters
  python backtest_strategy.py --symbol TSLA --lookback 30 --buy-threshold 0.003
        '''
    )
    
    parser.add_argument('--symbol', '-s', default='SPY', 
                        help='Stock symbol to backtest (default: SPY)')
    
    parser.add_argument('--start', type=str, default=None,
                        help='Start datetime (format: "YYYY-MM-DD HH:MM")')
    
    parser.add_argument('--end', type=str, default=None,
                        help='End datetime (format: "YYYY-MM-DD HH:MM")')
    
    parser.add_argument('--lookback', '-l', type=int, default=20,
                        help='Moving average lookback period (default: 20)')
    
    parser.add_argument('--buy-threshold', '-b', type=float, default=0.005,
                        help='Buy when price is this pct below MA (default: 0.0005 = 0.05 percent)')
    
    parser.add_argument('--sell-threshold', '-t', type=float, default=0.005,
                        help='Sell when price is this pct above MA (default: 0.0005 = 0.05 percent)')
    
    parser.add_argument('--cash', '-c', type=float, default=100000,
                        help='Starting cash amount (default: 100000)')
    
    return parser.parse_args()


def main():
    """Run the backtest"""
    args = parse_args()
    
    # Parse dates if provided
    start_date = None
    end_date = None
    
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
    
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d %H:%M")
    
    # If no dates provided, use today
    if not start_date and not end_date:
        today = datetime.now().date()
        start_date = datetime.combine(today, datetime_time(9, 30))
        end_date = datetime.combine(today, datetime_time(16, 0))
    elif start_date and not end_date:
        # If only start provided, end at market close same day
        end_date = start_date.replace(hour=16, minute=0)
    elif end_date and not start_date:
        # If only end provided, start at market open same day
        start_date = end_date.replace(hour=9, minute=30)
    
    print(f"\n🎯 Backtesting {args.symbol}")
    print(f"📅 Period: {start_date} to {end_date}")
    print(f"⚙️  Parameters:")
    print(f"   - MA Period: {args.lookback}")
    print(f"   - Buy Threshold: -{args.buy_threshold*100:.3f}%")
    print(f"   - Sell Threshold: +{args.sell_threshold*100:.3f}%")
    print(f"   - Starting Cash: ${args.cash:,.2f}")
    print("-" * 50)
    
    # Create backtester
    backtester = BacktestMeanReversion(
        symbol=args.symbol,
        lookback_periods=args.lookback,
        buy_threshold=-abs(args.buy_threshold),  # Ensure negative
        sell_threshold=abs(args.sell_threshold)   # Ensure positive
    )
    
    # Override starting cash
    backtester.cash = args.cash
    
    # Run backtest
    backtester.run_backtest(start_date, end_date)


if __name__ == "__main__":
    main()