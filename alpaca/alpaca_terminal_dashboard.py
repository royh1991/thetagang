import os
import time
import asyncio
from datetime import datetime
from collections import deque
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

# Load environment variables
load_dotenv()

API_KEY = os.environ.get('ALPACA_API_KEY')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY')

# Initialize clients
trading_client = TradingClient(API_KEY, SECRET_KEY)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

console = Console()

# Store price history for sparklines
price_history = {}
HISTORY_SIZE = 50


def get_sparkline(values, width=20):
    """Create a simple sparkline from values"""
    if not values or len(values) < 2:
        return " " * width
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return "─" * width
    
    chars = " ▁▂▃▄▅▆▇█"
    sparkline = ""
    
    # Sample values if we have too many
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    
    for val in sampled:
        normalized = (val - min_val) / (max_val - min_val)
        index = int(normalized * (len(chars) - 1))
        sparkline += chars[index]
    
    return sparkline


def create_stock_table(quotes_data):
    """Create a rich table with stock data"""
    table = Table(title="Live Stock Prices", border_style="blue")
    
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Last", justify="right", style="white")
    table.add_column("Change", justify="right")
    table.add_column("Change %", justify="right")
    table.add_column("Bid", justify="right", style="green")
    table.add_column("Ask", justify="right", style="red")
    table.add_column("Spread", justify="right", style="yellow")
    table.add_column("Trend (1min)", justify="center", no_wrap=True)
    
    for symbol, data in quotes_data.items():
        quote = data['quote']
        prev_close = data.get('prev_close', quote.ask_price)
        
        # Calculate change
        last_price = (quote.bid_price + quote.ask_price) / 2
        change = last_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close > 0 else 0
        
        # Update price history
        if symbol not in price_history:
            price_history[symbol] = deque(maxlen=HISTORY_SIZE)
        price_history[symbol].append(last_price)
        
        # Color for change
        change_color = "green" if change >= 0 else "red"
        change_str = f"[{change_color}]{change:+.2f}[/{change_color}]"
        change_pct_str = f"[{change_color}]{change_pct:+.2f}%[/{change_color}]"
        
        # Spread
        spread = quote.ask_price - quote.bid_price
        spread_bps = (spread / last_price) * 10000 if last_price > 0 else 0
        
        # Sparkline
        sparkline = get_sparkline(list(price_history.get(symbol, [])))
        
        table.add_row(
            symbol,
            f"${last_price:.2f}",
            change_str,
            change_pct_str,
            f"${quote.bid_price:.2f}",
            f"${quote.ask_price:.2f}",
            f"${spread:.2f} ({spread_bps:.0f}bps)",
            sparkline
        )
    
    return table


def create_account_panel(account):
    """Create account info panel"""
    text = Text()
    text.append("Account Information\n", style="bold cyan")
    text.append(f"Account: {account.account_number}\n")
    text.append(f"Equity: ${float(account.equity):,.2f}\n")
    text.append(f"Cash: ${float(account.cash):,.2f}\n")
    text.append(f"Buying Power: ${float(account.buying_power):,.2f}\n")
    
    # Calculate day change
    day_pl = float(account.equity) - float(account.last_equity)
    day_pl_pct = (day_pl / float(account.last_equity)) * 100 if float(account.last_equity) > 0 else 0
    
    pl_color = "green" if day_pl >= 0 else "red"
    text.append(f"Day P&L: ", style="white")
    text.append(f"${day_pl:+,.2f} ({day_pl_pct:+.2f}%)", style=pl_color)
    
    return Panel(text, border_style="green")


def create_positions_panel(positions):
    """Create positions panel"""
    if not positions:
        return Panel("No open positions", title="Positions", border_style="yellow")
    
    table = Table(show_header=True, header_style="bold yellow")
    table.add_column("Symbol", style="cyan")
    table.add_column("Qty", justify="right")
    table.add_column("Avg Cost", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("P&L", justify="right")
    
    for pos in positions[:5]:  # Show top 5
        pnl = float(pos.unrealized_pl)
        pnl_color = "green" if pnl >= 0 else "red"
        
        table.add_row(
            pos.symbol,
            str(pos.qty),
            f"${float(pos.avg_entry_price):.2f}",
            f"${float(pos.current_price):.2f}",
            f"[{pnl_color}]${pnl:+,.2f}[/{pnl_color}]"
        )
    
    return Panel(table, title="Positions", border_style="yellow")


def create_layout():
    """Create the dashboard layout"""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=2),
        Layout(name="footer", size=6)
    )
    
    layout["footer"].split_row(
        Layout(name="account"),
        Layout(name="positions")
    )
    
    return layout


async def update_dashboard(symbols):
    """Main update loop for dashboard"""
    layout = create_layout()
    
    with Live(layout, refresh_per_second=1, console=console) as live:
        while True:
            try:
                # Update header
                header_text = Text()
                header_text.append("Alpaca Live Market Dashboard", style="bold cyan")
                header_text.append(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="white")
                layout["header"].update(Panel(header_text, border_style="blue"))
                
                # Get quotes
                quotes_request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
                quotes = data_client.get_stock_latest_quote(quotes_request)
                
                # Get previous close prices (simplified - using current price as proxy)
                quotes_data = {}
                for symbol, quote in quotes.items():
                    quotes_data[symbol] = {
                        'quote': quote,
                        'prev_close': (quote.bid_price + quote.ask_price) / 2 * 0.99  # Simulated
                    }
                
                # Update main stock table
                layout["main"].update(create_stock_table(quotes_data))
                
                # Update account info
                account = trading_client.get_account()
                layout["account"].update(create_account_panel(account))
                
                # Update positions
                positions = trading_client.get_all_positions()
                layout["positions"].update(create_positions_panel(positions))
                
                await asyncio.sleep(1)  # Update every second
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                await asyncio.sleep(5)


def main():
    """Main entry point"""
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "SPY", "QQQ", "META", "AMZN", "AMD"]
    
    console.print("[bold cyan]Starting Alpaca Terminal Dashboard...[/bold cyan]")
    console.print(f"Tracking: {', '.join(symbols)}")
    console.print("Press Ctrl+C to exit\n")
    
    try:
        asyncio.run(update_dashboard(symbols))
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped[/yellow]")


if __name__ == "__main__":
    main()