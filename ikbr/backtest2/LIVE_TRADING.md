# Live Trading with Backtest2 Strategies

## Overview

The `main.py` script enables live trading with any backtest2 strategy, including the Nick strategy. It connects to IB Gateway and executes trades based on real-time market data.

## Prerequisites

1. **IB Gateway Running**
   - Paper trading: Port 4102 (default)
   - Live trading: Port 4101
   - Make sure to check the port mappings in docker-compose.yml

2. **Environment Setup**
   ```bash
   source venv/bin/activate
   ```

## Usage

### Test Connection First
```bash
python backtest2/test_live_connection.py
```

This will verify:
- IB Gateway connection
- Account access
- Market data subscription
- Current positions

### Paper Trading (Default)

```bash
# Basic usage with Nick strategy
python backtest2/main.py --symbol TSLA --strategy nick

# With custom parameters
python backtest2/main.py --symbol TSLA --strategy nick \
  --position-size 0.20 \
  --adx-threshold 25 \
  --debug
```

### Live Trading (Production)

⚠️ **WARNING**: This will trade with real money!

```bash
# Requires confirmation
python backtest2/main.py --symbol TSLA --strategy nick --trading-mode live

# With safety limits
python backtest2/main.py --symbol TSLA --strategy nick \
  --trading-mode live \
  --position-size 0.05 \
  --max-position 5000
```

## Command Line Options

### Required Arguments
- `--symbol`: Stock symbol to trade (e.g., TSLA, SPY)
- `--strategy`: Strategy to use (simple, enhanced, nick)

### Optional Arguments
- `--trading-mode`: 'paper' (default) or 'live'
- `--position-size`: Fraction of account to use per trade (default: 0.10)
- `--max-position`: Maximum position value in dollars (default: 10000)
- `--debug`: Enable debug logging

### Nick Strategy Parameters
- `--lookback-period`: Breakout lookback period (default: 20)
- `--rsi-threshold`: RSI threshold (default: 55)
- `--adx-threshold`: ADX trend threshold (default: 30)

## Features

### Safety Measures
1. **Paper Trading by Default**: Always starts in paper mode unless explicitly set to live
2. **Live Trading Confirmation**: Requires typing 'YES' to confirm live trading
3. **Position Limits**: Maximum position value to prevent oversized trades
4. **Order Rate Limiting**: Minimum 5 seconds between orders
5. **Duplicate Order Prevention**: Won't place new orders while one is pending

### Real-Time Processing
- Subscribes to 5-second real-time bars
- Processes each bar through the strategy
- Executes market orders based on signals
- Tracks position and P&L

### Status Updates
- Prints status every 5 minutes
- Shows account value, position, trades today
- Logs all activity to file

### Market Data
- Automatically fetches SPY data for strategies that need it (like Nick)
- Uses historical data for market context
- Could be extended to use real-time SPY data

## Log Files

Logs are saved to `backtest2/logs/` with format:
```
live_SYMBOL_YYYYMMDD_HHMMSS.log
```

## Differences from Backtesting

1. **Real-Time Bars**: Uses 5-second bars aggregated to 5-minute periods
2. **Market Orders**: Currently uses market orders (could add limit orders)
3. **Position Sizing**: Based on actual account value
4. **Slippage**: Real market slippage applies
5. **Partial Fills**: Possible in live trading

## Monitoring

While running, the script will show:
- Entry/exit signals with reasons
- Order placement and fills
- Current position and account value
- Strategy statistics

## Stopping the Bot

Press `Ctrl+C` to gracefully shutdown. The bot will:
- Cancel market data subscriptions
- Log final status
- Disconnect from IB Gateway

## Example Session

```bash
$ python backtest2/main.py --symbol TSLA --strategy nick --debug

17:30:00 | INFO | Connecting to IB Gateway (PAPER TRADING)
17:30:01 | INFO | Connected to IB Gateway at localhost:4102
17:30:01 | INFO | Contract qualified: Stock(symbol='TSLA', exchange='SMART', currency='USD')
17:30:01 | INFO | Using account: DU1234567
17:30:01 | INFO | Starting live trading for TSLA
17:30:01 | INFO | Current position: 0 shares
17:30:02 | INFO | Fetching SPY data for market context...
17:30:03 | INFO | Set SPY market data with 390 bars
...
17:35:00 | INFO | === Status Update ===
17:35:00 | INFO | Account Value: $100,000
17:35:00 | INFO | Position: 0 shares ($0)
17:35:00 | INFO | Trades Today: 0
...
17:42:15 | INFO | BUY signal: Breakout > $421.50 | ADX=31.2 | RSI=58.3 | Vol spike 2.1x | SPY bullish | SL=$419.85 | TP=$426.32
17:42:15 | INFO | Position sizing: Account=$100,000, Target=$10,000, Price=$421.85, Shares=23
17:42:15 | INFO | Order placed: BUY 23 TSLA
17:42:16 | INFO | Order filled at $421.87
```

## Next Steps

To extend this implementation:

1. **Add Limit Orders**: Use limit orders with smart pricing
2. **Real-Time SPY**: Subscribe to real-time SPY data for market context
3. **Multiple Symbols**: Trade multiple symbols simultaneously
4. **Risk Management**: Add daily loss limits, position limits
5. **Performance Tracking**: Track detailed metrics and generate reports
6. **Alerts**: Send notifications on trades or errors
7. **Web Dashboard**: Create a monitoring interface