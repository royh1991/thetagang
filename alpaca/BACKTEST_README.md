# Alpaca Backtest Framework

A modular backtesting framework for testing trading strategies with Alpaca market data.

## Architecture

The framework consists of several key components:

1. **`backtest_framework.py`** - Core engine and portfolio management
   - `BacktestEngine`: Orchestrates backtests
   - `Portfolio`: Tracks positions, cash, and P&L
   - `Order`: Represents trading orders
   - `Position`: Represents open positions
   - `Trade`: Represents completed trades

2. **`base_strategy.py`** - Abstract base class for strategies
   - Defines the interface all strategies must implement
   - Provides helper methods for order creation and position sizing

3. **`strategies/`** - Directory containing strategy implementations
   - `mean_reversion.py`: Buys dips, sells rallies
   - `momentum.py`: Follows trends with trailing stops

4. **`backtest.py`** - CLI interface for running backtests

## Usage

### List Available Strategies
```bash
python backtest.py --list-strategies
```

### Describe a Strategy
```bash
python backtest.py --strategy mean_reversion --describe
```

### Run a Simple Backtest
```bash
# Backtest mean reversion on SPY for today
python backtest.py --strategy mean_reversion --symbols SPY

# Multiple symbols
python backtest.py --strategy mean_reversion --symbols SPY,QQQ,AAPL
```

### Custom Date Range
```bash
# Specific date range
python backtest.py --strategy momentum --symbols TSLA \
  --start "2025-06-24 09:30" --end "2025-06-24 16:00"

# Multiple days
python backtest.py --strategy mean_reversion --symbols SPY \
  --start "2025-06-01 09:30" --end "2025-06-30 16:00"
```

### Custom Strategy Parameters
```bash
# Adjust mean reversion parameters
python backtest.py --strategy mean_reversion --symbols SPY \
  --param lookback_period=30 \
  --param buy_threshold=0.003 \
  --param sell_threshold=0.002

# Momentum with custom MAs
python backtest.py --strategy momentum --symbols AAPL,MSFT \
  --param fast_ma_period=5 \
  --param slow_ma_period=20 \
  --param trailing_stop_pct=0.03
```

### Verbose Mode
```bash
# Show progress bar
python backtest.py --strategy momentum --symbols SPY --verbose
```

## Creating New Strategies

1. Create a new file in `strategies/` directory
2. Import the base class:
   ```python
   from base_strategy import BaseStrategy
   from backtest_framework import Order, OrderSide, OrderType
   ```

3. Implement required methods:
   ```python
   class MyStrategy(BaseStrategy):
       def process_bars(self, bars, price_history, portfolio):
           # Your strategy logic
           return orders
       
       def get_required_history(self):
           return 20  # bars needed
   ```

4. The strategy will automatically be available in the CLI

## Strategy Parameters

Parameters can be passed via CLI and accessed in the strategy:

```python
# In strategy __init__:
self.my_param = self.get_param('my_param', default_value)

# From CLI:
--param my_param=123
```

## Performance Metrics

The framework tracks:
- Total trades and win rate
- Average win/loss
- Profit factor
- Sharpe ratio
- Maximum drawdown
- Total return

## Example Strategies

### Mean Reversion
- Buys when price drops X% below moving average
- Sells when price rises Y% above moving average
- Uses stop loss for risk management

### Momentum
- Buys stocks showing strong upward momentum
- Uses dual moving average crossover
- Implements trailing stop loss
- Considers volume surges

## Tips

1. Start with small date ranges to test quickly
2. Use verbose mode to see progress on large backtests
3. Adjust position sizing with `position_size_pct` parameter
4. Always validate strategy parameters before running
5. Monitor max drawdown to assess risk