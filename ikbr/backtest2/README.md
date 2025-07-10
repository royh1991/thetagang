# Simple Backtesting System for IKBR

A clean, modular backtesting system built from scratch for testing trading strategies with Interactive Brokers historical data.

## Features

- **Simple and Clean**: No complex event systems or async code - just straightforward logic
- **Real Historical Data**: Fetches actual historical data from Interactive Brokers
- **Caching**: Automatically caches historical data to avoid repeated API calls
- **Modular Design**: Easy to create new strategies by extending the base class
- **Performance Metrics**: Calculates Sharpe ratio, max drawdown, win rate, and more
- **CLI Interface**: Easy command-line interface for running backtests

## Quick Start

### 1. Basic Usage

Run a simple backtest:

```bash
python backtest2/run_backtest.py --symbol SPY --days 30
```

### 2. Advanced Usage

Run with custom parameters:

```bash
python backtest2/run_backtest.py \
    --symbol AAPL \
    --days 60 \
    --strategy enhanced \
    --capital 50000 \
    --commission 0.5 \
    --bar-size "1 hour" \
    --output results.json \
    --trades-csv trades.csv
```

### 3. Example Script

Run the example script to see how it works:

```bash
python backtest2/example_usage.py
```

## Architecture

### Components

1. **`data_fetcher.py`** - Fetches historical data from IB and caches it
2. **`broker_simulator.py`** - Simulates order execution with realistic fills
3. **`strategy_base.py`** - Base class for all trading strategies
4. **`simple_momentum.py`** - Example momentum strategies
5. **`backtest_engine.py`** - Main engine that coordinates everything
6. **`metrics.py`** - Calculates performance metrics
7. **`run_backtest.py`** - CLI runner

### Data Flow

```
IB Historical Data → DataFetcher → BacktestEngine → Strategy → BrokerSimulator → Metrics
```

## Creating Your Own Strategy

1. Create a new file in `backtest2/`:

```python
from backtest2.strategy_base import StrategyBase, Bar, Signal

class MyStrategy(StrategyBase):
    def __init__(self, symbol: str):
        super().__init__(symbol)
        # Your initialization
    
    def calculate_signal(self, bar: Bar) -> str:
        # Your logic here
        if some_condition:
            return Signal.BUY
        elif other_condition:
            return Signal.SELL
        return Signal.HOLD
```

2. Use it in the backtest:

```python
from backtest2.backtest_engine import BacktestEngine
from my_strategy import MyStrategy

engine = BacktestEngine()
strategy = MyStrategy('SPY')

results = engine.run(
    strategy=strategy,
    symbol='SPY',
    days=30,
    initial_capital=100000
)
```

## Available Strategies

### Simple Momentum Strategy
- Buys when price crosses above SMA
- Sells when price crosses below SMA
- Parameters: `sma_period` (default: 20)

### Enhanced Momentum Strategy
- Uses fast and slow SMA crossover
- Includes RSI filter
- Volume confirmation
- Parameters: `fast_sma`, `slow_sma`, `rsi_period`

## Command Line Options

```
Required:
  --symbol SYMBOL       Stock symbol (e.g., SPY)
  --days DAYS          Number of days to backtest

Optional:
  --strategy {simple,enhanced}  Strategy to use (default: simple)
  --capital CAPITAL            Initial capital (default: 100000)
  --commission COMMISSION      Commission per trade (default: 1.0)
  --bar-size BAR_SIZE         Bar size (default: "5 mins")
  --sma-period SMA_PERIOD     SMA period for simple strategy
  --fast-sma FAST_SMA         Fast SMA for enhanced strategy
  --slow-sma SLOW_SMA         Slow SMA for enhanced strategy
  --output OUTPUT             Output file for results (JSON)
  --trades-csv TRADES_CSV     Output file for trades (CSV)
  --verbose                   Enable verbose logging
```

## Performance Metrics

The system calculates:
- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Average Win/Loss**: Average profit and loss per trade
- **Expectancy**: Expected profit per trade

## Tips

1. **Start Simple**: Begin with the simple momentum strategy to verify everything works
2. **Check Data**: Use `--verbose` to see detailed data fetching and trading logs
3. **Cache Management**: Delete `backtest2/data_cache/` to force fresh data fetch
4. **Paper Trading**: Always test strategies in paper trading before going live
5. **Commission Impact**: Don't forget to include realistic commission costs

## Troubleshooting

1. **No data fetched**: Make sure IB Gateway is running on port 4102
2. **Connection errors**: Check that you're logged into IB Gateway
3. **No trades executed**: Strategy might need more historical data or different parameters
4. **Cache issues**: Delete cache files if data seems stale

## Next Steps

1. Add more sophisticated strategies
2. Implement stop-loss and take-profit orders
3. Add position sizing based on volatility
4. Create visualization of equity curves
5. Add multi-symbol portfolio backtesting