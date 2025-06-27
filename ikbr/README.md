# IBKR Trading Bot Framework

A modular, event-driven trading bot framework for Interactive Brokers using the `ib_async` library. This framework provides a complete infrastructure for developing, backtesting, and deploying automated trading strategies.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
- [Strategies](#strategies)
- [Backtesting](#backtesting)
- [Getting Started](#getting-started)
- [Testing](#testing)
- [Configuration](#configuration)
- [Examples](#examples)
- [Project Structure](#project-structure)

## Architecture Overview

The framework follows an event-driven architecture with loosely coupled components communicating through an async event bus. This design ensures:

- **Modularity**: Components can be developed and tested independently
- **Scalability**: Easy to add new strategies and components
- **Reliability**: Built-in error handling and risk management
- **Performance**: Async operations throughout for efficient execution

### Key Design Principles

1. **Event-Driven**: All components communicate via events (market data updates, signals, fills)
2. **Async-First**: Built on `asyncio` for concurrent operations
3. **Risk-Aware**: Every trade goes through risk checks before execution
4. **Testable**: Comprehensive unit tests and integration tests

## Core Components

### Event Bus (`core/event_bus.py`)

The central nervous system of the framework. Handles all inter-component communication.

```python
# Example usage
event_bus = get_event_bus()
await event_bus.emit(Event(EventTypes.SIGNAL_GENERATED, signal))
```

**Features:**
- Async event emission and handling
- Priority-based event processing
- Event filtering and routing
- Performance metrics

### Market Data Manager (`core/market_data.py`)

Manages real-time and historical market data with intelligent caching.

```python
market_data = MarketDataManager(ib, config)
await market_data.subscribe_ticker("SPY")
tick = market_data.get_latest_tick("SPY")
```

**Features:**
- Real-time market data streaming
- Historical data retrieval and caching
- Circular buffer for tick storage
- Automatic reconnection handling
- Support for delayed data fallback

### Order Manager (`core/order_manager.py`)

Handles the complete order lifecycle from signal to execution.

```python
order_manager = OrderManager(ib, config)
order_info = await order_manager.submit_order(signal)
```

**Features:**
- Multiple order types (Market, Limit, Stop, Stop-Limit)
- Bracket orders with stop-loss and take-profit
- Order modification and cancellation
- Timeout handling
- Fill tracking and reporting

### Risk Manager (`core/risk_manager.py`)

Enforces risk limits and position sizing rules.

```python
risk_manager = RiskManager(ib, limits)
is_allowed = await risk_manager.check_order(signal)
position_size = await risk_manager.calculate_position_size(signal)
```

**Features:**
- Position size limits
- Maximum position count
- Daily loss limits
- Exposure limits
- Risk-based position sizing
- Real-time P&L tracking

## Strategies

### Base Strategy (`strategies/base_strategy.py`)

Abstract base class that all strategies inherit from. Provides:

- Market data subscription management
- Signal generation framework
- Position tracking
- Performance metrics

### Example Strategies

#### Momentum Strategy (`strategies/examples/momentum_strategy.py`)

Trades based on price momentum over a configurable lookback period.

```python
config = MomentumConfig(
    symbols=["SPY", "QQQ"],
    lookback_period=20,
    momentum_threshold=0.02
)
strategy = MomentumStrategy(config, market_data, order_manager, risk_manager)
```

#### Mean Reversion Strategy (`strategies/examples/mean_reversion_strategy.py`)

Trades based on deviations from moving averages.

```python
config = MeanReversionConfig(
    symbols=["AAPL", "MSFT"],
    sma_period=20,
    entry_std=2.0,
    exit_std=0.5
)
```

### Creating Custom Strategies

1. Inherit from `BaseStrategy`
2. Implement `calculate_signals()` method
3. Define your strategy configuration
4. Handle position management logic

```python
class MyStrategy(BaseStrategy):
    async def calculate_signals(self, tick: TickData) -> List[Signal]:
        # Your strategy logic here
        pass
```

## Backtesting

### Backtesting Engine (`backtest/engine.py`)

Full-featured backtesting engine with synthetic data generation.

```python
engine = BacktestEngine(initial_capital=100000)
results = await engine.run_backtest(
    strategy,
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31)
)
```

**Features:**
- Synthetic market data generation
- Realistic order execution simulation
- Slippage and commission modeling
- Performance metrics calculation
- Trade-by-trade analysis

### Data Provider (`backtest/data_provider.py`)

Generates realistic synthetic market data for backtesting.

```python
provider = DataProvider()
tick_generator = await provider.get_ticker_stream(
    "SPY",
    start_date,
    end_date,
    volatility=0.15
)
```

### Mock Broker (`backtest/mock_broker.py`)

Simulates broker behavior for backtesting.

- Order matching logic
- Partial fills
- Market impact simulation
- Latency simulation

## Getting Started

### Prerequisites

1. Interactive Brokers account (paper or live)
2. IB Gateway or TWS running
3. Python 3.8+
4. Docker (optional, for IB Gateway)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ikbr.git
cd ikbr

# Install dependencies
pip install -r requirements.txt

# Copy and configure .env file
cp .env.example .env
# Edit .env with your settings
```

### Configuration

Create a `.env` file with:

```bash
# IB Gateway Configuration
IB_GATEWAY_PORT=4002  # Paper trading port
IB_GATEWAY_PORT_LIVE=4001  # Live trading port
TRADING_MODE=paper  # or 'live'
ACCOUNT_ID=YOUR_ACCOUNT_ID

# Strategy Configuration
MAX_POSITIONS=5
RISK_PER_TRADE_PCT=0.01
```

### Running the Framework

1. **Start IB Gateway**:
```bash
docker-compose up -d
```

2. **Test Connection**:
```bash
python tests/test_connection.py
```

3. **Run a Strategy**:
```python
from core.market_data import MarketDataManager
from strategies.examples.momentum_strategy import MomentumStrategy

# Initialize components
ib = IB()
ib.connect('localhost', 4002, clientId=1)

# Create and run strategy
strategy = MomentumStrategy(config, market_data, order_manager, risk_manager)
await strategy.start()
```

## Testing

### Unit Tests

Run all unit tests:
```bash
python -m pytest core/test_*.py -v
```

### Integration Tests

Test with paper trading account:
```bash
python explore/test_live_trading_simple.py
```

### Market Data Testing

Diagnose market data issues:
```bash
python tests/diagnose_market_data.py
```

## Configuration

### Market Data Configuration

```python
MarketDataConfig(
    tick_buffer_size=1000,  # Number of ticks to store
    cache_historical=True,  # Cache historical data
    delayed_data_fallback=True,  # Use delayed data if real-time unavailable
)
```

### Order Configuration

```python
OrderConfig(
    default_timeout=30,  # Order timeout in seconds
    use_adaptive_orders=True,  # Use IB's adaptive algo
    outside_rth=False,  # Trade outside regular hours
)
```

### Risk Limits

```python
RiskLimits(
    max_position_size=10000,  # Max $ per position
    max_positions=5,  # Max number of positions
    max_total_exposure=50000,  # Max total $ exposure
    max_daily_loss=1000,  # Daily loss limit
    risk_per_trade_pct=0.01,  # Risk 1% per trade
)
```

## Examples

### Simple Buy Signal

```python
signal = Signal(
    action="BUY",
    symbol="AAPL",
    quantity=100,
    order_type="LIMIT",
    limit_price=150.00,
    stop_loss=145.00,
    take_profit=160.00
)
```

### Running Multiple Strategies

```python
strategies = [
    MomentumStrategy(momentum_config, ...),
    MeanReversionStrategy(mean_reversion_config, ...)
]

for strategy in strategies:
    await strategy.start()
```

### Custom Event Handler

```python
@event_bus.on(EventTypes.MARKET_DATA_UPDATE)
async def handle_market_update(event: Event):
    tick = event.data
    logger.info(f"Price update: {tick.symbol} @ ${tick.last}")
```

## Performance Considerations

- Use async operations throughout for non-blocking execution
- Market data is buffered in circular buffers for efficiency
- Historical data is cached to reduce API calls
- Event bus uses priority queues for time-sensitive events

## Troubleshooting

### Common Issues

1. **No Market Data**
   - Check market data subscriptions in IB account
   - Ensure market hours for the symbol
   - Try delayed data with `reqMarketDataType(3)`

2. **Connection Failed**
   - Verify IB Gateway is running
   - Check port numbers (4001 for live, 4002 for paper)
   - Ensure API connections are enabled in TWS/Gateway

3. **Order Rejected**
   - Check account permissions
   - Verify market hours
   - Review risk limits

## Architecture Details

### Component Interaction Flow

1. **Market Data Flow**:
   - IB API → MarketDataManager → Event Bus → Strategies
   - Ticks are buffered and distributed to subscribed strategies

2. **Signal Generation Flow**:
   - Strategy → Signal → Event Bus → OrderManager
   - Each signal goes through risk checks before execution

3. **Order Execution Flow**:
   - OrderManager → RiskManager → IB API
   - Fills are reported back through the event bus

### Event Types

- `MARKET_DATA_UPDATE`: New tick data available
- `SIGNAL_GENERATED`: Strategy produced a trading signal
- `ORDER_SUBMITTED`: Order sent to broker
- `ORDER_FILLED`: Order execution complete
- `ORDER_REJECTED`: Order failed risk checks or broker validation
- `POSITION_UPDATE`: Position changed
- `RISK_LIMIT_BREACH`: Risk limit exceeded

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Project Structure

```
ikbr/
├── core/                    # Core framework components
│   ├── event_bus.py        # Event-driven messaging system
│   ├── market_data.py      # Market data management
│   ├── order_manager.py    # Order execution and management
│   └── risk_manager.py     # Risk controls and position limits
│
├── strategies/             # Trading strategies
│   ├── base_strategy.py    # Abstract base strategy class
│   └── examples/           # Example strategy implementations
│       ├── momentum_strategy.py
│       └── mean_reversion_strategy.py
│
├── backtest/               # Backtesting framework
│   ├── engine.py          # Main backtest engine
│   ├── data_provider.py   # Historical data management
│   └── mock_broker.py     # Simulated order execution
│
├── explore/                # Interactive exploration tools
│   ├── explore_fixed.py   # Main IB Gateway explorer
│   ├── explore_components.py  # Bot components explorer
│   ├── step_by_step.py   # Interactive tutorial
│   └── simple_test.py    # Basic connection test
│
├── tests/                  # Test suites
│   ├── unit/              # Unit tests for components
│   ├── integration/       # Integration tests
│   └── test_connection.py # Connection testing
│
├── config/                 # Configuration files
│   └── strategies/        # Strategy-specific configs
│
├── bot.py                 # Main bot entry point
├── run_backtest.py        # Backtest runner
├── verify_system.py       # System verification tool
├── docker-compose.yml     # IB Gateway Docker setup
└── requirements.txt       # Python dependencies
```

## License

This project is licensed under the MIT License.