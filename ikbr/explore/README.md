# IBKR Bot Exploration Tools

This folder contains interactive scripts for exploring and learning the IBKR bot system.

## Scripts Overview

### Connection & Basic Tests
- `simple_test.py` - Simplest connection test to verify IB Gateway is working
- `test_connection.py` - More comprehensive connection testing
- `test_market_data_simple.py` - Basic market data streaming test
- `test_live_trading_simple.py` - Simple live trading test (paper account)

### Interactive Explorers
- `explore_fixed.py` - Main exploration tool for IB Gateway features (no async complexity)
- `explore_components.py` - Explore bot components (EventBus, MarketData, etc.)
- `step_by_step.py` - Step-by-step tutorial that pauses at each operation
- `demo_components.py` - Demo of how components work together

### Other Tools
- `notebook_style.py` - Notebook-style exploration (for Jupyter-like experience)

## Quick Start

1. **First Time Setup**
   ```bash
   # Start IB Gateway
   docker-compose up -d
   
   # Wait 30-60 seconds for startup
   
   # Test connection
   python explore/simple_test.py
   ```

2. **Interactive Exploration**
   ```bash
   # For general IB Gateway features
   python explore/explore_fixed.py
   # Then: connect()
   
   # For bot components
   python explore/explore_components.py
   # Then: setup()
   ```

3. **Step-by-Step Tutorial**
   ```bash
   python explore/step_by_step.py
   # Follow the prompts
   ```

## Usage Tips

- Always start with `simple_test.py` to verify connection
- Use `explore_fixed.py` for market data and historical data exploration
- Use `explore_components.py` to understand how the bot architecture works
- The step-by-step tutorial is great for first-time users

## Common Issues

1. **Connection Failed**
   - Check Docker is running: `docker ps`
   - Check logs: `docker-compose logs ib-gateway`
   - Wait longer - IB Gateway takes 30-60 seconds to fully start

2. **No Market Data**
   - Market might be closed - use historical data functions
   - Try using 'frozen' or 'delayed' data types in `explore_fixed.py`

3. **Import Errors**
   - Make sure you're in the virtual environment
   - Run from the ikbr directory: `cd ikbr && python explore/script.py`