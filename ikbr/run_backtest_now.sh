#!/bin/bash
# Run enhanced momentum backtest with 30 days of data
cd /Users/rhu/thetagang/ikbr
python run_backtest.py --mode backtest --strategy enhanced_momentum --symbols AAPL TSLA NVDA --days 30