#!/usr/bin/env python3
"""
Test that duplicate trades are finally fixed
"""

import asyncio
import sys
from datetime import datetime, timedelta
import pandas as pd

sys.path.insert(0, '.')

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.examples.enhanced_momentum_strategy import EnhancedMomentumStrategy, EnhancedMomentumConfig

async def run_test():
    """Run a test and check for duplicates"""
    print("Testing duplicate trade fix...")
    
    # Configure a short backtest
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=3),
        end_date=datetime.now(),
        initial_capital=100000,
        commission_per_share=0.01,
        data_frequency="5min",
        use_ib_data=True
    )
    
    # Create engine
    engine = BacktestEngine(config)
    
    # Add strategy with low threshold to ensure trades
    strategy_config = EnhancedMomentumConfig(
        symbols=["AAPL", "TSLA"],
        max_positions=2,
        position_size_pct=0.3,
        metadata={
            'lookback_period': 20,
            'momentum_threshold': 0.002,  # Low threshold
            'use_trading_windows': False
        }
    )
    
    # DO NOT call _initialize()
    engine.add_strategy(EnhancedMomentumStrategy, strategy_config)
    
    # Run backtest
    result = await engine.run()
    
    # Check for duplicates
    print(f"\nTotal trades: {len(result.trades)}")
    
    if not result.trades.empty:
        # Check for exact duplicates
        duplicates = result.trades.duplicated(keep=False)
        num_duplicates = duplicates.sum()
        
        if num_duplicates > 0:
            print(f"❌ Found {num_duplicates} duplicate rows!")
            print("\nDuplicate trades:")
            print(result.trades[duplicates])
        else:
            print("✅ No exact duplicate rows found!")
        
        # Check for trades with same key fields
        key_cols = ['timestamp', 'symbol', 'action', 'quantity', 'price']
        grouped = result.trades.groupby(key_cols).size()
        duplicates = grouped[grouped > 1]
        
        if len(duplicates) > 0:
            print(f"\n⚠️  Found {len(duplicates)} groups of duplicate trades:")
            for key, count in duplicates.items():
                print(f"  {key} appears {count} times")
        else:
            print("✅ No duplicate trades by key fields!")
        
        # Show trades
        print("\nAll trades:")
        print(result.trades.to_string())
        
        # Save for inspection
        result.trades.to_csv("final_test_trades.csv", index=False)
        print("\nSaved to final_test_trades.csv")
    else:
        print("No trades generated")
    
    return result

async def main():
    """Run multiple tests to ensure consistency"""
    print("Running multiple tests to ensure fix is stable...\n")
    
    for i in range(3):
        print(f"\n{'='*60}")
        print(f"Test Run #{i+1}")
        print('='*60)
        
        await run_test()
        
        # Small delay between tests
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())