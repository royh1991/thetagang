#!/usr/bin/env python3
"""Simple test to check duplicate trades and exits"""

import asyncio
from datetime import datetime, timedelta
from loguru import logger
import os

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.examples.enhanced_momentum_strategy import EnhancedMomentumStrategy
from strategies.base_strategy import StrategyConfig

# Configure logger
logger.remove()
logger.add(lambda msg: print(msg), level="INFO")

# Set to paper trading mode
os.environ['TRADING_MODE'] = 'paper'


async def main():
    """Run test backtest"""
    # Use a short recent period
    start_date = datetime(2025, 6, 28, 9, 30)
    end_date = datetime(2025, 6, 28, 16, 0)
    
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        commission_per_share=0.01,
        slippage_pct=0.001,
        use_ib_data=True
    )
    
    # Create engine
    engine = BacktestEngine(config)
    
    # Simple strategy config
    strategy_config = StrategyConfig(
        name="TestMomentum",
        symbols=["AAPL"],
        max_positions=2,
        position_size_pct=0.1,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )
    
    # Set momentum parameters
    strategy_config.metadata = {
        'momentum_threshold': 0.001,  # Very low 0.1% for testing
        'use_trading_windows': False,
        'lookback_period': 10,
        'ma_period': 20
    }
    
    engine.add_strategy(EnhancedMomentumStrategy, strategy_config)
    
    print("Running backtest...")
    
    try:
        result = await engine.run()
        
        print(f"\nTotal trades: {result.total_trades}")
        
        if not result.trades.empty:
            print("\nTrades:")
            for _, trade in result.trades.iterrows():
                print(f"  {trade['timestamp']} - {trade['symbol']} {trade['action']} "
                      f"{trade['quantity']} @ ${trade['price']:.2f}")
            
            # Check for duplicates
            print("\nChecking for duplicate trades...")
            trades_df = result.trades.copy()
            trades_df['time_str'] = trades_df['timestamp'].astype(str)
            duplicates = trades_df.groupby(['time_str', 'symbol', 'action', 'price']).size()
            dups = duplicates[duplicates > 1]
            
            if len(dups) > 0:
                print(f"❌ Found {len(dups)} duplicate trade groups!")
                for key, count in dups.items():
                    print(f"   {key}: {count} trades")
            else:
                print("✅ No duplicate trades found!")
            
            # Check exits
            buy_count = len(trades_df[trades_df['action'] == 'BUY'])
            sell_count = len(trades_df[trades_df['action'] == 'SELL']) 
            print(f"\nBUY orders: {buy_count}")
            print(f"SELL orders: {sell_count}")
            
            if sell_count > 0:
                print("✅ Exit trades found!")
            else:
                print("❌ No exit trades found!")
                
        else:
            print("No trades executed")
            
    except Exception as e:
        import traceback
        logger.error(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())