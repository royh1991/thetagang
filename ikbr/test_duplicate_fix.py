#!/usr/bin/env python3
"""
Test script to verify duplicate trade records are fixed
"""

import asyncio
import sys
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, '.')

from backtest.engine import BacktestEngine, BacktestConfig
from backtest.report_generator import ReportGenerator
from strategies.examples.enhanced_momentum_strategy import EnhancedMomentumStrategy, EnhancedMomentumConfig

async def test_duplicate_fix():
    """Test that trades are only recorded once"""
    logger.info("Testing duplicate trade fix...")
    
    # Configure backtest
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=5),
        end_date=datetime.now(),
        initial_capital=100000,
        commission_per_share=0.01,
        slippage_pct=0.001,
        data_frequency="5min",
        use_ib_data=True
    )
    
    # Create engine
    engine = BacktestEngine(config)
    
    # Add enhanced momentum strategy
    strategy_config = EnhancedMomentumConfig(
        symbols=["AAPL"],
        max_positions=1,
        position_size_pct=0.3,
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        metadata={
            'lookback_period': 20,
            'momentum_threshold': 0.005,
            'use_trading_windows': False
        }
    )
    
    # Note: We're NOT calling engine._initialize() here
    # The run() method will handle initialization
    engine.add_strategy(EnhancedMomentumStrategy, strategy_config)
    
    # Run backtest
    result = await engine.run()
    
    # Check trades
    logger.info(f"Total trades recorded: {len(result.trades)}")
    
    # Count unique trades
    if not result.trades.empty:
        # Group by timestamp, symbol, action, quantity, price to find duplicates
        grouped = result.trades.groupby(['timestamp', 'symbol', 'action', 'quantity', 'price']).size()
        duplicates = grouped[grouped > 1]
        
        if len(duplicates) > 0:
            logger.error(f"Found {len(duplicates)} duplicate trade groups!")
            for idx, count in duplicates.items():
                logger.error(f"  Trade {idx} appears {count} times")
        else:
            logger.success("✅ No duplicate trades found! Fix is working.")
            
        # Show first few trades
        logger.info("\nFirst few trades:")
        print(result.trades.head(10))
    else:
        logger.warning("No trades generated in test")
    
    # Generate report
    report_gen = ReportGenerator()
    report_path = report_gen.generate_report(result, 'DuplicateFixTest')
    logger.info(f"Report saved to: {report_path}")
    
    return result

if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    # Run test
    asyncio.run(test_duplicate_fix())