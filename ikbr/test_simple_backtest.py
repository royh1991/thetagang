#!/usr/bin/env python3
"""Simple test to verify backtest functionality"""

import asyncio
from datetime import datetime, timedelta
from loguru import logger

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.examples.simple_momentum_strategy import SimpleMomentumStrategy, SimpleMomentumConfig

# Configure logger for detailed output
logger.remove()
logger.add(lambda msg: print(msg), level="INFO", 
          format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")


async def main():
    """Run a simple backtest"""
    # Use more recent dates
    start_date = datetime(2025, 6, 25, 9, 30)  # 2 days ago
    end_date = datetime(2025, 6, 26, 16, 0)    # 1 day period
    
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        commission_per_share=0.01,
        slippage_pct=0.001,
        use_ib_data=True,
        cache_dir="backtest/cache"
    )
    
    # Create backtest engine
    engine = BacktestEngine(config)
    
    # Add simple momentum strategy with lower thresholds
    strategy_config = SimpleMomentumConfig(
        symbols=["AAPL"],
        max_positions=1,
        position_size_pct=0.2,
        momentum_period=10,
        momentum_threshold=0.002,  # 0.2% - very low for testing
        volume_ma_period=10,
        volume_multiplier=1.1
    )
    
    engine.add_strategy(SimpleMomentumStrategy, strategy_config)
    
    logger.info("Starting simple backtest...")
    
    try:
        # Run backtest
        result = await engine.run()
        
        # Print results
        logger.info("\n=== BACKTEST RESULTS ===")
        logger.info(f"Total trades: {result.total_trades}")
        logger.info(f"Total return: {result.total_return:.2%}")
        
        # Check trade details
        if not result.trades.empty:
            logger.info("\n=== TRADE DETAILS ===")
            for idx, trade in result.trades.iterrows():
                logger.info(f"{trade['timestamp']} | {trade['symbol']} | {trade['action']} | "
                          f"Qty: {trade['quantity']} | Price: ${trade['price']:.2f}")
            
            # Count BUY and SELL orders
            buy_count = len(result.trades[result.trades['action'] == 'BUY'])
            sell_count = len(result.trades[result.trades['action'] == 'SELL'])
            logger.info(f"\nBUY orders: {buy_count}")
            logger.info(f"SELL orders: {sell_count}")
        else:
            logger.warning("No trades executed during backtest")
            
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())