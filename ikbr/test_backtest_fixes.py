#!/usr/bin/env python3
"""Test the backtest fixes for duplicate trades and missing exits"""

import asyncio
from datetime import datetime, timedelta
from loguru import logger

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.examples.enhanced_momentum_strategy import EnhancedMomentumStrategy, EnhancedMomentumConfig

# Configure logger for detailed output
logger.remove()
logger.add(lambda msg: print(msg), level="DEBUG", 
          format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")


async def main():
    """Run a simple backtest to verify fixes"""
    # Configure backtest
    start_date = datetime.now() - timedelta(days=5)
    end_date = datetime.now() - timedelta(days=1)
    
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
    
    # Add enhanced momentum strategy
    strategy_config = EnhancedMomentumConfig(
        symbols=["AAPL", "MSFT"],
        max_positions=2,
        position_size_pct=0.1
    )
    # Update metadata with strategy-specific parameters
    strategy_config.metadata.update({
        'momentum_threshold': 0.01,  # Lower threshold for testing
        'use_trading_windows': False,  # Disable for testing
        'allow_shorts': False
    })
    
    engine.add_strategy(EnhancedMomentumStrategy, strategy_config)
    
    logger.info("Starting backtest...")
    
    try:
        # Run backtest
        result = await engine.run()
        
        # Print results
        logger.info("\n=== BACKTEST RESULTS ===")
        logger.info(f"Total trades: {result.total_trades}")
        logger.info(f"Total return: {result.total_return:.2%}")
        logger.info(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"Max drawdown: {result.max_drawdown:.2%}")
        
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
            
            # Check for duplicates
            trades_by_time = result.trades.groupby(['timestamp', 'symbol', 'action']).size()
            duplicates = trades_by_time[trades_by_time > 1]
            if not duplicates.empty:
                logger.warning(f"\n⚠️  Found {len(duplicates)} duplicate trades:")
                for key, count in duplicates.items():
                    logger.warning(f"  {key}: {count} trades")
            else:
                logger.info("\n✓ No duplicate trades found!")
                
            # Check if positions were closed
            if sell_count > 0:
                logger.info(f"\n✓ Found {sell_count} closing trades!")
            else:
                logger.warning("\n⚠️  No closing trades found!")
        else:
            logger.warning("No trades executed during backtest")
            
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())