#!/usr/bin/env python3
"""
Run a single backtest properly to avoid state issues
"""

import asyncio
import sys
import argparse
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, '.')

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.examples.enhanced_momentum_strategy import EnhancedMomentumStrategy, EnhancedMomentumConfig

async def run_single_backtest(days: int):
    """Run a single backtest"""
    
    # Configure backtest
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=days),
        end_date=datetime.now(),
        initial_capital=100000,
        commission_per_share=0.01,
        slippage_pct=0.001,
        data_frequency="5min",
        use_ib_data=True
    )
    
    # Create engine
    engine = BacktestEngine(config)
    
    # Enhanced momentum with very relaxed parameters
    strategy_config = EnhancedMomentumConfig(
        symbols=["AAPL"],
        max_positions=1,
        position_size_pct=0.2,
        cooldown_period=300.0,  # 5 minutes
        metadata={
            'lookback_period': 20,
            'momentum_threshold': 0.003,  # 0.3% - very low
            'volume_multiplier': 0.8,     # Lower volume requirement
            'ma_period': 20,
            'regime_ma_period': 50,
            'volatility_period': 20,
            'min_adr_pct': 0.005,         # 0.5% minimum volatility
            'allow_shorts': False,
            'use_trading_windows': False
        }
    )
    
    # Add strategy
    engine.add_strategy(EnhancedMomentumStrategy, strategy_config)
    
    # Run backtest
    logger.info(f"Running {days}-day backtest...")
    result = await engine.run()
    
    # Print results
    logger.info(f"\nBacktest Results ({days} days):")
    logger.info(f"Total Return: {result.total_return:.2%}")
    logger.info(f"Total Trades: {result.total_trades}")
    logger.info(f"Win Rate: {result.win_rate:.2%}")
    logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {result.max_drawdown:.2%}")
    
    # Show unique trades (remove duplicates)
    if not result.trades.empty:
        # Remove duplicate trades
        unique_trades = result.trades.drop_duplicates(subset=['timestamp', 'symbol', 'action', 'quantity', 'price'])
        logger.info(f"\nUnique trades: {len(unique_trades)} (from {len(result.trades)} total)")
        
        logger.info("\nFirst 10 unique trades:")
        for i, (_, trade) in enumerate(unique_trades.head(10).iterrows()):
            logger.info(f"  {trade['timestamp']} - {trade['action']} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}")
    
    return result

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run single backtest")
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    # Run backtest
    asyncio.run(run_single_backtest(args.days))

if __name__ == "__main__":
    main()