#!/usr/bin/env python3
"""
Test simple momentum strategy
"""

import asyncio
import sys
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, '.')

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.examples.simple_momentum_strategy import SimpleMomentumStrategy, SimpleMomentumConfig

async def main():
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    # Backtest config
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now(),
        initial_capital=100000,
        data_frequency="5min",
        use_ib_data=True
    )
    
    # Create engine
    engine = BacktestEngine(config)
    
    # Strategy config
    strategy_config = SimpleMomentumConfig(
        symbols=['AAPL', 'TSLA', 'NVDA'],
        max_positions=3,
        position_size_pct=0.3,  # 30% per position
        stop_loss_pct=0.02,     # 2% stop
        take_profit_pct=0.04,   # 4% target
        cooldown_period=300.0,  # 5 minutes
        metadata={
            'lookback_period': 20,
            'momentum_threshold': 0.005,  # 0.5%
            'ma_period': 20,
            'allow_shorts': False
        }
    )
    
    await engine._initialize()
    engine.add_strategy(SimpleMomentumStrategy, strategy_config)
    
    logger.info("Starting simple momentum backtest...")
    result = await engine.run()
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("SIMPLE MOMENTUM STRATEGY RESULTS")
    logger.info("="*60)
    logger.info(f"Total Return: {result.total_return:.2%}")
    logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {result.max_drawdown:.2%}")
    logger.info(f"Total Trades: {result.total_trades}")
    logger.info(f"Win Rate: {result.win_rate:.2%}")
    logger.info(f"Profit Factor: {result.profit_factor:.2f}")
    logger.info(f"Average Win: ${result.avg_win:.2f}")
    logger.info(f"Average Loss: ${result.avg_loss:.2f}")
    
    await engine._cleanup()

if __name__ == "__main__":
    asyncio.run(main())