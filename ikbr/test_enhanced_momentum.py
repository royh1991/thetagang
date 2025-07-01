#!/usr/bin/env python3
"""
Quick test of enhanced momentum strategy without time restrictions
"""

import asyncio
import sys
from datetime import datetime, timedelta
from loguru import logger

sys.path.insert(0, '.')

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.examples.enhanced_momentum_strategy import EnhancedMomentumStrategy, EnhancedMomentumConfig

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
    
    # Strategy config with relaxed parameters
    strategy_config = EnhancedMomentumConfig(
        symbols=['AAPL', 'TSLA', 'NVDA'],
        max_positions=3,
        position_size_pct=0.2,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        cooldown_period=60.0,  # 1 minute
        metadata={
            'lookback_period': 10,        # Shorter lookback
            'momentum_threshold': 0.005,   # Lower threshold (0.5%)
            'volume_multiplier': 1.0,      # Lower volume requirement
            'ma_period': 20,              # Shorter MA
            'regime_ma_period': 50,       # Shorter regime MA
            'volatility_period': 10,      # Shorter volatility period
            'min_adr_pct': 0.01,         # Lower minimum volatility
            'allow_shorts': False
        }
    )
    
    await engine._initialize()
    
    # Temporarily override the trading window check
    original_is_trading_window = EnhancedMomentumStrategy._is_trading_window
    EnhancedMomentumStrategy._is_trading_window = lambda self, ts: True  # Always true
    
    try:
        engine.add_strategy(EnhancedMomentumStrategy, strategy_config)
        
        logger.info("Starting backtest...")
        result = await engine.run()
        
        # Print results
        logger.info("\n" + "="*60)
        logger.info("ENHANCED MOMENTUM STRATEGY RESULTS")
        logger.info("="*60)
        logger.info(f"Total Return: {result.total_return:.2%}")
        logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {result.max_drawdown:.2%}")
        logger.info(f"Total Trades: {result.total_trades}")
        logger.info(f"Win Rate: {result.win_rate:.2%}")
        logger.info(f"Profit Factor: {result.profit_factor:.2f}")
        
    finally:
        # Restore original method
        EnhancedMomentumStrategy._is_trading_window = original_is_trading_window
        await engine._cleanup()

if __name__ == "__main__":
    asyncio.run(main())