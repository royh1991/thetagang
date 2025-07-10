#!/usr/bin/env python3
"""Test with a very simple strategy to isolate the issue"""

import asyncio
import sys
from datetime import datetime, timedelta
from loguru import logger
from collections import deque

sys.path.insert(0, '.')

from backtest.engine import BacktestEngine, BacktestConfig
from strategies.base_strategy import BaseStrategy, StrategyConfig
from core.market_data import TickData
from core.order_manager import Signal

class SimpleTestStrategy(BaseStrategy):
    """Super simple strategy for testing"""
    
    def __init__(self, config: StrategyConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.tick_count = 0
        self.signal_count = 0
        self.prices = {}
        
    async def on_start(self):
        logger.info("SimpleTestStrategy started")
        for symbol in self.config.symbols:
            self.prices[symbol] = deque(maxlen=10)
    
    async def on_tick(self, tick: TickData):
        """Just count ticks"""
        self.tick_count += 1
        if tick.symbol in self.prices:
            self.prices[tick.symbol].append(tick.last)
        
        if self.tick_count % 1000 == 0:
            logger.info(f"Tick #{self.tick_count}: {tick.symbol} @ ${tick.last:.2f}")
    
    async def calculate_signals(self, tick: TickData) -> list:
        """Generate a signal every 100 ticks if no position"""
        if tick.symbol not in self.config.symbols:
            return []
            
        # Only generate signals when we don't have a position
        if tick.symbol in self._positions:
            return []
            
        # Generate a signal every 100 ticks
        if self.tick_count % 100 == 0 and len(self.prices.get(tick.symbol, [])) >= 5:
            self.signal_count += 1
            logger.info(f"🎯 Generating signal #{self.signal_count} at tick {self.tick_count}")
            
            signal = Signal(
                action="BUY",
                symbol=tick.symbol,
                quantity=0,  # Let risk manager calculate
                order_type="MARKET",
                stop_loss=tick.last * 0.98,
                take_profit=tick.last * 1.02
            )
            return [signal]
        
        return []
    
    async def should_close_position(self, tick: TickData, position) -> tuple:
        """Close after 50 ticks"""
        # Simple exit after some ticks
        if self.tick_count % 150 == 0:
            return True, "time_exit"
        return False, None

async def test_backtest(days: int):
    """Run test backtest"""
    logger.info(f"\nTesting {days}-day backtest with SimpleTestStrategy")
    
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=days),
        end_date=datetime.now(),
        initial_capital=100000,
        commission_per_share=0.01,
        data_frequency="5min",
        use_ib_data=True
    )
    
    engine = BacktestEngine(config)
    
    strategy_config = StrategyConfig(
        name="SimpleTest",
        symbols=["AAPL"],
        max_positions=1,
        position_size_pct=0.2,
        cooldown_period=0.0  # No cooldown!
    )
    
    engine.add_strategy(SimpleTestStrategy, strategy_config)
    result = await engine.run()
    
    logger.info(f"\nResults:")
    logger.info(f"Total trades: {result.total_trades}")
    
    if not result.trades.empty:
        logger.info("Trades:")
        for _, trade in result.trades.iterrows():
            logger.info(f"  {trade['timestamp']} - {trade['action']} @ ${trade['price']:.2f}")
    
    return result

async def main():
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    # Test both periods
    result_10 = await test_backtest(10)
    result_100 = await test_backtest(100)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"10-day: {result_10.total_trades} trades")
    logger.info(f"100-day: {result_100.total_trades} trades")
    
    if result_10.total_trades == result_100.total_trades:
        logger.error("❌ Still getting same number of trades!")
    else:
        logger.success("✅ Different number of trades!")

if __name__ == "__main__":
    asyncio.run(main())