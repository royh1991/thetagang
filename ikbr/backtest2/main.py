#!/usr/bin/env python3
"""
Live Trading Main Script for Backtest2 Strategies

Usage:
    # Paper trading (default)
    python main.py --symbol TSLA --strategy nick
    
    # Production trading (requires confirmation)
    python main.py --symbol TSLA --strategy nick --trading-mode live
    
    # With custom parameters
    python main.py --symbol TSLA --strategy nick --position-size 0.5 --adx-threshold 25
"""

import asyncio
import sys
import os
import argparse
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger
from ib_async import IB, Stock, MarketOrder, util, Contract

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backtest2.strategy_base import StrategyBase, Bar, Signal, SignalInfo
from backtest2.simple_momentum import SimpleMomentumStrategy, EnhancedMomentumStrategy
from backtest2.nick_strategy import NickStrategy


class LiveTrader:
    """Live trading engine for backtest2 strategies"""
    
    def __init__(self, 
                 strategy: StrategyBase,
                 symbol: str,
                 position_size_pct: float = 0.10,
                 trading_mode: str = 'paper',
                 max_position_value: float = 10000,
                 min_order_interval: float = 5.0):
        """
        Initialize live trader
        
        Args:
            strategy: Trading strategy instance
            symbol: Stock symbol to trade
            position_size_pct: Position size as fraction of equity
            trading_mode: 'paper' or 'live'
            max_position_value: Maximum position value allowed
            min_order_interval: Minimum seconds between orders
        """
        self.strategy = strategy
        self.symbol = symbol
        self.position_size_pct = position_size_pct
        self.trading_mode = trading_mode
        self.max_position_value = max_position_value
        self.min_order_interval = min_order_interval
        
        self.ib = IB()
        self.contract: Optional[Contract] = None
        self.ticker = None  # Store ticker subscription
        self.running = False
        self.last_order_time = datetime.min
        self.current_position = 0
        self.in_order = False
        
        # Performance tracking
        self.trades_today = 0
        self.pnl_today = 0.0
        
    async def connect(self):
        """Connect to IB Gateway"""
        # Use environment variables or defaults
        host = os.getenv('IB_GATEWAY_HOST', 'localhost')
        
        if self.trading_mode == 'paper':
            port = int(os.getenv('IB_GATEWAY_PORT', 4102))
            logger.info("Connecting to IB Gateway (PAPER TRADING)")
        else:
            port = int(os.getenv('IB_GATEWAY_PORT_LIVE', 4101))
            logger.warning("Connecting to IB Gateway (LIVE TRADING)")
            
            # Require confirmation for live trading
            confirm = input("⚠️  LIVE TRADING MODE - Type 'YES' to confirm: ")
            if confirm != 'YES':
                logger.error("Live trading not confirmed. Exiting.")
                sys.exit(1)
        
        # Connect with a unique client ID
        client_id = int(os.getenv('IB_CLIENT_ID', 99))
        await self.ib.connectAsync(host, port, clientId=client_id)
        logger.info(f"Connected to IB Gateway at {host}:{port}")
        
        # Create and qualify contract
        self.contract = Stock(self.symbol, 'SMART', 'USD')
        await self.ib.qualifyContractsAsync(self.contract)
        logger.info(f"Contract qualified: {self.contract}")
        
        # Subscribe to market data for this contract
        self.ticker = self.ib.reqMktData(self.contract, '', False, False)
        logger.info(f"Subscribed to market data for {self.symbol}")
        
        # Wait a moment for ticker to populate
        await asyncio.sleep(2)
        
        # Get account info
        self.account = self.ib.managedAccounts()[0]
        logger.info(f"Using account: {self.account}")
        
    async def get_account_value(self) -> float:
        """Get current account value"""
        account_values = self.ib.accountSummary(self.account)
        for av in account_values:
            if av.tag == 'NetLiquidation':
                return float(av.value)
        return 100000.0  # Default fallback
        
    async def get_position(self) -> int:
        """Get current position in shares"""
        positions = self.ib.positions(self.account)
        for pos in positions:
            if pos.contract.symbol == self.symbol:
                return int(pos.position)
        return 0
        
    async def calculate_position_size(self) -> int:
        """Calculate position size based on account value"""
        try:
            account_value = await self.get_account_value()
            
            # Use stored ticker with better validation
            if not self.ticker:
                logger.error("No ticker subscription available")
                return 0
                
            # Try multiple price fields in order of preference
            price = None
            if self.ticker.last and self.ticker.last > 0:
                price = self.ticker.last
            elif self.ticker.bid and self.ticker.ask and self.ticker.bid > 0 and self.ticker.ask > 0:
                price = (self.ticker.bid + self.ticker.ask) / 2
                logger.info(f"Using mid price: ${price:.2f} (bid: ${self.ticker.bid:.2f}, ask: ${self.ticker.ask:.2f})")
            elif self.ticker.close and self.ticker.close > 0:
                price = self.ticker.close
                logger.info(f"Using close price: ${price:.2f}")
            else:
                logger.warning("No valid price available for position sizing")
                logger.debug(f"Ticker data: last={self.ticker.last}, bid={self.ticker.bid}, "
                           f"ask={self.ticker.ask}, close={self.ticker.close}")
                return 0
                
            # Calculate position value
            position_value = min(
                account_value * self.position_size_pct,
                self.max_position_value
            )
            
            # Calculate shares
            shares = int(position_value / price)
            
            logger.info(f"Position sizing: Account=${account_value:,.0f}, "
                       f"Target=${position_value:,.0f}, Price=${price:.2f}, "
                       f"Shares={shares}")
            
            return shares
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}", exc_info=True)
            return 0
            
    async def place_order(self, action: str, quantity: int):
        """Place an order"""
        # Check minimum order interval
        time_since_last = (datetime.now() - self.last_order_time).total_seconds()
        if time_since_last < self.min_order_interval:
            logger.warning(f"Order rejected: Too soon since last order ({time_since_last:.1f}s)")
            return
            
        # Prevent duplicate orders
        if self.in_order:
            logger.warning("Order rejected: Another order is pending")
            return
            
        try:
            self.in_order = True
            
            # Create market order for simplicity
            # In production, consider limit orders with proper price calculation
            order = MarketOrder(action, quantity)
            
            # Place order
            trade = self.ib.placeOrder(self.contract, order)
            logger.info(f"Order placed: {action} {quantity} {self.symbol}")
            
            # Wait for fill
            await asyncio.sleep(1)
            
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                logger.info(f"Order filled at ${fill_price:.2f}")
                self.trades_today += 1
                self.last_order_time = datetime.now()
            else:
                logger.warning(f"Order status: {trade.orderStatus.status}")
                
        except Exception as e:
            logger.error(f"Order error: {e}")
        finally:
            self.in_order = False
            
    async def process_signal(self, bar: Bar, signal_info: SignalInfo):
        """Process trading signal"""
        signal = signal_info.signal
        reason = signal_info.reason
        
        if signal == Signal.BUY and self.current_position == 0:
            # Calculate position size
            quantity = await self.calculate_position_size()
            if quantity > 0:
                logger.info(f"BUY signal: {reason}")
                await self.place_order('BUY', quantity)
                self.current_position = quantity
                
        elif signal == Signal.SELL and self.current_position > 0:
            logger.info(f"SELL signal: {reason}")
            await self.place_order('SELL', self.current_position)
            self.current_position = 0
            
    async def process_bar(self, bar_update):
        """Process a single bar update"""
        try:
            # Convert to our Bar format
            bar = Bar(
                timestamp=bar_update.time,
                open=bar_update.open_,
                high=bar_update.high,
                low=bar_update.low,
                close=bar_update.close,
                volume=int(bar_update.volume)
            )
            
            # Process through strategy
            result = self.strategy.on_bar(bar)
            
            # Handle signal
            if isinstance(result, SignalInfo):
                await self.process_signal(bar, result)
                
        except Exception as e:
            logger.error(f"Error processing bar: {e}")
            
    async def run(self):
        """Main trading loop"""
        self.running = True
        logger.info(f"Starting live trading for {self.symbol}")
        
        # Get initial position
        self.current_position = await self.get_position()
        logger.info(f"Current position: {self.current_position} shares")
        
        # Subscribe to real-time bars
        self.bars_subscription = self.ib.reqRealTimeBars(
            self.contract, 
            5,  # 5 second bars
            'TRADES', 
            False
        )
        
        # Status update counter
        self.status_counter = 0
        
        # Set up bar update handler
        def on_bar_update(bars, hasNewBar):
            if hasNewBar and self.running:
                asyncio.create_task(self.process_bar(bars[-1]))
                
                # Status update every 60 bars (5 minutes)
                self.status_counter += 1
                if self.status_counter >= 60:
                    asyncio.create_task(self.print_status())
                    self.status_counter = 0
                    
        # Connect the handler
        self.bars_subscription.updateEvent += on_bar_update
        
        # Keep running until stopped
        while self.running:
            await asyncio.sleep(1)
                
    async def print_status(self):
        """Print current status"""
        account_value = await self.get_account_value()
        
        # Calculate position value safely
        position_value = 0
        if self.current_position and self.ticker:
            if self.ticker.last and self.ticker.last > 0:
                position_value = self.current_position * self.ticker.last
            elif self.ticker.bid and self.ticker.ask:
                mid_price = (self.ticker.bid + self.ticker.ask) / 2
                position_value = self.current_position * mid_price
        
        logger.info(f"=== Status Update ===")
        logger.info(f"Account Value: ${account_value:,.0f}")
        logger.info(f"Position: {self.current_position} shares (${position_value:,.0f})")
        logger.info(f"Trades Today: {self.trades_today}")
        logger.info(f"Strategy Stats: {self.strategy.get_stats()}")
        
    async def shutdown(self):
        """Shutdown gracefully"""
        logger.info("Shutting down...")
        self.running = False
        
        # Cancel market data subscriptions
        if hasattr(self, 'ticker') and self.ticker:
            self.ib.cancelMktData(self.ticker)
            
        if hasattr(self, 'bars_subscription') and self.bars_subscription:
            self.ib.cancelRealTimeBars(self.bars_subscription)
            
        # Wait a moment for cleanup
        await asyncio.sleep(0.5)
            
        # Disconnect
        if self.ib.isConnected():
            self.ib.disconnect()
            
        logger.info("Shutdown complete")


async def main():
    parser = argparse.ArgumentParser(description='Live trading with backtest2 strategies')
    
    # Required arguments
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol to trade')
    parser.add_argument('--strategy', type=str, required=True, 
                       choices=['simple', 'enhanced', 'nick'], help='Strategy to use')
    
    # Optional arguments
    parser.add_argument('--trading-mode', type=str, default='paper',
                       choices=['paper', 'live'], help='Trading mode')
    parser.add_argument('--position-size', type=float, default=0.10,
                       help='Position size as fraction of equity')
    parser.add_argument('--max-position', type=float, default=10000,
                       help='Maximum position value')
    
    # Strategy parameters
    parser.add_argument('--sma-period', type=int, default=20, help='SMA period for simple strategy')
    parser.add_argument('--lookback-period', type=int, default=20, help='Lookback for Nick strategy')
    parser.add_argument('--rsi-threshold', type=int, default=55, help='RSI threshold for Nick strategy')
    parser.add_argument('--adx-threshold', type=int, default=30, help='ADX threshold for Nick strategy')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, 
               format="{time:HH:mm:ss} | {level} | {message}")
    
    # Create log file
    log_dir = Path("backtest2/logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"live_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_file, level="DEBUG")
    
    # Create strategy
    if args.strategy == 'simple':
        strategy = SimpleMomentumStrategy(args.symbol, sma_period=args.sma_period)
    elif args.strategy == 'enhanced':
        strategy = EnhancedMomentumStrategy(args.symbol)
    else:  # nick
        strategy = NickStrategy(
            args.symbol,
            lookback_period=args.lookback_period,
            rsi_threshold=args.rsi_threshold,
            adx_trend_threshold=args.adx_threshold,
            debug=args.debug
        )
    
    # Create trader
    trader = LiveTrader(
        strategy=strategy,
        symbol=args.symbol,
        position_size_pct=args.position_size,
        trading_mode=args.trading_mode,
        max_position_value=args.max_position
    )
    
    # Setup signal handlers
    shutdown_event = asyncio.Event()
    
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal")
        shutdown_event.set()
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Connection retry logic
    max_connect_retries = 3
    connect_retry_delay = 30
    
    for attempt in range(max_connect_retries):
        try:
            # Connect to IB
            logger.info(f"Connection attempt {attempt + 1}/{max_connect_retries}")
            await trader.connect()
            
            # Connection successful, break retry loop
            break
            
        except asyncio.TimeoutError:
            logger.error(f"Connection timeout on attempt {attempt + 1}")
            if attempt < max_connect_retries - 1:
                logger.info(f"Waiting {connect_retry_delay} seconds before retry...")
                await asyncio.sleep(connect_retry_delay)
            else:
                logger.error("Max connection attempts reached")
                return
        except Exception as e:
            logger.error(f"Connection error: {e}")
            if attempt < max_connect_retries - 1:
                logger.info(f"Waiting {connect_retry_delay} seconds before retry...")
                await asyncio.sleep(connect_retry_delay)
            else:
                logger.error("Max connection attempts reached")
                return
    
    try:
        # Check if strategy needs market data (SPY)
        if hasattr(strategy, 'needs_market_data') and strategy.needs_market_data():
            logger.info("Fetching SPY data for market context...")
            # For live trading, we might want to subscribe to SPY real-time data
            # For now, fetch recent historical data
            from backtest2.data_fetcher import DataFetcher
            fetcher = DataFetcher()
            
            # Retry SPY data fetch with better error handling
            spy_fetch_retries = 3
            for i in range(spy_fetch_retries):
                try:
                    spy_data = fetcher.fetch_historical_data('SPY', days=5, bar_size='5 mins')
                    if not spy_data.empty:
                        strategy.set_market_data('SPY', spy_data)
                        logger.info(f"Set SPY market data with {len(spy_data)} bars")
                        break
                    else:
                        logger.warning(f"Empty SPY data on attempt {i + 1}")
                except Exception as e:
                    logger.error(f"Error fetching SPY data: {e}")
                    if i < spy_fetch_retries - 1:
                        await asyncio.sleep(5)
                    else:
                        logger.warning("Continuing without SPY data")
        
        # Run trading loop with shutdown monitoring
        run_task = asyncio.create_task(trader.run())
        shutdown_task = asyncio.create_task(shutdown_event.wait())
        
        # Wait for either trading to stop or shutdown signal
        done, pending = await asyncio.wait(
            [run_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await trader.shutdown()


if __name__ == "__main__":
    # IMPORTANT: Start the event loop for ib_async
    util.startLoop()
    
    # Run main
    asyncio.run(main())