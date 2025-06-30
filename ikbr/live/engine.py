"""
Live Trading Engine

Real-time trading engine that uses actual IB connection for paper or production trading.
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Type
from loguru import logger

from ib_async import IB, util

from core.event_bus import EventBus, Event, EventTypes, get_event_bus
from core.market_data import MarketDataManager
from core.order_manager import OrderManager
from core.risk_manager import RiskManager, RiskLimits
from strategies.base_strategy import BaseStrategy, StrategyConfig


@dataclass
class LiveConfig:
    """Live trading configuration"""
    trading_mode: str = "paper"  # paper or prod
    initial_capital: Optional[float] = None  # Use actual account value
    use_adaptive_orders: bool = True
    log_trades: bool = True
    enable_notifications: bool = True
    safety_check_prod: bool = True  # Require explicit confirmation for prod mode


class LiveEngine:
    """
    Live trading engine using real IB connection
    
    Features:
    - Real-time market data
    - Actual order execution (paper or production)
    - Live risk management
    - Real account tracking
    """
    
    def __init__(self, config: LiveConfig):
        self.config = config
        
        # Initialize components
        self.event_bus = get_event_bus()
        self.ib = IB()
        
        # Managers will be initialized after connection
        self.market_data: Optional[MarketDataManager] = None
        self.order_manager: Optional[OrderManager] = None
        self.risk_manager: Optional[RiskManager] = None
        
        # Strategy tracking
        self.strategies: List[BaseStrategy] = []
        self._running = False
        
        # Performance tracking
        self._start_time: Optional[datetime] = None
        self._initial_account_value: Optional[float] = None
        
    async def connect(self):
        """Connect to IB Gateway/TWS"""
        # Determine port based on trading mode
        if self.config.trading_mode == "paper":
            port = int(os.getenv('IB_GATEWAY_PORT', 4102))
            logger.warning("🏦 PAPER TRADING MODE - Using simulated account")
        else:
            port = int(os.getenv('IB_GATEWAY_PORT_LIVE', 4101))
            logger.warning("💰 PRODUCTION MODE - Using REAL MONEY account")
            
            if self.config.safety_check_prod:
                # Safety check for production mode
                logger.warning("⚠️  YOU ARE ABOUT TO TRADE WITH REAL MONEY ⚠️")
                confirmation = input("Type 'CONFIRM REAL TRADING' to proceed: ")
                if confirmation != "CONFIRM REAL TRADING":
                    logger.error("Production trading not confirmed. Exiting.")
                    raise ValueError("Production trading not confirmed")
        
        # Connect to IB
        logger.info(f"Connecting to IB on port {port}...")
        try:
            await self.ib.connectAsync('localhost', port, clientId=1)
            logger.info("✅ Connected to Interactive Brokers")
            
            # Log account info
            account_values = self.ib.accountValues()
            for av in account_values:
                if av.tag == 'NetLiquidation':
                    self._initial_account_value = float(av.value)
                    logger.info(f"Account Value: ${self._initial_account_value:,.2f}")
                    
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            raise
    
    async def initialize(self):
        """Initialize trading components"""
        # Start event bus
        await self.event_bus.start()
        
        # Create managers with real IB connection
        self.market_data = MarketDataManager(self.ib)
        self.order_manager = OrderManager(self.ib)
        
        # Create risk limits appropriate for live trading
        live_risk_limits = RiskLimits(
            # Conservative limits for live trading
            max_position_size=10000,  # Max $10k per position
            max_positions=5,          # Max 5 concurrent positions
            max_daily_loss=1000,      # Max $1k daily loss
            max_drawdown_pct=0.05,    # Max 5% drawdown
            min_order_interval=1.0,   # 1 second between orders (keep for live)
            max_daily_trades=50,      # Max 50 trades per day
        )
        
        # Override limits if in production mode (even more conservative)
        if self.config.trading_mode == "prod":
            live_risk_limits.max_position_size = 5000
            live_risk_limits.max_daily_loss = 500
            live_risk_limits.max_drawdown_pct = 0.02
            
        self.risk_manager = RiskManager(self.ib, live_risk_limits)
        
        # Link components
        self.order_manager.set_risk_manager(self.risk_manager)
        
        # Start managers
        await self.market_data.start()
        await self.risk_manager.start()
        
        # Subscribe to important events
        self.event_bus.subscribe(EventTypes.ORDER_FILLED, self._on_order_filled)
        self.event_bus.subscribe(EventTypes.ORDER_REJECTED, self._on_order_rejected)
        self.event_bus.subscribe(EventTypes.STRATEGY_ERROR, self._on_strategy_error)
        
        logger.info("Live trading engine initialized")
    
    def add_strategy(self, strategy_class: Type[BaseStrategy], 
                    config: StrategyConfig):
        """Add a strategy to trade live"""
        # Create strategy instance
        strategy = strategy_class(
            config=config,
            market_data=self.market_data,
            order_manager=self.order_manager,
            risk_manager=self.risk_manager
        )
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {config.name} for symbols: {config.symbols}")
    
    async def run(self):
        """Run live trading"""
        logger.info("="*60)
        logger.info(f"Starting LIVE {'PAPER' if self.config.trading_mode == 'paper' else 'PRODUCTION'} Trading")
        logger.info(f"Time: {datetime.now()}")
        logger.info(f"Strategies: {[s.config.name for s in self.strategies]}")
        logger.info("="*60)
        
        self._running = True
        self._start_time = datetime.now()
        
        try:
            # Initialize if not already done
            if not self.market_data:
                await self.initialize()
            
            # Start all strategies
            for strategy in self.strategies:
                await strategy.start()
                
            logger.info("All strategies started. Trading is LIVE.")
            
            # Keep running until stopped
            while self._running:
                # Log periodic status updates
                await self._log_status()
                await asyncio.sleep(60)  # Status every minute
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Error in live trading: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop live trading"""
        logger.info("Stopping live trading...")
        self._running = False
        
        # Stop all strategies
        for strategy in self.strategies:
            await strategy.stop()
        
        # Log final performance
        await self._log_final_performance()
        
        # Stop managers
        if self.market_data:
            await self.market_data.stop()
        if self.risk_manager:
            await self.risk_manager.stop()
        
        # Disconnect from IB
        if self.ib.isConnected():
            self.ib.disconnect()
        
        # Stop event bus
        await self.event_bus.stop()
        
        logger.info("Live trading stopped")
    
    async def _log_status(self):
        """Log current trading status"""
        if not self._running:
            return
            
        try:
            # Get current account value
            account_values = self.ib.accountValues()
            current_value = None
            for av in account_values:
                if av.tag == 'NetLiquidation':
                    current_value = float(av.value)
                    break
            
            if current_value and self._initial_account_value:
                pnl = current_value - self._initial_account_value
                pnl_pct = (pnl / self._initial_account_value) * 100
                
                # Get positions
                positions = self.ib.positions()
                
                logger.info(f"📊 Status Update: "
                           f"P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%) | "
                           f"Positions: {len(positions)} | "
                           f"Account: ${current_value:,.2f}")
                
        except Exception as e:
            logger.error(f"Error logging status: {e}")
    
    async def _log_final_performance(self):
        """Log final performance metrics"""
        if not self._start_time or not self._initial_account_value:
            return
            
        try:
            # Get final account value
            account_values = self.ib.accountValues()
            final_value = None
            for av in account_values:
                if av.tag == 'NetLiquidation':
                    final_value = float(av.value)
                    break
            
            if final_value:
                total_pnl = final_value - self._initial_account_value
                total_pnl_pct = (total_pnl / self._initial_account_value) * 100
                
                runtime = datetime.now() - self._start_time
                
                logger.info("="*60)
                logger.info("FINAL PERFORMANCE SUMMARY")
                logger.info(f"Runtime: {runtime}")
                logger.info(f"Initial Value: ${self._initial_account_value:,.2f}")
                logger.info(f"Final Value: ${final_value:,.2f}")
                logger.info(f"Total P&L: ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")
                
                # Get trade metrics from order manager
                metrics = self.order_manager.get_metrics()
                logger.info(f"Orders Submitted: {metrics['orders_submitted']}")
                logger.info(f"Orders Filled: {metrics['orders_filled']}")
                logger.info(f"Fill Rate: {metrics['fill_rate']:.1%}")
                logger.info(f"Total Commission: ${metrics['total_commission']:.2f}")
                logger.info("="*60)
                
        except Exception as e:
            logger.error(f"Error logging final performance: {e}")
    
    async def _on_order_filled(self, event: Event):
        """Handle order filled events"""
        order_info = event.data.get('order_info')
        if order_info and self.config.log_trades:
            logger.info(f"✅ Order Filled: {order_info.signal.symbol} "
                       f"{order_info.signal.action} {order_info.signal.quantity} "
                       f"@ ${order_info.fill_price:.2f}")
    
    async def _on_order_rejected(self, event: Event):
        """Handle order rejected events"""
        data = event.data
        logger.warning(f"❌ Order Rejected: {data.get('signal')} - {data.get('reason')}")
    
    async def _on_strategy_error(self, event: Event):
        """Handle strategy errors"""
        error = event.data.get('error')
        logger.error(f"⚠️ Strategy Error: {error}")