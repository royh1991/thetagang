#!/usr/bin/env python3
"""
IBKR Trading Bot - Main Entry Point
Optimized for low latency trading with ib_async
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from ib_async import *
from loguru import logger

# Load environment variables
load_dotenv()


class IBKRBot:
    """Main bot class for IBKR trading"""
    
    def __init__(self):
        self.ib = IB()
        self.account_id = os.getenv('ACCOUNT_ID')
        self.host = os.getenv('IB_GATEWAY_HOST', 'localhost')
        
        # Use correct port based on trading mode
        trading_mode = os.getenv('TRADING_MODE', 'paper')
        if trading_mode == 'paper':
            self.port = int(os.getenv('IB_GATEWAY_PORT', 4002))
        else:
            self.port = int(os.getenv('IB_GATEWAY_PORT_LIVE', 4001))
            
        self.client_id = 1
        self.connected = False
        
        # Configure logging
        logger.remove()
        logger.add(
            sys.stderr,
            level=os.getenv('LOG_LEVEL', 'INFO'),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        )
        
    async def connect(self) -> bool:
        """Connect to IB Gateway"""
        try:
            logger.info(f"Connecting to IB Gateway at {self.host}:{self.port}")
            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=20
            )
            
            # Wait for connection to stabilize
            await asyncio.sleep(1)
            
            if self.ib.isConnected():
                self.connected = True
                logger.success("Connected to IB Gateway")
                
                # Request market data type
                market_data_type = os.getenv('MARKET_DATA_TYPE', 'DELAYED')
                if market_data_type == 'DELAYED':
                    self.ib.reqMarketDataType(3)  # Delayed data
                elif market_data_type == 'FROZEN':
                    self.ib.reqMarketDataType(2)  # Frozen data
                else:
                    self.ib.reqMarketDataType(1)  # Real-time data
                
                # Log account info
                await self._log_account_info()
                return True
            else:
                logger.error("Failed to connect to IB Gateway")
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    async def _log_account_info(self):
        """Log account information"""
        try:
            # Get account summary
            account_summary = await self.ib.accountSummaryAsync()
            logger.info(f"Account ID: {self.account_id}")
            
            for item in account_summary:
                if item.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower']:
                    logger.info(f"{item.tag}: ${float(item.value):,.2f}")
                    
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
    
    async def get_positions(self):
        """Get current positions"""
        try:
            positions = self.ib.positions()  # This is synchronous in ib_async
            if positions:
                logger.info(f"Current positions: {len(positions)}")
                for pos in positions:
                    logger.info(f"  {pos.contract.symbol}: {pos.position} @ ${pos.avgCost:.2f}")
            else:
                logger.info("No open positions")
            return positions
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    async def get_market_data(self, symbol: str, exchange: str = 'SMART', currency: str = 'USD'):
        """Get market data for a symbol"""
        try:
            # Create contract
            contract = Stock(symbol, exchange, currency)
            
            # Qualify the contract to get conId
            self.ib.qualifyContracts(contract)
            
            # Request market data
            ticker = self.ib.reqMktData(contract, '', False, False)
            
            # Use ib.sleep instead of asyncio.sleep to work with the IB event loop
            self.ib.sleep(2)
            
            if ticker.last:
                logger.info(f"{symbol}: Last=${ticker.last:.2f}, Bid=${ticker.bid:.2f}, Ask=${ticker.ask:.2f}")
            else:
                logger.warning(f"No market data received for {symbol} - Delayed data may need to be requested")
                
            # Cancel market data subscription
            self.ib.cancelMktData(contract)
                
            return ticker
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    async def place_order(self, contract, order):
        """Place an order"""
        try:
            # Validate order
            trade = self.ib.placeOrder(contract, order)
            
            # Wait for order to be placed
            self.ib.sleep(1)
            
            logger.info(f"Order placed: {order.action} {order.totalQuantity} {contract.symbol} @ {order.orderType}")
            return trade
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    async def run_strategy(self):
        """Main strategy execution loop"""
        logger.info("Starting strategy execution...")
        
        while self.connected:
            try:
                # Example: Get positions every 30 seconds
                await self.get_positions()
                
                # Example: Get market data for SPY
                await self.get_market_data('SPY')
                
                # Add your strategy logic here
                # ...
                
                # Sleep before next iteration
                self.ib.sleep(30)
                
            except Exception as e:
                logger.error(f"Strategy error: {e}")
                self.ib.sleep(5)
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down...")
        if self.connected:
            self.ib.disconnect()
        logger.info("Shutdown complete")


async def main():
    """Main entry point"""
    bot = IBKRBot()
    
    # Connect to IB Gateway
    if not await bot.connect():
        logger.error("Failed to establish connection. Exiting.")
        return
    
    try:
        # Run the bot
        await bot.run_strategy()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await bot.shutdown()


if __name__ == "__main__":
    # Run the bot
    asyncio.run(main())