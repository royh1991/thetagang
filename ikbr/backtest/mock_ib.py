"""
Mock IB-async implementation for backtesting

Provides a mock implementation of the IB-async interface
that works with the backtesting engine.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import asyncio
from loguru import logger

from ib_async import Contract, Stock, Option, Ticker, BarData


@dataclass
class MockTicker(Ticker):
    """Mock ticker that simulates IB ticker behavior"""
    def __init__(self, contract: Contract):
        super().__init__()
        self.contract = contract
        self.time = datetime.now()
        self.bid = float('nan')
        self.bidSize = 0
        self.ask = float('nan')
        self.askSize = 0
        self.last = float('nan')
        self.lastSize = 0
        self.volume = 0
        self.open = float('nan')
        self.high = float('nan')
        self.low = float('nan')
        self.close = float('nan')
        self.halted = False
        self.modelGreeks = None
    
    def marketPrice(self) -> float:
        """Get the market price (mid if no last)"""
        if self.last:
            return self.last
        elif self.bid and self.ask:
            return (self.bid + self.ask) / 2
        elif self.bid:
            return self.bid
        elif self.ask:
            return self.ask
        else:
            return None


class MockIB:
    """
    Mock IB connection that implements the ib_async interface
    for backtesting purposes
    """
    
    def __init__(self):
        self._connected = True
        self._next_con_id = 1000
        self._tickers: Dict[Contract, MockTicker] = {}
        self._current_prices: Dict[str, Dict[str, float]] = {}
        # Create mock event attributes
        class MockEvent:
            def __init__(self):
                self._handlers = []
            
            def __iadd__(self, handler):
                self._handlers.append(handler)
                return self
            
            def __isub__(self, handler):
                if handler in self._handlers:
                    self._handlers.remove(handler)
                return self
            
            def emit(self, *args):
                for handler in self._handlers:
                    handler(*args)
        
        # Initialize events
        self.orderStatusEvent = MockEvent()
        self.execDetailsEvent = MockEvent()
        self.errorEvent = MockEvent()
        self.positionEvent = MockEvent()
        self.accountValueEvent = MockEvent()
        self.pnlEvent = MockEvent()
        self.pendingTickersEvent = MockEvent()
        self.barUpdateEvent = MockEvent()
        self.newOrderEvent = MockEvent()
        
        # Mock account info
        self._account_values = {
            'NetLiquidation': 100000.0,
            'TotalCashValue': 100000.0,
            'TotalCashBalance': 100000.0,
            'BuyingPower': 400000.0,
            'UnrealizedPnL': 0.0,
            'RealizedPnL': 0.0
        }
        
        self._positions = {}
        self._orders = {}
        self._next_order_id = 1
        self._pending_order_fills = []
        
        logger.info("MockIB initialized for backtesting")
    
    def isConnected(self) -> bool:
        """Check if connected"""
        return self._connected
    
    def connect(self, host: str, port: int, clientId: int, timeout: float = 10):
        """Mock connect - always succeeds"""
        self._connected = True
        logger.info(f"MockIB connected to {host}:{port} with clientId={clientId}")
    
    def disconnect(self):
        """Mock disconnect"""
        self._connected = False
        logger.info("MockIB disconnected")
    
    def qualifyContracts(self, *contracts: Contract) -> List[Contract]:
        """
        Qualify contracts by adding conId
        This is essential for contracts to be hashable
        """
        qualified = []
        for contract in contracts:
            if not contract.conId:
                contract.conId = self._next_con_id
                self._next_con_id += 1
                
                # Set default values if missing
                if isinstance(contract, Stock):
                    if not contract.primaryExchange:
                        contract.primaryExchange = 'NASDAQ'
                    if not contract.secType:
                        contract.secType = 'STK'
                
                logger.debug(f"Qualified {contract.symbol} with conId={contract.conId}")
            
            qualified.append(contract)
        
        return qualified[0] if len(qualified) == 1 else qualified
    
    def reqMktData(self, contract: Contract, genericTickList: str = '', 
                   snapshot: bool = False, regulatorySnapshot: bool = False) -> Ticker:
        """
        Request market data - returns a mock ticker
        """
        # Ensure contract is qualified
        if not contract.conId:
            self.qualifyContracts(contract)
        
        # Create or get ticker
        if contract not in self._tickers:
            ticker = MockTicker(contract)
            self._tickers[contract] = ticker
            
            # Set initial prices if available
            if contract.symbol in self._current_prices:
                prices = self._current_prices[contract.symbol]
                ticker.last = prices.get('last')
                ticker.bid = prices.get('bid', ticker.last * 0.999 if ticker.last else None)
                ticker.ask = prices.get('ask', ticker.last * 1.001 if ticker.last else None)
                ticker.open = prices.get('open', ticker.last)
                ticker.high = prices.get('high', ticker.last)
                ticker.low = prices.get('low', ticker.last)
                ticker.close = prices.get('close', ticker.last)
                ticker.volume = prices.get('volume', 0)
            
            logger.debug(f"Created ticker for {contract.symbol}")
        
        return self._tickers[contract]
    
    def cancelMktData(self, contract: Contract):
        """Cancel market data subscription"""
        if contract in self._tickers:
            del self._tickers[contract]
            logger.debug(f"Cancelled market data for {contract.symbol}")
    
    def reqHistoricalData(self, contract: Contract, endDateTime: str, 
                         durationStr: str, barSizeSetting: str, 
                         whatToShow: str, useRTH: bool = True, 
                         formatDate: int = 1, keepUpToDate: bool = False,
                         chartOptions: List = None) -> List[BarData]:
        """Mock historical data request - returns empty list in backtest"""
        logger.debug(f"Mock historical data requested for {contract.symbol}")
        return []
    
    async def reqHistoricalDataAsync(self, contract: Contract, endDateTime: str, 
                                    durationStr: str, barSizeSetting: str, 
                                    whatToShow: str, useRTH: bool = True, 
                                    formatDate: int = 1, keepUpToDate: bool = False,
                                    chartOptions: List = None) -> List[BarData]:
        """Mock async historical data request - returns empty list in backtest"""
        logger.debug(f"Mock async historical data requested for {contract.symbol}")
        return []
    
    def positions(self) -> List:
        """Get current positions"""
        return list(self._positions.values())
    
    def pendingTickers(self) -> List[MockTicker]:
        """Get all pending tickers"""
        return list(self._tickers.values())
    
    def accountValues(self, account: str = '') -> List:
        """Get account values"""
        from types import SimpleNamespace
        values = []
        for tag, value in self._account_values.items():
            # Create object with attributes instead of dict
            av = SimpleNamespace(
                tag=tag,
                value=str(value),
                currency='USD',
                account=account or 'DU123456'
            )
            values.append(av)
        return values
    
    def accountSummary(self, account: str = '') -> List:
        """Get account summary"""
        return self.accountValues(account)
    
    def placeOrder(self, contract: Contract, order: Any) -> Any:
        """Place an order"""
        if not contract.conId:
            self.qualifyContracts(contract)
        
        # Assign order ID if not set
        if not hasattr(order, 'orderId') or not order.orderId:
            order.orderId = self._next_order_id
            self._next_order_id += 1
        
        # Store order
        self._orders[order.orderId] = {
            'contract': contract,
            'order': order,
            'status': 'PreSubmitted'
        }
        
        logger.info(f"Placed order {order.orderId}: {order.action} {order.totalQuantity} {contract.symbol}")
        
        # Return trade object
        from ib_async import Trade, OrderStatus as IBOrderStatus
        trade = Trade(contract=contract, order=order)
        
        # Create order status
        order_status = IBOrderStatus(
            orderId=order.orderId,
            status='PreSubmitted',
            filled=0,
            remaining=order.totalQuantity,
            avgFillPrice=0.0,
            permId=0,
            parentId=0,
            lastFillPrice=0.0,
            clientId=0,
            whyHeld='',
            mktCapPrice=0.0
        )
        trade.orderStatus = order_status
        
        # Store trade for processing
        self._orders[order.orderId]['trade'] = trade
        
        # In backtesting, execute orders immediately
        # We'll call this synchronously from process_pending_orders
        self._pending_order_fills.append(trade)
        
        return trade
    
    async def _simulate_order_submission(self, trade):
        """Simulate order submission and execution"""
        logger.debug(f"Simulating order submission for {trade.contract.symbol}")
        
        # Update status to Submitted
        trade.orderStatus.status = 'Submitted'
        self.orderStatusEvent.emit(trade)
        
        # Check if we should fill the order
        contract = trade.contract
        order = trade.order
        
        # Get current price
        current_price = None
        logger.debug(f"Looking for price data for {contract.symbol}. Available symbols: {list(self._current_prices.keys())}")
        if contract.symbol in self._current_prices:
            current_price = self._current_prices[contract.symbol].get('last')
            logger.info(f"Current price for {contract.symbol}: {current_price}")
        
        if not current_price:
            # No price available, reject order
            logger.warning(f"No market data available for {contract.symbol}")
            self.errorEvent.emit(order.orderId, 201, "No market data available", contract)
            trade.orderStatus.status = 'Cancelled'
            self.orderStatusEvent.emit(trade)
            return
        
        # For market orders, fill immediately
        order_type = getattr(order, 'orderType', 'MKT')
        logger.info(f"Order details - type: {order_type}, action: {order.action}, qty: {order.totalQuantity}")
        # MIDPRICE is an adaptive market order
        if order_type in ['MKT', 'MARKET', 'MIDPRICE']:
            logger.info(f"Filling market order for {contract.symbol} at {current_price}")
            await self._fill_order(trade, current_price)
        # For limit orders, check if price allows fill
        elif order_type == 'LMT':
            if (order.action == 'BUY' and order.lmtPrice >= current_price) or \
               (order.action == 'SELL' and order.lmtPrice <= current_price):
                logger.info(f"Filling limit order for {contract.symbol} at {order.lmtPrice}")
                await self._fill_order(trade, order.lmtPrice)
    
    async def _fill_order(self, trade, fill_price):
        """Fill an order"""
        from ib_async import Fill, Execution, CommissionReport
        import time
        
        order = trade.order
        contract = trade.contract
        
        logger.info(f"Filling order {order.orderId}: {order.action} {order.totalQuantity} {contract.symbol} @ {fill_price}")
        
        # Update order status
        trade.orderStatus.status = 'Filled'
        trade.orderStatus.filled = order.totalQuantity
        trade.orderStatus.remaining = 0
        trade.orderStatus.avgFillPrice = fill_price
        trade.orderStatus.lastFillPrice = fill_price
        
        # Emit order status event
        self.orderStatusEvent.emit(trade)
        
        # Get historical timestamp if available
        exec_time = datetime.now()
        if contract.symbol in self._current_prices:
            price_data = self._current_prices[contract.symbol]
            if isinstance(price_data, dict) and 'timestamp' in price_data:
                exec_time = datetime.fromtimestamp(price_data['timestamp'])
        
        # Create execution
        execution = Execution(
            execId=f"exec_{order.orderId}_{time.time()}",
            time=exec_time.strftime("%Y%m%d %H:%M:%S"),
            acctNumber='DU123456',
            exchange='SMART',
            side=order.action,
            shares=order.totalQuantity,
            price=fill_price,
            permId=order.permId if hasattr(order, 'permId') else 0,
            orderId=order.orderId,
            cumQty=order.totalQuantity,
            avgPrice=fill_price,
            lastLiquidity=1
        )
        
        # Create commission report
        commission = CommissionReport(
            execId=execution.execId,
            commission=0.01 * order.totalQuantity,  # $0.01 per share
            currency='USD',
            realizedPNL=0.0,
            yield_=0.0,
            yieldRedemptionDate=0
        )
        
        # Create fill
        fill = Fill(
            contract=contract,
            execution=execution,
            commissionReport=commission,
            time=datetime.now()
        )
        
        # Emit execution details event
        self.execDetailsEvent.emit(trade, fill)
        
        # Update order status in our records
        if order.orderId in self._orders:
            self._orders[order.orderId]['status'] = 'Filled'
    
    def cancelOrder(self, order: Any):
        """Cancel an order"""
        if hasattr(order, 'orderId') and order.orderId in self._orders:
            self._orders[order.orderId]['status'] = 'Cancelled'
            logger.info(f"Cancelled order {order.orderId}")
    
    def reqExecutions(self) -> List:
        """Get executions"""
        return []
    
    def reqOpenOrders(self) -> List:
        """Get open orders"""
        open_orders = []
        for order_id, order_info in self._orders.items():
            if order_info['status'] not in ['Filled', 'Cancelled']:
                open_orders.append(order_info)
        return open_orders
    
    def update_prices(self, prices: Dict[str, Dict[str, float]]):
        """
        Update current market prices
        Called by backtesting engine to simulate market data
        """
        self._current_prices.update(prices)
        
        # Update all active tickers
        for contract, ticker in self._tickers.items():
            if contract.symbol in prices:
                price_data = prices[contract.symbol]
                
                # Update ticker with new prices
                ticker.last = price_data.get('last')
                ticker.bid = price_data.get('bid', ticker.last * 0.999 if ticker.last else None)
                ticker.ask = price_data.get('ask', ticker.last * 1.001 if ticker.last else None)
                ticker.open = ticker.last  # MockTicker doesn't track open
                ticker.high = price_data.get('high', ticker.last)
                ticker.low = price_data.get('low', ticker.last)
                ticker.close = price_data.get('close', ticker.last)
                ticker.volume = price_data.get('volume', 0)
                
                # Use historical timestamp if provided, otherwise current time
                if 'timestamp' in price_data:
                    # Convert float timestamp to datetime
                    ticker.time = datetime.fromtimestamp(price_data['timestamp'])
                else:
                    ticker.time = datetime.now()
        
        # Trigger pending tickers event with all updated tickers
        if self._tickers:
            # Pass tickers as a list - the handler will iterate through them
            self.pendingTickersEvent.emit(self._tickers.values())
    
    def sleep(self, secs: float):
        """IB-specific sleep that doesn't block event loop"""
        # In mock mode, we don't actually sleep
        pass
    
    async def sleepAsync(self, secs: float):
        """Async version of sleep"""
        await asyncio.sleep(secs)
    
    def run(self):
        """Run the IB event loop - no-op in mock mode"""
        pass
    
    async def process_pending_orders(self):
        """Process any pending order fills"""
        if self._pending_order_fills:
            logger.info(f"Processing {len(self._pending_order_fills)} pending order fills")
        while self._pending_order_fills:
            trade = self._pending_order_fills.pop(0)
            logger.debug(f"About to simulate order for {trade.contract.symbol}")
            await self._simulate_order_submission(trade)
    
    def __getattr__(self, name):
        """Catch-all for any missing IB methods"""
        def mock_method(*args, **kwargs):
            logger.debug(f"Mock method called: {name}")
            return None
        return mock_method


# For backward compatibility
MockIBConnection = MockIB