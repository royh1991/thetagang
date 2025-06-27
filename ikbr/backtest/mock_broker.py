"""
Mock Broker for Backtesting

Simulates broker functionality including order execution,
position tracking, and commission calculation.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np
from loguru import logger

from core.order_manager import OrderInfo, OrderStatus, Signal
from core.market_data import TickData


class FillModel(Enum):
    """Order fill models"""
    MARKET = "market"  # Fill at current market price
    LIMIT = "limit"    # Fill only if limit price met
    REALISTIC = "realistic"  # Add slippage and partial fills


@dataclass
class MockPosition:
    """Position in mock broker"""
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class MockBroker:
    """
    Mock broker for backtesting
    
    Features:
    - Realistic order execution with slippage
    - Commission calculation
    - Position tracking
    - Margin simulation
    """
    
    def __init__(self, config):
        self.config = config
        self.fill_model = FillModel.REALISTIC
        
        # Account state
        self.cash: float = 0.0
        self.initial_capital: float = 0.0
        self.positions: Dict[str, MockPosition] = {}
        self.pending_orders: List[OrderInfo] = []
        
        # Market data
        self.current_prices: Dict[str, TickData] = {}
        
        # Execution tracking
        self.filled_orders: List[OrderInfo] = []
        self.order_id_counter: int = 1
        
        # IB mock
        self._mock_ib = None
    
    def initialize(self, initial_capital: float):
        """Initialize broker with capital"""
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions.clear()
        self.pending_orders.clear()
        self.filled_orders.clear()
        logger.info(f"Mock broker initialized with ${initial_capital:,.2f}")
    
    def get_mock_ib(self):
        """Get mock IB instance for managers"""
        if not self._mock_ib:
            self._mock_ib = MockIB(self)
        return self._mock_ib
    
    def update_prices(self, tick_data: Dict[str, TickData]):
        """Update current market prices"""
        self.current_prices.update(tick_data)
        
        # Update position values
        for symbol, position in self.positions.items():
            if symbol in tick_data:
                current_price = tick_data[symbol].last
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_cost) * position.quantity
    
    def submit_order(self, order_info: OrderInfo) -> bool:
        """Submit an order to the mock broker"""
        # Basic validation
        signal = order_info.signal
        
        # Check available cash for buy orders
        if signal.is_buy:
            required_cash = self._calculate_required_cash(signal)
            if required_cash > self.cash:
                logger.warning(f"Insufficient cash for order: required=${required_cash:.2f}, "
                             f"available=${self.cash:.2f}")
                order_info.status = OrderStatus.REJECTED
                order_info.error_message = "Insufficient funds"
                return False
        
        # Check position for sell orders
        if signal.is_sell:
            position = self.positions.get(signal.symbol)
            if not position or position.quantity < signal.quantity:
                logger.warning(f"Insufficient position for sell order")
                order_info.status = OrderStatus.REJECTED
                order_info.error_message = "Insufficient position"
                return False
        
        # Accept order
        order_info.status = OrderStatus.SUBMITTED
        self.pending_orders.append(order_info)
        logger.debug(f"Order submitted: {signal.symbol} {signal.action} {signal.quantity}")
        return True
    
    def cancel_order(self, order_info: OrderInfo) -> bool:
        """Cancel a pending order"""
        if order_info in self.pending_orders:
            self.pending_orders.remove(order_info)
            order_info.status = OrderStatus.CANCELLED
            logger.debug(f"Order cancelled: {order_info.order_id}")
            return True
        return False
    
    async def process_orders(self):
        """Process pending orders with current market data"""
        filled_orders = []
        
        for order_info in self.pending_orders[:]:  # Copy list to allow modification
            signal = order_info.signal
            
            if signal.symbol not in self.current_prices:
                continue
            
            tick = self.current_prices[signal.symbol]
            
            # Check if order should be filled
            should_fill, fill_price = self._check_fill(order_info, tick)
            
            if should_fill:
                # Apply slippage
                if self.fill_model == FillModel.REALISTIC:
                    slippage = self.config.slippage_pct * fill_price
                    if signal.is_buy:
                        fill_price += slippage
                    else:
                        fill_price -= slippage
                
                # Execute fill
                self._execute_fill(order_info, fill_price)
                filled_orders.append(order_info)
                self.pending_orders.remove(order_info)
        
        return filled_orders
    
    def _check_fill(self, order_info: OrderInfo, tick: TickData) -> tuple[bool, float]:
        """Check if order should be filled"""
        signal = order_info.signal
        
        if signal.order_type == "MARKET":
            # Market orders always fill
            if signal.is_buy:
                return True, tick.ask or tick.last
            else:
                return True, tick.bid or tick.last
        
        elif signal.order_type == "LIMIT":
            # Limit orders fill if price is favorable
            if signal.is_buy and tick.ask <= signal.limit_price:
                return True, min(tick.ask, signal.limit_price)
            elif signal.is_sell and tick.bid >= signal.limit_price:
                return True, max(tick.bid, signal.limit_price)
        
        elif signal.order_type == "STOP":
            # Stop orders trigger and convert to market
            if signal.is_buy and tick.last >= signal.stop_price:
                return True, tick.ask or tick.last
            elif signal.is_sell and tick.last <= signal.stop_price:
                return True, tick.bid or tick.last
        
        return False, 0.0
    
    def _execute_fill(self, order_info: OrderInfo, fill_price: float):
        """Execute order fill"""
        signal = order_info.signal
        
        # Calculate commission
        commission = signal.quantity * self.config.commission_per_share
        
        # Update order info
        order_info.status = OrderStatus.FILLED
        order_info.fill_price = fill_price
        order_info.fill_time = time.time()
        order_info.commission = commission
        
        # Update positions
        if signal.is_buy:
            self._add_position(signal.symbol, signal.quantity, fill_price)
            self.cash -= (fill_price * signal.quantity + commission)
        else:
            pnl = self._reduce_position(signal.symbol, signal.quantity, fill_price)
            self.cash += (fill_price * signal.quantity - commission)
            order_info.realized_pnl = pnl
        
        self.filled_orders.append(order_info)
        
        logger.info(f"Order filled: {signal.symbol} {signal.action} {signal.quantity} "
                   f"@ ${fill_price:.2f}, commission=${commission:.2f}")
    
    def _add_position(self, symbol: str, quantity: int, price: float):
        """Add to or create position"""
        if symbol in self.positions:
            position = self.positions[symbol]
            total_cost = position.avg_cost * position.quantity + price * quantity
            position.quantity += quantity
            position.avg_cost = total_cost / position.quantity
        else:
            self.positions[symbol] = MockPosition(
                symbol=symbol,
                quantity=quantity,
                avg_cost=price
            )
    
    def _reduce_position(self, symbol: str, quantity: int, price: float) -> float:
        """Reduce position and calculate P&L"""
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        
        # Calculate realized P&L
        pnl = (price - position.avg_cost) * quantity
        position.realized_pnl += pnl
        
        # Reduce position
        position.quantity -= quantity
        
        # Remove if position is closed
        if position.quantity == 0:
            del self.positions[symbol]
        
        return pnl
    
    def _calculate_required_cash(self, signal: Signal) -> float:
        """Calculate cash required for order"""
        if signal.symbol not in self.current_prices:
            return float('inf')
        
        tick = self.current_prices[signal.symbol]
        
        if signal.order_type == "MARKET":
            price = tick.ask or tick.last
        elif signal.order_type == "LIMIT":
            price = signal.limit_price
        else:
            price = tick.last
        
        return price * signal.quantity + signal.quantity * self.config.commission_per_share
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value
    
    def get_positions(self) -> Dict[str, MockPosition]:
        """Get current positions"""
        return self.positions.copy()
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        portfolio_value = self.get_portfolio_value()
        positions_value = sum(p.market_value for p in self.positions.values())
        
        return {
            'cash': self.cash,
            'portfolio_value': portfolio_value,
            'positions_value': positions_value,
            'unrealized_pnl': sum(p.unrealized_pnl for p in self.positions.values()),
            'realized_pnl': sum(p.realized_pnl for p in self.positions.values()),
            'initial_capital': self.initial_capital,
            'total_return': (portfolio_value / self.initial_capital - 1) * 100
        }


class MockIB:
    """Mock IB API for backtesting"""
    
    def __init__(self, broker: MockBroker):
        self.broker = broker
        self.pendingTickersEvent = MockEvent()
        self.orderStatusEvent = MockEvent()
        self.execDetailsEvent = MockEvent()
        self.errorEvent = MockEvent()
        self.newOrderEvent = MockEvent()
    
    def connect(self, *args, **kwargs):
        """Mock connection"""
        pass
    
    def disconnect(self):
        """Mock disconnect"""
        pass
    
    def qualifyContracts(self, contract):
        """Mock qualify contracts"""
        pass
    
    def placeOrder(self, contract, order):
        """Mock place order"""
        # Create order info
        from ikbr.core.order_manager import Signal, OrderInfo
        
        signal = Signal(
            action=order.action,
            symbol=contract.symbol,
            quantity=order.totalQuantity,
            order_type=order.orderType,
            limit_price=getattr(order, 'lmtPrice', None),
            stop_price=getattr(order, 'auxPrice', None)
        )
        
        order_info = OrderInfo(
            order_id=str(self.broker.order_id_counter),
            signal=signal,
            ib_order=order
        )
        
        self.broker.order_id_counter += 1
        
        # Submit to broker
        if self.broker.submit_order(order_info):
            # Create mock trade object
            trade = MockTrade(order, order_info)
            return trade
        else:
            # Trigger error event
            self.errorEvent.emit(order.orderId, 201, "Order rejected", contract)
            return None
    
    def cancelOrder(self, order):
        """Mock cancel order"""
        # Find order in pending orders
        for order_info in self.broker.pending_orders:
            if order_info.ib_order == order:
                self.broker.cancel_order(order_info)
                break
    
    def reqMktData(self, contract, *args, **kwargs):
        """Mock market data request"""
        return MockTicker(contract)
    
    def cancelMktData(self, contract):
        """Mock cancel market data"""
        pass
    
    def positions(self):
        """Mock positions"""
        positions = []
        for symbol, pos in self.broker.positions.items():
            mock_pos = type('Position', (), {
                'contract': type('Contract', (), {'symbol': symbol}),
                'position': pos.quantity,
                'marketValue': pos.market_value,
                'unrealizedPNL': pos.unrealized_pnl,
                'realizedPNL': pos.realized_pnl,
                'avgCost': pos.avg_cost,
                'marketPrice': pos.market_value / pos.quantity if pos.quantity else 0
            })
            positions.append(mock_pos)
        return positions
    
    def accountValues(self):
        """Mock account values"""
        values = []
        account_info = self.broker.get_account_info()
        
        # Create mock account values
        for tag, value in [
            ('TotalCashBalance', account_info['cash']),
            ('NetLiquidation', account_info['portfolio_value']),
            ('UnrealizedPnL', account_info['unrealized_pnl']),
            ('RealizedPnL', account_info['realized_pnl'])
        ]:
            mock_av = type('AccountValue', (), {
                'tag': tag,
                'value': str(value),
                'currency': 'USD'
            })
            values.append(mock_av)
        
        return values
    
    async def reqHistoricalDataAsync(self, contract, *args, **kwargs):
        """Mock historical data request"""
        # Return empty list for backtesting
        return []


class MockEvent:
    """Mock event for IB API compatibility"""
    
    def __init__(self):
        self.handlers = []
    
    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self
    
    def emit(self, *args, **kwargs):
        for handler in self.handlers:
            handler(*args, **kwargs)


class MockTrade:
    """Mock trade object"""
    
    def __init__(self, order, order_info):
        self.order = order
        self.contract = type('Contract', (), {'symbol': order_info.signal.symbol})
        self.orderStatus = type('OrderStatus', (), {
            'status': 'Submitted',
            'filled': 0,
            'remaining': order.totalQuantity,
            'avgFillPrice': 0.0
        })


class MockTicker:
    """Mock ticker object"""
    
    def __init__(self, contract):
        self.contract = contract
        self.bid = float('nan')
        self.ask = float('nan')
        self.last = float('nan')
        self.bidSize = 0
        self.askSize = 0
        self.volume = 0
        self.high = float('nan')
        self.low = float('nan')
        self.close = float('nan')