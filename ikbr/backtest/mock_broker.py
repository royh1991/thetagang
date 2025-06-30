"""
Mock Broker for Backtesting

Simulates realistic order execution with slippage, commission, and market impact.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from loguru import logger

from core.event_bus import EventBus, Event, EventTypes, get_event_bus
from core.order_manager import Signal, OrderInfo, OrderStatus
from core.market_data import TickData
from .mock_ib import MockIB


class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class Position:
    """Position tracking"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_price(self, price: float):
        """Update current price and unrealized PnL"""
        self.current_price = price
        self.unrealized_pnl = (price - self.avg_price) * self.quantity


@dataclass
class MockOrder:
    """Mock order for backtesting"""
    order_id: str
    signal: Signal
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0
    remaining_quantity: int = 0
    
    def __post_init__(self):
        self.remaining_quantity = self.signal.quantity


class MockBroker:
    """
    Simulates a broker for backtesting
    
    Features:
    - Realistic order execution with slippage
    - Commission calculation
    - Position tracking
    - P&L calculation
    - Market impact simulation
    """
    
    def __init__(self, initial_capital: float = 100000, 
                 commission_per_share: float = 0.01,
                 min_commission: float = 1.0,
                 slippage_bps: float = 5.0):  # 5 basis points default slippage
        
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.slippage_bps = slippage_bps / 10000  # Convert to decimal
        
        # Tracking
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, MockOrder] = {}
        self.order_history: List[MockOrder] = []
        self.current_prices: Dict[str, TickData] = {}
        
        # Performance metrics
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.num_trades = 0
        
        # Event bus
        self.event_bus = get_event_bus()
        
        # Order ID counter
        self._order_counter = 0
        
        # Create mock IB instance
        self._mock_ib = MockIB()
        
        # Set reference so MockIB can execute orders through this broker
        self._mock_ib._mock_broker = self
        
        # Initialize MockIB account values
        self._update_mock_ib_account_values()
        
    def get_next_order_id(self) -> str:
        """Generate unique order ID"""
        self._order_counter += 1
        return f"MOCK_{self._order_counter:06d}"
    
    def update_prices(self, tick_data: Dict[str, TickData]):
        """Update current market prices"""
        self.current_prices.update(tick_data)
        
        # Update position unrealized P&L
        for symbol, position in self.positions.items():
            if symbol in tick_data:
                position.update_price(tick_data[symbol].last)
        
        # Update MockIB prices
        price_dict = {}
        for symbol, tick in tick_data.items():
            price_dict[symbol] = {
                'last': tick.last,
                'bid': tick.bid,
                'ask': tick.ask,
                'high': tick.high,
                'low': tick.low,
                'close': tick.close,
                'volume': tick.volume
            }
        self._mock_ib.update_prices(price_dict)
        
        # Update MockIB account values to reflect current state
        self._update_mock_ib_account_values()
    
    async def submit_order(self, signal: Signal) -> Optional[OrderInfo]:
        """Submit an order for execution"""
        # Generate order ID
        order_id = self.get_next_order_id()
        
        # Validate order
        if not self._validate_order(signal):
            await self._emit_order_rejected(order_id, signal, "Order validation failed")
            return None
        
        # Create mock order
        mock_order = MockOrder(
            order_id=order_id,
            signal=signal,
            status=OrderStatus.PENDING
        )
        
        self.orders[order_id] = mock_order
        
        # Create order info
        order_info = OrderInfo(
            order_id=order_id,
            signal=signal,
            status=OrderStatus.PENDING,
            timestamp=datetime.now()
        )
        
        # Emit order submitted event
        await self._emit_order_submitted(order_info)
        
        # Process immediately if market order
        if signal.order_type == "MARKET":
            await self._execute_order(mock_order)
        
        return order_info
    
    def _validate_order(self, signal: Signal) -> bool:
        """Validate order before submission"""
        # Check if symbol has price data
        if signal.symbol not in self.current_prices:
            logger.warning(f"No price data for {signal.symbol}")
            return False
        
        # Check buying power
        if signal.is_buy:
            tick = self.current_prices[signal.symbol]
            required_cash = tick.ask * signal.quantity * 1.02  # Include buffer
            if required_cash > self.cash:
                logger.warning(f"Insufficient buying power: need ${required_cash:.2f}, have ${self.cash:.2f}")
                return False
        else:
            # Check if we have position to sell
            position = self.positions.get(signal.symbol)
            if not position or position.quantity < signal.quantity:
                available = position.quantity if position else 0
                logger.warning(f"Insufficient position: trying to sell {signal.quantity}, have {available}")
                return False
        
        return True
    
    async def _execute_order(self, mock_order: MockOrder):
        """Execute an order"""
        signal = mock_order.signal
        tick = self.current_prices.get(signal.symbol)
        
        if not tick:
            mock_order.status = OrderStatus.REJECTED
            await self._emit_order_rejected(mock_order.order_id, signal, "No market data")
            return
        
        # Calculate fill price with slippage
        if signal.is_buy:
            base_price = tick.ask
            slippage = base_price * self.slippage_bps * np.random.uniform(0.5, 1.5)
            fill_price = base_price + slippage
        else:
            base_price = tick.bid
            slippage = base_price * self.slippage_bps * np.random.uniform(0.5, 1.5)
            fill_price = base_price - slippage
        
        # Calculate commission
        commission = max(
            self.commission_per_share * signal.quantity,
            self.min_commission
        )
        
        # Update order
        mock_order.fill_price = fill_price
        mock_order.fill_time = tick.timestamp
        mock_order.commission = commission
        mock_order.slippage = slippage * signal.quantity
        mock_order.status = OrderStatus.FILLED
        mock_order.remaining_quantity = 0
        
        # Update cash
        if signal.is_buy:
            self.cash -= (fill_price * signal.quantity + commission)
        else:
            self.cash += (fill_price * signal.quantity - commission)
        
        # Update positions
        self._update_positions(signal, fill_price)
        
        # Track metrics
        self.total_commission += commission
        self.total_slippage += abs(slippage * signal.quantity)
        self.num_trades += 1
        
        # Move to history
        self.order_history.append(mock_order)
        del self.orders[mock_order.order_id]
        
        # Create order info with fill details
        order_info = OrderInfo(
            order_id=mock_order.order_id,
            signal=signal,
            status=OrderStatus.FILLED,
            timestamp=mock_order.fill_time,
            fill_price=fill_price,
            commission=commission
        )
        
        # Emit order filled event
        await self._emit_order_filled(order_info)
        
        # Update MockIB account values after execution
        self._update_mock_ib_account_values()
        
        logger.info(f"Executed {signal.action} {signal.quantity} {signal.symbol} @ ${fill_price:.2f}")
    
    def _update_positions(self, signal: Signal, fill_price: float):
        """Update positions after fill"""
        symbol = signal.symbol
        
        if signal.is_buy:
            if symbol in self.positions:
                # Add to existing position
                position = self.positions[symbol]
                total_cost = position.avg_price * position.quantity + fill_price * signal.quantity
                position.quantity += signal.quantity
                position.avg_price = total_cost / position.quantity
            else:
                # New position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=signal.quantity,
                    avg_price=fill_price
                )
        else:
            # Sell
            if symbol in self.positions:
                position = self.positions[symbol]
                
                # Calculate realized P&L
                realized_pnl = (fill_price - position.avg_price) * signal.quantity
                position.realized_pnl += realized_pnl
                
                # Update position
                position.quantity -= signal.quantity
                
                # Remove if fully closed
                if position.quantity == 0:
                    del self.positions[symbol]
    
    async def process_orders(self):
        """Process pending orders (limit, stop, etc.)"""
        for order_id, mock_order in list(self.orders.items()):
            if mock_order.status != OrderStatus.PENDING:
                continue
            
            signal = mock_order.signal
            tick = self.current_prices.get(signal.symbol)
            
            if not tick:
                continue
            
            # Check limit orders
            if signal.order_type == "LIMIT":
                if signal.is_buy and tick.ask <= signal.limit_price:
                    await self._execute_order(mock_order)
                elif not signal.is_buy and tick.bid >= signal.limit_price:
                    await self._execute_order(mock_order)
            
            # Check stop orders
            elif signal.order_type == "STOP":
                if signal.stop_price:
                    if signal.is_buy and tick.ask >= signal.stop_price:
                        await self._execute_order(mock_order)
                    elif not signal.is_buy and tick.bid <= signal.stop_price:
                        await self._execute_order(mock_order)
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            position.quantity * position.current_price 
            for position in self.positions.values()
        )
        return self.cash + positions_value
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        return self.positions.copy()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        portfolio_value = self.get_portfolio_value()
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        
        return {
            "portfolio_value": portfolio_value,
            "cash": self.cash,
            "total_return": total_return,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "num_trades": self.num_trades,
            "avg_commission": self.total_commission / max(self.num_trades, 1),
            "avg_slippage": self.total_slippage / max(self.num_trades, 1)
        }
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if order_id in self.orders:
            mock_order = self.orders[order_id]
            if mock_order.status == OrderStatus.PENDING:
                mock_order.status = OrderStatus.CANCELLED
                self.order_history.append(mock_order)
                del self.orders[order_id]
                
                # Emit cancelled event
                order_info = OrderInfo(
                    order_id=order_id,
                    signal=mock_order.signal,
                    status=OrderStatus.CANCELLED,
                    timestamp=datetime.now()
                )
                await self._emit_order_cancelled(order_info)
                return True
        return False
    
    # Event emission methods
    async def _emit_order_submitted(self, order_info: OrderInfo):
        """Emit order submitted event"""
        await self.event_bus.emit(Event(
            EventTypes.ORDER_SUBMITTED,
            {"order_info": order_info},
            source="MockBroker"
        ))
    
    async def _emit_order_filled(self, order_info: OrderInfo):
        """Emit order filled event"""
        await self.event_bus.emit(Event(
            EventTypes.ORDER_FILLED,
            {"order_info": order_info},
            source="MockBroker"
        ))
    
    async def _emit_order_rejected(self, order_id: str, signal: Signal, reason: str):
        """Emit order rejected event"""
        await self.event_bus.emit(Event(
            EventTypes.ORDER_REJECTED,
            {
                "order_id": order_id,
                "signal": signal,
                "reason": reason
            },
            source="MockBroker"
        ))
    
    async def _emit_order_cancelled(self, order_info: OrderInfo):
        """Emit order cancelled event"""
        await self.event_bus.emit(Event(
            EventTypes.ORDER_CANCELLED,
            {"order_info": order_info},
            source="MockBroker"
        ))
    
    def reset(self, initial_capital: Optional[float] = None):
        """Reset broker state"""
        self.cash = initial_capital or self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.order_history.clear()
        self.current_prices.clear()
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.num_trades = 0
        self._order_counter = 0
    
    def get_mock_ib(self):
        """Return mock IB connection for compatibility"""
        return self._mock_ib
    
    def _update_mock_ib_account_values(self):
        """Update MockIB account values to reflect current broker state"""
        # Calculate portfolio value
        portfolio_value = self.get_portfolio_value()
        
        # Calculate positions value with safety check for current_price
        positions_value = 0.0
        for position in self.positions.values():
            if position.current_price > 0:
                positions_value += position.quantity * position.current_price
            else:
                # Use average price if current price not set
                positions_value += position.quantity * position.avg_price
        
        # Update MockIB's account values
        self._mock_ib._account_values.update({
            'NetLiquidation': portfolio_value,
            'TotalCashValue': self.cash,
            'TotalCashBalance': self.cash,
            'BuyingPower': self.cash * 4,  # Assuming 4x margin
            'UnrealizedPnL': sum(p.unrealized_pnl for p in self.positions.values()),
            'RealizedPnL': sum(p.realized_pnl for p in self.positions.values()),
            'GrossPositionValue': positions_value
        })
        
        logger.debug(f"Updated MockIB account values - NetLiq: ${portfolio_value:.2f}, Cash: ${self.cash:.2f}")
    
    def _execute_order_fill(self, symbol: str, action: str, quantity: int, price: float, commission: float):
        """Execute an order fill from MockIB"""
        logger.info(f"Executing fill: {action} {quantity} {symbol} @ ${price:.2f}")
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        position = self.positions[symbol]
        
        # Calculate cost
        total_cost = quantity * price + commission
        
        if action == "BUY":
            # Check if we have enough cash
            if total_cost > self.cash:
                logger.warning(f"Insufficient cash for {action} {quantity} {symbol}: ${total_cost:.2f} > ${self.cash:.2f}")
                return
            
            # Update cash and position
            self.cash -= total_cost
            position.add_shares(quantity, price, commission)
        else:  # SELL
            # Update cash and position
            self.cash += (quantity * price - commission)
            position.remove_shares(quantity, price, commission)
        
        # Update metrics
        self.total_commission += commission
        self.num_trades += 1
        
        # Update MockIB account values
        self._update_mock_ib_account_values()
        
        logger.info(f"Fill executed. New position: {position.quantity} {symbol}, Cash: ${self.cash:.2f}")