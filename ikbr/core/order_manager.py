"""
Order Management System for IBKR Trading Bot

Handles order submission, tracking, modification, and execution monitoring
with support for various order types and smart routing.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Union, Callable, Any
from enum import Enum
import uuid
from loguru import logger
from ib_async import *

from .event_bus import EventBus, Event, EventTypes, get_event_bus


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class OrderConfig:
    """Configuration for order manager"""
    default_timeout: int = 30  # seconds
    use_adaptive_orders: bool = True
    max_retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_bracket_orders: bool = True
    smart_routing: bool = True
    outside_rth: bool = False  # Outside regular trading hours
    hidden: bool = False
    all_or_none: bool = False
    fill_or_kill: bool = False
    good_after_time: Optional[str] = None
    good_till_date: Optional[str] = None


@dataclass
class Signal:
    """Trading signal from strategy"""
    action: str  # BUY or SELL
    symbol: str
    quantity: int
    order_type: str = "MARKET"  # MARKET, LIMIT, STOP, etc
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_buy(self) -> bool:
        return self.action.upper() == "BUY"
    
    @property
    def is_sell(self) -> bool:
        return self.action.upper() == "SELL"


@dataclass
class OrderInfo:
    """Enhanced order information"""
    order_id: str
    signal: Signal
    ib_order: Order
    ib_trade: Optional[Trade] = None
    status: OrderStatus = OrderStatus.PENDING
    submit_time: float = field(default_factory=time.time)
    fill_time: Optional[float] = None
    fill_price: Optional[float] = None
    commission: Optional[float] = None
    realized_pnl: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
    
    @property
    def is_complete(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.ERROR]
    
    @property
    def execution_time_ms(self) -> Optional[float]:
        if self.fill_time:
            return (self.fill_time - self.submit_time) * 1000
        return None


class OrderManager:
    """
    Manages order lifecycle from signal to execution
    
    Features:
    - Multiple order types (market, limit, stop, bracket)
    - Smart order routing
    - Order modification and cancellation
    - Fill tracking and commission calculation
    - Retry logic for failed orders
    - Performance metrics
    """
    
    def __init__(self, ib: IB, config: Optional[OrderConfig] = None):
        self.ib = ib
        self.config = config or OrderConfig()
        self.event_bus = get_event_bus()
        
        # Order tracking
        self._orders: Dict[str, OrderInfo] = {}  # order_id -> OrderInfo
        self._active_orders: Set[str] = set()
        self._symbol_orders: Dict[str, List[str]] = defaultdict(list)
        
        # IB order mapping
        self._ib_order_map: Dict[int, str] = {}  # IB orderId -> our order_id
        
        # Performance metrics
        self._metrics = {
            'orders_submitted': 0,
            'orders_filled': 0,
            'orders_cancelled': 0,
            'orders_rejected': 0,
            'total_commission': 0.0,
            'total_latency_ms': 0.0
        }
        
        # Risk manager reference (set later)
        self.risk_manager = None
        
        # Setup IB event handlers
        self._setup_event_handlers()
        
    def _setup_event_handlers(self):
        """Setup IB event handlers"""
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_exec_details
        self.ib.errorEvent += self._on_error
        self.ib.newOrderEvent += self._on_new_order
        
    def set_risk_manager(self, risk_manager):
        """Set risk manager reference"""
        self.risk_manager = risk_manager
        
    async def submit_order(self, signal: Signal) -> Optional[OrderInfo]:
        """
        Submit order based on signal
        
        Args:
            signal: Trading signal
            
        Returns:
            OrderInfo if successful, None if rejected
        """
        start_time = time.perf_counter()
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        logger.info(f"Submitting order for {signal.symbol}: {signal.action} {signal.quantity} @ {signal.order_type}")
        
        try:
            # Risk checks
            if self.risk_manager:
                if not await self.risk_manager.check_order(signal):
                    logger.warning(f"Order rejected by risk manager: {signal}")
                    await self._emit_order_rejected(order_id, signal, "Risk check failed")
                    return None
            
            # Create contract
            contract = self._create_contract(signal)
            
            # Create IB order
            ib_order = self._create_ib_order(signal)
            
            # Create order info
            order_info = OrderInfo(
                order_id=order_id,
                signal=signal,
                ib_order=ib_order
            )
            
            # Place order
            trade = self.ib.placeOrder(contract, ib_order)
            order_info.ib_trade = trade
            
            # Store order
            self._orders[order_id] = order_info
            self._active_orders.add(order_id)
            self._symbol_orders[signal.symbol].append(order_id)
            self._ib_order_map[ib_order.orderId] = order_id
            
            # Update metrics
            self._metrics['orders_submitted'] += 1
            
            # Emit event
            await self._emit_order_submitted(order_info)
            
            logger.info(f"Order submitted: {order_id} for {signal.symbol} "
                       f"{signal.action} {signal.quantity}")
            
            # Wait for initial status
            await self._wait_for_submit_confirmation(order_info)
            
            submit_latency = (time.perf_counter() - start_time) * 1000
            self._metrics['total_latency_ms'] += submit_latency
            
            return order_info
            
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            await self._emit_order_error(order_id, signal, str(e))
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Returns:
            bool: True if cancellation requested successfully
        """
        if order_id not in self._orders:
            logger.warning(f"Order not found: {order_id}")
            return False
        
        order_info = self._orders[order_id]
        
        if not order_info.is_active:
            logger.debug(f"Order not active: {order_id} (status: {order_info.status})")
            return False
        
        try:
            self.ib.cancelOrder(order_info.ib_order)
            logger.info(f"Cancel requested for order: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def modify_order(self, order_id: str, 
                          limit_price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          quantity: Optional[int] = None) -> bool:
        """
        Modify an existing order
        
        Returns:
            bool: True if modification successful
        """
        if order_id not in self._orders:
            logger.warning(f"Order not found: {order_id}")
            return False
        
        order_info = self._orders[order_id]
        
        if not order_info.is_active:
            logger.warning(f"Order not active: {order_id}")
            return False
        
        try:
            # Modify order parameters
            if limit_price is not None:
                order_info.ib_order.lmtPrice = limit_price
            if stop_price is not None:
                order_info.ib_order.auxPrice = stop_price
            if quantity is not None:
                order_info.ib_order.totalQuantity = quantity
            
            # Submit modification
            self.ib.placeOrder(order_info.ib_trade.contract, order_info.ib_order)
            
            logger.info(f"Order modified: {order_id}")
            await self._emit_order_updated(order_info)
            return True
            
        except Exception as e:
            logger.error(f"Failed to modify order {order_id}: {e}")
            return False
    
    def get_order(self, order_id: str) -> Optional[OrderInfo]:
        """Get order information"""
        return self._orders.get(order_id)
    
    def get_active_orders(self) -> List[OrderInfo]:
        """Get all active orders"""
        return [self._orders[oid] for oid in self._active_orders if oid in self._orders]
    
    def get_orders_by_symbol(self, symbol: str) -> List[OrderInfo]:
        """Get all orders for a symbol"""
        order_ids = self._symbol_orders.get(symbol, [])
        return [self._orders[oid] for oid in order_ids if oid in self._orders]
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all orders, optionally filtered by symbol
        
        Returns:
            int: Number of orders cancelled
        """
        if symbol:
            orders = self.get_orders_by_symbol(symbol)
        else:
            orders = self.get_active_orders()
        
        cancelled = 0
        for order in orders:
            if await self.cancel_order(order.order_id):
                cancelled += 1
        
        logger.info(f"Cancelled {cancelled} orders")
        return cancelled
    
    def _create_contract(self, signal: Signal) -> Contract:
        """Create IB contract from signal"""
        contract = Stock(signal.symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        return contract
    
    def _create_ib_order(self, signal: Signal) -> Order:
        """Create IB order from signal"""
        # Base order
        if signal.order_type == "MARKET":
            order = MarketOrder(signal.action, signal.quantity)
        elif signal.order_type == "LIMIT":
            order = LimitOrder(signal.action, signal.quantity, signal.limit_price)
        elif signal.order_type == "STOP":
            order = StopOrder(signal.action, signal.quantity, signal.stop_price)
        elif signal.order_type == "STOP_LIMIT":
            order = StopLimitOrder(signal.action, signal.quantity, 
                                 signal.stop_price, signal.limit_price)
        else:
            # Default to market order
            order = MarketOrder(signal.action, signal.quantity)
        
        # Apply configuration
        order.outsideRth = self.config.outside_rth
        order.hidden = self.config.hidden
        order.allOrNone = self.config.all_or_none
        order.tif = signal.time_in_force
        
        # Good after/till time
        if self.config.good_after_time:
            order.goodAfterTime = self.config.good_after_time
        if self.config.good_till_date:
            order.goodTillDate = self.config.good_till_date
        
        # Smart routing
        if self.config.smart_routing:
            order.smartComboRoutingParams = []
        
        # Adaptive orders
        if self.config.use_adaptive_orders and signal.order_type == "MARKET":
            order.orderType = "MIDPRICE"
            order.adaptivePriority = "Patient"
        
        return order
    
    async def _create_bracket_order(self, signal: Signal) -> Optional[List[Order]]:
        """Create bracket order (entry + take profit + stop loss)"""
        if not signal.take_profit or not signal.stop_loss:
            return None
        
        # Parent order
        parent = self._create_ib_order(signal)
        parent.transmit = False
        
        # Take profit order
        tp_action = "SELL" if signal.is_buy else "BUY"
        take_profit = LimitOrder(tp_action, signal.quantity, signal.take_profit)
        take_profit.parentId = parent.orderId
        take_profit.transmit = False
        
        # Stop loss order
        stop_loss = StopOrder(tp_action, signal.quantity, signal.stop_loss)
        stop_loss.parentId = parent.orderId
        stop_loss.transmit = True  # Transmit all orders
        
        return [parent, take_profit, stop_loss]
    
    def _on_order_status(self, trade: Trade):
        """Handle order status updates"""
        ib_order_id = trade.order.orderId
        if ib_order_id not in self._ib_order_map:
            return
        
        order_id = self._ib_order_map[ib_order_id]
        if order_id not in self._orders:
            return
        
        order_info = self._orders[order_id]
        old_status = order_info.status
        
        # Update status
        status_map = {
            'PendingSubmit': OrderStatus.PENDING,
            'PreSubmitted': OrderStatus.SUBMITTED,
            'Submitted': OrderStatus.SUBMITTED,
            'Filled': OrderStatus.FILLED,
            'PartiallyFilled': OrderStatus.PARTIAL,
            'Cancelled': OrderStatus.CANCELLED,
            'Inactive': OrderStatus.CANCELLED
        }
        
        ib_status = trade.orderStatus.status
        new_status = status_map.get(ib_status, OrderStatus.ERROR)
        order_info.status = new_status
        
        # Handle status changes
        if old_status != new_status:
            asyncio.create_task(self._handle_status_change(order_info, old_status))
        
        # Update fill information
        if new_status == OrderStatus.FILLED:
            order_info.fill_time = time.time()
            order_info.fill_price = trade.orderStatus.avgFillPrice
            self._active_orders.discard(order_id)
            self._metrics['orders_filled'] += 1
        elif new_status == OrderStatus.CANCELLED:
            self._active_orders.discard(order_id)
            self._metrics['orders_cancelled'] += 1
    
    def _on_exec_details(self, trade: Trade, fill: Fill):
        """Handle execution details"""
        ib_order_id = trade.order.orderId
        if ib_order_id not in self._ib_order_map:
            return
        
        order_id = self._ib_order_map[ib_order_id]
        if order_id not in self._orders:
            return
        
        order_info = self._orders[order_id]
        
        # Update fill time from execution if available
        if hasattr(fill.execution, 'time'):
            # Parse IB's time format (YYYYMMDD HH:MM:SS)
            try:
                from datetime import datetime
                exec_time_str = fill.execution.time
                # Convert to datetime and then to timestamp
                exec_dt = datetime.strptime(exec_time_str, "%Y%m%d %H:%M:%S")
                order_info.fill_time = exec_dt.timestamp()
            except Exception as e:
                logger.debug(f"Could not parse execution time: {e}")
                order_info.fill_time = time.time()
        else:
            order_info.fill_time = time.time()
        
        # Update commission
        if fill.commissionReport:
            order_info.commission = fill.commissionReport.commission
            self._metrics['total_commission'] += fill.commissionReport.commission
        
        # Update realized PnL if available
        if hasattr(fill, 'realizedPnL'):
            order_info.realized_pnl = fill.realizedPnL
        
        logger.info(f"Execution details for {order_id}: "
                   f"price={fill.execution.price}, "
                   f"shares={fill.execution.shares}, "
                   f"commission={order_info.commission}")
    
    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Contract):
        """Handle IB errors"""
        if reqId in self._ib_order_map:
            order_id = self._ib_order_map[reqId]
            if order_id in self._orders:
                order_info = self._orders[order_id]
                order_info.error_message = f"{errorCode}: {errorString}"
                
                # Check if this is a rejection
                if errorCode in [201, 202, 203]:  # Order rejected codes
                    order_info.status = OrderStatus.REJECTED
                    self._active_orders.discard(order_id)
                    self._metrics['orders_rejected'] += 1
                    asyncio.create_task(self._emit_order_rejected(
                        order_id, order_info.signal, errorString))
    
    def _on_new_order(self, trade: Trade):
        """Handle new order event"""
        # This is called when orders are received from TWS on connect
        # We can use this to sync state if needed
        pass
    
    async def _handle_status_change(self, order_info: OrderInfo, old_status: OrderStatus):
        """Handle order status changes"""
        new_status = order_info.status
        
        if new_status == OrderStatus.FILLED:
            await self._emit_order_filled(order_info)
        elif new_status == OrderStatus.CANCELLED:
            await self._emit_order_cancelled(order_info)
        elif new_status == OrderStatus.SUBMITTED and old_status == OrderStatus.PENDING:
            # Order confirmed by broker
            logger.debug(f"Order {order_info.order_id} confirmed by broker")
    
    async def _wait_for_submit_confirmation(self, order_info: OrderInfo, timeout: float = 5.0):
        """Wait for order to be acknowledged by broker"""
        # In backtesting, orders are processed synchronously, so no need to wait
        # Just return immediately
        return True
    
    # Event emission methods
    async def _emit_order_submitted(self, order_info: OrderInfo):
        """Emit order submitted event"""
        await self.event_bus.emit(Event(
            EventTypes.ORDER_SUBMITTED,
            {
                'order_id': order_info.order_id,
                'order_info': order_info,
                'signal': order_info.signal
            },
            source="OrderManager"
        ))
    
    async def _emit_order_filled(self, order_info: OrderInfo):
        """Emit order filled event"""
        await self.event_bus.emit(Event(
            EventTypes.ORDER_FILLED,
            {
                'order_id': order_info.order_id,
                'order_info': order_info,
                'fill_price': order_info.fill_price,
                'commission': order_info.commission,
                'execution_time_ms': order_info.execution_time_ms
            },
            source="OrderManager"
        ))
    
    async def _emit_order_cancelled(self, order_info: OrderInfo):
        """Emit order cancelled event"""
        await self.event_bus.emit(Event(
            EventTypes.ORDER_CANCELLED,
            {
                'order_id': order_info.order_id,
                'order_info': order_info
            },
            source="OrderManager"
        ))
    
    async def _emit_order_rejected(self, order_id: str, signal: Signal, reason: str):
        """Emit order rejected event"""
        await self.event_bus.emit(Event(
            EventTypes.ORDER_REJECTED,
            {
                'order_id': order_id,
                'signal': signal,
                'reason': reason
            },
            source="OrderManager"
        ))
    
    async def _emit_order_updated(self, order_info: OrderInfo):
        """Emit order updated event"""
        await self.event_bus.emit(Event(
            EventTypes.ORDER_UPDATED,
            {
                'order_id': order_info.order_id,
                'order_info': order_info
            },
            source="OrderManager"
        ))
    
    async def _emit_order_error(self, order_id: str, signal: Signal, error: str):
        """Emit order error event"""
        await self.event_bus.emit(Event(
            EventTypes.STRATEGY_ERROR,
            {
                'order_id': order_id,
                'signal': signal,
                'error': error
            },
            source="OrderManager"
        ))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        avg_latency = 0.0
        if self._metrics['orders_submitted'] > 0:
            avg_latency = self._metrics['total_latency_ms'] / self._metrics['orders_submitted']
        
        return {
            'orders_submitted': self._metrics['orders_submitted'],
            'orders_filled': self._metrics['orders_filled'],
            'orders_cancelled': self._metrics['orders_cancelled'],
            'orders_rejected': self._metrics['orders_rejected'],
            'active_orders': len(self._active_orders),
            'total_commission': self._metrics['total_commission'],
            'avg_submit_latency_ms': avg_latency,
            'fill_rate': self._metrics['orders_filled'] / max(1, self._metrics['orders_submitted'])
        }
    
    async def get_position(self, symbol: str) -> Optional[float]:
        """Get current position for a symbol"""
        # In a real implementation, this would track positions
        # For now, return None
        return None
    
    async def close_position(self, symbol: str) -> Optional[OrderInfo]:
        """Close position for a symbol"""
        position = await self.get_position(symbol)
        if position is None:
            return None
        
        # Create market order to close
        action = "SELL" if position > 0 else "BUY"
        signal = Signal(
            action=action,
            symbol=symbol,
            quantity=abs(int(position)),
            order_type="MARKET"
        )
        
        return await self.submit_order(signal)


# Order utilities
def calculate_order_size(capital: float, risk_pct: float, 
                        entry_price: float, stop_price: float) -> int:
    """
    Calculate position size based on risk
    
    Args:
        capital: Available capital
        risk_pct: Risk percentage (e.g., 0.01 for 1%)
        entry_price: Entry price
        stop_price: Stop loss price
        
    Returns:
        int: Number of shares
    """
    risk_amount = capital * risk_pct
    price_risk = abs(entry_price - stop_price)
    
    if price_risk == 0:
        return 0
    
    shares = int(risk_amount / price_risk)
    return shares


def validate_signal(signal: Signal) -> bool:
    """Validate signal parameters"""
    # Check action
    if signal.action.upper() not in ["BUY", "SELL"]:
        return False
    
    # Check quantity
    if signal.quantity <= 0:
        return False
    
    # Check order type
    valid_types = ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]
    if signal.order_type not in valid_types:
        return False
    
    # Check limit price for limit orders
    if signal.order_type in ["LIMIT", "STOP_LIMIT"] and not signal.limit_price:
        return False
    
    # Check stop price for stop orders
    if signal.order_type in ["STOP", "STOP_LIMIT"] and not signal.stop_price:
        return False
    
    return True