"""
High-performance async event bus for HFT trading system

This module implements a low-latency event dispatcher that allows
decoupled components to communicate efficiently.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from weakref import WeakSet
import traceback
from loguru import logger


@dataclass
class Event:
    """Base event class with timing information"""
    event_type: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    
    @property
    def age_ms(self) -> float:
        """Age of event in milliseconds"""
        return (time.time() - self.timestamp) * 1000


class EventBus:
    """
    High-performance async event bus for trading system
    
    Features:
    - Async event emission
    - Priority handling
    - Performance metrics
    - Error isolation
    - Weak references to prevent memory leaks
    """
    
    def __init__(self, name: str = "main"):
        self.name = name
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._async_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._subscribers: WeakSet = WeakSet()
        self._metrics = {
            'events_emitted': 0,
            'events_processed': 0,
            'errors': 0,
            'total_latency_ms': 0.0
        }
        self._running = True
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the event bus worker"""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._process_events())
            logger.info(f"EventBus '{self.name}' started")
    
    async def stop(self):
        """Stop the event bus gracefully"""
        self._running = False
        if self._worker_task:
            await self._event_queue.put(None)  # Sentinel value
            await self._worker_task
            self._worker_task = None
        logger.info(f"EventBus '{self.name}' stopped")
    
    def subscribe(self, event_type: str, handler: Callable, priority: int = 0):
        """
        Subscribe to an event type
        
        Args:
            event_type: Type of event to subscribe to
            handler: Callback function (can be sync or async)
            priority: Higher priority handlers are called first
        """
        if asyncio.iscoroutinefunction(handler):
            handlers = self._async_handlers[event_type]
        else:
            handlers = self._handlers[event_type]
        
        # Insert handler based on priority
        handlers.append((priority, handler))
        handlers.sort(key=lambda x: x[0], reverse=True)
        
        self._subscribers.add(handler)
        logger.debug(f"Handler {handler.__name__} subscribed to {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from an event type"""
        for handlers_dict in [self._handlers, self._async_handlers]:
            if event_type in handlers_dict:
                handlers_dict[event_type] = [
                    (p, h) for p, h in handlers_dict[event_type] 
                    if h != handler
                ]
        logger.debug(f"Handler {handler.__name__} unsubscribed from {event_type}")
    
    async def emit(self, event: Event, wait: bool = False):
        """
        Emit an event asynchronously
        
        Args:
            event: Event to emit
            wait: If True, wait for all handlers to complete
        """
        start_time = time.perf_counter()
        self._metrics['events_emitted'] += 1
        
        if wait:
            # Process immediately and wait
            await self._process_event(event)
        else:
            # Queue for async processing
            await self._event_queue.put(event)
        
        emit_latency = (time.perf_counter() - start_time) * 1000
        if emit_latency > 1.0:  # Log if emission takes > 1ms
            logger.warning(f"Slow event emission: {event.event_type} took {emit_latency:.2f}ms")
    
    async def _process_events(self):
        """Worker coroutine to process events from queue"""
        while self._running:
            try:
                event = await self._event_queue.get()
                if event is None:  # Sentinel value
                    break
                    
                await self._process_event(event)
                
            except Exception as e:
                logger.error(f"Error in event processing worker: {e}")
                self._metrics['errors'] += 1
    
    async def _process_event(self, event: Event):
        """Process a single event"""
        start_time = time.perf_counter()
        
        # Get all handlers for this event type
        sync_handlers = [(p, h) for p, h in self._handlers.get(event.event_type, [])]
        async_handlers = [(p, h) for p, h in self._async_handlers.get(event.event_type, [])]
        
        # Execute sync handlers
        for priority, handler in sync_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in sync handler {handler.__name__}: {e}")
                logger.error(traceback.format_exc())
                self._metrics['errors'] += 1
        
        # Execute async handlers concurrently
        if async_handlers:
            tasks = []
            for priority, handler in async_handlers:
                tasks.append(self._call_async_handler(handler, event))
            
            # Wait for all async handlers to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    handler_name = async_handlers[i][1].__name__
                    logger.error(f"Error in async handler {handler_name}: {result}")
                    self._metrics['errors'] += 1
        
        # Update metrics
        self._metrics['events_processed'] += 1
        latency = (time.perf_counter() - start_time) * 1000
        self._metrics['total_latency_ms'] += latency
        
        if latency > 10.0:  # Log if processing takes > 10ms
            logger.warning(f"Slow event processing: {event.event_type} took {latency:.2f}ms")
    
    async def _call_async_handler(self, handler: Callable, event: Event):
        """Call an async handler with error handling"""
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Error in async handler {handler.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        avg_latency = 0.0
        if self._metrics['events_processed'] > 0:
            avg_latency = self._metrics['total_latency_ms'] / self._metrics['events_processed']
        
        return {
            'events_emitted': self._metrics['events_emitted'],
            'events_processed': self._metrics['events_processed'],
            'events_pending': self._event_queue.qsize(),
            'errors': self._metrics['errors'],
            'avg_latency_ms': avg_latency,
            'subscribers': len(self._subscribers)
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self._metrics = {
            'events_emitted': 0,
            'events_processed': 0,
            'errors': 0,
            'total_latency_ms': 0.0
        }


# Global event bus instance
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create the global event bus"""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus("global")
    return _global_event_bus


# Event type constants
class EventTypes:
    """Standard event types for the trading system"""
    # Market data events
    TICK = "market.tick"
    BAR = "market.bar"
    QUOTE = "market.quote"
    TRADE = "market.trade"
    
    # Order events
    ORDER_SUBMITTED = "order.submitted"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"
    ORDER_UPDATED = "order.updated"
    
    # Position events
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_UPDATED = "position.updated"
    
    # Strategy events
    SIGNAL_GENERATED = "strategy.signal"
    STRATEGY_ERROR = "strategy.error"
    
    # Risk events
    RISK_LIMIT_BREACHED = "risk.limit_breached"
    RISK_CHECK_FAILED = "risk.check_failed"
    
    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    CONNECTION_LOST = "system.connection_lost"
    CONNECTION_RESTORED = "system.connection_restored"