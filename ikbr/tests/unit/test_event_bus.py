"""
Unit tests for EventBus - High-performance async event dispatcher
"""

import asyncio
import time
import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.event_bus import EventBus, Event, EventTypes, get_event_bus


class TestEventBus:
    """Test suite for EventBus functionality"""
    
    @pytest_asyncio.fixture
    async def event_bus(self):
        """Create a fresh event bus for each test"""
        bus = EventBus("test")
        await bus.start()
        yield bus
        await bus.stop()
    
    @pytest.mark.asyncio
    async def test_basic_event_emission(self, event_bus):
        """Test basic event emission and handling"""
        received_events = []
        
        def handler(event: Event):
            received_events.append(event)
        
        # Subscribe to event
        event_bus.subscribe("test.event", handler)
        
        # Emit event
        test_event = Event("test.event", {"value": 42})
        await event_bus.emit(test_event, wait=True)
        
        # Verify event was received
        assert len(received_events) == 1
        assert received_events[0].data["value"] == 42
    
    @pytest.mark.asyncio
    async def test_async_handler(self, event_bus):
        """Test async event handlers"""
        received_events = []
        
        async def async_handler(event: Event):
            await asyncio.sleep(0.001)  # Simulate async work
            received_events.append(event)
        
        event_bus.subscribe("test.async", async_handler)
        
        test_event = Event("test.async", {"async": True})
        await event_bus.emit(test_event, wait=True)
        
        assert len(received_events) == 1
        assert received_events[0].data["async"] is True
    
    @pytest.mark.asyncio
    async def test_multiple_handlers(self, event_bus):
        """Test multiple handlers for same event"""
        results = []
        
        def handler1(event: Event):
            results.append(1)
        
        def handler2(event: Event):
            results.append(2)
        
        async def handler3(event: Event):
            results.append(3)
        
        event_bus.subscribe("test.multi", handler1)
        event_bus.subscribe("test.multi", handler2)
        event_bus.subscribe("test.multi", handler3)
        
        await event_bus.emit(Event("test.multi", {}), wait=True)
        
        assert len(results) == 3
        assert set(results) == {1, 2, 3}
    
    @pytest.mark.asyncio
    async def test_priority_handling(self, event_bus):
        """Test handler priority execution order"""
        execution_order = []
        
        def low_priority(event: Event):
            execution_order.append("low")
        
        def medium_priority(event: Event):
            execution_order.append("medium")
        
        def high_priority(event: Event):
            execution_order.append("high")
        
        # Subscribe with different priorities
        event_bus.subscribe("test.priority", low_priority, priority=1)
        event_bus.subscribe("test.priority", medium_priority, priority=5)
        event_bus.subscribe("test.priority", high_priority, priority=10)
        
        await event_bus.emit(Event("test.priority", {}), wait=True)
        
        # Should execute in priority order (high to low)
        assert execution_order == ["high", "medium", "low"]
    
    @pytest.mark.asyncio
    async def test_error_isolation(self, event_bus):
        """Test that errors in one handler don't affect others"""
        results = []
        
        def failing_handler(event: Event):
            raise Exception("Handler error")
        
        def working_handler(event: Event):
            results.append("success")
        
        event_bus.subscribe("test.error", failing_handler)
        event_bus.subscribe("test.error", working_handler)
        
        await event_bus.emit(Event("test.error", {}), wait=True)
        
        # Working handler should still execute
        assert "success" in results
        
        # Check error metrics
        metrics = event_bus.get_metrics()
        assert metrics['errors'] == 1
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events"""
        call_count = 0
        
        def handler(event: Event):
            nonlocal call_count
            call_count += 1
        
        event_bus.subscribe("test.unsub", handler)
        
        # First emission
        await event_bus.emit(Event("test.unsub", {}), wait=True)
        assert call_count == 1
        
        # Unsubscribe
        event_bus.unsubscribe("test.unsub", handler)
        
        # Second emission should not call handler
        await event_bus.emit(Event("test.unsub", {}), wait=True)
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, event_bus):
        """Test performance metrics collection"""
        event_bus.reset_metrics()
        
        async def slow_handler(event: Event):
            await asyncio.sleep(0.02)  # 20ms delay
        
        event_bus.subscribe("test.perf", slow_handler)
        
        # Emit multiple events
        for i in range(5):
            await event_bus.emit(Event("test.perf", {"index": i}), wait=True)
        
        metrics = event_bus.get_metrics()
        assert metrics['events_emitted'] == 5
        assert metrics['events_processed'] == 5
        assert metrics['avg_latency_ms'] > 20.0  # Should be > 20ms due to slow handler
    
    @pytest.mark.asyncio
    async def test_concurrent_emissions(self, event_bus):
        """Test concurrent event emissions"""
        received_events = []
        lock = asyncio.Lock()
        
        async def handler(event: Event):
            async with lock:
                received_events.append(event.data["id"])
            await asyncio.sleep(0.001)  # Simulate work
        
        event_bus.subscribe("test.concurrent", handler)
        
        # Emit multiple events concurrently
        tasks = []
        for i in range(100):
            event = Event("test.concurrent", {"id": i})
            tasks.append(event_bus.emit(event))
        
        await asyncio.gather(*tasks)
        
        # Wait for processing
        await asyncio.sleep(0.2)
        
        # All events should be processed
        assert len(received_events) == 100
        assert set(received_events) == set(range(100))
    
    @pytest.mark.asyncio
    async def test_event_age(self, event_bus):
        """Test event age calculation"""
        event = Event("test.age", {})
        original_timestamp = event.timestamp
        
        # Wait a bit
        await asyncio.sleep(0.05)  # 50ms
        
        age_ms = event.age_ms
        assert age_ms >= 50.0
        assert age_ms < 100.0  # Should be close to 50ms
    
    @pytest.mark.asyncio
    async def test_queue_based_processing(self, event_bus):
        """Test queue-based async processing"""
        processed_events = []
        
        async def handler(event: Event):
            processed_events.append(event.data["seq"])
        
        event_bus.subscribe("test.queue", handler)
        
        # Emit events without waiting
        for i in range(10):
            await event_bus.emit(Event("test.queue", {"seq": i}), wait=False)
        
        # Wait for queue processing
        await asyncio.sleep(0.1)
        
        assert len(processed_events) == 10
        # Events should be processed in order
        assert processed_events == list(range(10))
    
    @pytest.mark.asyncio
    async def test_global_event_bus(self):
        """Test global event bus singleton"""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        
        assert bus1 is bus2  # Should be same instance
        assert bus1.name == "global"
    
    @pytest.mark.asyncio
    async def test_standard_event_types(self, event_bus):
        """Test standard event types for trading system"""
        market_events = []
        order_events = []
        
        def market_handler(event: Event):
            market_events.append(event)
        
        def order_handler(event: Event):
            order_events.append(event)
        
        # Subscribe to different event types
        event_bus.subscribe(EventTypes.TICK, market_handler)
        event_bus.subscribe(EventTypes.ORDER_FILLED, order_handler)
        
        # Emit different events
        await event_bus.emit(Event(EventTypes.TICK, {"symbol": "SPY", "price": 450.0}), wait=True)
        await event_bus.emit(Event(EventTypes.ORDER_FILLED, {"orderId": 123}), wait=True)
        
        assert len(market_events) == 1
        assert len(order_events) == 1
        assert market_events[0].data["symbol"] == "SPY"
        assert order_events[0].data["orderId"] == 123


class TestEventBusPerformance:
    """Performance tests for EventBus"""
    
    @pytest.mark.asyncio
    async def test_high_throughput(self):
        """Test event bus can handle high throughput"""
        bus = EventBus("perf_test")
        await bus.start()
        
        events_received = 0
        
        def fast_handler(event: Event):
            nonlocal events_received
            events_received += 1
        
        bus.subscribe("perf.test", fast_handler)
        
        # Measure throughput
        start_time = time.perf_counter()
        
        # Emit 10,000 events
        for i in range(10000):
            await bus.emit(Event("perf.test", {"seq": i}), wait=False)
        
        # Wait for processing
        while events_received < 10000 and time.perf_counter() - start_time < 5.0:
            await asyncio.sleep(0.01)
        
        elapsed = time.perf_counter() - start_time
        throughput = events_received / elapsed
        
        await bus.stop()
        
        assert events_received == 10000
        assert throughput > 1000  # Should handle > 1000 events/second
        print(f"Throughput: {throughput:.0f} events/second")
    
    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test event processing latency"""
        bus = EventBus("latency_test")
        await bus.start()
        
        latencies = []
        
        async def latency_handler(event: Event):
            latency = event.age_ms
            latencies.append(latency)
        
        bus.subscribe("latency.test", latency_handler)
        
        # Emit events and measure latency
        for i in range(100):
            await bus.emit(Event("latency.test", {"seq": i}), wait=True)
        
        await bus.stop()
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 10.0  # Average should be < 10ms
        assert max_latency < 50.0  # Max should be < 50ms
        print(f"Average latency: {avg_latency:.2f}ms, Max: {max_latency:.2f}ms")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])