"""
Core infrastructure for HFT-grade IBKR trading system
"""

from .event_bus import EventBus, Event

__all__ = ['EventBus', 'Event']