"""
Trading strategies for the backtest framework
"""

# Import strategy classes for easy access
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy

__all__ = ['MeanReversionStrategy', 'MomentumStrategy']