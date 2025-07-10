"""
Performance metrics calculation for backtesting
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime


class PerformanceMetrics:
    """Calculate various performance metrics for backtest results"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of trading periods per year
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        
        if excess_returns.std() == 0:
            return 0.0
        
        return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Dict[str, Any]:
        """
        Calculate maximum drawdown
        
        Args:
            equity_curve: Series of portfolio values
            
        Returns:
            Dictionary with drawdown metrics
        """
        if len(equity_curve) < 2:
            return {'max_drawdown_pct': 0.0, 'max_drawdown_duration': 0}
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        
        # Calculate drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i in range(len(drawdown)):
            if drawdown.iloc[i] < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                drawdown_start = None
                current_duration = 0
        
        return {
            'max_drawdown_pct': abs(max_dd) * 100,
            'max_drawdown_duration': max_duration
        }
    
    @staticmethod
    def calculate_win_rate_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate win rate and related metrics
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with win rate metrics
        """
        if not trades:
            return {
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0
            }
        
        # Filter trades with P&L
        trades_with_pnl = [t for t in trades if t.get('pnl') is not None]
        
        if not trades_with_pnl:
            return {
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0
            }
        
        wins = [t['pnl'] for t in trades_with_pnl if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in trades_with_pnl if t['pnl'] <= 0]
        
        win_rate = len(wins) / len(trades_with_pnl) if trades_with_pnl else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }
    
    @staticmethod
    def calculate_returns_metrics(equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate return-based metrics
        
        Args:
            equity_curve: DataFrame with 'timestamp' and 'total_value' columns
            
        Returns:
            Dictionary with return metrics
        """
        if len(equity_curve) < 2:
            return {
                'total_return_pct': 0.0,
                'annualized_return_pct': 0.0,
                'volatility_pct': 0.0,
                'calmar_ratio': 0.0
            }
        
        # Calculate returns
        returns = equity_curve['total_value'].pct_change().dropna()
        
        # Total return
        total_return = (equity_curve['total_value'].iloc[-1] - equity_curve['total_value'].iloc[0]) / equity_curve['total_value'].iloc[0]
        
        # Calculate time period in years
        time_period = (equity_curve['timestamp'].iloc[-1] - equity_curve['timestamp'].iloc[0]).total_seconds() / (365.25 * 24 * 3600)
        
        # Annualized return
        annualized_return = (1 + total_return) ** (1 / time_period) - 1 if time_period > 0 else 0
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Max drawdown for Calmar ratio
        max_dd = PerformanceMetrics.calculate_max_drawdown(equity_curve['total_value'])
        
        # Calmar ratio (annualized return / max drawdown)
        calmar_ratio = annualized_return / (max_dd['max_drawdown_pct'] / 100) if max_dd['max_drawdown_pct'] > 0 else 0
        
        return {
            'total_return_pct': total_return * 100,
            'annualized_return_pct': annualized_return * 100,
            'volatility_pct': volatility * 100,
            'calmar_ratio': calmar_ratio
        }
    
    @staticmethod
    def calculate_all_metrics(backtest_results: Dict[str, Any], equity_curve: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all performance metrics
        
        Args:
            backtest_results: Results from backtest engine
            equity_curve: Equity curve DataFrame
            
        Returns:
            Dictionary with all metrics
        """
        # Get trades
        trades = backtest_results.get('trades', [])
        
        # Calculate returns if we have equity curve
        if not equity_curve.empty and 'total_value' in equity_curve.columns:
            returns = equity_curve['total_value'].pct_change().dropna()
            
            # Calculate metrics
            sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns)
            max_dd = PerformanceMetrics.calculate_max_drawdown(equity_curve['total_value'])
            returns_metrics = PerformanceMetrics.calculate_returns_metrics(equity_curve)
        else:
            sharpe = 0.0
            max_dd = {'max_drawdown_pct': 0.0, 'max_drawdown_duration': 0}
            returns_metrics = {
                'total_return_pct': backtest_results.get('total_return_pct', 0.0),
                'annualized_return_pct': 0.0,
                'volatility_pct': 0.0,
                'calmar_ratio': 0.0
            }
        
        # Win rate metrics
        win_metrics = PerformanceMetrics.calculate_win_rate_metrics(trades)
        
        # Combine all metrics
        all_metrics = {
            'sharpe_ratio': sharpe,
            **max_dd,
            **returns_metrics,
            **win_metrics,
            'total_trades': len(trades),
            'avg_trades_per_day': len(trades) / backtest_results.get('days', 1)
        }
        
        return all_metrics
    
    @staticmethod
    def format_metrics(metrics: Dict[str, Any]) -> str:
        """
        Format metrics for display
        
        Args:
            metrics: Dictionary of metrics
            
        Returns:
            Formatted string
        """
        output = []
        output.append("=== Performance Metrics ===")
        output.append(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        output.append(f"Annualized Return: {metrics.get('annualized_return_pct', 0):.2f}%")
        output.append(f"Volatility: {metrics.get('volatility_pct', 0):.2f}%")
        output.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        output.append(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
        output.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        output.append("")
        output.append("=== Trading Statistics ===")
        output.append(f"Total Trades: {metrics.get('total_trades', 0)}")
        output.append(f"Win Rate: {metrics.get('win_rate', 0):.1%}")
        output.append(f"Average Win: ${metrics.get('avg_win', 0):.2f}")
        output.append(f"Average Loss: ${metrics.get('avg_loss', 0):.2f}")
        output.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        output.append(f"Expectancy: ${metrics.get('expectancy', 0):.2f}")
        
        return "\n".join(output)