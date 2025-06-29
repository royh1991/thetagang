"""
Backtest Report Generator

Generates comprehensive reports with analytics, charts, and metrics.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import json

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logger.warning("Matplotlib/Seaborn not installed. Plotting disabled.")

from .engine import BacktestResult


class ReportGenerator:
    """
    Generates detailed backtest reports
    
    Features:
    - Performance metrics summary
    - Trade analysis
    - Risk metrics
    - Equity curves and drawdown charts
    - Monthly/yearly returns
    - Trade distribution analysis
    """
    
    def __init__(self, output_dir: str = "backtest/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        if HAS_PLOTTING:
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")
    
    def generate_report(self, result: BacktestResult, 
                       strategy_name: str = "Strategy",
                       save_plots: bool = True) -> Dict:
        """Generate comprehensive backtest report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"{strategy_name}_{timestamp}"
        report_dir = self.output_dir / report_name
        report_dir.mkdir(exist_ok=True)
        
        logger.info(f"Generating report: {report_name}")
        
        # Calculate all metrics
        result.calculate_metrics()
        
        # Generate report sections
        report = {
            "metadata": self._generate_metadata(strategy_name),
            "performance_summary": self._generate_performance_summary(result),
            "trade_analysis": self._generate_trade_analysis(result),
            "risk_metrics": self._generate_risk_metrics(result),
            "monthly_returns": self._generate_monthly_returns(result),
            "yearly_returns": self._generate_yearly_returns(result)
        }
        
        # Generate plots if available
        if HAS_PLOTTING and save_plots:
            self._generate_plots(result, report_dir)
        
        # Save report as JSON
        report_file = report_dir / "report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save detailed trade log
        if not result.trades.empty:
            result.trades.to_csv(report_dir / "trades.csv", index=False)
        
        # Save equity curve
        if not result.equity_curve.empty:
            result.equity_curve.to_csv(report_dir / "equity_curve.csv")
        
        # Generate HTML report
        self._generate_html_report(report, result, report_dir)
        
        logger.info(f"Report saved to: {report_dir}")
        return report
    
    def _generate_metadata(self, strategy_name: str) -> Dict:
        """Generate report metadata"""
        return {
            "strategy_name": strategy_name,
            "generated_at": datetime.now().isoformat(),
            "report_version": "1.0"
        }
    
    def _generate_performance_summary(self, result: BacktestResult) -> Dict:
        """Generate performance summary"""
        return {
            "total_return": f"{result.total_return:.2%}",
            "annualized_return": f"{result.annualized_return:.2%}",
            "sharpe_ratio": f"{result.sharpe_ratio:.2f}",
            "sortino_ratio": f"{result.sortino_ratio:.2f}",
            "max_drawdown": f"{result.max_drawdown:.2%}",
            "calmar_ratio": f"{result.calmar_ratio:.2f}",
            "value_at_risk_95": f"{result.var_95:.2%}",
            "conditional_var_95": f"{result.cvar_95:.2%}"
        }
    
    def _generate_trade_analysis(self, result: BacktestResult) -> Dict:
        """Generate trade analysis"""
        analysis = {
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate": f"{result.win_rate:.2%}",
            "average_win": f"${result.avg_win:.2f}",
            "average_loss": f"${result.avg_loss:.2f}",
            "profit_factor": f"{result.profit_factor:.2f}",
            "kelly_criterion": f"{result.kelly_criterion:.2%}"
        }
        
        # Add trade duration analysis if we have trade data
        if not result.trades.empty and 'entry_time' in result.trades.columns and 'exit_time' in result.trades.columns:
            result.trades['duration'] = pd.to_datetime(result.trades['exit_time']) - pd.to_datetime(result.trades['entry_time'])
            avg_duration = result.trades['duration'].mean()
            analysis['average_trade_duration'] = str(avg_duration)
        
        return analysis
    
    def _generate_risk_metrics(self, result: BacktestResult) -> Dict:
        """Generate risk metrics"""
        if result.equity_curve.empty:
            return {}
        
        # Calculate additional risk metrics
        returns = result.equity_curve['value'].pct_change().dropna()
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Omega ratio (probability weighted ratio of gains vs losses)
        threshold = 0
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        omega_ratio = gains.sum() / losses.sum() if losses.sum() > 0 else np.inf
        
        # Maximum consecutive losses
        is_loss = returns < 0
        loss_streaks = is_loss.astype(int).groupby((~is_loss).cumsum()).cumsum()
        max_consecutive_losses = loss_streaks.max()
        
        return {
            "downside_deviation": f"{downside_deviation:.2%}",
            "omega_ratio": f"{omega_ratio:.2f}",
            "max_consecutive_losses": max_consecutive_losses,
            "recovery_factor": f"{result.recovery_factor:.2f}",
            "risk_adjusted_return": f"{result.annualized_return / (returns.std() * np.sqrt(252)):.2f}" if returns.std() > 0 else "N/A"
        }
    
    def _generate_monthly_returns(self, result: BacktestResult) -> Dict:
        """Generate monthly returns table"""
        if result.equity_curve.empty:
            return {}
        
        # Calculate monthly returns
        equity_df = result.equity_curve.copy()
        equity_df['month'] = equity_df.index.to_period('M')
        
        monthly_returns = equity_df.groupby('month')['value'].agg(['first', 'last'])
        monthly_returns['return'] = (monthly_returns['last'] / monthly_returns['first'] - 1) * 100
        
        # Pivot to create year x month table
        monthly_returns.index = pd.to_datetime(monthly_returns.index.to_timestamp())
        monthly_returns['year'] = monthly_returns.index.year
        monthly_returns['month_name'] = monthly_returns.index.strftime('%b')
        
        pivot_table = monthly_returns.pivot(index='year', columns='month_name', values='return')
        
        # Calculate yearly returns
        yearly_returns = monthly_returns.groupby('year')['return'].apply(lambda x: (1 + x/100).prod() - 1) * 100
        pivot_table['Year'] = yearly_returns
        
        return pivot_table.round(2).to_dict()
    
    def _generate_yearly_returns(self, result: BacktestResult) -> Dict:
        """Generate yearly returns summary"""
        if result.equity_curve.empty:
            return {}
        
        equity_df = result.equity_curve.copy()
        equity_df['year'] = equity_df.index.year
        
        yearly_data = equity_df.groupby('year')['value'].agg(['first', 'last'])
        yearly_data['return'] = (yearly_data['last'] / yearly_data['first'] - 1) * 100
        
        return yearly_data['return'].round(2).to_dict()
    
    def _generate_plots(self, result: BacktestResult, output_dir: Path):
        """Generate all plots"""
        # Set up the figure
        fig = plt.figure(figsize=(15, 20))
        
        # 1. Equity Curve
        ax1 = plt.subplot(4, 2, 1)
        result.equity_curve['value'].plot(ax=ax1, color='blue', linewidth=2)
        ax1.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = plt.subplot(4, 2, 2)
        rolling_max = result.equity_curve['value'].expanding().max()
        drawdown = (result.equity_curve['value'] - rolling_max) / rolling_max * 100
        drawdown.plot(ax=ax2, color='red', linewidth=2)
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Daily Returns Distribution
        ax3 = plt.subplot(4, 2, 3)
        returns = result.equity_curve['value'].pct_change().dropna() * 100
        returns.hist(ax=ax3, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax3.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Daily Return (%)')
        ax3.set_ylabel('Frequency')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # 4. Rolling Sharpe Ratio (252-day)
        ax4 = plt.subplot(4, 2, 4)
        rolling_sharpe = (returns.rolling(252).mean() / returns.rolling(252).std()) * np.sqrt(252)
        rolling_sharpe.plot(ax=ax4, color='purple', linewidth=2)
        ax4.set_title('Rolling Sharpe Ratio (252-day)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        # 5. Monthly Returns Heatmap
        if not result.equity_curve.empty:
            ax5 = plt.subplot(4, 2, 5)
            monthly_data = self._generate_monthly_returns(result)
            if monthly_data:
                # Convert to DataFrame for heatmap
                monthly_df = pd.DataFrame(monthly_data)
                if 'Year' in monthly_df.columns:
                    monthly_df = monthly_df.drop('Year', axis=1)
                
                # Reorder months
                month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                available_months = [m for m in month_order if m in monthly_df.columns]
                monthly_df = monthly_df[available_months]
                
                sns.heatmap(monthly_df, annot=True, fmt='.1f', cmap='RdYlGn', 
                           center=0, ax=ax5, cbar_kws={'label': 'Return (%)'})
                ax5.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        
        # 6. Trade P&L Distribution
        if not result.trades.empty and 'pnl' in result.trades.columns:
            ax6 = plt.subplot(4, 2, 6)
            result.trades['pnl'].hist(ax=ax6, bins=30, color='orange', alpha=0.7, edgecolor='black')
            ax6.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
            ax6.set_xlabel('P&L ($)')
            ax6.set_ylabel('Frequency')
            ax6.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # 7. Cumulative Returns by Year
        ax7 = plt.subplot(4, 2, 7)
        equity_df = result.equity_curve.copy()
        equity_df['year'] = equity_df.index.year
        for year in equity_df['year'].unique():
            year_data = equity_df[equity_df['year'] == year]['value']
            year_returns = (year_data / year_data.iloc[0] - 1) * 100
            year_returns.plot(ax=ax7, label=str(year), linewidth=2)
        ax7.set_title('Cumulative Returns by Year', fontsize=14, fontweight='bold')
        ax7.set_ylabel('Cumulative Return (%)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Risk-Return Scatter (if multiple strategies)
        # For now, just show current strategy point
        ax8 = plt.subplot(4, 2, 8)
        ax8.scatter(returns.std() * np.sqrt(252) * 100, result.annualized_return * 100, 
                   s=200, color='red', edgecolor='black', linewidth=2)
        ax8.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Annualized Volatility (%)')
        ax8.set_ylabel('Annualized Return (%)')
        ax8.grid(True, alpha=0.3)
        
        # Add strategy name as text
        ax8.text(returns.std() * np.sqrt(252) * 100, result.annualized_return * 100, 
                'Strategy', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'backtest_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Plots saved successfully")
    
    def _generate_html_report(self, report: Dict, result: BacktestResult, output_dir: Path):
        """Generate HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report - {report['metadata']['strategy_name']}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2 {{
                    color: #333;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border: 1px solid #dee2e6;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #0066cc;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #666;
                    margin-top: 5px;
                }}
                .section {{
                    margin: 30px 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 10px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f8f9fa;
                    font-weight: bold;
                }}
                .positive {{
                    color: #28a745;
                }}
                .negative {{
                    color: #dc3545;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Backtest Report: {report['metadata']['strategy_name']}</h1>
                <p>Generated: {report['metadata']['generated_at']}</p>
                
                <div class="section">
                    <h2>Performance Summary</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{report['performance_summary']['total_return']}</div>
                            <div class="metric-label">Total Return</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{report['performance_summary']['sharpe_ratio']}</div>
                            <div class="metric-label">Sharpe Ratio</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{report['performance_summary']['max_drawdown']}</div>
                            <div class="metric-label">Max Drawdown</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{report['performance_summary']['calmar_ratio']}</div>
                            <div class="metric-label">Calmar Ratio</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Trade Analysis</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{report['trade_analysis']['total_trades']}</div>
                            <div class="metric-label">Total Trades</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{report['trade_analysis']['win_rate']}</div>
                            <div class="metric-label">Win Rate</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{report['trade_analysis']['profit_factor']}</div>
                            <div class="metric-label">Profit Factor</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{report['trade_analysis']['average_win']}</div>
                            <div class="metric-label">Average Win</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Charts</h2>
                    <img src="backtest_report.png" alt="Backtest Charts">
                </div>
                
                <div class="section">
                    <h2>Detailed Metrics</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
        """
        
        # Add all metrics to the table
        all_metrics = {
            **report['performance_summary'],
            **report['trade_analysis'],
            **report['risk_metrics']
        }
        
        for key, value in all_metrics.items():
            formatted_key = key.replace('_', ' ').title()
            html_content += f"""
                        <tr>
                            <td>{formatted_key}</td>
                            <td>{value}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        
        html_file = output_dir / "report.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {html_file}")