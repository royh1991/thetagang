"""
Professional plotting for backtest results
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np


class BacktestPlotter:
    """Create professional charts for backtest results"""
    
    def __init__(self):
        # Professional color scheme
        self.colors = {
            'background': '#0e1117',
            'paper': '#1a1a2e',
            'text': '#ffffff',
            'grid': '#2a2a3e',
            'buy': '#00ff88',
            'sell': '#ff3366',
            'price': '#3366ff',
            'sma': '#ffaa00',
            'volume': '#666666',
            'profit': '#00ff88',
            'loss': '#ff3366'
        }
        
    def plot_backtest_results(self, 
                            data: pd.DataFrame,
                            trades: List[Dict],
                            strategy_name: str = "Strategy",
                            show_volume: bool = True,
                            show_indicators: Dict[str, List[float]] = None,
                            save_path: Optional[str] = None):
        """
        Create comprehensive backtest visualization
        
        Args:
            data: DataFrame with OHLCV data
            trades: List of trade dictionaries
            strategy_name: Name of the strategy
            show_volume: Whether to show volume subplot
            show_indicators: Dictionary of indicator names to values
            save_path: Path to save HTML file
        """
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Create subplots
        n_subplots = 3 if show_volume else 2
        row_heights = [0.6, 0.2, 0.2] if show_volume else [0.7, 0.3]
        
        fig = make_subplots(
            rows=n_subplots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=row_heights,
            subplot_titles=[
                f'{strategy_name} - Price Action',
                'Volume' if show_volume else 'Portfolio Value',
                'Portfolio Value' if show_volume else None
            ]
        )
        
        # 1. Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                increasing_line_color=self.colors['profit'],
                decreasing_line_color=self.colors['loss']
            ),
            row=1, col=1
        )
        
        # 2. Add buy/sell signals
        if not trades_df.empty:
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            # Buy signals
            if not buy_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(buy_trades['timestamp']),
                        y=buy_trades['price'],
                        mode='markers',
                        name='Buy',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color=self.colors['buy'],
                            line=dict(width=2, color='white')
                        ),
                        text=[f"Buy {t['quantity']} @ ${t['price']:.2f}" 
                              for _, t in buy_trades.iterrows()],
                        hovertemplate='%{text}<br>%{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Sell signals
            if not sell_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(sell_trades['timestamp']),
                        y=sell_trades['price'],
                        mode='markers',
                        name='Sell',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color=self.colors['sell'],
                            line=dict(width=2, color='white')
                        ),
                        text=[f"Sell {t['quantity']} @ ${t['price']:.2f}<br>P&L: ${t.get('pnl', 0):.2f}" 
                              for _, t in sell_trades.iterrows()],
                        hovertemplate='%{text}<br>%{x}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # 3. Add technical indicators
        if show_indicators:
            for name, values in show_indicators.items():
                if len(values) == len(data):
                    fig.add_trace(
                        go.Scatter(
                            x=data['timestamp'],
                            y=values,
                            mode='lines',
                            name=name,
                            line=dict(width=2)
                        ),
                        row=1, col=1
                    )
        
        # 4. Volume subplot
        if show_volume:
            colors = ['red' if data.iloc[i]['close'] < data.iloc[i]['open'] 
                     else 'green' for i in range(len(data))]
            
            fig.add_trace(
                go.Bar(
                    x=data['timestamp'],
                    y=data['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 5. Portfolio value / Equity curve
        equity_row = 3 if show_volume else 2
        if not trades_df.empty:
            # Calculate cumulative P&L
            equity_curve = self._calculate_equity_curve(trades_df, initial_capital=100000)
            
            fig.add_trace(
                go.Scatter(
                    x=equity_curve['timestamp'],
                    y=equity_curve['value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color=self.colors['price'], width=2),
                    fill='tozeroy',
                    fillcolor='rgba(51, 102, 255, 0.1)'
                ),
                row=equity_row, col=1
            )
            
            # Add benchmark line at initial capital
            fig.add_hline(
                y=100000, 
                line_dash="dash", 
                line_color="gray",
                annotation_text="Initial Capital",
                row=equity_row, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{strategy_name} Backtest Results',
                font=dict(size=24, color=self.colors['text'])
            ),
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_dark',
            paper_bgcolor=self.colors['paper'],
            plot_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            height=900,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(
            gridcolor=self.colors['grid'],
            rangeslider_visible=False,
            type='date'
        )
        
        fig.update_yaxes(
            gridcolor=self.colors['grid'],
            title_text="Price ($)",
            row=1, col=1
        )
        
        if show_volume:
            fig.update_yaxes(
                gridcolor=self.colors['grid'],
                title_text="Volume",
                row=2, col=1
            )
            
        fig.update_yaxes(
            gridcolor=self.colors['grid'],
            title_text="Portfolio Value ($)",
            row=equity_row, col=1
        )
        
        # Save or show
        if save_path:
            fig.write_html(save_path)
            print(f"Chart saved to: {save_path}")
        else:
            fig.show()
        
        return fig
    
    def plot_performance_metrics(self, 
                               metrics: Dict[str, float],
                               trades_df: pd.DataFrame,
                               save_path: Optional[str] = None):
        """Create a dashboard of performance metrics"""
        
        # Create subplots for metrics dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Return Distribution',
                'Win/Loss Distribution', 
                'Monthly Returns',
                'Drawdown',
                'Trade P&L',
                'Key Metrics'
            ],
            specs=[
                [{"type": "histogram"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "table"}]
            ]
        )
        
        if not trades_df.empty and 'pnl' in trades_df.columns:
            pnl_trades = trades_df[trades_df['pnl'].notna()]
            
            # 1. Return distribution
            fig.add_trace(
                go.Histogram(
                    x=pnl_trades['pnl'],
                    name='P&L Distribution',
                    marker_color=self.colors['price'],
                    nbinsx=20
                ),
                row=1, col=1
            )
            
            # 2. Win/Loss bars
            wins = len(pnl_trades[pnl_trades['pnl'] > 0])
            losses = len(pnl_trades[pnl_trades['pnl'] <= 0])
            
            fig.add_trace(
                go.Bar(
                    x=['Wins', 'Losses'],
                    y=[wins, losses],
                    marker_color=[self.colors['profit'], self.colors['loss']],
                    text=[f'{wins}', f'{losses}'],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # 3. Monthly returns (simplified)
            if 'timestamp' in trades_df.columns:
                trades_df['month'] = pd.to_datetime(trades_df['timestamp']).dt.to_period('M')
                monthly_pnl = trades_df.groupby('month')['pnl'].sum()
                
                fig.add_trace(
                    go.Bar(
                        x=monthly_pnl.index.astype(str),
                        y=monthly_pnl.values,
                        marker_color=[self.colors['profit'] if x > 0 else self.colors['loss'] 
                                    for x in monthly_pnl.values]
                    ),
                    row=1, col=3
                )
            
            # 5. Individual trade P&L
            fig.add_trace(
                go.Bar(
                    x=list(range(len(pnl_trades))),
                    y=pnl_trades['pnl'],
                    marker_color=[self.colors['profit'] if x > 0 else self.colors['loss'] 
                                for x in pnl_trades['pnl']],
                    name='Trade P&L'
                ),
                row=2, col=2
            )
        
        # 6. Key metrics table
        metrics_data = [
            ['Total Return', f"{metrics.get('total_return_pct', 0):.2f}%"],
            ['Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"],
            ['Max Drawdown', f"{metrics.get('max_drawdown_pct', 0):.2f}%"],
            ['Win Rate', f"{metrics.get('win_rate', 0):.1%}"],
            ['Profit Factor', f"{metrics.get('profit_factor', 0):.2f}"],
            ['Total Trades', f"{metrics.get('total_trades', 0)}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Metric', 'Value'],
                    fill_color=self.colors['paper'],
                    font=dict(color=self.colors['text'], size=14)
                ),
                cells=dict(
                    values=list(zip(*metrics_data)),
                    fill_color=self.colors['background'],
                    font=dict(color=self.colors['text'], size=12)
                )
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title='Performance Metrics Dashboard',
            template='plotly_dark',
            paper_bgcolor=self.colors['paper'],
            plot_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text']),
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Metrics dashboard saved to: {save_path}")
        else:
            fig.show()
        
        return fig
    
    def _calculate_equity_curve(self, trades_df: pd.DataFrame, initial_capital: float = 100000):
        """Calculate equity curve from trades"""
        equity_data = []
        capital = initial_capital
        
        for _, trade in trades_df.iterrows():
            if trade['action'] == 'BUY':
                # Capital decreases on buy
                capital = trade.get('cash_after', capital)
            else:  # SELL
                # Capital increases on sell
                capital = trade.get('cash_after', capital)
            
            equity_data.append({
                'timestamp': trade['timestamp'],
                'value': capital
            })
        
        return pd.DataFrame(equity_data)