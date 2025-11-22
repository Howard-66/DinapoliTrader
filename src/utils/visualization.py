import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class Visualizer:
    """
    Visualization tools for DiNapoli charts.
    """

    @staticmethod
    def plot_chart(df: pd.DataFrame, 
                   indicators: dict = None, 
                   equity: pd.Series = None,
                   drawdown: pd.Series = None,
                   title: str = "DiNapoli Chart"):
        """
        Create an interactive candlestick chart with indicators.
        
        Args:
            df (pd.DataFrame): OHLCV data.
            indicators (dict): Dictionary of {name: series} to plot on the chart.
            title (str): Chart title.
            
        Returns:
            plotly.graph_objects.Figure
        """

        rows = 2
        row_heights = [0.7, 0.3]
        subplot_titles = [title, 'Volume']
        
        if equity is not None and drawdown is not None:
            rows = 4
            row_heights = [0.5, 0.15, 0.15, 0.15]
            subplot_titles = [title, 'Volume', 'Equity Curve', 'Drawdown %']
            
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=subplot_titles,
                            row_heights=row_heights)

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ), row=1, col=1)

        # Indicators
        if indicators:
            for name, series in indicators.items():
                fig.add_trace(go.Scatter(
                    x=series.index, 
                    y=series.values, 
                    name=name,
                    line=dict(width=1)
                ), row=1, col=1)

        # Volume
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume'
        ), row=2, col=1)

        # Equity & Drawdown
        if equity is not None and drawdown is not None:
            fig.add_trace(go.Scatter(
                x=equity.index,
                y=equity.values,
                name='Equity',
                line=dict(color='cyan', width=1.5)
            ), row=3, col=1)
            
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100, # Convert to %
                name='Drawdown %',
                line=dict(color='red', width=1),
                fill='tozeroy'
            ), row=4, col=1)

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=800,
            template='plotly_dark'
        )
        
        return fig
