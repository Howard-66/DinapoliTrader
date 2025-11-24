import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class Visualizer:
    """
    Visualization tools for DiNapoli charts.
    """

    @staticmethod
    @staticmethod
    def plot_chart(df: pd.DataFrame, 
                   indicators: dict = None, 
                   equity: pd.Series = None,
                   drawdown: pd.Series = None,
                   trades: pd.DataFrame = None,
                   title: str = "DiNapoli Chart"):
        """
        Create an interactive chart with indicators and trade execution lines.
        
        Args:
            df (pd.DataFrame): OHLCV data.
            indicators (dict): Dictionary of {name: series} to plot on the chart.
            equity (pd.Series): Equity curve.
            drawdown (pd.Series): Drawdown curve.
            trades (pd.DataFrame): Trade log containing Entry/Exit Date/Price and PnL.
            title (str): Chart title.
            
        Returns:
            plotly.graph_objects.Figure
        """

        rows = 1
        row_heights = [1.0]
        subplot_titles = [title, 'Volume']
        
        if equity is not None and drawdown is not None:
            rows = 3
            row_heights = [0.75, 0.15, 0.10]
            subplot_titles = [title, 'Equity Curve', 'Drawdown %']
            
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=subplot_titles,
                            row_heights=row_heights)

        # Area Chart (using Close price)
        # Using Scatter with fill='tozeroy' creates an area chart
        # To make it look nice, we often use a gradient or semi-transparent fill
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['close'],
            name='Price',
            mode='lines',
            line=dict(color='rgba(150, 150, 150, 0.5)', width=2),
            fill='tozeroy',
            fillcolor='rgba(150, 150, 150, 0.1)'
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
        # fig.add_trace(go.Bar(
        #     x=df.index,
        #     y=df['volume'],
        #     name='Volume',
        #     marker_color='rgba(100, 100, 100, 0.5)'
        # ), row=2, col=1)

        # Trade Execution Lines
        if trades is not None and not trades.empty:
            for _, trade in trades.iterrows():
                # Determine color based on PnL
                color = 'green' if trade['PnL Amount'] > 0 else 'red'
                
                # Draw Line Segment
                fig.add_trace(go.Scatter(
                    x=[trade['Entry Date'], trade['Exit Date']],
                    y=[trade['Entry Price'], trade['Exit Price']],
                    mode='lines+markers',
                    line=dict(color=color, width=2, dash='dash'),
                    # marker=dict(size=6, symbol=['circle', 'x']), # Circle for entry, X for exit
                    marker=dict(size=4, symbol=['circle', 'circle']), # Circle for entry, X for exit
                    name=f"Trade {trade['PnL %']:.1%}",
                    showlegend=False, # Too many trades will clutter legend
                    hoverinfo='text',
                    text=f"PnL: {trade['PnL Amount']:.2f} ({trade['PnL %']:.2%})"
                ), row=1, col=1)

        # Equity & Drawdown
        if equity is not None and drawdown is not None:
            fig.add_trace(go.Scatter(
                x=equity.index,
                y=equity.values,
                name='Equity',
                line=dict(color='cyan', width=1.5)
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100, # Convert to %
                name='Drawdown %',
                line=dict(color='red', width=1),
                fill='tozeroy'
            ), row=3, col=1)

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=550, # Reduced height
            template='plotly_dark',
            hovermode='x unified',
            showlegend=False,
            margin=dict(t=20, b=0)
        )
        
        return fig

    @staticmethod
    def plot_heatmap(monthly_returns: pd.DataFrame, title: str = "Monthly Returns Heatmap"):
        """
        Plots a heatmap of monthly returns.
        
        Args:
            monthly_returns (pd.DataFrame): Matrix of returns (Year x Month).
            title (str): Chart title.
            
        Returns:
            plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go
        
        # Month names for x-axis
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Ensure columns are 1-12 and reindex if necessary
        # The input df has columns 1, 2, ... 12 (integers)
        # We map them to names for display
        
        z = monthly_returns.values
        x = month_names
        y = monthly_returns.index
        
        # Create annotation text (percentage)
        z_text = [[f"{val:.1%}" for val in row] for row in z]
        
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale='RdYlGn',
            zmid=0.0,
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_nticks=12,
            yaxis_nticks=len(y),
            height=400 + (len(y) * 20), # Dynamic height based on years
            margin=dict(t=50, b=20, l=50, r=50)
        )
        
        return fig
