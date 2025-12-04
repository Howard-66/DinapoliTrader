import pandas as pd
import numpy as np

class Visualizer:
    """
    Visualization tools for DiNapoli charts using ECharts5.
    """

    @staticmethod
    def plot_chart(df: pd.DataFrame, 
                   indicators: dict = None, 
                   equity: pd.Series = None,
                   floating_equity: pd.Series = None,
                   drawdown: pd.Series = None,
                   trades: pd.DataFrame = None,
                   signals: dict = None,
                   title: str = "DiNapoli Chart"):
        """
        Create an interactive chart with indicators and trade execution lines using ECharts.
        
        Args:
            df (pd.DataFrame): OHLCV data.
            indicators (dict): Dictionary of {name: series} to plot on the chart.
            equity (pd.Series): Realized equity curve (closed trades only).
            floating_equity (pd.Series): Floating equity curve (includes unrealized P&L).
            drawdown (pd.Series): Drawdown curve.
            trades (pd.DataFrame): Trade log containing Entry/Exit Date/Price and PnL.
            signals (dict): Dictionary of {pattern_name: DataFrame} with signal markers.
            title (str): Chart title.
            
        Returns:
            dict: ECharts option dictionary
        """
        
        # Prepare data
        dates = df.index.strftime('%Y-%m-%d').tolist()
        prices = df['close'].tolist()
        
        # Determine grid layout based on whether we have equity/drawdown
        if (equity is not None or floating_equity is not None) and drawdown is not None:
            grid_config = [
                {'left': '5%', 'right': '5%', 'top': '8%', 'height': '50%'},
                {'left': '5%', 'right': '5%', 'top': '62%', 'height': '18%'},
                {'left': '5%', 'right': '5%', 'top': '84%', 'height': '12%'}
            ]
        else:
            grid_config = [
                {'left': '5%', 'right': '5%', 'top': '8%', 'bottom': '10%'}
            ]
        
        # Base option
        option = {
            'backgroundColor': '#1e1e1e',
            'title': {
                'text': title,
                'left': 'center',
                'top': '1%',
                'textStyle': {'color': '#fff', 'fontSize': 16}
            },
            'tooltip': {
                'trigger': 'axis',
                'axisPointer': {'type': 'cross'},
                'backgroundColor': 'rgba(50, 50, 50, 0.9)',
                'borderColor': '#777',
                'textStyle': {'color': '#fff'}
            },
            'legend': {
                'data': [],
                'top': '3%',
                'left': '10%',
                'textStyle': {'color': '#fff'}
            },
            'grid': grid_config,
            'xAxis': [],
            'yAxis': [],
            'dataZoom': [
                {
                    'type': 'inside',
                    'xAxisIndex': [0, 1, 2] if len(grid_config) == 3 else [0],
                    'start': 0,
                    'end': 100
                },
                {
                    'show': True,
                    'xAxisIndex': [0, 1, 2] if len(grid_config) == 3 else [0],
                    'type': 'slider',
                    'bottom': '1%',
                    'start': 0,
                    'end': 100
                }
            ],
            'series': []
        }
        
        # X-Axis for main chart
        option['xAxis'].append({
            'type': 'category',
            'data': dates,
            'gridIndex': 0,
            'axisLine': {'lineStyle': {'color': '#666'}},
            'axisLabel': {'color': '#999'}
        })
        
        # Y-Axis for main chart
        option['yAxis'].append({
            'type': 'value',
            'gridIndex': 0,
            'scale': True,
            'splitLine': {'lineStyle': {'color': '#333'}},
            'axisLine': {'lineStyle': {'color': '#666'}},
            'axisLabel': {'color': '#999'}
        })
        
        # Price Area Chart
        option['series'].append({
            'name': 'Price',
            'type': 'line',
            'data': prices,
            'xAxisIndex': 0,
            'yAxisIndex': 0,
            'smooth': False,
            'lineStyle': {'color': 'rgba(150, 150, 150, 0.8)', 'width': 2},
            'areaStyle': {'color': 'rgba(150, 150, 150, 0.1)'},
            'showSymbol': False
        })
        option['legend']['data'].append('Price')
        
        # Add indicators
        if indicators:
            colors = ['#00ff00', '#ff00ff', '#00ffff', '#ffff00', '#ff8800']
            for idx, (name, series) in enumerate(indicators.items()):
                indicator_data = [x if pd.notna(x) else None for x in series.reindex(df.index)]
                option['series'].append({
                    'name': name,
                    'type': 'line',
                    'data': indicator_data,
                    'xAxisIndex': 0,
                    'yAxisIndex': 0,
                    'smooth': False,
                    'lineStyle': {'color': colors[idx % len(colors)], 'width': 1},
                    'showSymbol': False
                })
                option['legend']['data'].append(name)
        
        # Add signal markers
        if signals:
            signal_colors = {
                'Double Repo': '#00ff00',
                'Single Penetration': '#0088ff',
                'Railroad Tracks': '#ff00ff',
                'Failure to Penetrate': '#ff8800'
            }
            
            for pattern_name, signal_df in signals.items():
                if not signal_df.empty:
                    marker_data = []
                    for idx in signal_df.index:
                        if idx in df.index:
                            date_str = idx.strftime('%Y-%m-%d')
                            if date_str in dates:
                                date_idx = dates.index(date_str)
                                price = df.loc[idx, 'low'] * 0.99
                                marker_data.append([date_idx, price])
                    
                    if marker_data:
                        option['series'].append({
                            'name': f'{pattern_name} Buy',
                            'type': 'scatter',
                            'data': marker_data,
                            'xAxisIndex': 0,
                            'yAxisIndex': 0,
                            'symbol': 'triangle',
                            'symbolSize': 10,
                            'symbolRotate': 0,
                            'itemStyle': {
                                'color': signal_colors.get(pattern_name, '#ffffff')
                            }
                        })
                        option['legend']['data'].append(f'{pattern_name} Buy')
        
        # Add trade execution lines
        if trades is not None and not trades.empty:
            for _, trade in trades.iterrows():
                color = '#00ff00' if trade['PnL Amount'] > 0 else '#ff0000'
                entry_idx = dates.index(trade['Entry Date'].strftime('%Y-%m-%d'))
                exit_idx = dates.index(trade['Exit Date'].strftime('%Y-%m-%d'))
                
                option['series'].append({
                    'type': 'line',
                    'data': [[entry_idx, trade['Entry Price']], [exit_idx, trade['Exit Price']]],
                    'xAxisIndex': 0,
                    'yAxisIndex': 0,
                    'lineStyle': {'color': color, 'width': 2, 'type': 'dashed'},
                    'symbol': 'circle',
                    'symbolSize': 4,
                    'showSymbol': True,
                    'silent': True
                })
        
        # Add equity curve
        if equity is not None and drawdown is not None:
            # X-Axis for equity
            option['xAxis'].append({
                'type': 'category',
                'data': dates,
                'gridIndex': 1,
                'axisLine': {'lineStyle': {'color': '#666'}},
                'axisLabel': {'show': False}
            })
            
            # Y-Axis for equity
            option['yAxis'].append({
                'type': 'value',
                'gridIndex': 1,
                'scale': True,
                'splitLine': {'lineStyle': {'color': '#333'}},
                'axisLine': {'lineStyle': {'color': '#666'}},
                'axisLabel': {'color': '#999'}
            })
            
            # Add realized equity curve
            if equity is not None:
                equity_data = equity.reindex(df.index).ffill().tolist()
                option['series'].append({
                    'name': 'Realized Equity',
                    'type': 'line',
                    'data': equity_data,
                    'xAxisIndex': 1,
                    'yAxisIndex': 1,
                    'smooth': False,
                    'lineStyle': {'color': '#00ffff', 'width': 1.5},
                    'showSymbol': False
                })
                option['legend']['data'].append('Realized Equity')
            
            # Add floating equity curve
            if floating_equity is not None:
                floating_equity_data = floating_equity.reindex(df.index).ffill().tolist()
                option['series'].append({
                    'name': 'Floating Equity',
                    'type': 'line',
                    'data': floating_equity_data,
                    'xAxisIndex': 1,
                    'yAxisIndex': 1,
                    'smooth': False,
                    'lineStyle': {'color': '#ffff00', 'width': 1.5, 'type': 'dashed'},
                    'showSymbol': False
                })
                option['legend']['data'].append('Floating Equity')
            
            # X-Axis for drawdown
            option['xAxis'].append({
                'type': 'category',
                'data': dates,
                'gridIndex': 2,
                'axisLine': {'lineStyle': {'color': '#666'}},
                'axisLabel': {'color': '#999'}
            })
            
            # Y-Axis for drawdown
            option['yAxis'].append({
                'type': 'value',
                'gridIndex': 2,
                'splitLine': {'lineStyle': {'color': '#333'}},
                'axisLine': {'lineStyle': {'color': '#666'}},
                'axisLabel': {'color': '#999', 'formatter': '{value}%'}
            })
            
            drawdown_data = (drawdown.reindex(df.index).fillna(0) * 100).tolist()
            option['series'].append({
                'name': 'Drawdown %',
                'type': 'line',
                'data': drawdown_data,
                'xAxisIndex': 2,
                'yAxisIndex': 2,
                'smooth': False,
                'lineStyle': {'color': '#ff0000', 'width': 1},
                'areaStyle': {'color': 'rgba(255, 0, 0, 0.3)'},
                'showSymbol': False
            })
            option['legend']['data'].append('Drawdown %')
        
        return option

    @staticmethod
    def plot_heatmap(monthly_returns: pd.DataFrame, title: str = "Monthly Returns Heatmap"):
        """
        Plots a heatmap of monthly returns using ECharts.
        
        Args:
            monthly_returns (pd.DataFrame): Matrix of returns (Year x Month).
            title (str): Chart title.
            
        Returns:
            dict: ECharts option dictionary
        """
        
        # Month names for x-axis
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        years = monthly_returns.index.tolist()
        
        # Prepare data in [month_idx, year_idx, value, formatted_text] format
        # Pre-format the percentage strings in Python for exact control
        data = []
        raw_values = []  # For color mapping
        for year_idx, year in enumerate(years):
            for month_idx in range(12):
                month_num = month_idx + 1
                if month_num in monthly_returns.columns:
                    value = monthly_returns.loc[year, month_num]
                    if pd.notna(value):
                        # Convert to percentage
                        percentage_value = float(value * 100)
                        # Format as string with 2 decimal places
                        formatted_text = f"{percentage_value:.2f}%"
                        data.append({
                            'value': [month_idx, year_idx, percentage_value],
                            'label': {'formatter': formatted_text}
                        })
                        raw_values.append(percentage_value)
        
        # Find min/max for color scale
        if raw_values:
            min_val = min(raw_values)
            max_val = max(raw_values)
        else:
            min_val, max_val = -10, 10
        
        option = {
            'backgroundColor': '#1e1e1e',
            'title': {
                'text': title,
                'left': 'center',
                'top': '2%',
                'textStyle': {'color': '#fff', 'fontSize': 16}
            },
            'tooltip': {
                'position': 'top',
                'formatter': '{b0} {b1}: {c}%'
            },
            'grid': {
                'left': '80px',
                'right': '80px',
                'top': '80px',
                'bottom': '80px',
                'containLabel': True
            },
            'xAxis': {
                'type': 'category',
                'data': month_names,
                'splitArea': {'show': True},
                'axisLine': {'lineStyle': {'color': '#666'}},
                'axisLabel': {'color': '#999'}
            },
            'yAxis': {
                'type': 'category',
                'data': years,
                'splitArea': {'show': True},
                'axisLine': {'lineStyle': {'color': '#666'}},
                'axisLabel': {'color': '#999'}
            },
            'visualMap': {
                'min': min_val,
                'max': max_val,
                'calculable': True,
                'orient': 'horizontal',
                'left': 'center',
                'bottom': '5%',
                'type': 'piecewise',
                'pieces': [
                    {'min': 5, 'color': '#1a9850', 'label': '> 5%'},           # Dark green
                    {'min': 2, 'max': 5, 'color': '#91cf60', 'label': '2-5%'}, # Light green
                    {'min': 0, 'max': 2, 'color': '#d9ef8b', 'label': '0-2%'}, # Very light green
                    {'min': -2, 'max': 0, 'color': '#fee08b', 'label': '-2-0%'}, # Very light red
                    {'min': -5, 'max': -2, 'color': '#fc8d59', 'label': '-5--2%'}, # Light red
                    {'max': -5, 'color': '#d73027', 'label': '< -5%'}          # Dark red
                ],
                'textStyle': {'color': '#fff'}
            },
            'series': [{
                'name': 'Monthly Returns',
                'type': 'heatmap',
                'data': data,
                'label': {
                    'show': True,
                    'color': '#000',
                    'fontSize': 9
                },
                'emphasis': {
                    'itemStyle': {
                        'shadowBlur': 10,
                        'shadowColor': 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }]
        }
        
        return option

    @staticmethod
    def plot_bar_chart(data: pd.DataFrame, x_col: str, y_col: str, title: str = "Bar Chart"):
        """
        Create a bar chart using ECharts.
        
        Args:
            data (pd.DataFrame): Data to plot
            x_col (str): Column name for x-axis
            y_col (str): Column name for y-axis (values)
            title (str): Chart title
            
        Returns:
            dict: ECharts option dictionary
        """
        
        x_data = data[x_col].tolist()
        y_data = data[y_col].tolist()
        
        # Color bars based on positive/negative values
        colors = ['#00ff00' if val > 0 else '#ff0000' for val in y_data]
        
        option = {
            'backgroundColor': '#1e1e1e',
            'title': {
                'text': title,
                'left': 'center',
                'textStyle': {'color': '#fff'}
            },
            'tooltip': {
                'trigger': 'axis',
                'axisPointer': {'type': 'shadow'},
                'backgroundColor': 'rgba(50, 50, 50, 0.9)',
                'textStyle': {'color': '#fff'}
            },
            'grid': {
                'left': '10%',
                'right': '10%',
                'top': '15%',
                'bottom': '15%'
            },
            'xAxis': {
                'type': 'category',
                'data': x_data,
                'axisLine': {'lineStyle': {'color': '#666'}},
                'axisLabel': {'color': '#999', 'rotate': 0}
            },
            'yAxis': {
                'type': 'value',
                'axisLine': {'lineStyle': {'color': '#666'}},
                'axisLabel': {'color': '#999', 'formatter': '{value}%'},
                'splitLine': {'lineStyle': {'color': '#333'}}
            },
            'series': [{
                'type': 'bar',
                'data': [
                    {
                        'value': val,
                        'itemStyle': {'color': color},
                        'label': {
                            'show': True,
                            'position': 'top' if val > 0 else 'bottom',
                            'formatter': f'{val:.1f}%',
                            'color': '#fff'
                        }
                    }
                    for val, color in zip(y_data, colors)
                ],
                'barWidth': '60%'
            }]
        }
        
        return option
    @staticmethod
    def plot_trade_detail(df: pd.DataFrame, trade_row: pd.Series, lookback: int = 30, lookforward: int = 10):
        """
        Create a detailed chart for a single trade with pattern visualization.
        
        Args:
            df (pd.DataFrame): Full OHLCV data.
            trade_row (pd.Series): A row from the trade log containing Entry/Exit info and Metadata.
            lookback (int): Number of bars before pattern start to show.
            lookforward (int): Number of bars after exit to show.
            
        Returns:
            dict: ECharts option dictionary.
        """
        # Extract Trade Info
        entry_date = trade_row['Entry Date']
        exit_date = trade_row['Exit Date']
        pattern_name = trade_row.get('Pattern', 'Unknown')
        metadata = trade_row.get('Metadata', {})
        
        entry_price = trade_row['Entry Price']
        exit_price = trade_row['Exit Price']
        
        # Ensure dates are timestamps
        if isinstance(entry_date, str):
            entry_date = pd.to_datetime(entry_date)
        if isinstance(exit_date, str):
            exit_date = pd.to_datetime(exit_date)
            
        # Determine Chart Window
        # Start: A bit before the pattern starts (if available in metadata) or Entry Date
        start_date = entry_date
        
        # Try to find the earliest date in metadata
        if isinstance(metadata, dict):
            dates = []
            for key, val in metadata.items():
                if isinstance(val, dict) and 'date' in val:
                    dates.append(pd.to_datetime(val['date']))
            if dates:
                start_date = min(dates)
        
        # Calculate indices
        try:
            start_idx = df.index.get_loc(start_date)
            entry_idx = df.index.get_loc(entry_date)
            exit_idx = df.index.get_loc(exit_date)
        except KeyError:
            # Fallback if exact dates not found (shouldn't happen with proper indexing)
            return {}
            
        plot_start_idx = max(0, start_idx - lookback)
        plot_end_idx = min(len(df), exit_idx + lookforward)
        
        # Slice Data
        sub_df = df.iloc[plot_start_idx:plot_end_idx].copy()
        dates = sub_df.index.strftime('%Y-%m-%d').tolist()
        
        # Indicators
        # Calculate locally to ensure alignment or slice existing if passed (but here we calc for simplicity)
        from src.indicators.basics import Indicators
        dma_3x3 = Indicators.displaced_ma(df['close'], 3, 3).iloc[plot_start_idx:plot_end_idx]
        dma_25x5 = Indicators.displaced_ma(df['close'], 25, 5).iloc[plot_start_idx:plot_end_idx]
        
        # Base Option
        option = {
            'backgroundColor': '#1e1e1e',
            'title': {
                'text': f"{pattern_name} Trade Details",
                'subtext': f"PnL: {trade_row['PnL Amount']:.2f} ({trade_row['PnL %']:.2%})",
                'left': 'center',
                'textStyle': {'color': '#fff'},
                'subtextStyle': {'color': '#00ff00' if trade_row['PnL Amount'] > 0 else '#ff0000', 'fontSize': 14}
            },
            'tooltip': {
                'trigger': 'axis',
                'axisPointer': {'type': 'cross'},
                'backgroundColor': 'rgba(50, 50, 50, 0.9)',
                'textStyle': {'color': '#fff'}
            },
            'legend': {
                'data': ['Price', 'Volume', '3x3 DMA', '25x5 DMA', 'Pattern', 'SL', 'TP'],
                'top': '10%',
                'textStyle': {'color': '#fff'}
            },
            'axisPointer': {
                'link': [{'xAxisIndex': 'all'}]
            },
            'grid': [
                {'left': '5%', 'right': '5%', 'top': '20%', 'height': '55%'},
                {'left': '5%', 'right': '5%', 'top': '80%', 'height': '15%'}
            ],
            'xAxis': [
                {
                    'type': 'category',
                    'data': dates,
                    'gridIndex': 0,
                    'axisLine': {'lineStyle': {'color': '#666'}},
                    'axisLabel': {'show': False}
                },
                {
                    'type': 'category',
                    'data': dates,
                    'gridIndex': 1,
                    'axisLine': {'lineStyle': {'color': '#666'}},
                    'axisLabel': {'color': '#999'}
                }
            ],
            'yAxis': [
                {
                    'type': 'value',
                    'scale': True,
                    'gridIndex': 0,
                    'splitLine': {'lineStyle': {'color': '#333'}},
                    'axisLine': {'lineStyle': {'color': '#666'}},
                    'axisLabel': {'color': '#999'}
                },
                {
                    'type': 'value',
                    'scale': True,
                    'gridIndex': 1,
                    'splitLine': {'show': False},
                    'axisLine': {'lineStyle': {'color': '#666'}},
                    'axisLabel': {'show': False}
                }
            ],
            'series': []
        }
        
        # 1. Candlestick
        k_data = sub_df[['open', 'close', 'low', 'high']].values.tolist()
        option['series'].append({
            'name': 'Price',
            'type': 'candlestick',
            'data': k_data,
            'xAxisIndex': 0,
            'yAxisIndex': 0,
            'itemStyle': {
                'color': '#ef232a',
                'color0': '#14b143',
                'borderColor': '#ef232a',
                'borderColor0': '#14b143'
            }
        })
        
        # 2. Volume
        vol_data = sub_df['volume'].tolist()
        # Color volume bars based on price change
        vol_colors = ['#ef232a' if row['close'] >= row['open'] else '#14b143' for _, row in sub_df.iterrows()]
        
        option['series'].append({
            'name': 'Volume',
            'type': 'bar',
            'data': [{'value': v, 'itemStyle': {'color': c}} for v, c in zip(vol_data, vol_colors)],
            'xAxisIndex': 1,
            'yAxisIndex': 1
        })
        
        # 3. Indicators
        option['series'].append({
            'name': '3x3 DMA',
            'type': 'line',
            'data': [x if pd.notna(x) else None for x in dma_3x3],
            'xAxisIndex': 0,
            'yAxisIndex': 0,
            'smooth': True,
            'lineStyle': {'color': '#00ffff', 'width': 1},
            'symbol': 'none'
        })
        
        option['series'].append({
            'name': '25x5 DMA',
            'type': 'line',
            'data': [x if pd.notna(x) else None for x in dma_25x5],
            'xAxisIndex': 0,
            'yAxisIndex': 0,
            'smooth': True,
            'lineStyle': {'color': '#ff00ff', 'width': 2},
            'symbol': 'none'
        })
        
        # 3. Pattern Structure (Double Repo A-B-C)
        if pattern_name == 'Double Repo' and isinstance(metadata, dict):
            # Extract points
            pts = []
            for pt_name in ['A', 'B', 'C']:
                if pt_name in metadata:
                    pt_date = pd.to_datetime(metadata[pt_name]['date'])
                    if pt_date in sub_df.index:
                        idx = sub_df.index.get_loc(pt_date)
                        price = metadata[pt_name]['price']
                        pts.append([idx, price])
            
            # Connect A-B-C
            if len(pts) >= 3:
                option['series'].append({
                    'name': 'Pattern',
                    'type': 'line',
                    'data': pts,
                    'lineStyle': {'color': '#ffff00', 'width': 2, 'type': 'dashed'},
                    'symbol': 'circle',
                    'symbolSize': 8,
                    'label': {'show': False, 'formatter': '{b}'} # Need custom formatter or separate series for labels
                })
                
                # Add labels for A, B, C
                # We can use markPoint
                mark_data = []
                labels = ['A', 'B', 'C']
                for i, pt in enumerate(pts):
                    mark_data.append({
                        'coord': pt,
                        'value': labels[i],
                        'symbolSize': 24,
                        'itemStyle': {'color': '#ffff00'}
                    })
                
                option['series'][-1]['markPoint'] = {
                    'data': mark_data,
                    'label': {'color': '#000'}
                }
                
                # Visualize RRT/FTP Confirmation
                if metadata.get('rrt_detected'):
                    # Add marker at C or Entry
                    c_date = pd.to_datetime(metadata['C']['date']) if 'C' in metadata else entry_date
                    if c_date in sub_df.index:
                        idx = sub_df.index.get_loc(c_date)
                        price = metadata['C']['price'] if 'C' in metadata else entry_price
                        option['series'].append({
                            'name': 'RRT Confirmation',
                            'type': 'scatter',
                            'data': [[idx, price]],
                            'symbol': 'pin',
                            'symbolSize': 20,
                            'itemStyle': {'color': '#ff00ff'},
                            'label': {'show': True, 'formatter': 'RRT', 'position': 'top', 'color': '#fff'}
                        })
                
                if metadata.get('ftp_detected'):
                    c_date = pd.to_datetime(metadata['C']['date']) if 'C' in metadata else entry_date
                    if c_date in sub_df.index:
                        idx = sub_df.index.get_loc(c_date)
                        price = metadata['C']['price'] if 'C' in metadata else entry_price
                        option['series'].append({
                            'name': 'FTP Confirmation',
                            'type': 'scatter',
                            'data': [[idx, price]],
                            'symbol': 'pin',
                            'symbolSize': 20,
                            'itemStyle': {'color': '#00ffff'},
                            'label': {'show': True, 'formatter': 'FTP', 'position': 'bottom', 'color': '#fff'}
                        })

        elif pattern_name == 'Single Penetration' and isinstance(metadata, dict):
            # Visualize Thrust and Penetration
            if 'thrust_start' in metadata and 'thrust_end' in metadata:
                t_start = metadata['thrust_start']
                t_end = metadata['thrust_end']
                
                t_start_date = pd.to_datetime(t_start['date'])
                t_end_date = pd.to_datetime(t_end['date'])
                
                if t_start_date in sub_df.index and t_end_date in sub_df.index:
                    idx1 = sub_df.index.get_loc(t_start_date)
                    idx2 = sub_df.index.get_loc(t_end_date)
                    
                    option['series'].append({
                        'name': 'Thrust',
                        'type': 'line',
                        'data': [[idx1, t_start['price']], [idx2, t_end['price']]],
                        'lineStyle': {'color': '#ffff00', 'width': 2, 'type': 'solid'},
                        'symbol': 'none'
                    })
            
            if 'penetration' in metadata:
                pen = metadata['penetration']
                pen_date = pd.to_datetime(pen['date'])
                if pen_date in sub_df.index:
                    idx = sub_df.index.get_loc(pen_date)
                    option['series'].append({
                        'name': 'Penetration',
                        'type': 'scatter',
                        'data': [[idx, pen['price']]],
                        'symbol': 'circle',
                        'symbolSize': 8,
                        'itemStyle': {'color': '#ff00ff'}
                    })

            # Visualize RRT/FTP Confirmation
            if metadata.get('rrt_detected'):
                pen = metadata.get('penetration', {})
                pen_date = pd.to_datetime(pen.get('date', entry_date))
                if pen_date in sub_df.index:
                    idx = sub_df.index.get_loc(pen_date)
                    price = pen.get('price', entry_price)
                    option['series'].append({
                        'name': 'RRT Confirmation',
                        'type': 'scatter',
                        'data': [[idx, price]],
                        'symbol': 'pin',
                        'symbolSize': 20,
                        'itemStyle': {'color': '#ff00ff'},
                        'label': {'show': True, 'formatter': 'RRT', 'position': 'top', 'color': '#fff'}
                    })

            if metadata.get('ftp_detected'):
                pen = metadata.get('penetration', {})
                pen_date = pd.to_datetime(pen.get('date', entry_date))
                if pen_date in sub_df.index:
                    idx = sub_df.index.get_loc(pen_date)
                    price = pen.get('price', entry_price)
                    option['series'].append({
                        'name': 'FTP Confirmation',
                        'type': 'scatter',
                        'data': [[idx, price]],
                        'symbol': 'pin',
                        'symbolSize': 20,
                        'itemStyle': {'color': '#00ffff'},
                        'label': {'show': True, 'formatter': 'FTP', 'position': 'bottom', 'color': '#fff'}
                    })

        elif pattern_name == 'Railroad Tracks' and isinstance(metadata, dict):
            # Highlight the two bars
            if 'bar1' in metadata and 'bar2' in metadata:
                b1 = metadata['bar1']
                b2 = metadata['bar2']
                d1 = pd.to_datetime(b1['date'])
                d2 = pd.to_datetime(b2['date'])
                
                if d1 in sub_df.index and d2 in sub_df.index:
                    idx1 = sub_df.index.get_loc(d1)
                    idx2 = sub_df.index.get_loc(d2)
                    
                    # Draw a box or just mark them
                    option['series'].append({
                        'name': 'RRT',
                        'type': 'scatter',
                        'data': [
                            [idx1, (b1['open'] + b1['close'])/2],
                            [idx2, (b2['open'] + b2['close'])/2]
                        ],
                        'symbol': 'rect',
                        'symbolSize': [10, 20], # Approximate
                        'itemStyle': {'color': 'rgba(255, 255, 0, 0.3)', 'borderColor': '#ffff00'}
                    })

        elif pattern_name == 'Failure to Penetrate' and isinstance(metadata, dict):
            if 'penetration' in metadata:
                pen = metadata['penetration']
                pen_date = pd.to_datetime(pen['date'])
                if pen_date in sub_df.index:
                    idx = sub_df.index.get_loc(pen_date)
                    # Mark the failure point
                    option['series'].append({
                        'name': 'FTP',
                        'type': 'scatter',
                        'data': [[idx, pen.get('close', 0)]],
                        'symbol': 'diamond',
                        'symbolSize': 12,
                        'itemStyle': {'color': '#00ffff'}
                    })

        # 4. Trade Execution (Entry -> Exit)
        # Map dates to indices in the sub_df
        try:
            sub_entry_idx = sub_df.index.get_loc(entry_date)
            sub_exit_idx = sub_df.index.get_loc(exit_date)
            
            # entry_price and exit_price are already defined at the top
            
            # Entry Marker
            option['series'].append({
                'name': 'Entry',
                'type': 'scatter',
                'data': [[sub_entry_idx, entry_price]],
                'symbol': 'arrow',
                'symbolSize': 10,
                'symbolRotate': 0 if entry_price < exit_price else 180, # Point up for buy? No, arrow usually points to location
                'itemStyle': {'color': '#ffffff'},
                'label': {'show': True, 'position': 'bottom', 'formatter': 'Entry', 'color': '#fff'}
            })
            
            # Exit Marker
            option['series'].append({
                'name': 'Exit',
                'type': 'scatter',
                'data': [[sub_exit_idx, exit_price]],
                'symbol': 'arrow',
                'symbolRotate': 180 if entry_price < exit_price else 0,
                'symbolSize': 10,
                'itemStyle': {'color': '#ffffff'},
                'label': {'show': True, 'position': 'top', 'formatter': 'Exit', 'color': '#fff'}
            })
            
            # SL and TP Lines
            sl_price = trade_row['Stop Loss']
            tp_price = trade_row['Take Profit']
            
            # Draw horizontal lines from Entry to Exit (or slightly beyond)
            # SL
            option['series'].append({
                'name': 'SL',
                'type': 'line',
                'data': [[sub_entry_idx, sl_price], [sub_exit_idx, sl_price]],
                'lineStyle': {'color': '#ff0000', 'width': 2, 'type': 'dotted'},
                'symbol': 'none',
                'markPoint': {
                    'data': [{
                        'coord': [sub_entry_idx, sl_price],
                        'value': 'SL',
                        'symbolSize': 0,
                        'label': {'position': 'left', 'color': '#ff0000'}
                    }]
                }
            })
            
            # TP
            option['series'].append({
                'name': 'TP',
                'type': 'line',
                'data': [[sub_entry_idx, tp_price], [sub_exit_idx, tp_price]],
                'lineStyle': {'color': '#00ff00', 'width': 2, 'type': 'dotted'},
                'symbol': 'none',
                'markPoint': {
                    'data': [{
                        'coord': [sub_entry_idx, tp_price],
                        'value': 'TP',
                        'symbolSize': 0,
                        'label': {'position': 'left', 'color': '#00ff00'}
                    }]
                }
            })
            
        except KeyError:
            pass
            
        return option
