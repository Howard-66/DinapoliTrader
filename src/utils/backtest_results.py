import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


class BacktestResultsManager:
    """
    Manages saving, loading, and comparing backtest results.
    Results are stored as JSON files in the backtest_results directory.
    """
    
    def __init__(self, results_dir: str = "backtest_results"):
        """
        Initialize the BacktestResultsManager.
        
        Args:
            results_dir: Directory to store backtest results (relative to project root)
        """
        # Get project root (assuming this file is in src/utils/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.results_dir = os.path.join(project_root, results_dir)
        
        # Create directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _clean_metadata(self, metadata: dict) -> dict:
        """
        Recursively clean metadata dictionary by converting Timestamp objects to strings.
        
        Args:
            metadata: Dictionary that may contain Timestamp objects
            
        Returns:
            Cleaned dictionary with Timestamps converted to strings
        """
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, pd.Timestamp):
                cleaned[key] = value.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(value, dict):
                cleaned[key] = self._clean_metadata(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    v.strftime('%Y-%m-%d %H:%M:%S') if isinstance(v, pd.Timestamp) else v
                    for v in value
                ]
            else:
                cleaned[key] = value
        return cleaned
    
    def _serialize_pandas_object(self, obj: Any) -> Any:
        """
        Convert pandas Series/DataFrame to JSON-serializable format.
        
        Args:
            obj: Object to serialize (can be Series, DataFrame, or other types)
            
        Returns:
            JSON-serializable representation
        """
        if isinstance(obj, pd.Series):
            return {
                '_type': 'Series',
                'data': obj.tolist(),
                'index': obj.index.strftime('%Y-%m-%d').tolist() if isinstance(obj.index, pd.DatetimeIndex) else obj.index.tolist()
            }
        elif isinstance(obj, pd.DataFrame):
            return {
                '_type': 'DataFrame',
                'data': obj.to_dict(orient='split'),
                'index_is_datetime': isinstance(obj.index, pd.DatetimeIndex),
                'columns_info': {col: str(obj[col].dtype) for col in obj.columns}
            }
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def _deserialize_pandas_object(self, obj: Any) -> Any:
        """
        Convert JSON representation back to pandas Series/DataFrame.
        
        Args:
            obj: JSON object to deserialize
            
        Returns:
            Original pandas object or unchanged object
        """
        if isinstance(obj, dict) and '_type' in obj:
            if obj['_type'] == 'Series':
                index = pd.to_datetime(obj['index']) if obj['index'] and '-' in str(obj['index'][0]) else obj['index']
                return pd.Series(obj['data'], index=index)
            elif obj['_type'] == 'DataFrame':
                df = pd.DataFrame(**obj['data'])
                if obj.get('index_is_datetime', False):
                    df.index = pd.to_datetime(df.index)
                return df
        return obj
    
    def save_result(self, 
                   symbol: str,
                   start_date: str,
                   end_date: str,
                   parameters: Dict[str, Any],
                   metrics: Dict[str, Any],
                   description: str = "") -> str:
        """
        Save backtest results to a JSON file.
        
        Args:
            symbol: Trading symbol
            start_date: Start date of backtest
            end_date: End date of backtest
            parameters: Dictionary of backtest parameters
            metrics: Dictionary of backtest metrics (from PerformanceAnalyzer.calculate_metrics)
            description: Optional description of this backtest run
            
        Returns:
            Filename of saved result
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_{start_date}_{end_date}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Prepare data structure
        result_data = {
            'metadata': {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'timestamp': timestamp,
                'description': description
            },
            'parameters': parameters,
            'metrics': {}
        }
        
        # Serialize metrics (handle pandas objects)
        for key, value in metrics.items():
            if key == 'Trade Log' and isinstance(value, pd.DataFrame):
                # Special handling for Trade Log - convert timestamps to strings
                trade_log_copy = value.copy()
                for col in trade_log_copy.columns:
                    if pd.api.types.is_datetime64_any_dtype(trade_log_copy[col]):
                        trade_log_copy[col] = trade_log_copy[col].dt.strftime('%Y-%m-%d')
                    elif col == 'Metadata':
                        # Handle Metadata column - convert any Timestamp objects in dictionaries
                        trade_log_copy[col] = trade_log_copy[col].apply(
                            lambda x: self._clean_metadata(x) if isinstance(x, dict) else x
                        )
                result_data['metrics'][key] = self._serialize_pandas_object(trade_log_copy)
            else:
                result_data['metrics'][key] = self._serialize_pandas_object(value)
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def load_result(self, filename: str) -> Dict[str, Any]:
        """
        Load backtest results from a JSON file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Dictionary containing metadata, parameters, and metrics
        """
        filepath = os.path.join(self.results_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Result file not found: {filename}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        # Deserialize metrics
        metrics = {}
        for key, value in result_data['metrics'].items():
            metrics[key] = self._deserialize_pandas_object(value)
            
            # Special handling for Trade Log - convert date strings back to datetime
            if key == 'Trade Log' and isinstance(metrics[key], pd.DataFrame):
                for col in ['Signal Date', 'Entry Date', 'Exit Date']:
                    if col in metrics[key].columns:
                        metrics[key][col] = pd.to_datetime(metrics[key][col])
        
        result_data['metrics'] = metrics
        return result_data
    
    def list_results(self, 
                    symbol: Optional[str] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available backtest results with optional filtering.
        
        Args:
            symbol: Filter by symbol (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)
            
        Returns:
            List of dictionaries containing metadata for each result
        """
        results = []
        
        for filename in os.listdir(self.results_dir):
            if not filename.endswith('.json'):
                continue
            
            try:
                filepath = os.path.join(self.results_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                metadata = data['metadata']
                
                # Apply filters
                if symbol and metadata['symbol'] != symbol:
                    continue
                if start_date and metadata['start_date'] != start_date:
                    continue
                if end_date and metadata['end_date'] != end_date:
                    continue
                
                # Extract key metrics for preview
                metrics_summary = {
                    'Total Trades': data['metrics'].get('Total Trades', 0),
                    'Win Rate': data['metrics'].get('Win Rate', 0.0),
                    'Total Return': data['metrics'].get('Total Return', 0.0),
                    'Sharpe Ratio': data['metrics'].get('Sharpe Ratio', 0.0)
                }
                
                results.append({
                    'filename': filename,
                    'metadata': metadata,
                    'metrics_summary': metrics_summary,
                    'parameters': data.get('parameters', {})
                })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x['metadata']['timestamp'], reverse=True)
        return results
    
    def delete_result(self, filename: str) -> bool:
        """
        Delete a saved backtest result.
        
        Args:
            filename: Name of the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        filepath = os.path.join(self.results_dir, filename)
        
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            return False
        except Exception as e:
            print(f"Error deleting {filename}: {e}")
            return False
    
    def prepare_comparison_data(self, filenames: List[str]) -> Dict[str, Any]:
        """
        Prepare data for comparing multiple backtest results.
        
        Args:
            filenames: List of result filenames to compare
            
        Returns:
            Dictionary containing comparison data
        """
        if not filenames:
            return {}
        
        comparison_data = {
            'results': [],
            'metrics_comparison': [],
            'equity_curves': {},
            'parameter_differences': []
        }
        
        # Load all results
        for filename in filenames:
            try:
                result = self.load_result(filename)
                comparison_data['results'].append({
                    'filename': filename,
                    'description': result['metadata'].get('description', ''),
                    'timestamp': result['metadata']['timestamp'],
                    'result': result
                })
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        if not comparison_data['results']:
            return comparison_data
        
        # Build metrics comparison table
        metric_keys = ['Total Trades', 'Win Rate', 'Avg Return', 'Total Return', 
                      'Annualized Return', 'Max Drawdown', 'Sharpe Ratio', 
                      'Sortino Ratio', 'Profit Factor']
        
        for metric_key in metric_keys:
            row = {'Metric': metric_key}
            for i, item in enumerate(comparison_data['results']):
                label = item['description'] or f"Run {i+1}"
                row[label] = item['result']['metrics'].get(metric_key, 0)
            comparison_data['metrics_comparison'].append(row)
        
        # Extract equity curves for overlay
        for i, item in enumerate(comparison_data['results']):
            label = item['description'] or f"Run {i+1}"
            # Try Floating Equity Curve first, then Equity Curve
            equity_curve = item['result']['metrics'].get('Floating Equity Curve')
            if equity_curve is None:
                equity_curve = item['result']['metrics'].get('Equity Curve')
            if equity_curve is not None:
                comparison_data['equity_curves'][label] = equity_curve
        
        # Identify parameter differences
        if len(comparison_data['results']) > 1:
            base_params = comparison_data['results'][0]['result']['parameters']
            
            for key in base_params.keys():
                values = [r['result']['parameters'].get(key) for r in comparison_data['results']]
                # Convert values to strings for comparison (handles lists, etc.)
                str_values = [str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v for v in values]
                # Check if values differ
                if len(set(str(v) for v in str_values)) > 1:
                    diff_row = {'Parameter': key}
                    for i, item in enumerate(comparison_data['results']):
                        label = item['description'] or f"Run {i+1}"
                        diff_row[label] = item['result']['parameters'].get(key)
                    comparison_data['parameter_differences'].append(diff_row)
        
        return comparison_data
