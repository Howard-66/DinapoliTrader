import unittest
import os
import sys
import json
import tempfile
import shutil
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.backtest_results import BacktestResultsManager


class TestBacktestResultsManager(unittest.TestCase):
    """Test cases for BacktestResultsManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for test results
        self.test_dir = tempfile.mkdtemp()
        self.manager = BacktestResultsManager(results_dir=self.test_dir)
        
        # Create sample metrics data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.sample_metrics = {
            'Total Trades': 10,
            'Win Rate': 0.6,
            'Avg Return': 0.02,
            'Total Return': 0.15,
            'Annualized Return': 0.20,
            'Max Drawdown': -0.10,
            'Sharpe Ratio': 1.5,
            'Sortino Ratio': 2.0,
            'Profit Factor': 2.5,
            'Equity Curve': pd.Series(np.linspace(100000, 115000, 100), index=dates),
            'Floating Equity Curve': pd.Series(np.linspace(100000, 115000, 100), index=dates),
            'Drawdown Curve': pd.Series(np.linspace(0, -0.05, 100), index=dates),
            'Monthly Returns': pd.DataFrame({
                1: [0.01, 0.02],
                2: [0.015, 0.025]
            }, index=[2020, 2021]),
            'Trade Log': pd.DataFrame({
                'Signal Date': pd.to_datetime(['2020-01-05', '2020-02-10']),
                'Pattern': ['Double Repo', 'Single Penetration'],
                'Entry Date': pd.to_datetime(['2020-01-06', '2020-02-11']),
                'Entry Price': [100.0, 105.0],
                'Stop Loss': [98.0, 103.0],
                'Take Profit': [105.0, 110.0],
                'Exit Date': pd.to_datetime(['2020-01-15', '2020-02-20']),
                'Exit Price': [104.0, 108.0],
                'Quantity': [100, 100],
                'PnL Amount': [400.0, 300.0],
                'PnL %': [0.04, 0.0286],
                'Confidence': [0.75, 0.80],
                'Metadata': [{}, {}]
            }),
            'Strategy Breakdown': pd.DataFrame({
                'Strategy': ['Double Repo', 'Single Penetration'],
                'Trades': [5, 5],
                'Total PnL': [2000.0, 1500.0],
                'Win Rate': [0.6, 0.6],
                'Contribution %': [57.14, 42.86]
            })
        }
        
        self.sample_parameters = {
            'selected_strategies': ['Double Repo', 'Single Penetration'],
            'enable_trend_filter': True,
            'enable_mtf_filter': False,
            'min_confidence': 0.5,
            'sl_mode': 'Pattern Based',
            'tp_mode': 'Pattern Based (Fib)',
            'holding_period': 10,
            'stop_loss': 0.02,
            'atr_multiplier': 2.0,
            'take_profit': 0.05,
            'initial_capital': 100000.0,
            'use_dynamic_sizing': True,
            'risk_per_trade': 0.02
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_save_result(self):
        """Test saving backtest results"""
        filename = self.manager.save_result(
            symbol='TEST.SH',
            start_date='2020-01-01',
            end_date='2020-12-31',
            parameters=self.sample_parameters,
            metrics=self.sample_metrics,
            description='Test backtest'
        )
        
        # Check that file was created
        self.assertTrue(filename.endswith('.json'))
        filepath = os.path.join(self.test_dir, filename)
        self.assertTrue(os.path.exists(filepath))
        
        # Check file contents
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertEqual(data['metadata']['symbol'], 'TEST.SH')
        self.assertEqual(data['metadata']['description'], 'Test backtest')
        self.assertEqual(data['parameters']['holding_period'], 10)
        self.assertEqual(data['metrics']['Total Trades'], 10)
    
    def test_load_result(self):
        """Test loading backtest results"""
        # First save a result
        filename = self.manager.save_result(
            symbol='TEST.SH',
            start_date='2020-01-01',
            end_date='2020-12-31',
            parameters=self.sample_parameters,
            metrics=self.sample_metrics,
            description='Test backtest'
        )
        
        # Load it back
        loaded_data = self.manager.load_result(filename)
        
        # Verify metadata
        self.assertEqual(loaded_data['metadata']['symbol'], 'TEST.SH')
        self.assertEqual(loaded_data['metadata']['description'], 'Test backtest')
        
        # Verify parameters
        self.assertEqual(loaded_data['parameters']['holding_period'], 10)
        
        # Verify metrics
        self.assertEqual(loaded_data['metrics']['Total Trades'], 10)
        self.assertAlmostEqual(loaded_data['metrics']['Win Rate'], 0.6)
        
        # Verify pandas objects were reconstructed
        self.assertIsInstance(loaded_data['metrics']['Equity Curve'], pd.Series)
        self.assertIsInstance(loaded_data['metrics']['Trade Log'], pd.DataFrame)
        
        # Verify Trade Log dates are datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(
            loaded_data['metrics']['Trade Log']['Entry Date']
        ))
    
    def test_list_results(self):
        """Test listing saved results"""
        # Save multiple results
        self.manager.save_result(
            symbol='TEST1.SH',
            start_date='2020-01-01',
            end_date='2020-12-31',
            parameters=self.sample_parameters,
            metrics=self.sample_metrics,
            description='Test 1'
        )
        
        self.manager.save_result(
            symbol='TEST2.SH',
            start_date='2020-01-01',
            end_date='2020-12-31',
            parameters=self.sample_parameters,
            metrics=self.sample_metrics,
            description='Test 2'
        )
        
        # List all results
        all_results = self.manager.list_results()
        self.assertEqual(len(all_results), 2)
        
        # List filtered by symbol
        filtered_results = self.manager.list_results(symbol='TEST1.SH')
        self.assertEqual(len(filtered_results), 1)
        self.assertEqual(filtered_results[0]['metadata']['symbol'], 'TEST1.SH')
    
    def test_delete_result(self):
        """Test deleting a saved result"""
        # Save a result
        filename = self.manager.save_result(
            symbol='TEST.SH',
            start_date='2020-01-01',
            end_date='2020-12-31',
            parameters=self.sample_parameters,
            metrics=self.sample_metrics,
            description='Test backtest'
        )
        
        # Verify it exists
        filepath = os.path.join(self.test_dir, filename)
        self.assertTrue(os.path.exists(filepath))
        
        # Delete it
        success = self.manager.delete_result(filename)
        self.assertTrue(success)
        
        # Verify it's gone
        self.assertFalse(os.path.exists(filepath))
    
    def test_serialization(self):
        """Test pandas object serialization and deserialization"""
        # Save and load
        filename = self.manager.save_result(
            symbol='TEST.SH',
            start_date='2020-01-01',
            end_date='2020-12-31',
            parameters=self.sample_parameters,
            metrics=self.sample_metrics,
            description='Test serialization'
        )
        
        loaded_data = self.manager.load_result(filename)
        
        # Check Series
        original_equity = self.sample_metrics['Equity Curve']
        loaded_equity = loaded_data['metrics']['Equity Curve']
        
        self.assertIsInstance(loaded_equity, pd.Series)
        self.assertEqual(len(loaded_equity), len(original_equity))
        np.testing.assert_array_almost_equal(
            loaded_equity.values,
            original_equity.values
        )
        
        # Check DataFrame
        original_trades = self.sample_metrics['Trade Log']
        loaded_trades = loaded_data['metrics']['Trade Log']
        
        self.assertIsInstance(loaded_trades, pd.DataFrame)
        self.assertEqual(len(loaded_trades), len(original_trades))
        self.assertEqual(list(loaded_trades.columns), list(original_trades.columns))
    
    def test_comparison_data_preparation(self):
        """Test comparison data preparation"""
        # Save two results with different parameters
        params1 = self.sample_parameters.copy()
        params1['holding_period'] = 10
        
        params2 = self.sample_parameters.copy()
        params2['holding_period'] = 20
        
        filename1 = self.manager.save_result(
            symbol='TEST.SH',
            start_date='2020-01-01',
            end_date='2020-12-31',
            parameters=params1,
            metrics=self.sample_metrics,
            description='HP 10'
        )
        
        filename2 = self.manager.save_result(
            symbol='TEST.SH',
            start_date='2020-01-01',
            end_date='2020-12-31',
            parameters=params2,
            metrics=self.sample_metrics,
            description='HP 20'
        )
        
        # Prepare comparison data
        comparison_data = self.manager.prepare_comparison_data([filename1, filename2])
        
        # Verify structure
        self.assertEqual(len(comparison_data['results']), 2)
        self.assertTrue(len(comparison_data['metrics_comparison']) > 0)
        self.assertEqual(len(comparison_data['equity_curves']), 2)
        
        # Verify parameter differences detected
        self.assertTrue(len(comparison_data['parameter_differences']) > 0)
        param_diff_df = pd.DataFrame(comparison_data['parameter_differences'])
        self.assertIn('holding_period', param_diff_df['Parameter'].values)


if __name__ == '__main__':
    unittest.main()
