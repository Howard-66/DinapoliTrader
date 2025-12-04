import sys
import os
import json
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.backtest_results import BacktestResultsManager

def test_json_serialization():
    print("Testing JSON Serialization of Numpy Types...")
    
    manager = BacktestResultsManager()
    
    # Test cases
    data = {
        'bool_true': np.bool_(True),
        'bool_false': np.bool_(False),
        'int64': np.int64(123),
        'float64': np.float64(123.456),
        'nan': np.nan,
        'list_with_bool': [np.bool_(True), 1, 2],
        'dict_with_bool': {'a': np.bool_(False)}
    }
    
    # We need to manually apply the serialization logic as save_result does
    # save_result iterates over keys and calls _serialize_pandas_object
    
    serialized_data = {}
    for key, value in data.items():
        # For lists and dicts, _serialize_pandas_object doesn't recurse by default unless it's a Series/DataFrame
        # But let's check if _serialize_pandas_object handles the scalar types correctly
        serialized_data[key] = manager._serialize_pandas_object(value)
        
    # Check if it dumps to JSON without error
    try:
        json_str = json.dumps(serialized_data)
        print("✅ JSON dump successful.")
        print(f"Serialized: {json_str}")
    except TypeError as e:
        print(f"❌ JSON dump failed: {e}")
        raise e

    # Verify values
    if serialized_data['bool_true'] is True:
        print("✅ np.bool_(True) -> True")
    else:
        print(f"❌ np.bool_(True) -> {type(serialized_data['bool_true'])}")

if __name__ == "__main__":
    test_json_serialization()
