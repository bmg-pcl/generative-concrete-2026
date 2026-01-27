import os
import pytest
import pandas as pd
from src.data_fetcher import load_data, LOCAL_FILE

def test_dataset_exists():
    """Verify the dataset file is downloaded."""
    assert os.path.exists(LOCAL_FILE), f"Dataset file {LOCAL_FILE} not found. Run 'python -m src.data_fetcher' first."

def test_data_loading():
    """Verify data loads with correct columns and non-empty."""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    
    expected_columns = [
        "cement", "slag", "ash", "water", "superplasticizer", 
        "coarse_agg", "fine_agg", "age", "strength"
    ]
    assert list(df.columns) == expected_columns

def test_data_types():
    """Verify data types are numeric."""
    df = load_data()
    for col in df.columns:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} is not numeric"
