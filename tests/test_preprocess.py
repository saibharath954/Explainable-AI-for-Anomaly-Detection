import pytest
import numpy as np
import pandas as pd
from src.preprocess import Preprocessor
from tests.test_config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, RANDOM_STATE

def test_preprocessor_fit_transform():
    """Test that preprocessor can fit and transform data"""
    # Create test data
    np.random.seed(RANDOM_STATE)
    n_samples = 100
    data = {
        'Time': np.random.normal(0, 1, n_samples),
        'V1': np.random.normal(0, 1, n_samples),
        'Amount': np.random.exponential(10, n_samples),
        'Class': np.random.randint(0, 2, n_samples)
    }
    df = pd.DataFrame(data)
    
    # Initialize preprocessor
    preprocessor = Preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    
    # Fit and transform
    X = preprocessor.fit_transform(df)
    
    # Check output shape
    assert X.shape[0] == n_samples
    assert X.shape[1] == len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    
    # Check no NaN values
    assert not np.any(np.isnan(X))

def test_preprocessor_save_load(tmp_path):
    """Test that preprocessor can be saved and loaded"""
    # Create test data
    np.random.seed(RANDOM_STATE)
    n_samples = 50
    data = {
        'Time': np.random.normal(0, 1, n_samples),
        'V1': np.random.normal(0, 1, n_samples),
        'Amount': np.random.exponential(10, n_samples),
        'Class': np.random.randint(0, 2, n_samples)
    }
    df = pd.DataFrame(data)
    
    # Initialize and fit preprocessor
    preprocessor = Preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    preprocessor.fit(df)
    
    # Save preprocessor
    save_path = tmp_path / "test_preprocessor.joblib"
    preprocessor.save(save_path)
    
    # Load preprocessor
    loaded_preprocessor = Preprocessor.load(save_path)
    
    # Transform with both preprocessors and compare
    X_original = preprocessor.transform(df)
    X_loaded = loaded_preprocessor.transform(df)
    
    assert np.allclose(X_original, X_loaded)