import pytest
import numpy as np
from src.detector import IsfDetector
from src.config import RANDOM_STATE

def test_detector_fit_predict():
    """Test that detector can fit and predict"""
    # Create test data
    np.random.seed(RANDOM_STATE)
    n_samples = 100
    n_features = 5
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # Initialize detector
    detector = IsfDetector(n_estimators=10, random_state=RANDOM_STATE)
    
    # Fit detector
    detector.fit(X)
    
    # Predict
    predictions, scores = detector.predict(X)
    
    # Check output shapes
    assert predictions.shape[0] == n_samples
    assert scores.shape[0] == n_samples
    
    # Check predictions are binary
    assert np.all(np.isin(predictions, [0, 1]))

def test_detector_threshold():
    """Test that detector threshold works correctly"""
    # Create test data
    np.random.seed(RANDOM_STATE)
    n_samples = 50
    n_features = 5
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # Initialize detector
    detector = IsfDetector(n_estimators=10, random_state=RANDOM_STATE)
    detector.fit(X)
    
    # Set custom threshold
    custom_threshold = detector.threshold_ + 1.0
    detector.set_threshold(custom_threshold)
    
    # Predict with custom threshold
    predictions, _ = detector.predict(X)
    
    # Should have fewer anomalies with higher threshold
    predictions_default, _ = detector.predict(X, threshold=detector.threshold_)
    assert np.sum(predictions) <= np.sum(predictions_default)