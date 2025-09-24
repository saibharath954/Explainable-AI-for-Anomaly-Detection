import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator
from src.config import MODELS_DIR, ANOMALY_THRESHOLD
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IsfDetector(BaseEstimator):
    def __init__(self, n_estimators=200, max_samples='auto', contamination='auto',
                 random_state=42, n_jobs=None, threshold_percentile=ANOMALY_THRESHOLD):
        """Isolation Forest wrapper with robust anomaly thresholding"""
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.threshold_percentile = threshold_percentile
        self.threshold_ = None
        self.is_fitted_ = False

    def fit(self, X, y=None):
        logger.info("Fitting Isolation Forest...")
        self.model.fit(X)

        # Calculate anomaly scores
        scores = self.decision_function(X)

        # Threshold based on training data distribution
        self.threshold_ = np.percentile(scores, self.threshold_percentile)

        self.is_fitted_ = True
        logger.info(f"Fitted Isolation Forest with threshold: {self.threshold_:.4f}")
        return self

    def decision_function(self, X):
        """Return anomaly scores (higher = more anomalous)."""
        return -self.model.decision_function(X)  # invert sklearn convention

    def predict(self, X, threshold=None):
        """Predict anomalies (1 = anomaly, 0 = normal)."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        scores = self.decision_function(X)

        if threshold is None:
            threshold = self.threshold_

        preds = (scores >= threshold).astype(int)
        return preds, scores

    def predict_proba(self, X):
        """Return normalized anomaly scores in [0,1] (not true probabilities)."""
        scores = self.decision_function(X)
        min_score, max_score = np.min(scores), np.max(scores)
        if max_score == min_score:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def set_threshold(self, threshold):
        """Set custom threshold for anomaly detection."""
        self.threshold_ = threshold

    def save(self, filename="isolation_forest.joblib"):
        filepath = MODELS_DIR / filename
        joblib.dump(self, filepath)
        logger.info(f"Detector saved to {filepath}")

    @classmethod
    def load(cls, filename="isolation_forest.joblib"):
        filepath = MODELS_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Detector not found at {filepath}")
        return joblib.load(filepath)
