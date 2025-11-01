"""
Defines the IsfDetector class.
"""

from sklearn.ensemble import IsolationForest
import joblib
import logging
import config

logger = logging.getLogger(__name__)

class IsfDetector:
    """
    Wrapper around the Isolation Forest anomaly detection model.
    Provides fitting, scoring, predicting, and persistence utilities.
    """

    def __init__(self, **params):
        """
        Initializes the Isolation Forest model with validated parameters.
        Ensures 'contamination' is within the allowed range (0.0, 0.5].
        """
        # Extract contamination safely from params
        contamination = params.pop("contamination", "auto")

        # Validate contamination value
        if isinstance(contamination, (int, float)):
            if contamination <= 0.0:
                logger.warning(f"Invalid contamination={contamination}, setting to 0.001")
                contamination = 0.001
            elif contamination > 0.5:
                logger.warning(f"Contamination too high ({contamination}), capping at 0.5")
                contamination = 0.5

        # Initialize model
        self.model = IsolationForest(contamination=contamination, **params)
        self.threshold_ = None
        logger.info(f"Isolation Forest detector initialized with contamination={contamination}")

    def fit(self, X):
        """
        Fits the Isolation Forest model on input data.
        Typically trained on the mixed (normal + fraud) dataset.
        """
        if X is None or len(X) == 0:
            raise ValueError("Cannot fit IsolationForest on empty dataset.")

        logger.info("Fitting Isolation Forest...")
        self.model.fit(X)

        # Save learned threshold (offset)
        self.threshold_ = getattr(self.model, "offset_", None)
        logger.info(f"Fitted Isolation Forest with score threshold: {self.threshold_:.4f}")

    def get_scores(self, X):
        """
        Returns raw anomaly scores (higher = more normal).
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        if self.threshold_ is None:
            raise RuntimeError("Detector must be fitted before getting scores.")

        return self.model.decision_function(X)

    def predict(self, X, threshold=None):
        """
        Predicts anomalies: 1 for anomalous, 0 for normal.
        Uses the learned threshold unless overridden.
        """
        if self.threshold_ is None:
            raise RuntimeError("Detector must be fitted before predicting.")

        raw_scores = self.get_scores(X)
        anomaly_scores = -raw_scores  # invert: higher = more anomalous

        if threshold is None:
            threshold = -self.threshold_

        predictions = (anomaly_scores > threshold).astype(int)
        return predictions

    def save(self, path=config.DETECTOR_PATH):
        """Saves the trained detector to disk."""
        try:
            joblib.dump(self, path)
            logger.info(f"Detector saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save detector: {e}")

    @classmethod
    def load(cls, path=config.DETECTOR_PATH):
        """Loads a detector instance from disk."""
        try:
            logger.info(f"Loading detector from {path}")
            return joblib.load(path)
        except FileNotFoundError:
            logger.error(f"Detector file not found at {path}")
            return None
        except Exception as e:
            logger.error(f"Failed to load detector: {e}")
            return None
