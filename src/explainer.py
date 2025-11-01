# src/explainer.py
"""
Defines the SHAPExplainer class for model interpretability.
Handles both Tree and Deep Learning models.
"""

import shap
import numpy as np
import pandas as pd
import joblib
import logging
import matplotlib.pyplot as plt
import config
from tensorflow.keras.models import Model
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """
    Wrapper for SHAP to explain the detection model.
    """
    def __init__(self, detector, background_data, feature_names):
        """
        Initializes the explainer.
        
        Args:
            detector: The fitted detector object (e.g., AutoencoderDetector).
            background_data: A sample of data (e.g., from X_train) 
                             for the explainer to use as a background.
            feature_names: List of feature names.
        """
        self.model = detector.model
        self.feature_names = feature_names
        self.background_data = background_data
        
        # *** NEW: Select the correct SHAP explainer ***
        if isinstance(self.model, Model): # Keras Model
            logger.info("Initializing shap.DeepExplainer for Keras model.")
            # We must provide background data for DeepExplainer
            self.explainer = shap.DeepExplainer(self.model, self.background_data)
        elif isinstance(self.model, IsolationForest):
             logger.info("Initializing shap.TreeExplainer for Isolation Forest.")
             self.explainer = shap.TreeExplainer(self.model)
        else:
            logger.warning("Model type not recognized for SHAP. Using KernelExplainer.")
            self.explainer = shap.KernelExplainer(self.model, self.background_data)
            
        logger.info("SHAP explainer initialized.")

    def explain_batch(self, X):
        """
        Generates SHAP values for a batch of instances.
        X should be a pandas DataFrame or 2D numpy array.
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        # For Autoencoder (DeepExplainer), shap_values will have multiple
        # outputs if the model has multiple layers. We are interested in
        # the sum of SHAP values across all features for the *reconstruction*,
        # which is what shap.DeepExplainer(model, data) returns.
        # For an autoencoder, this explains the *reconstructed output*.
        shap_values = self.explainer.shap_values(X_values)
        
        if isinstance(self.model, Model):
            # DeepExplainer for AE returns shap_values per output feature.
            # We sum them to get a single contribution value per input feature.
            # This represents the feature's contribution to its own reconstruction.
            # A high SHAP value (positive or negative) means the feature
            # strongly influenced the (likely poor) reconstruction.
            
            # The output of shap_values is a list (one per output layer)
            # Our model has one output layer, so we take shap_values[0]
            return shap_values[0]
        else:
            # TreeExplainer returns a single array
            return shap_values

    def generate_natural_language_explanation(self, shap_values_instance, instance_values):
        """
        Generates a plain-language explanation for a single anomaly.
        """
        # For Autoencoders, we look for the features with the
        # LARGEST absolute SHAP values. These are the features that
        # most contributed to the reconstruction error (the anomaly score).
        
        shap_series = pd.Series(np.abs(shap_values_instance), index=self.feature_names)
        contributions = shap_series.sort_values(ascending=False)
        
        top_contributors = contributions.head(3)
        
        explanation = "Instance flagged as anomalous. Top 3 contributing factors:\n"
        for feature, shap_val in top_contributors.items():
            feature_idx = self.feature_names.index(feature)
            feature_val = instance_values[feature_idx]
            explanation += (
                f"  - **{feature}** (value: {feature_val:.4f}) "
                f"strongly contributed to the anomaly (SHAP contribution: {shap_val:.4f}).\n"
            )
        
        return explanation

    def summary_plot(self, shap_values, X, save_path=None):
        """
        Generates and saves a SHAP summary plot.
        """
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names)

        plt.figure()
        # Use plot_type='bar' for a clear view of global feature importance
        shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def save(self, path=config.EXPLAINER_PATH):
        """Saves the explainer (or its config) to disk."""
        # The explainer itself can be large, we'll save the config
        # and rebuild it on load if needed.
        # For simplicity in this step, we just save the feature names.
        joblib.dump(self.feature_names, path)
        logger.info(f"SHAP configuration saved to {path}")

    @classmethod
    def load(cls, detector, background_data, path=config.EXPLAINER_PATH):
        """Loads the explainer from disk."""
        logger.info(f"Loading SHAP explainer configuration from {path}")
        feature_names = joblib.load(path)
        return cls(detector, background_data, feature_names)