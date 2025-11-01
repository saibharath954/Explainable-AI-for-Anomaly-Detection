# src/explainer.py
"""
Defines the SHAPExplainer class for model interpretability.
"""

import shap
import numpy as np
import pandas as pd
import joblib
import logging
import matplotlib.pyplot as plt
import config

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """
    Wrapper for the SHAP TreeExplainer [cite: 61] to explain the Isolation Forest.
    """
    def __init__(self, detector, feature_names):
        if not hasattr(detector, 'model'):
            raise ValueError("Detector object must have a 'model' attribute.")
            
        # Use TreeExplainer for tree-based models like Isolation Forest [cite: 61]
        self.explainer = shap.TreeExplainer(detector.model)
        self.feature_names = feature_names
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
            
        shap_values = self.explainer.shap_values(X_values)
        return shap_values

    def generate_natural_language_explanation(self, shap_values_instance, instance_values):
        """
        Generates a plain-language explanation for a single anomaly[cite: 108].
        """
        
        # Per[cite: 63], negative SHAP values drive the score down (more anomalous)
        # We find the features with the most negative SHAP values
        
        # Create a Series for easy sorting
        shap_series = pd.Series(shap_values_instance, index=self.feature_names)
        
        # Sort by SHAP value (most negative first)
        contributions = shap_series.sort_values()
        
        # Get top 3 negative contributors
        top_negative_contributors = contributions[contributions < 0].head(3)
        
        if top_negative_contributors.empty:
            return "This instance was not flagged as anomalous by SHAP analysis (no negative contributors)."

        explanation = "Instance flagged as anomalous. Top 3 contributing factors:\n"
        for feature, shap_val in top_negative_contributors.items():
            feature_idx = self.feature_names.index(feature)
            feature_val = instance_values[feature_idx]
            explanation += (
                f"  - **{feature}** (value: {feature_val:.4f}) "
                f"strongly contributed to the anomaly (SHAP: {shap_val:.4f}).\n"
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
        shap.summary_plot(shap_values, X_df, show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    def save(self, path=config.EXPLAINER_PATH):
        """Saves the explainer to disk."""
        joblib.dump(self, path)
        logger.info(f"SHAP explainer saved to {path}")

    @classmethod
    def load(cls, path=config.EXPLAINER_PATH):
        """Loads the explainer from disk."""
        logger.info(f"Loading SHAP explainer from {path}")
        return joblib.load(path)