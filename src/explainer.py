import shap
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SHAPExplainer:
    def __init__(self, detector, feature_names, model_dir="models"):
        self.detector = detector
        self.feature_names = feature_names
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize SHAP explainer
        try:
            # Try newer API first
            self.explainer = shap.Explainer(
                detector.model, 
                masker=shap.maskers.Independent(np.zeros((1, len(feature_names))), 
                feature_names=feature_names)
            )
        except:
            # Fall back to TreeExplainer
            self.explainer = shap.TreeExplainer(
                detector.model, 
                feature_names=feature_names
            )
        
        logger.info("SHAP explainer initialized")

    def explain_batch(self, X, feature_names=None):
        """Explain a batch of instances"""
        if feature_names is None:
            feature_names = self.feature_names
            
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        try:
            # Try newer API
            shap_values = self.explainer(X_array).values
        except:
            # Fall back to older API
            shap_values = self.explainer.shap_values(X_array)
            
        return shap_values

    def explain_instance(self, instance, feature_names=None):
        """Explain a single instance"""
        if feature_names is None:
            feature_names = self.feature_names
            
        if isinstance(instance, pd.Series):
            instance_array = instance.values.reshape(1, -1)
        elif isinstance(instance, pd.DataFrame):
            instance_array = instance.values
        else:
            instance_array = instance.reshape(1, -1)
            
        shap_values = self.explain_batch(instance_array, feature_names)
        return shap_values[0]  # Return for single instance

    def generate_natural_language_explanation(self, shap_values, instance, feature_names, top_k=3):
        """Generate natural language explanation from SHAP values"""
        # Get top contributing features
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(-abs_shap)[:top_k]
        
        explanations = []
        for idx in top_indices:
            feature_name = feature_names[idx]
            shap_value = shap_values[idx]
            feature_value = instance[idx] if hasattr(instance, '__getitem__') else instance.iloc[idx]
            
            direction = "increases" if shap_value > 0 else "decreases"
            explanations.append(
                f"Feature '{feature_name}' (value: {feature_value:.4f}) {direction} "
                f"anomaly score by {abs(shap_value):.4f}"
            )
        
        return "Top contributors: " + "; ".join(explanations)

    def summary_plot(self, shap_values, features, max_display=20, plot_type="dot", save_path=None):
        """Create SHAP summary plot"""
        plt.figure(figsize=(10, 8))
        
        if plot_type == "bar":
            shap.summary_plot(shap_values, features, plot_type="bar", max_display=max_display, show=False)
        else:
            shap.summary_plot(shap_values, features, max_display=max_display, show=False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Summary plot saved to {save_path}")
        
        return plt.gcf()

    def force_plot(self, base_value, shap_values, instance, feature_names, save_path=None):
        """Create SHAP force plot for a single instance"""
        plt.figure(figsize=(12, 4))
        
        # Try to create force plot
        try:
            shap.force_plot(base_value, shap_values, instance, feature_names=feature_names, matplotlib=True, show=False)
        except:
            # Fallback if matplotlib doesn't work
            force_plot = shap.force_plot(base_value, shap_values, instance, feature_names=feature_names)
            shap.save_html(str(save_path).replace('.png', '.html'), force_plot)
            logger.info(f"Force plot saved as HTML to {save_path.replace('.png', '.html')}")
            return None
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Force plot saved to {save_path}")
        
        return plt.gcf()

    def save(self, filename="shap_explainer.joblib"):
        filepath = self.model_dir / filename
        joblib.dump(self, filepath)
        logger.info(f"SHAP explainer saved to {filepath}")

    @classmethod
    def load(cls, filename="shap_explainer.joblib", model_dir="models"):
        filepath = Path(model_dir) / filename
        if not filepath.exists():
            raise FileNotFoundError(f"SHAP explainer not found at {filepath}")
        return joblib.load(filepath)