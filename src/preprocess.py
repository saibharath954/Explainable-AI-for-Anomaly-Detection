# src/preprocess.py
"""
Defines the Preprocessor class for scaling data.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import logging
import config

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Preprocessor for the XAD framework.
    Applies StandardScaler to 'Time' and 'Amount' features as per.
    """
    def __init__(self, features_to_scale):
        self.features_to_scale = features_to_scale
        self.preprocessor = None
        self.feature_names = None

    def _get_column_transformer(self):
        """Creates the ColumnTransformer."""
        # 'passthrough' ensures that all other columns (V1-V28) are kept unchanged
        return ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), self.features_to_scale)
            ],
            remainder='passthrough'
        )

    def fit(self, df):
        """Fits the preprocessor on the training data."""
        logger.info("Fitting preprocessor...")
        self.preprocessor = self._get_column_transformer()
        
        # We only fit on the features, not the target
        X = df.drop(config.TARGET, axis=1, errors='ignore')
        self.preprocessor.fit(X)
        
        # Get feature names in the correct order after transformation
        self._set_feature_names_out(X)
        logger.info("Preprocessor fitted.")

    def transform(self, df):
        """Transforms the data using the fitted preprocessor."""
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        
        X = df.drop(config.TARGET, axis=1, errors='ignore')
        X_transformed = self.preprocessor.transform(X)
        
        # Return as a DataFrame to maintain feature names
        return pd.DataFrame(X_transformed, columns=self.get_feature_names_out(), index=X.index)

    def fit_transform(self, df):
        """Fits and transforms the data."""
        self.fit(df)
        return self.transform(df)

    def _set_feature_names_out(self, df):
        """Helper to get the correct order of feature names after transformation."""
        # Get names of scaled features
        scaled_feature_names = self.features_to_scale
        
        # Get names of passthrough features
        all_feature_names = df.columns.tolist()
        passthrough_feature_names = [
            f for f in all_feature_names if f not in self.features_to_scale
        ]
        
        # The transformer outputs scaled features first, then passthrough features
        self.feature_names = scaled_feature_names + passthrough_feature_names

    def get_feature_names_out(self):
        """Returns the list of feature names in the order of the transform."""
        if self.feature_names is None:
            raise RuntimeError("Must fit preprocessor to get feature names.")
        return self.feature_names

    def save(self, path=config.PREPROCESSOR_PATH):
        """Saves the preprocessor to disk."""
        joblib.dump(self.preprocessor, path)
        joblib.dump(self.feature_names, str(path) + "_features")
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path=config.PREPROCESSOR_PATH):
        """Loads the preprocessor from disk."""
        preprocessor_obj = joblib.load(path)
        feature_names = joblib.load(str(path) + "_features")
        
        instance = cls(features_to_scale=feature_names[:len(config.FEATURES_TO_SCALE)])
        instance.preprocessor = preprocessor_obj
        instance.feature_names = feature_names
        logger.info(f"Preprocessor loaded from {path}")
        return instance