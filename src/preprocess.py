import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, MODELS_DIR, RANDOM_STATE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_features, categorical_features, scaling_method='standard', seed=RANDOM_STATE):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.scaling_method = scaling_method
        self.seed = seed
        
        # Initialize transformers
        self.num_imputer = SimpleImputer(strategy='median')
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("scaling_method must be 'standard' or 'robust'")
        
        # For categorical features (though creditcard dataset has none)
        if categorical_features:
            from sklearn.preprocessing import OrdinalEncoder
            self.cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        else:
            self.cat_encoder = None
        
        self.feature_names_out_ = None

    def fit(self, df, y=None):
        logger.info("Fitting preprocessor...")
        
        # Fit numeric transformers
        numeric_data = df[self.numeric_features]
        self.num_imputer.fit(numeric_data)
        num_imputed = self.num_imputer.transform(numeric_data)
        self.scaler.fit(num_imputed)
        
        # Fit categorical transformers if needed
        if self.categorical_features and self.cat_encoder is not None:
            categorical_data = df[self.categorical_features].fillna('MISSING')
            self.cat_encoder.fit(categorical_data)
        
        # Set feature names for output
        self.feature_names_out_ = self.numeric_features.copy()
        if self.categorical_features:
            self.feature_names_out_.extend(self.categorical_features)
        
        return self

    def transform(self, df):
        logger.info("Transforming data...")
        
        # Transform numeric features
        numeric_data = df[self.numeric_features]
        num_imputed = self.num_imputer.transform(numeric_data)
        num_scaled = self.scaler.transform(num_imputed)
        
        # Transform categorical features if needed
        if self.categorical_features and self.cat_encoder is not None:
            categorical_data = df[self.categorical_features].fillna('MISSING')
            cat_encoded = self.cat_encoder.transform(categorical_data)
            X = np.hstack([num_scaled, cat_encoded])
        else:
            X = num_scaled
        
        return X

    def fit_transform(self, df, y=None):
        return self.fit(df).transform(df)

    def get_feature_names_out(self):
        if self.feature_names_out_ is None:
            raise ValueError("Preprocessor has not been fitted yet")
        return self.feature_names_out_

    def save(self, filename="preprocessor.joblib"):
        filepath = MODELS_DIR / filename
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filename="preprocessor.joblib"):
        filepath = MODELS_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor not found at {filepath}")
        return joblib.load(filepath)