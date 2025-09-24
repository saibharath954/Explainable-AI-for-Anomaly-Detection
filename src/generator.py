import pandas as pd
import numpy as np
from ctgan import CTGAN   # standalone CTGAN package
from pathlib import Path
import joblib
import logging

# Try importing SDV CTGAN depending on version
try:
    # New API (SDV >= 1.0)
    from sdv.single_table import CTGANSynthesizer as SDV_CTGAN
    SDV_API = "new"
except ImportError:
    try:
        # Old API (SDV < 1.0)
        from sdv.tabular import CTGAN as SDV_CTGAN
        SDV_API = "old"
    except ImportError:
        SDV_CTGAN = None
        SDV_API = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudCTGAN:
    def __init__(self, categorical_columns=None, epochs=300, batch_size=500,
                 cuda=False, verbose=True, use_sdv=False, random_state=42,
                 metadata=None):
        self.categorical_columns = categorical_columns or []
        self.epochs = epochs
        self.batch_size = batch_size
        self.cuda = cuda
        self.verbose = verbose
        self.use_sdv = use_sdv
        self.random_state = random_state
        self.metadata = metadata  # required for new SDV API

        if use_sdv:
            if SDV_CTGAN is None:
                raise ImportError("SDV CTGAN not available. Install sdv or disable use_sdv.")
            
            if SDV_API == "old":
                # Old SDV CTGAN API
                self.ctgan = SDV_CTGAN(
                    epochs=epochs,
                    batch_size=batch_size,
                    cuda=cuda,
                    verbose=verbose,
                    random_state=random_state
                )
            else:
                # New SDV CTGAN API (>=1.0)
                if metadata is None:
                    raise ValueError("metadata is required when using SDV >=1.0 CTGAN.")
                self.ctgan = SDV_CTGAN(metadata)
        else:
            # Standalone CTGAN
            self.ctgan = CTGAN(
                epochs=epochs,
                batch_size=batch_size,
                cuda=cuda,
                verbose=verbose,
            )

        self.is_fitted = False

    def fit(self, df, discrete_columns=None):
        """Fit CTGAN on the provided dataframe"""
        logger.info("Fitting CTGAN...")

        if discrete_columns is None:
            discrete_columns = self.categorical_columns

        if self.use_sdv:
            if SDV_API == "old":
                self.ctgan.fit(df)
            else:
                self.ctgan.fit(df)
        else:
            self.ctgan.fit(df, discrete_columns)

        self.is_fitted = True
        logger.info("CTGAN fitted successfully")
        return self

    def sample(self, n_samples):
        """Generate synthetic samples"""
        if not self.is_fitted:
            raise ValueError("CTGAN must be fitted before sampling")

        logger.info(f"Generating {n_samples} synthetic samples...")
        synthetic_data = self.ctgan.sample(n_samples)
        return synthetic_data

    def evaluate_quality(self, real_data, synthetic_data, n_samples=1000):
        """Evaluate the quality of synthetic data"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        # Sample from both datasets
        real_sample = real_data.sample(min(n_samples, len(real_data)), random_state=self.random_state)
        synthetic_sample = synthetic_data.sample(min(n_samples, len(synthetic_data)), random_state=self.random_state)

        # Create labels
        real_sample['is_real'] = 1
        synthetic_sample['is_real'] = 0

        # Combine and shuffle
        combined = pd.concat([real_sample, synthetic_sample], ignore_index=True)
        combined = combined.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # Prepare features and target
        X = combined.drop('is_real', axis=1)
        y = combined['is_real']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )

        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)

        logger.info(f"Quality evaluation AUC: {auc_score:.4f}")
        return auc_score

    def save(self, filename="ctgan_model.pkl"):
        filepath = Path("models") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"CTGAN model saved to {filepath}")

    @classmethod
    def load(cls, filename="ctgan_model.pkl"):
        filepath = Path("models") / filename
        if not filepath.exists():
            raise FileNotFoundError(f"CTGAN model not found at {filepath}")
        return joblib.load(filepath)
