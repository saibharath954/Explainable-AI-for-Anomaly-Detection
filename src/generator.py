"""
Defines the FraudCTGAN class for synthetic data generation and evaluation.
"""

import pandas as pd
import logging
import pickle
from ctgan import CTGAN
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import config

logger = logging.getLogger(__name__)

class FraudCTGAN:
    """
    Wrapper for the CTGAN synthesizer to generate synthetic fraud data.
    """

    def __init__(self, **params):
        self.model = CTGAN(**params)
        self.metadata = None
        logger.info("CTGAN synthesizer initialized.")

    def _build_metadata(self, df):
        """
        Automatically infer metadata from the given fraud DataFrame.
        Required for SDV evaluation.
        """
        logger.info("Building metadata from real fraud data...")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)
        self.metadata = metadata
        logger.info("Metadata successfully created and stored.")
        return metadata

    def fit(self, fraud_df):
        """
        Trains the CTGAN model on real fraud data.
        fraud_df should not contain the target column.
        """
        if fraud_df.empty:
            logger.warning("No fraud data to train CTGAN. Skipping.")
            return

        logger.info(f"Fitting CTGAN on {len(fraud_df)} real fraud samples...")
        try:
            # Build metadata before fitting (used later in evaluation)
            self._build_metadata(fraud_df)

            self.model.fit(fraud_df)
            logger.info("CTGAN fitted successfully.")
        except Exception as e:
            logger.error(f"Error fitting CTGAN: {e}")
            logger.error("This can happen if there is too little data or no variance.")

    def sample(self, n_samples):
        """Generates new synthetic fraud samples."""
        if self.model is None:
            logger.warning("CTGAN not fitted, cannot sample.")
            return pd.DataFrame()

        logger.info(f"Generating {n_samples} synthetic fraud samples...")
        try:
            synthetic_data = self.model.sample(n_samples)
            logger.info(f"Generated {len(synthetic_data)} synthetic samples successfully.")
            return synthetic_data
        except Exception as e:
            logger.error(f"Error generating synthetic samples: {e}")
            return pd.DataFrame()

    def evaluate_quality(self, real_data, synthetic_data):
        """
        Evaluates the quality of the synthetic data against the real data using SDV.
        """
        if real_data.empty or synthetic_data.empty:
            logger.warning("Cannot evaluate quality with empty data.")
            return 0.0

        if self.metadata is None:
            logger.info("Metadata not found, rebuilding metadata...")
            self._build_metadata(real_data)

        logger.info("Evaluating synthetic data quality...")
        try:
            quality_report = evaluate_quality(
                real_data=real_data,
                synthetic_data=synthetic_data,
                metadata=self.metadata
            )

            quality_score = quality_report.get_score()
            logger.info(f"Synthetic data quality score: {quality_score:.4f}")
            return quality_score

        except Exception as e:
            logger.error(f"Error during synthetic data evaluation: {e}")
            return 0.0

    def save(self, path=config.CTGAN_PATH):
        """Saves the CTGAN model and metadata."""
        with open(path, 'wb') as f:
            pickle.dump({
                "model": self.model,
                "metadata": self.metadata
            }, f)
