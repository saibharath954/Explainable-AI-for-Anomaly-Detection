# src/data_loader.py
"""
Handles loading and splitting the credit card fraud dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import config

logger = logging.getLogger(__name__)

def load_creditcard_data(path=config.CREDIT_CARD_DATASET):
    """Loads the dataset from the specified path."""
    logger.info(f"Loading dataset from {path}")
    try:
        df = pd.read_csv(path)
        logger.info(f"Dataset shape: {df.shape}")
        fraud_perc = 100 * df[config.TARGET].mean()
        logger.info(f"Fraud percentage: {fraud_perc:.4f}%")
        return df
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {path}")
        return None

def split_data(df, test_size=0.2, val_size=0.1):
    """
    Splits the data into train, validation, and test sets.
    The split is stratified by the target variable.
    """
    if df is None:
        return None, None, None
    
    # First split: Train + Val vs Test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=42, 
        stratify=df[config.TARGET]
    )
    
    # Second split: Train vs Val
    # Val size is relative to the original full dataset,
    # so we adjust the test_size for the second split.
    val_test_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_test_size,
        random_state=42,
        stratify=train_val_df[config.TARGET]
    )
    
    logger.info(f"Train shape: {train_df.shape}, Fraud: {100*train_df[config.TARGET].mean():.4f}%")
    logger.info(f"Validation shape: {val_df.shape}, Fraud: {100*val_df[config.TARGET].mean():.4f}%")
    logger.info(f"Test shape: {test_df.shape}, Fraud: {100*test_df[config.TARGET].mean():.4f}%")
    
    return train_df, val_df, test_df

def save_processed_data(train_df, val_df, test_df):
    """Saves the split data to the processed data directory."""
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(config.TRAIN_DATA, index=False)
    val_df.to_csv(config.VAL_DATA, index=False)
    test_df.to_csv(config.TEST_DATA, index=False)
    logger.info(f"Processed data saved to {config.PROCESSED_DATA_DIR}")