import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_creditcard_data(filename="creditcard.csv"):
    """Load the credit card fraud detection dataset"""
    filepath = RAW_DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            f"Please download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            f"and place it in the {RAW_DATA_DIR} directory."
        )
    
    logger.info(f"Loading dataset from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Fraud percentage: {df['Class'].mean() * 100:.4f}%")
    
    return df

def split_data(df, test_size=TEST_SIZE, validation_size=VALIDATION_SIZE, random_state=RANDOM_STATE):
    """Split data into train, validation, and test sets"""
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['Class']
    )
    
    # Second split: separate validation from train
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=validation_size/(1-test_size), 
        random_state=random_state, 
        stratify=train_val_df['Class']
    )
    
    logger.info(f"Train shape: {train_df.shape}, Fraud: {train_df['Class'].mean() * 100:.4f}%")
    logger.info(f"Validation shape: {val_df.shape}, Fraud: {val_df['Class'].mean() * 100:.4f}%")
    logger.info(f"Test shape: {test_df.shape}, Fraud: {test_df['Class'].mean() * 100:.4f}%")
    
    return train_df, val_df, test_df

def save_processed_data(train_df, val_df, test_df):
    """Save processed data to disk"""
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "validation.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    logger.info(f"Processed data saved to {PROCESSED_DATA_DIR}")

def load_processed_data():
    """Load processed data from disk"""
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "validation.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    
    logger.info(f"Loaded processed data: "
                f"Train={train_df.shape}, Validation={val_df.shape}, Test={test_df.shape}")
    
    return train_df, val_df, test_df