import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
DATASET_FILENAME = "creditcard.csv"

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Features
NUMERIC_FEATURES = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

CATEGORICAL_FEATURES = []  # No categorical features in creditcard dataset
TARGET = 'Class'

# Isolation Forest parameters
ISOLATION_FOREST_PARAMS = {
    'n_estimators': 200,
    'max_samples': 'auto',
    'contamination': 'auto',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# CTGAN parameters
CTGAN_PARAMS = {
    'epochs': 300,
    'batch_size': 500,
    'cuda': False,
    'verbose': True
}

# Threshold for anomaly detection (percentile)
ANOMALY_THRESHOLD = 95