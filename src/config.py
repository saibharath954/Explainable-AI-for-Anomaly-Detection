# config.py
"""
Configuration file for the Explainable Anomaly Detection (XAD) project.
"""

from pathlib import Path
import os

# --- File Paths ---
BASE_DIR = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CREDIT_CARD_DATASET = RAW_DATA_DIR / "creditcard.csv"

# Processed data files
TRAIN_DATA = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA = PROCESSED_DATA_DIR / "test.csv"

# Model artifact paths
# *** UPDATED: New path for the Autoencoder model ***
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
DETECTOR_PATH = MODELS_DIR / "autoencoder_detector.h5" # Keras models use .h5
EXPLAINER_PATH = MODELS_DIR / "shap_explainer.joblib"

# Report and figure paths
EVALUATION_REPORT = REPORTS_DIR / "evaluation_report.txt"
PR_CURVE_PLOT = REPORTS_DIR / "precision_recall_curve.png"
ROC_CURVE_PLOT = REPORTS_DIR / "roc_curve.png"
CONFUSION_MATRIX_PLOT = REPORTS_DIR / "confusion_matrix.png"
SCORE_DIST_PLOT = REPORTS_DIR / "score_distribution.png"
SHAP_SUMMARY_PLOT = REPORTS_DIR / "shap_summary.png"


# --- Data and Feature Settings ---
TARGET = 'Class'
FEATURES_TO_SCALE = ['Time', 'Amount']

# --- Model Parameters ---

# *** NEW: Autoencoder parameters ***
AUTOENCODER_PARAMS = {
    'encoding_dim': 14,  # Intermediate dimension
    'epochs': 20,        # Increase this for better performance (e.g., 50)
    'batch_size': 256
}

# --- SHAP Explainer Settings ---
N_EXPLANATIONS = 5
N_SHAP_BACKGROUND = 100  # Number of samples for SHAP background