# config.py
"""
Configuration file for the Explainable Anomaly Detection (XAD) project.
"""

from pathlib import Path

# --- File Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Original dataset file
CREDIT_CARD_DATASET = RAW_DATA_DIR / "creditcard.csv"

# Processed data files
TRAIN_DATA = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA = PROCESSED_DATA_DIR / "test.csv"

# Model artifact paths
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
DETECTOR_PATH = MODELS_DIR / "isolation_forest.joblib"
EXPLAINER_PATH = MODELS_DIR / "shap_explainer.joblib"
CTGAN_PATH = MODELS_DIR / "ctgan_model.pkl"

# Report and figure paths
EVALUATION_REPORT = REPORTS_DIR / "evaluation_report.txt"
PR_CURVE_PLOT = REPORTS_DIR / "precision_recall_curve.png"
ROC_CURVE_PLOT = REPORTS_DIR / "roc_curve.png"
CONFUSION_MATRIX_PLOT = REPORTS_DIR / "confusion_matrix.png"
SCORE_DIST_PLOT = REPORTS_DIR / "score_distribution.png"
SHAP_SUMMARY_PLOT = REPORTS_DIR / "shap_summary.png"


# --- Data and Feature Settings ---
TARGET = 'Class'
# As per the paper[cite: 77], V1-V28 are PCA features.
# 'Time' and 'Amount' are the raw features needing scaling.
FEATURES_TO_SCALE = ['Time', 'Amount']

# --- Model Parameters ---

# Isolation Forest parameters
# The contamination is set to the known fraud rate in the dataset [cite: 77, 95]
ISOLATION_FOREST_PARAMS = {
    'n_estimators': 100,
    'contamination': 0.00172, # From paper [cite: 95]
    'max_samples': 'auto',
    'n_jobs': -1,
    'random_state': 42
}

# CTGAN parameters [cite: 66]
CTGAN_PARAMS = {
    'epochs': 300,
    'batch_size': 50,
    'verbose': True
}

# --- SHAP Explainer Settings ---
N_EXPLANATIONS = 5 # Number of anomalies to explain