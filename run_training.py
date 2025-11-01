#!/usr/bin/env python3
# run_training.py
"""
Main training pipeline for the Explainable Anomaly Detection (XAD) framework.
Uses an Autoencoder detection engine.
"""

import os, sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

# Force base directory to project root
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

os.environ["PROJECT_ROOT"] = str(BASE_DIR)

# Add src to path
SRC_DIR = Path(__file__).parent / "src"
sys.path.append(str(SRC_DIR))

import config
from data_loader import load_creditcard_data, split_data, save_processed_data
from preprocess import Preprocessor
from detector import AutoencoderDetector # *** UPDATED ***
from explainer import SHAPExplainer
from evaluator import Evaluator

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Set up logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_autoencoder.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("--- Starting XAD Training Pipeline (Autoencoder) ---")

    # === STEP 1: LOAD AND SPLIT DATA ===
    logger.info("--- STEP 1: Loading and Splitting Data ---")
    df = load_creditcard_data()
    if df is None:
        logger.error("Failed to load data. Exiting.")
        return
        
    train_df, val_df, test_df = split_data(df)
    save_processed_data(train_df, val_df, test_df)

    y_train = train_df[config.TARGET]
    y_val = val_df[config.TARGET]
    y_test = test_df[config.TARGET]

    # === STEP 2: PREPROCESS DATA ===
    logger.info("--- STEP 2: Preprocessing Data ---")
    preprocessor = Preprocessor(features_to_scale=config.FEATURES_TO_SCALE)
    
    # Fit on training data
    X_train_df = preprocessor.fit_transform(train_df)
    # Transform validation and test data
    X_val_df = preprocessor.transform(val_df)
    X_test_df = preprocessor.transform(test_df)
    
    preprocessor.save()
    feature_names = preprocessor.get_feature_names_out()
    input_dim = len(feature_names)

    # *** CRITICAL: Train Autoencoder ONLY on NORMAL data ***
    # This is the correct methodology for this type of anomaly detection.
    X_train_normal = X_train_df[y_train == 0].values
    X_val_normal = X_val_df[y_val == 0].values
    
    # Convert test sets to numpy for faster processing
    X_test = X_test_df.values
    
    logger.info(f"Training normal data shape: {X_train_normal.shape}")
    logger.info(f"Validation normal data shape: {X_val_normal.shape}")

    # === STEP 3: TRAIN DETECTOR ===
    logger.info("--- STEP 3: Training Autoencoder Detector ---")
    detector = AutoencoderDetector(
        input_dim=input_dim, 
        **config.AUTOENCODER_PARAMS
    )
    
    # Fit on normal data
    detector.fit(X_train_normal, X_val_normal)
    detector.save()

    # === STEP 4: EVALUATE ON VALIDATION SET (to find threshold) ===
    logger.info("--- STEP 4: Evaluating on Validation Set ---")
    # Get scores for the *full* validation set (normal + anomalies)
    val_scores = detector.get_scores(X_val_df.values)
    
    # The evaluator will find the best F1-score and threshold
    evaluator_val = Evaluator(y_val, val_scores, "Autoencoder (Validation)")
    val_metrics = evaluator_val.find_best_threshold()
    best_threshold = val_metrics['best_threshold']
    
    logger.info(f"Best validation F1: {val_metrics['best_f1']:.4f} at threshold {best_threshold:.4f}")

    # === STEP 5: FINAL EVALUATION ON TEST SET ===
    logger.info("--- STEP 5: Final Evaluation on Test Set ---")
    
    # 1. Get scores on the TEST set
    test_scores = detector.get_scores(X_test)
    
    # 2. Evaluate using the best_threshold from the validation set
    test_evaluator = Evaluator(y_test, test_scores, "Autoencoder (Test)")
    final_metrics, y_pred_test = test_evaluator.get_metrics_at_threshold(best_threshold)
    
    # 3. Generate final plots
    test_evaluator.plot_precision_recall_curve(save_path=config.PR_CURVE_PLOT)
    test_evaluator.plot_roc_curve(save_path=config.ROC_CURVE_PLOT)
    test_evaluator.plot_confusion_matrix(y_pred_test, save_path=config.CONFUSION_MATRIX_PLOT)
    
    # 4. Generate report
    with open(config.EVALUATION_REPORT, 'w') as f:
        f.write("--- Final Evaluation Report for Autoencoder ---\n\n")
        f.write(f"Threshold (from validation set): {best_threshold:.4f}\n\n")
        f.write("--- Test Set Metrics ---\n")
        for key, val in final_metrics.items():
            f.write(f"{key}: {val:.4f}\n")
            
    logger.info(f"Final evaluation report saved to {config.EVALUATION_REPORT}")

    # === STEP 6: GENERATE SHAP EXPLANATIONS ===
    logger.info("--- STEP 6: Generating SHAP Explanations ---")
    
    # We need a background dataset for the DeepExplainer
    # Use a sample of the normal training data
    background_sample_idx = np.random.choice(
        X_train_normal.shape[0], 
        config.N_SHAP_BACKGROUND, 
        replace=False
    )
    background_data = X_train_normal[background_sample_idx]
    
    explainer = SHAPExplainer(detector, background_data, feature_names)
    
    # Find anomalies in the test set
    anomaly_indices = np.where(y_pred_test == 1)[0]
    
    if len(anomaly_indices) > 0:
        n_explain = min(config.N_EXPLANATIONS, len(anomaly_indices))
        logger.info(f"Generating explanations for top {n_explain} anomalies...")
        
        anomalies_to_explain = X_test[anomaly_indices[:n_explain]]
        
        # Explain the anomalies
        shap_values = explainer.explain_batch(anomalies_to_explain)
        
        for i in range(n_explain):
            instance_values = anomalies_to_explain[i]
            explanation = explainer.generate_natural_language_explanation(
                shap_values[i], instance_values
            )
            logger.info(f"Explanation for Anomaly {i+1}:\n{explanation}")
        
        # Create summary plot based on a sample of the test set
        test_sample_idx = np.random.choice(
            X_test.shape[0], 
            config.N_SHAP_BACKGROUND, 
            replace=False
        )
        test_sample = X_test[test_sample_idx]
        
        all_shap_values = explainer.explain_batch(test_sample)
        explainer.summary_plot(
            all_shap_values, 
            pd.DataFrame(test_sample, columns=feature_names), 
            save_path=config.SHAP_SUMMARY_PLOT
        )
        
    explainer.save()

    logger.info("--- XAD Training Pipeline (Autoencoder) Completed Successfully ---")


if __name__ == "__main__":
    main()