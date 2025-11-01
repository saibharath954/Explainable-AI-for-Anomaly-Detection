#!/usr/bin/env python3
# run_training.py
"""
Main training pipeline for the Explainable Anomaly Detection (XAD) framework.
"""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# Add src to path
SRC_DIR = Path(__file__).parent / "src"
sys.path.append(str(SRC_DIR))

import config
from data_loader import load_creditcard_data, split_data, save_processed_data
from preprocess import Preprocessor
from detector import IsfDetector
from explainer import SHAPExplainer
from generator import FraudCTGAN
from evaluator import Evaluator

# --- Set up logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("--- Starting XAD Training Pipeline ---")

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
    X_train = preprocessor.fit_transform(train_df)
    # Transform validation and test data
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)
    
    preprocessor.save()
    feature_names = preprocessor.get_feature_names_out()

    # === STEP 3: TRAIN & EVALUATE BASELINE DETECTOR ===
    logger.info("--- STEP 3: Training and Evaluating Baseline Detector ---")
    detector_base = IsfDetector(**config.ISOLATION_FOREST_PARAMS)
    detector_base.fit(X_train)
    
    # Get scores for validation set and evaluate
    val_scores_base = detector_base.get_scores(X_val)
    evaluator_base = Evaluator(y_val, val_scores_base, "Baseline IF")
    
    # Find the best threshold on the validation set
    val_metrics_base = evaluator_base.find_best_threshold()
    baseline_threshold = val_metrics_base['best_threshold']
    baseline_f1 = val_metrics_base['best_f1']
    
    detector_base.save(config.MODELS_DIR / "isolation_forest_base.joblib")

    # === STEP 4: GENERATE SHAP EXPLANATIONS (from baseline) ===
    logger.info("--- STEP 4: Generating SHAP Explanations ---")
    explainer = SHAPExplainer(detector_base, feature_names)
    
    # Find anomalies in the validation set using our new threshold
    val_anomaly_scores = -val_scores_base
    anomaly_indices = np.where(val_anomaly_scores > baseline_threshold)[0]
    
    if len(anomaly_indices) > 0:
        n_explain = min(config.N_EXPLANATIONS, len(anomaly_indices))
        logger.info(f"Generating explanations for top {n_explain} anomalies...")
        
        # Get the instances to explain
        # Need to use .iloc to select by integer position
        anomalies_to_explain_df = X_val.iloc[anomaly_indices[:n_explain]]
        
        shap_values = explainer.explain_batch(anomalies_to_explain_df)
        
        for i in range(n_explain):
            instance_values = anomalies_to_explain_df.iloc[i].values
            explanation = explainer.generate_natural_language_explanation(
                shap_values[i], instance_values
            )
            logger.info(f"Explanation for Anomaly {i+1}:\n{explanation}")
        
        # Create summary plot based on all validation data
        all_shap_values = explainer.explain_batch(X_val)
        explainer.summary_plot(all_shap_values, X_val, save_path=config.SHAP_SUMMARY_PLOT)
        
    explainer.save()

    # === STEP 5: TRAIN & EVALUATE AUGMENTED DETECTOR ===
    logger.info("--- STEP 5: Training and Evaluating Augmented Detector (CTGAN) ---")
    
    # Get real fraud samples from the training set [cite: 69]
    fraud_df = train_df[train_df[config.TARGET] == 1].drop(config.TARGET, axis=1)
    
    # Get the *preprocessed* fraud data for training the new detector
    X_train_fraud = X_train[y_train == 1]
    
    if len(fraud_df) < 10:
        logger.warning("Not enough fraud samples to train CTGAN. Skipping augmentation.")
        detector_aug = None
        augmented_f1 = -1
    else:
        # 1. Train CTGAN
        ctgan = FraudCTGAN(**config.CTGAN_PARAMS)
        # We train CTGAN on the *original*, *un-scaled* fraud data
        ctgan.fit(fraud_df[feature_names]) 
        
        # 2. Generate synthetic samples [cite: 71]
        n_synthetic = len(X_train) - len(X_train_fraud) # Create a 50/50 balance
        synthetic_fraud_df = ctgan.sample(n_synthetic)
        
        # 3. Evaluate synthetic data
        ctgan.evaluate_quality(fraud_df[feature_names], synthetic_fraud_df)
        ctgan.save()

        # 4. Preprocess synthetic data
        # We need to add a dummy 'Class' column so the preprocessor can drop it
        synthetic_fraud_df[config.TARGET] = 1 
        X_synthetic_fraud = preprocessor.transform(synthetic_fraud_df)
        
        # 5. Create augmented training set [cite: 72]
        X_train_aug = pd.concat([X_train, X_synthetic_fraud], ignore_index=True)
        y_synthetic_fraud = pd.Series([1] * len(X_synthetic_fraud))
        y_train_aug = pd.concat([y_train, y_synthetic_fraud], ignore_index=True)
        
        logger.info(f"Augmented training set: {len(X_train_aug)} samples")
        
        # 6. Train augmented detector
        detector_aug = IsfDetector(**config.ISOLATION_FOREST_PARAMS)
        # Note: We must adjust contamination for the new, balanced dataset
        new_contamination = min(max(y_train_aug.mean(), 0.001), 0.5)
        detector_aug.model.contamination = new_contamination
        logger.info(f"Adjusted contamination for augmented data: {new_contamination:.6f}")

        detector_aug.fit(X_train_aug)
        
        # 7. Evaluate augmented detector on validation set
        val_scores_aug = detector_aug.get_scores(X_val)
        evaluator_aug = Evaluator(y_val, val_scores_aug, "Augmented IF")
        val_metrics_aug = evaluator_aug.find_best_threshold()
        augmented_f1 = val_metrics_aug['best_f1']
        
        detector_aug.save(config.MODELS_DIR / "isolation_forest_augmented.joblib")

    # === STEP 6: FINAL EVALUATION ON TEST SET ===
    logger.info("--- STEP 6: Final Evaluation on Test Set ---")
    
    # --- THIS IS THE CRITICAL FIX ---
    # Select the best model based on validation F1 score
    if augmented_f1 > baseline_f1:
        logger.info("Augmented model performed better on validation set.")
        final_detector = detector_aug
        final_threshold = val_metrics_aug['best_threshold']
        final_model_name = "Augmented Isolation Forest"
    else:
        logger.info("Baseline model performed better or augmentation was skipped.")
        final_detector = detector_base
        final_threshold = baseline_threshold
        final_model_name = "Baseline Isolation Forest"
        
    logger.info(f"Using model: {final_model_name}")
    logger.info(f"Using threshold (from validation): {final_threshold:.4f}")

    # 1. Get scores on the TEST set
    test_scores = final_detector.get_scores(X_test)
    
    # 2. Evaluate using the threshold from the validation set
    test_evaluator = Evaluator(y_test, test_scores, final_model_name)
    final_metrics, y_pred_test = test_evaluator.get_metrics_at_threshold(final_threshold)
    
    # 3. Generate final plots
    test_evaluator.plot_precision_recall_curve(save_path=config.PR_CURVE_PLOT)
    test_evaluator.plot_roc_curve(save_path=config.ROC_CURVE_PLOT)
    test_evaluator.plot_confusion_matrix(y_pred_test, save_path=config.CONFUSION_MATRIX_PLOT)
    
    # 4. Generate report
    with open(config.EVALUATION_REPORT, 'w') as f:
        f.write(f"--- Final Evaluation Report for {final_model_name} ---\n\n")
        f.write(f"Threshold (from validation set): {final_threshold:.4f}\n\n")
        f.write("--- Test Set Metrics ---\n")
        for key, val in final_metrics.items():
            f.write(f"{key}: {val:.4f}\n")
            
    logger.info(f"Final evaluation report saved to {config.EVALUATION_REPORT}")
    logger.info("--- XAD Training Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()