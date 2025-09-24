import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

import numpy as np
import pandas as pd
import logging
from data_loader import load_creditcard_data, split_data, save_processed_data
from preprocess import Preprocessor
from detector import IsfDetector
from explainer import SHAPExplainer
from generator import FraudCTGAN
from evaluator import Evaluator
from config import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, 
    ISOLATION_FOREST_PARAMS, CTGAN_PARAMS, MODELS_DIR
)
import joblib
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting fraud detection training pipeline...")
    
    # Step 1: Load and prepare data
    logger.info("Step 1: Loading data...")
    df = load_creditcard_data()
    train_df, val_df, test_df = split_data(df)
    save_processed_data(train_df, val_df, test_df)
    
    # Step 2: Preprocess data
    logger.info("Step 2: Preprocessing data...")
    preprocessor = Preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, scaling_method='robust')
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)
    
    y_train = train_df[TARGET].values
    y_val = val_df[TARGET].values
    y_test = test_df[TARGET].values
    
    # Save preprocessor
    preprocessor.save()
    
    # Step 3: Train Isolation Forest
    logger.info("Step 3: Training Isolation Forest...")
    detector = IsfDetector(**ISOLATION_FOREST_PARAMS)
    detector.fit(X_train)
    
    # Evaluate on validation set
    val_preds, val_scores = detector.predict(X_val)
    val_evaluator = Evaluator(y_val, val_scores, "Isolation Forest (Validation)")
    val_metrics = val_evaluator.calculate_metrics()
    
    # Save detector
    detector.save()
    
    # Step 4: SHAP explanations
    logger.info("Step 4: Generating SHAP explanations...")
    feature_names = preprocessor.get_feature_names_out()
    explainer = SHAPExplainer(detector, feature_names)
    
    # Explain top anomalies in validation set
    anomaly_indices = np.where(val_scores > detector.threshold_)[0]
    if len(anomaly_indices) > 0:
        top_anomalies = X_val[anomaly_indices[:5]]  # Explain top 5 anomalies
        shap_values = explainer.explain_batch(top_anomalies)
        
        # Generate natural language explanations
        for i, (instance, shap_vals) in enumerate(zip(top_anomalies, shap_values)):
            explanation = explainer.generate_natural_language_explanation(
                shap_vals, instance, feature_names
            )
            logger.info(f"Explanation for anomaly {i+1}:\n{explanation}")
        
        # Create and save summary plot
        explainer.summary_plot(shap_values, top_anomalies, 
                              save_path=MODELS_DIR / "shap_summary.png")
    
    # Save explainer
    explainer.save()
    
    # Step 5: CTGAN for fraud augmentation (optional)
    logger.info("Step 5: Training CTGAN for fraud augmentation...")
    
    # Get fraud samples from training data
    fraud_df = train_df[train_df[TARGET] == 1].drop(TARGET, axis=1)
    
    if len(fraud_df) > 10:  # Only train if we have enough fraud samples
        ctgan = FraudCTGAN(**CTGAN_PARAMS)
        ctgan.fit(fraud_df)
        
        # Generate synthetic fraud samples
        n_synthetic = min(1000, len(fraud_df) * 10)  # Generate up to 10x original fraud
        synthetic_fraud = ctgan.sample(n_synthetic)
        synthetic_fraud[TARGET] = 1  # Add fraud label
        
        # Evaluate synthetic data quality
        quality_score = ctgan.evaluate_quality(fraud_df, synthetic_fraud.drop(TARGET, axis=1))
        logger.info(f"CTGAN synthetic data quality score: {quality_score:.4f}")
        
        # Save CTGAN model
        ctgan.save()
        
        # Augment training data with synthetic fraud
        augmented_train_df = pd.concat([train_df, synthetic_fraud], ignore_index=True)
        logger.info(f"Training data augmented: {len(train_df)} -> {len(augmented_train_df)} samples")
        
        # Preprocess augmented data
        X_train_aug = preprocessor.transform(augmented_train_df)
        y_train_aug = augmented_train_df[TARGET].values
        
        # Retrain detector on augmented data
        logger.info("Retraining detector on augmented data...")
        detector_aug = IsfDetector(**ISOLATION_FOREST_PARAMS)
        detector_aug.fit(X_train_aug)
        
        # Evaluate augmented model
        val_preds_aug, val_scores_aug = detector_aug.predict(X_val)
        val_evaluator_aug = Evaluator(y_val, val_scores_aug, "Isolation Forest (Augmented)")
        val_metrics_aug = val_evaluator_aug.calculate_metrics()
        
        # Compare performance
        logger.info("Performance comparison:")
        logger.info(f"Original F1: {val_metrics['f1']:.4f}")
        logger.info(f"Augmented F1: {val_metrics_aug['f1']:.4f}")
        
        # Save augmented detector if it performs better
        if val_metrics_aug['f1'] > val_metrics['f1']:
            detector_aug.save("isolation_forest_augmented.joblib")
            logger.info("Augmented model saved (better performance)")
        else:
            logger.info("Original model performs better, keeping it")
    
    # Step 6: Final evaluation on test set
    logger.info("Step 6: Final evaluation on test set...")
    test_preds, test_scores = detector.predict(X_test)
    test_evaluator = Evaluator(y_test, test_scores, "Isolation Forest (Final)")
    test_metrics = test_evaluator.calculate_metrics()
    
    # Generate evaluation plots
    test_evaluator.plot_precision_recall_curve(save_path=MODELS_DIR / "precision_recall_curve.png")
    test_evaluator.plot_roc_curve(save_path=MODELS_DIR / "roc_curve.png")
    test_evaluator.plot_confusion_matrix(save_path=MODELS_DIR / "confusion_matrix.png")
    test_evaluator.plot_score_distribution(save_path=MODELS_DIR / "score_distribution.png")
    test_evaluator.generate_report(save_path=MODELS_DIR / "evaluation_report.txt")
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()