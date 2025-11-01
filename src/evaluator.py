# src/evaluator.py
"""
Defines the Evaluator class for assessing model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import logging
import config

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Handles evaluation of the anomaly detection model.
    
    IMPORTANT:
    - Isolation Forest gives LOWER scores for anomalies, so we invert them.
    - Autoencoder gives HIGHER reconstruction errors for anomalies, so we keep them as-is.
    """

    def __init__(self, y_true, y_scores, model_name="Model", invert_scores=False):
        self.y_true = y_true
        self.y_scores = y_scores
        self.model_name = model_name
        self.metrics = {}
        self.best_threshold = 0.0

        # Normalize anomaly score direction
        if invert_scores:
            # For models like Isolation Forest (low = anomalous)
            self.anomaly_scores = -self.y_scores
        else:
            # For models like Autoencoder (high = anomalous)
            self.anomaly_scores = self.y_scores

    def calculate_auprc(self):
        """Calculates the Area Under the Precision-Recall Curve (AUPRC)."""
        auprc = average_precision_score(self.y_true, self.anomaly_scores)
        self.metrics['auprc'] = auprc
        logger.info(f"[{self.model_name}] AUPRC: {auprc:.4f}")
        return auprc

    def find_best_threshold(self):
        """
        Finds the best threshold from the P-R curve that maximizes the F1 score.
        """
        precision, recall, thresholds = precision_recall_curve(
            self.y_true, self.anomaly_scores
        )

        # Compute F1 score for each threshold
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)

        # Find the best threshold index
        best_f1_idx = np.argmax(f1_scores)
        self.best_threshold = thresholds[best_f1_idx]

        # Compute AUPRC as well
        auprc = auc(recall, precision)

        # Store metrics
        self.metrics.update({
            'best_f1': f1_scores[best_f1_idx],
            'best_precision': precision[best_f1_idx],
            'best_recall': recall[best_f1_idx],
            'best_threshold': self.best_threshold,
            'auprc': auprc
        })

        logger.info(f"[{self.model_name}] Best Threshold (via F1): {self.best_threshold:.4f}")
        logger.info(f"[{self.model_name}] Best F1: {self.metrics['best_f1']:.4f}")
        logger.info(f"[{self.model_name}] Best Precision: {self.metrics['best_precision']:.4f}")
        logger.info(f"[{self.model_name}] Best Recall: {self.metrics['best_recall']:.4f}")
        logger.info(f"[{self.model_name}] AUPRC: {self.metrics['auprc']:.4f}")

        return self.metrics

    def get_metrics_at_threshold(self, threshold):
        """
        Calculates P, R, and F1 at a specific, given threshold.
        """
        y_pred = (self.anomaly_scores > threshold).astype(int)
        
        precision = precision_score(self.y_true, y_pred)
        recall = recall_score(self.y_true, y_pred)
        f1 = f1_score(self.y_true, y_pred)
        auprc = average_precision_score(self.y_true, self.anomaly_scores)
        roc_auc = roc_auc_score(self.y_true, self.anomaly_scores)
        
        metrics = {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auprc': auprc,
            'roc_auc': roc_auc
        }
        
        logger.info(f"[{self.model_name}] Metrics at threshold {threshold:.4f}:")
        for key, val in metrics.items():
            logger.info(f"  {key}: {val:.4f}")
            
        return metrics, y_pred

    def plot_precision_recall_curve(self, save_path=None):
        """Plots the Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(self.y_true, self.anomaly_scores)
        auprc = self.metrics.get('auprc', self.calculate_auprc())
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{self.model_name} (AUPRC = {auprc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Precision-recall curve saved to {save_path}")
        plt.close()

    def plot_roc_curve(self, save_path=None):
        """Plots the ROC curve."""
        fpr, tpr, _ = roc_curve(self.y_true, self.anomaly_scores)
        roc_auc = roc_auc_score(self.y_true, self.anomaly_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{self.model_name} (ROC AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"ROC curve saved to {save_path}")
        plt.close()

    def plot_confusion_matrix(self, y_pred, save_path=None):
        """Plots the confusion matrix."""
        cm = confusion_matrix(self.y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
        plt.close()