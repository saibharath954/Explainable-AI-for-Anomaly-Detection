import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, f1_score, 
    precision_score, recall_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, PrecisionRecallDisplay
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, y_true, scores, model_name="Isolation Forest"):
        self.y_true = y_true
        self.scores = scores
        self.model_name = model_name
        self.metrics = {}
        
    def calculate_metrics(self, threshold=None):
        """Calculate various evaluation metrics"""
        if threshold is None:
            # Find optimal threshold using precision-recall curve
            precision, recall, thresholds = precision_recall_curve(self.y_true, self.scores)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
            best_idx = np.argmax(f1_scores)
            threshold = thresholds[best_idx]
        
        y_pred = (self.scores >= threshold).astype(int)
        
        self.metrics = {
            'threshold': threshold,
            'precision': precision_score(self.y_true, y_pred, zero_division=0),
            'recall': recall_score(self.y_true, y_pred, zero_division=0),
            'f1': f1_score(self.y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_true, self.scores),
            'average_precision': average_precision_score(self.y_true, self.scores),
            'confusion_matrix': confusion_matrix(self.y_true, y_pred)
        }
        
        logger.info(f"Evaluation metrics at threshold {threshold:.4f}:")
        for metric, value in self.metrics.items():
            if metric != 'confusion_matrix':
                logger.info(f"  {metric}: {value:.4f}")
        
        return self.metrics
    
    def plot_precision_recall_curve(self, save_path=None):
        """Plot precision-recall curve"""
        precision, recall, _ = precision_recall_curve(self.y_true, self.scores)
        avg_precision = average_precision_score(self.y_true, self.scores)
        
        plt.figure(figsize=(10, 8))
        display = PrecisionRecallDisplay(precision=precision, recall=recall, 
                                        average_precision=avg_precision)
        display.plot()
        plt.title(f'Precision-Recall Curve ({self.model_name})\nAP={avg_precision:.4f}')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curve saved to {save_path}")
        
        return plt.gcf()
    
    def plot_roc_curve(self, save_path=None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(self.y_true, self.scores)
        roc_auc = roc_auc_score(self.y_true, self.scores)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({self.model_name})')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        return plt.gcf()
    
    def plot_confusion_matrix(self, threshold=None, save_path=None):
        """Plot confusion matrix"""
        if threshold is None:
            threshold = self.metrics.get('threshold', 
                                       np.percentile(self.scores, 95))
        
        y_pred = (self.scores >= threshold).astype(int)
        cm = confusion_matrix(self.y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix (Threshold = {threshold:.4f})')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return plt.gcf()
    
    def plot_score_distribution(self, save_path=None):
        """Plot distribution of anomaly scores"""
        plt.figure(figsize=(10, 6))
        
        # Plot scores for each class
        for label, name in [(0, 'Normal'), (1, 'Fraud')]:
            mask = self.y_true == label
            sns.histplot(self.scores[mask], label=name, alpha=0.7, kde=True)
        
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title('Distribution of Anomaly Scores by Class')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Score distribution plot saved to {save_path}")
        
        return plt.gcf()
    
    def generate_report(self, save_path=None):
        """Generate comprehensive evaluation report"""
        report = f"Model Evaluation Report: {self.model_name}\n"
        report += "=" * 50 + "\n\n"
        
        # Add metrics
        report += "Performance Metrics:\n"
        report += "-" * 20 + "\n"
        for metric, value in self.metrics.items():
            if metric != 'confusion_matrix':
                report += f"{metric}: {value:.4f}\n"
        
        # Add confusion matrix
        report += "\nConfusion Matrix:\n"
        report += "-" * 20 + "\n"
        cm = self.metrics['confusion_matrix']
        report += f"True Negatives: {cm[0, 0]}\n"
        report += f"False Positives: {cm[0, 1]}\n"
        report += f"False Negatives: {cm[1, 0]}\n"
        report += f"True Positives: {cm[1, 1]}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report