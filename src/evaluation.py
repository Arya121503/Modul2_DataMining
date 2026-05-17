"""
evaluation.py
Modul untuk evaluasi model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score, auc
)

class ModelEvaluator:
    """Kelas untuk evaluasi model"""
    
    def __init__(self):
        """Inisialisasi ModelEvaluator"""
        self.results = {}
    
    def evaluate_classification(self, y_true, y_pred, y_pred_proba=None):
        """
        Evaluasi model klasifikasi
        
        Parameters:
        -----------
        y_true : array-like
            Actual labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like, optional
            Predicted probabilities
            
        Returns:
        --------
        dict
            Metrics hasil evaluasi
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except:
                pass
        
        print("="*50)
        print("CLASSIFICATION METRICS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric:15}: {value:.4f}")
        print("="*50)
        
        self.results['metrics'] = metrics
        return metrics
    
    def evaluate_regression(self, y_true, y_pred):
        """
        Evaluasi model regresi
        
        Parameters:
        -----------
        y_true : array-like
            Actual values
        y_pred : array-like
            Predicted values
            
        Returns:
        --------
        dict
            Metrics hasil evaluasi
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print("="*50)
        print("REGRESSION METRICS")
        print("="*50)
        print(f"{'MSE':15}: {mse:.4f}")
        print(f"{'RMSE':15}: {rmse:.4f}")
        print(f"{'MAE':15}: {mae:.4f}")
        print(f"{'R² Score':15}: {r2:.4f}")
        print("="*50)
        
        self.results['metrics'] = metrics
        return metrics
    
    def get_classification_report(self, y_true, y_pred, target_names=None):
        """
        Dapatkan detailed classification report
        
        Parameters:
        -----------
        y_true : array-like
            Actual labels
        y_pred : array-like
            Predicted labels
        target_names : list
            Nama-nama class
            
        Returns:
        --------
        str
            Detailed classification report
        """
        report = classification_report(y_true, y_pred, target_names=target_names)
        print("\nDETAILED CLASSIFICATION REPORT:")
        print(report)
        return report
    
    def get_confusion_matrix(self, y_true, y_pred):
        """
        Dapatkan confusion matrix
        
        Parameters:
        -----------
        y_true : array-like
            Actual labels
        y_pred : array-like
            Predicted labels
            
        Returns:
        --------
        array
            Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        return cm
    
    def plot_confusion_matrix(self, y_true, y_pred, figsize=(8, 6)):
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        y_true : array-like
            Actual labels
        y_pred : array-like
            Predicted labels
        figsize : tuple
            Ukuran figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba, figsize=(8, 6)):
        """
        Plot ROC Curve
        
        Parameters:
        -----------
        y_true : array-like
            Actual labels
        y_pred_proba : array-like
            Predicted probabilities
        figsize : tuple
            Ukuran figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, feature_importance_df, top_n=10, figsize=(10, 6)):
        """
        Plot feature importance
        
        Parameters:
        -----------
        feature_importance_df : pd.DataFrame
            DataFrame dengan 'feature' dan 'importance' columns
        top_n : int
            Jumlah top features yang diplot
        figsize : tuple
            Ukuran figure
        """
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_vs_actual(self, y_true, y_pred, figsize=(8, 6)):
        """
        Plot predictions vs actual (untuk regresi)
        
        Parameters:
        -----------
        y_true : array-like
            Actual values
        y_pred : array-like
            Predicted values
        figsize : tuple
            Ukuran figure
        """
        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predictions vs Actual Values')
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, models_dict, X_test, y_test):
        """
        Bandingkan beberapa model
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary dengan nama model dan model object
        X_test : array-like
            Test features
        y_test : array-like
            Test target
            
        Returns:
        --------
        pd.DataFrame
            Comparison results
        """
        results = []
        
        for name, model in models_dict.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append({'Model': name, 'Accuracy': accuracy})
        
        results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
        
        print("\n" + "="*40)
        print("MODEL COMPARISON")
        print("="*40)
        print(results_df.to_string(index=False))
        print("="*40)
        
        return results_df

if __name__ == "__main__":
    # Contoh penggunaan akan dilakukan di notebook
    print("Model evaluator module ready!")
