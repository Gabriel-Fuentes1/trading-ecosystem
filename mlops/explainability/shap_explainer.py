"""
SHAP-based Model Explainability for Trading Models
Provides feature importance and prediction explanations.
"""

import shap
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
import joblib
import warnings
warnings.filterwarnings('ignore')

class TradingModelExplainer:
    """
    SHAP-based explainer for trading models.
    Provides both global and local explanations.
    """
    
    def __init__(self, model: BaseEstimator, model_type: str = 'tree'):
        """
        Initialize explainer.
        
        :param model: Trained model to explain
        :param model_type: Type of model ('tree', 'linear', 'deep', 'kernel')
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.background_data = None
        
        logging.info(f"SHAP explainer initialized for {model_type} model")
    
    def fit_explainer(self, background_data: pd.DataFrame, max_samples: int = 100) -> None:
        """
        Fit SHAP explainer on background data.
        
        :param background_data: Background dataset for SHAP
        :param max_samples: Maximum samples for background (for performance)
        """
        try:
            # Sample background data if too large
            if len(background_data) > max_samples:
                self.background_data = background_data.sample(n=max_samples, random_state=42)
            else:
                self.background_data = background_data.copy()
            
            # Initialize appropriate explainer
            if self.model_type == 'tree':
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == 'linear':
                self.explainer = shap.LinearExplainer(self.model, self.background_data)
            elif self.model_type == 'deep':
                self.explainer = shap.DeepExplainer(self.model, self.background_data.values)
            else:  # kernel explainer as fallback
                self.explainer = shap.KernelExplainer(self.model.predict, self.background_data)
            
            logging.info("SHAP explainer fitted successfully")
            
        except Exception as e:
            logging.error(f"Error fitting SHAP explainer: {e}")
            raise
    
    def explain_prediction(
        self,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Explain individual predictions.
        
        :param X: Input data to explain
        :param feature_names: Feature names for explanation
        :return: Explanation results
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer first.")
        
        try:
            # Get SHAP values
            if self.model_type == 'deep':
                shap_values = self.explainer.shap_values(X.values)
            else:
                shap_values = self.explainer.shap_values(X)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary classification
            
            # Get predictions
            predictions = self.model.predict(X)
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)
            else:
                probabilities = None
            
            # Prepare feature names
            if feature_names is None:
                feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
            
            # Create explanation dictionary
            explanations = []
            for i in range(len(X)):
                explanation = {
                    'prediction': predictions[i],
                    'probability': probabilities[i] if probabilities is not None else None,
                    'shap_values': dict(zip(feature_names, shap_values[i])),
                    'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                    'top_features': self._get_top_features(shap_values[i], feature_names, top_k=10)
                }
                explanations.append(explanation)
            
            return {
                'explanations': explanations,
                'feature_importance': self._calculate_feature_importance(shap_values, feature_names),
                'summary_stats': self._calculate_summary_stats(shap_values, feature_names)
            }
            
        except Exception as e:
            logging.error(f"Error explaining predictions: {e}")
            raise
    
    def _get_top_features(self, shap_values: np.ndarray, feature_names: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top contributing features for a prediction."""
        feature_importance = [(name, abs(value), value) for name, value in zip(feature_names, shap_values)]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {
                'feature': name,
                'importance': abs_value,
                'contribution': value,
                'direction': 'positive' if value > 0 else 'negative'
            }
            for name, abs_value, value in feature_importance[:top_k]
        ]
    
    def _calculate_feature_importance(self, shap_values: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate global feature importance."""
        importance = np.mean(np.abs(shap_values), axis=0)
        return dict(zip(feature_names, importance))
    
    def _calculate_summary_stats(self, shap_values: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Calculate summary statistics for SHAP values."""
        return {
            'mean_abs_shap': dict(zip(feature_names, np.mean(np.abs(shap_values), axis=0))),
            'std_shap': dict(zip(feature_names, np.std(shap_values, axis=0))),
            'max_shap': dict(zip(feature_names, np.max(shap_values, axis=0))),
            'min_shap': dict(zip(feature_names, np.min(shap_values, axis=0)))
        }
    
    def generate_explanation_plots(
        self,
        X: pd.DataFrame,
        shap_values: np.ndarray,
        output_dir: str = './plots',
        max_display: int = 20
    ) -> None:
        """Generate and save SHAP explanation plots."""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, max_display=max_display, show=False)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/shap_summary_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Feature importance plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display, show=False)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/shap_importance_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Waterfall plot for first prediction
            if len(X) > 0:
                plt.figure(figsize=(10, 8))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                        data=X.iloc[0].values,
                        feature_names=X.columns.tolist()
                    ),
                    show=False
                )
                plt.tight_layout()
                plt.savefig(f'{output_dir}/shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            logging.info(f"SHAP plots saved to {output_dir}")
            
        except Exception as e:
            logging.error(f"Error generating SHAP plots: {e}")
    
    def save_explainer(self, filepath: str) -> None:
        """Save fitted explainer to disk."""
        try:
            explainer_data = {
                'explainer': self.explainer,
                'model_type': self.model_type,
                'background_data': self.background_data
            }
            joblib.dump(explainer_data, filepath)
            logging.info(f"Explainer saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving explainer: {e}")
    
    @classmethod
    def load_explainer(cls, filepath: str, model: BaseEstimator) -> 'TradingModelExplainer':
        """Load explainer from disk."""
        try:
            explainer_data = joblib.load(filepath)
            instance = cls(model, explainer_data['model_type'])
            instance.explainer = explainer_data['explainer']
            instance.background_data = explainer_data['background_data']
            logging.info(f"Explainer loaded from {filepath}")
            return instance
        except Exception as e:
            logging.error(f"Error loading explainer: {e}")
            raise
