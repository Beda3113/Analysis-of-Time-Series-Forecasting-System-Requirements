import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Реальный SHAP объяснитель для XGBoost и других моделей"""
    
    def __init__(self, model, background_sample_size: int = 100):
        self.model = model
        self.background_sample_size = background_sample_size
        self._explainer = None
        self._shap_values = None
        self._expected_value = None
        self._has_shap = False
        
        try:
            import shap
            self.shap = shap
            self._has_shap = True
            logger.info("SHAP library loaded")
        except ImportError as e:
            logger.error(f"SHAP not installed: {e}")
            self._has_shap = False
    
    def fit(self, X):
        if not self._has_shap:
            raise ImportError("SHAP not installed. Run: pip install shap")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        if len(X) > self.background_sample_size:
            X_background = X.sample(n=self.background_sample_size, random_state=42)
        else:
            X_background = X
        
        model_type = str(type(self.model)).lower()
        
        if 'xgboost' in model_type:
            self._explainer = self.shap.TreeExplainer(self.model)
        elif 'lgbm' in model_type or 'lightgbm' in model_type:
            self._explainer = self.shap.TreeExplainer(self.model)
        else:
            def predict_fn(x):
                return self.model.predict(x)
            self._explainer = self.shap.KernelExplainer(predict_fn, X_background)
        
        self._expected_value = self._explainer.expected_value
        logger.info("SHAP explainer created")
        return self
    
    def explain(self, X):
        if not self._has_shap:
            raise ImportError("SHAP not installed")
        
        if self._explainer is None:
            raise RuntimeError("SHAP explainer not fitted. Call fit() first.")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        logger.info(f"Calculating SHAP values for {len(X)} samples")
        self._shap_values = self._explainer.shap_values(X)
        
        if isinstance(self._shap_values, list):
            global_importance = np.abs(self._shap_values[0]).mean(axis=0)
        else:
            global_importance = np.abs(self._shap_values).mean(axis=0)
        
        if global_importance.sum() > 0:
            global_importance = global_importance / global_importance.sum()
        
        if hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        return {
            "shap_values": self._shap_values.tolist(),
            "expected_value": float(self._expected_value),
            "global_importance": dict(zip(feature_names, global_importance.tolist())),
            "feature_names": feature_names,
            "n_samples": X.shape[0]
        }