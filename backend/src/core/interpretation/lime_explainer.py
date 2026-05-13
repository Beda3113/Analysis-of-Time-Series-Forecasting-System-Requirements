import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """Реальный LIME объяснитель для любых моделей"""
    
    def __init__(self, model, training_data, feature_names=None, mode='regression', num_features=10):
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.mode = mode
        self.num_features = num_features
        self._explainer = None
        self._has_lime = False
        
        try:
            import lime
            import lime.lime_tabular
            self.lime = lime
            self._has_lime = True
            logger.info("LIME library loaded")
        except ImportError:
            logger.error("LIME not installed. Run: pip install lime")
            self._has_lime = False
    
    def fit(self):
        if not self._has_lime:
            raise ImportError("LIME not installed")
        
        if isinstance(self.training_data, pd.DataFrame):
            data_array = self.training_data.values
            if self.feature_names is None:
                self.feature_names = self.training_data.columns.tolist()
        else:
            data_array = self.training_data
            if self.feature_names is None:
                self.feature_names = [f"feature_{i}" for i in range(data_array.shape[1])]
        
        self._explainer = self.lime.lime_tabular.LimeTabularExplainer(
            training_data=data_array,
            feature_names=self.feature_names,
            mode=self.mode,
            random_state=42,
            verbose=False
        )
        
        logger.info(f"LIME explainer created with {len(self.feature_names)} features")
        return self
    
    def explain_instance(self, instance, num_features=None):
        if not self._has_lime:
            raise ImportError("LIME not installed")
        
        if self._explainer is None:
            raise RuntimeError("LIME explainer not fitted")
        
        if isinstance(instance, pd.Series):
            instance_array = instance.values
        elif isinstance(instance, list):
            instance_array = np.array(instance)
        else:
            instance_array = instance
        
        if len(instance_array.shape) == 1:
            instance_array = instance_array.reshape(1, -1)
        
        num_feat = num_features or self.num_features
        
        def predict_fn(x):
            return self.model.predict(x)
        
        explanation = self._explainer.explain_instance(
            data_row=instance_array[0],
            predict_fn=predict_fn,
            num_features=num_feat
        )
        
        local_importance = []
        for feature, weight in explanation.as_list():
            local_importance.append({
                "feature": feature,
                "weight": float(weight),
                "direction": "positive" if weight > 0 else "negative"
            })
        
        return {
            "local_importance": local_importance,
            "prediction": float(explanation.predicted_value) if hasattr(explanation, 'predicted_value') else None,
            "intercept": float(explanation.intercept) if hasattr(explanation, 'intercept') else 0,
            "num_features": len(local_importance)
        }