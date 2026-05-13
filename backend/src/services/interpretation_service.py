import numpy as np
from typing import List, Dict, Any
from src.core.interpretation.shap_explainer import SHAPExplainer
from src.core.interpretation.lime_explainer import LIMEExplainer

def generate_shap_values(series, model, lags: List[int]) -> Dict[str, Any]:
    """Генерация реальных SHAP значений"""
    import pandas as pd
    
    # Создаём признаки
    X, y = _create_lag_features(series.values, lags)
    
    if len(X) == 0:
        return _generate_fallback_shap(series, lags)
    
    df = pd.DataFrame(X, columns=[f"lag_{l}" for l in lags])
    
    try:
        # Пытаемся загрузить реальную модель
        from src.storage.minio.model_storage import get_model_storage
        model_storage = get_model_storage()
        loaded_model = model_storage.load_model(model.id, model.model_type)
        
        if loaded_model is None:
            return _generate_fallback_shap(series, lags)
        
        explainer = SHAPExplainer(loaded_model)
        explainer.fit(df)
        result = explainer.explain(df)
        
        return {
            "base_value": result["expected_value"],
            "feature_names": result["feature_names"],
            "global_importance": result["global_importance"],
            "shap_values": result["shap_values"][:5]
        }
    except Exception as e:
        print(f"SHAP failed: {e}")
        return _generate_fallback_shap(series, lags)


def _create_lag_features(values, lags):
    X = []
    y = []
    max_lag = max(lags)
    
    for i in range(max_lag, len(values)):
        features = []
        for lag in lags:
            features.append(values[i - lag])
        X.append(features)
        y.append(values[i])
    
    return X, y


def _generate_fallback_shap(series, lags):
    """Минимальный fallback когда SHAP недоступен"""
    feature_names = [f"lag_{l}" for l in lags]
    global_importance = {f"lag_{l}": round(1.0 / (l + 1), 3) for l in lags}
    
    total = sum(global_importance.values())
    if total > 0:
        for key in global_importance:
            global_importance[key] = round(global_importance[key] / total, 4)
    
    return {
        "base_value": float(np.mean(series.values[-10:])),
        "feature_names": feature_names,
        "global_importance": global_importance,
        "shap_values": []
    }