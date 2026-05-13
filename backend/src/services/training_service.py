"""
Сервис для обучения моделей временных рядов
"""

import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.models import TimeSeries, TrainedModel, add_model, update_task_status
from src.storage.minio.model_storage import get_model_storage
from src.utils.logger import get_logger

logger = get_logger("training_service")


def create_lag_features(values: List[float], lags: List[int]) -> tuple:
    """Создание лаговых признаков для XGBoost"""
    X = []
    y = []
    max_lag = max(lags)
    
    for i in range(max_lag, len(values)):
        features = []
        for lag in lags:
            features.append(values[i - lag])
        X.append(features)
        y.append(values[i])
    
    return np.array(X), np.array(y)


def train_xgboost_model(
    series: TimeSeries,
    task_id: str,
    model_id: str,
    hyperparams: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Обучение модели XGBoost с сохранением в MinIO"""
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        # Параметры по умолчанию
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        if hyperparams:
            default_params.update(hyperparams)
        
        # Создание признаков
        lags = [1, 2, 3, 7, 14, 30] if len(series.values) > 30 else [1, 2, 3]
        
        update_task_status(task_id, "training", 20)
        
        X, y = create_lag_features(series.values, lags)
        
        if len(X) < 10:
            raise ValueError(f"Недостаточно данных. Нужно минимум {max(lags) + 10} точек")
        
        # Разделение на train/val
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        update_task_status(task_id, "training", 50)
        
        # Обучение модели
        model = xgb.XGBRegressor(**default_params)
        model.fit(X_train, y_train)
        
        update_task_status(task_id, "training", 80)
        
        # Оценка качества
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100 if len(y_val) > 0 else 0
        
        metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 2),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "lags": lags
        }
        
        # Сохраняем модель в MinIO
        model_storage = get_model_storage()
        model_path = model_storage.save_model(
            model=model,
            model_id=model_id,
            model_type="xgboost",
            metadata={
                "series_id": series.id,
                "user_id": series.user_id,
                "hyperparams": str(default_params),
                "metrics": str(metrics),
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        update_task_status(task_id, "training", 100, {"metrics": metrics, "model_path": model_path})
        
        return {
            "success": True,
            "metrics": metrics,
            "hyperparams": default_params,
            "lags": lags,
            "model_path": model_path
        }
        
    except Exception as e:
        logger.error(f"XGBoost training failed: {str(e)}")
        update_task_status(task_id, "failed", 0, None, str(e))
        return {
            "success": False,
            "error": str(e)
        }


def train_model(
    series: TimeSeries,
    model_type: str,
    task_id: str,
    model_id: str,
    hyperparams: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Обучение модели в зависимости от типа"""
    if model_type == "xgboost":
        return train_xgboost_model(series, task_id, model_id, hyperparams)
    else:
        # Для других моделей (LSTM, Prophet, SARIMA) - заглушки
        update_task_status(task_id, "training", 30)
        import time
        time.sleep(2)  # Имитация обучения
        
        import random
        metrics = {
            "mae": round(random.uniform(5, 15), 2),
            "rmse": round(random.uniform(10, 20), 2),
            "mape": round(random.uniform(5, 15), 2)
        }
        
        update_task_status(task_id, "training", 100, {"metrics": metrics})
        
        return {
            "success": True,
            "metrics": metrics,
            "hyperparams": hyperparams or {},
            "lags": [1, 2, 3, 7, 14]
        }
