"""
Сервис для прогнозирования временных рядов
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from src.models import TimeSeries, TrainedModel
from src.storage.minio.model_storage import get_model_storage
from src.utils.logger import get_logger

logger = get_logger("forecast_service")


def load_model_from_storage(model: TrainedModel):
    """Загрузка модели из MinIO"""
    model_storage = get_model_storage()
    loaded_model = model_storage.load_model(model.id, model.model_type)
    return loaded_model


def create_prediction_features(values: List[float], lags: List[int], horizon: int) -> np.ndarray:
    """Создание признаков для прогнозирования"""
    features = []
    current_values = values.copy()
    
    for step in range(horizon):
        row = []
        for lag in lags:
            if len(current_values) >= lag:
                row.append(current_values[-lag])
            else:
                row.append(current_values[0])
        features.append(row)
    
    return np.array(features)


def simple_predict_with_trend(values: List[float], horizon: int) -> Tuple[List[float], List[float], List[float]]:
    """
    Простое прогнозирование на основе тренда (для демо, пока нет реальных моделей)
    """
    if len(values) < 5:
        return [values[-1]] * horizon, [values[-1] * 0.9] * horizon, [values[-1] * 1.1] * horizon
    
    x = np.arange(len(values))
    z = np.polyfit(x, values, 1)
    trend = np.poly1d(z)
    
    predictions = []
    lower_bounds = []
    upper_bounds = []
    
    for i in range(1, horizon + 1):
        pred = trend(len(values) + i)
        predictions.append(round(pred, 2))
        
        std = np.std(values[-20:]) if len(values) >= 20 else np.std(values)
        lower_bounds.append(round(pred - 1.96 * std, 2))
        upper_bounds.append(round(pred + 1.96 * std, 2))
    
    return predictions, lower_bounds, upper_bounds


def get_model_predictions(
    series: TimeSeries,
    model: TrainedModel,
    horizon: int,
    alpha: float = 0.05
) -> Tuple[List[float], List[float], List[float]]:
    """
    Получение прогноза от модели (загружает модель из MinIO)
    """
    logger.info(f" Generating forecast for series {series.id} using model {model.id}")
    
    # Загружаем модель из MinIO
    loaded_model = load_model_from_storage(model)
    
    if loaded_model is None:
        logger.warning(f" Model {model.id} not found in storage, using fallback")
        return simple_predict_with_trend(series.values, horizon)
    
    try:
        # Получаем лаги из гиперпараметров модели
        lags = model.hyperparams.get('lags', [1, 2, 3, 7, 14]) if model.hyperparams else [1, 2, 3, 7, 14]
        
        # Создаём признаки для прогноза
        X_pred = create_prediction_features(series.values, lags, horizon)
        
        # Делаем прогноз
        predictions = loaded_model.predict(X_pred).tolist()
        
        # Простая аппроксимация доверительных интервалов
        std_error = model.metrics.get('rmse', np.std(series.values) * 0.1) if model.metrics else np.std(series.values) * 0.1
        z_score = 1.96
        
        lower_bounds = [p - z_score * std_error for p in predictions]
        upper_bounds = [p + z_score * std_error for p in predictions]
        
        predictions = [round(p, 2) for p in predictions]
        lower_bounds = [round(l, 2) for l in lower_bounds]
        upper_bounds = [round(u, 2) for u in upper_bounds]
        
        logger.info(f" Forecast generated successfully")
        return predictions, lower_bounds, upper_bounds
        
    except Exception as e:
        logger.error(f" Prediction failed: {str(e)}")
        return simple_predict_with_trend(series.values, horizon)


def calculate_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Расчёт метрик качества прогноза"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    mae = mean_absolute_error(y_true_np, y_pred_np)
    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    mape = np.mean(np.abs((y_true_np - y_pred_np) / y_true_np)) * 100
    
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2)
    }
