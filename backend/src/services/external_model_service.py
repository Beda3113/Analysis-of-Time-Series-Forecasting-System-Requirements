"""
Сервис для работы с внешними (клиентскими) моделями
"""

import os
import uuid
import tempfile
import pickle
import hashlib
import time
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import numpy as np

from src.models import get_series_by_id
from src.utils.logger import get_logger

logger = get_logger("external_model")


class ExternalModelStorage:
    """Хранилище для внешних моделей (in-memory для демо)"""
    
    def __init__(self):
        self.models: Dict[str, dict] = {}  # model_id -> model_data
        self.models_metadata: Dict[str, dict] = {}  # model_id -> metadata
    
    def add_model(self, model_id: str, model_data: dict, metadata: dict) -> None:
        self.models[model_id] = model_data
        self.models_metadata[model_id] = metadata
    
    def get_model(self, model_id: str) -> Optional[dict]:
        return self.models.get(model_id)
    
    def get_metadata(self, model_id: str) -> Optional[dict]:
        return self.models_metadata.get(model_id)
    
    def delete_model(self, model_id: str) -> bool:
        if model_id in self.models:
            del self.models[model_id]
            del self.models_metadata[model_id]
            return True
        return False
    
    def list_models(self, user_id: str) -> List[dict]:
        return [meta for meta in self.models_metadata.values() if meta.get("user_id") == user_id]


# Глобальное хранилище
external_model_storage = ExternalModelStorage()


def validate_model_file(file_content: bytes, model_type: str) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """
    Валидация загруженного файла модели
    
    Returns:
        (is_valid, errors, warnings, metadata)
    """
    errors = []
    warnings = []
    metadata = {}
    
    try:
        # Попытка загрузить модель
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        with open(tmp_path, 'rb') as f:
            model = pickle.load(f)
        
        # Проверка наличия метода predict
        if hasattr(model, 'predict'):
            metadata['has_predict'] = True
        else:
            errors.append("Модель не имеет метода 'predict'")
        
        # Проверка наличия метода predict_proba
        if hasattr(model, 'predict_proba'):
            metadata['has_predict_proba'] = True
        
        # Проверка типа модели
        model_class_name = model.__class__.__name__
        metadata['class_name'] = model_class_name
        
        # Проверка соответствия типа
        expected_types = {
            'xgboost': ['XGBRegressor', 'XGBClassifier', 'Booster'],
            'sklearn': ['LinearRegression', 'RandomForestRegressor', 'SVR', 'KNeighborsRegressor'],
            'lightgbm': ['LGBMRegressor', 'LGBMClassifier'],
            'catboost': ['CatBoostRegressor', 'CatBoostClassifier']
        }
        
        expected = expected_types.get(model_type, [])
        if expected and not any(exp in model_class_name for exp in expected):
            warnings.append(f"Тип модели '{model_class_name}' может не соответствовать заявленному '{model_type}'")
        
        # Тестовый запуск predict на небольшом наборе данных
        try:
            test_input = np.random.rand(5, 10).tolist()
            test_output = model.predict(test_input)
            metadata['test_predict_success'] = True
            metadata['test_output_shape'] = len(test_output) if hasattr(test_output, '__len__') else 1
        except Exception as e:
            warnings.append(f"Тестовый вызов predict() не удался: {str(e)}")
            metadata['test_predict_success'] = False
        
        # Очистка
        os.unlink(tmp_path)
        
        is_valid = len(errors) == 0
        
    except Exception as e:
        errors.append(f"Ошибка загрузки модели: {str(e)}")
        is_valid = False
    
    return is_valid, errors, warnings, metadata


def generate_model_id() -> str:
    """Генерация уникального ID для модели"""
    return str(uuid.uuid4())


def get_model_cache_key(model_id: str, series_id: str, horizon: int) -> str:
    """Генерация ключа для кэша прогнозов"""
    return f"external_forecast:{model_id}:{series_id}:{horizon}"


def predict_with_external_model(
    model_id: str,
    series_id: str,
    horizon: int,
    features: Optional[List[List[float]]] = None
) -> Tuple[List[float], Optional[Dict[str, List[float]]], float]:
    """
    Прогнозирование с использованием внешней модели
    """
    start_time = time.time()
    
    model_data = external_model_storage.get_model(model_id)
    metadata = external_model_storage.get_metadata(model_id)
    
    if not model_data:
        raise ValueError(f"Модель {model_id} не найдена")
    
    model = model_data.get('model')
    if not model:
        raise ValueError(f"Модель {model_id} не содержит загруженный объект")
    
    series = get_series_by_id(series_id)
    if not series:
        raise ValueError(f"Ряд {series_id} не найден")
    
    # Если признаки не предоставлены, создаём простые лаги
    if features is None:
        features = []
        values = series.values
        lags = [1, 2, 3, 7, 14]
        max_lag = max(lags)
        
        # Используем последние значения для прогноза
        for i in range(horizon):
            row = []
            for lag in lags:
                if len(values) - i - lag >= 0:
                    row.append(values[-i - lag] if i + lag <= len(values) else values[0])
                else:
                    row.append(values[0])
            features.append(row)
        
        # Первый прогноз на основе последних значений
        try:
            predictions = model.predict(features).tolist()
        except Exception as e:
            logger.warning(f"Predict failed: {e}, using fallback")
            predictions = [series.values[-1] + (i * 0.01) for i in range(horizon)]
    else:
        try:
            predictions = model.predict(features).tolist()
        except Exception as e:
            logger.error(f"Predict failed: {e}")
            predictions = [0] * horizon
    
    execution_time = (time.time() - start_time) * 1000
    
    return predictions, None, execution_time
