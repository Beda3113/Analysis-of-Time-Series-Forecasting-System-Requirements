"""
C01-07: EnsembleForecaster - Ансамбль моделей для прогнозирования
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from src.core.forecasting.base import BaseForecaster


class EnsembleForecaster(BaseForecaster):
    """
    Ансамблевая модель, объединяющая прогнозы нескольких базовых моделей.
    
    Поддерживаемые методы объединения:
    - 'mean': среднее арифметическое
    - 'median': медиана
    - 'weighted': взвешенное среднее (веса по метрикам)
    - 'best': выбор лучшей модели по заданной метрике
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        models: Optional[List[BaseForecaster]] = None,
        method: str = 'mean',
        weights: Optional[List[float]] = None,
        metric: str = 'mae'
    ):
        """
        Инициализация ансамблевой модели
        
        Args:
            name: Название ансамбля
            models: Список базовых моделей
            method: Метод объединения ('mean', 'median', 'weighted', 'best')
            weights: Веса для взвешенного среднего
            metric: Метрика для выбора лучшей модели ('mae', 'rmse', 'mape')
        """
        super().__init__(name)
        
        self.models = models or []
        self.method = method
        self.weights = weights
        self.metric = metric
        self._model_metrics: Dict[int, Dict[str, float]] = {}
    
    def add_model(self, model: BaseForecaster, weight: Optional[float] = None) -> None:
        """
        Добавление модели в ансамбль
        
        Args:
            model: Модель для добавления
            weight: Вес модели (для взвешенного среднего)
        """
        self.models.append(model)
        if weight is not None:
            if self.weights is None:
                self.weights = []
            self.weights.append(weight)
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'EnsembleForecaster':
        """
        Обучение всех моделей в ансамбле
        """
        self._validate_input(y)
        
        for i, model in enumerate(self.models):
            model.fit(y, X)
            
            # Сохраняем метрики модели
            interp = model.get_interpretation()
            if 'metrics' in interp:
                self._model_metrics[i] = interp['metrics']
        
        self.is_fitted = True
        return self
    
    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Объединённый прогноз ансамбля
        """
        if not self.is_fitted:
            raise RuntimeError("Ансамбль не обучен. Вызовите fit() сначала.")
        
        if not self.models:
            raise ValueError("Ансамбль не содержит моделей")
        
        # Получение прогнозов от всех моделей
        predictions = []
        for model in self.models:
            pred = model.predict(horizon, X_future)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Объединение прогнозов
        if self.method == 'mean':
            result = np.mean(predictions, axis=0)
        elif self.method == 'median':
            result = np.median(predictions, axis=0)
        elif self.method == 'weighted':
            if self.weights is None:
                # Автоматический расчёт весов на основе метрик
                weights = self._calculate_weights()
            else:
                weights = np.array(self.weights)
            weights = weights / np.sum(weights)
            result = np.average(predictions, axis=0, weights=weights)
        elif self.method == 'best':
            best_idx = self._get_best_model_index()
            result = predictions[best_idx]
        else:
            raise ValueError(f"Неизвестный метод объединения: {self.method}")
        
        return result
    
    def predict_interval(
        self, 
        horizon: int, 
        alpha: float = 0.05,
        X_future: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Прогнозирование с доверительными интервалами (минимальный/максимальный прогноз)
        """
        if not self.is_fitted:
            raise RuntimeError("Ансамбль не обучен. Вызовите fit() сначала.")
        
        # Получение прогнозов от всех моделей
        predictions = []
        for model in self.models:
            pred = model.predict(horizon, X_future)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Доверительный интервал как диапазон прогнозов
        lower = np.min(predictions, axis=0)
        upper = np.max(predictions, axis=0)
        
        return lower, upper
    
    def _calculate_weights(self) -> np.ndarray:
        """Расчёт весов на основе метрик качества"""
        if not self._model_metrics:
            return np.ones(len(self.models)) / len(self.models)
        
        # Обратная зависимость от метрики (чем меньше метрика, тем больше вес)
        scores = []
        for i in range(len(self.models)):
            metrics = self._model_metrics.get(i, {})
            score = metrics.get(self.metric, 1.0)
            if score == 0:
                score = 1e-6
            scores.append(1.0 / score)
        
        weights = np.array(scores)
        return weights / np.sum(weights)
    
    def _get_best_model_index(self) -> int:
        """Получение индекса лучшей модели по метрике"""
        if not self._model_metrics:
            return 0
        
        best_idx = 0
        best_score = float('inf')
        
        for i in range(len(self.models)):
            metrics = self._model_metrics.get(i, {})
            score = metrics.get(self.metric, float('inf'))
            if score < best_score:
                best_score = score
                best_idx = i
        
        return best_idx
    
    def get_interpretation(self) -> Dict[str, Any]:
        """
        Получение интерпретации ансамблевой модели
        """
        interpretation = super().get_interpretation()
        
        models_info = []
        for i, model in enumerate(self.models):
            models_info.append({
                "index": i,
                "type": model.__class__.__name__,
                "name": model.name,
                "metrics": self._model_metrics.get(i, {})
            })
        
        interpretation.update({
            "model_type": "ensemble",
            "method": self.method,
            "n_models": len(self.models),
            "models": models_info,
            "weights": self.weights if self.weights else self._calculate_weights().tolist()
        })
        
        return interpretation
    
    def get_best_model(self) -> Optional[BaseForecaster]:
        """
        Получение лучшей модели из ансамбля
        """
        if not self.models:
            return None
        
        best_idx = self._get_best_model_index()
        return self.models[best_idx]
