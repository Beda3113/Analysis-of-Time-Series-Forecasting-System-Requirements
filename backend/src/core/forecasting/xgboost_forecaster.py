"""
C01-02: XGBoostForecaster - Реализация модели XGBoost для прогнозирования временных рядов
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.core.forecasting.base import BaseForecaster


class XGBoostForecaster(BaseForecaster):
    """
    Модель XGBoost для прогнозирования временных рядов
    
    Использует лаговые значения в качестве признаков для обучения.
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        lags: Optional[List[int]] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs
    ):
        """
        Инициализация XGBoost модели
        
        Args:
            name: Название модели
            lags: Список лагов для использования в качестве признаков
            n_estimators: Количество деревьев
            max_depth: Максимальная глубина дерева
            learning_rate: Скорость обучения
            subsample: Доля выборки для обучения каждого дерева
            colsample_bytree: Доля признаков для каждого дерева
            random_state: Случайное зерно для воспроизводимости
        """
        super().__init__(name)
        
        # Параметры модели
        self.lags = lags or [1, 2, 3, 7, 14, 30]
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Внутренние переменные
        self._model = None
        self._feature_names = None
        self._last_values = None
        self._metrics = {}
    
    def _create_lag_features(self, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Создание лаговых признаков из временного ряда
        
        Args:
            y: Временной ряд
            
        Returns:
            X: Матрица признаков
            y_target: Целевая переменная
            feature_names: Названия признаков
        """
        values = y.values
        X = []
        y_target = []
        max_lag = max(self.lags)
        
        for i in range(max_lag, len(values)):
            features = []
            for lag in self.lags:
                features.append(values[i - lag])
            X.append(features)
            y_target.append(values[i])
        
        feature_names = [f"lag_{lag}" for lag in self.lags]
        
        return np.array(X), np.array(y_target), feature_names
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'XGBoostForecaster':
        """
        Обучение модели XGBoost
        """
        import xgboost as xgb
        
        self._validate_input(y)
        
        # Создание признаков
        X_train, y_train, self._feature_names = self._create_lag_features(y)
        
        if len(X_train) < 10:
            raise ValueError(f"Недостаточно данных для обучения. Нужно минимум {max(self.lags) + 10} точек")
        
        # Обучение модели
        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            **self.kwargs
        )
        
        self._model.fit(X_train, y_train)
        
        # Сохранение последних значений для прогноза
        self._last_values = y.values.tolist()
        self.is_fitted = True
        
        # Расчёт метрик на обучающих данных
        y_pred = self._model.predict(X_train)
        self._metrics = {
            "mae": float(mean_absolute_error(y_train, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_train, y_pred))),
            "train_size": len(X_train)
        }
        
        return self
    
    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Рекурсивное прогнозирование с помощью XGBoost
        """
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена. Вызовите fit() сначала.")
        
        self._validate_horizon(horizon)
        
        predictions = []
        current_values = self._last_values.copy()
        
        for step in range(horizon):
            # Создание признаков из последних значений
            features = []
            for lag in self.lags:
                if len(current_values) >= lag:
                    features.append(current_values[-lag])
                else:
                    features.append(current_values[0])
            
            # Прогноз
            pred = self._model.predict([features])[0]
            predictions.append(pred)
            current_values.append(pred)
        
        return np.array(predictions)
    
    def predict_interval(
        self, 
        horizon: int, 
        alpha: float = 0.05,
        X_future: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Прогнозирование с доверительными интервалами (используется бутстрап)
        """
        predictions = self.predict(horizon)
        
        # Простая аппроксимация доверительного интервала на основе метрик
        std_error = self._metrics.get("rmse", 1.0) * 1.5
        z_score = 1.96  # для alpha=0.05
        
        lower = predictions - z_score * std_error
        upper = predictions + z_score * std_error
        
        return lower, upper
    
    def get_interpretation(self) -> Dict[str, Any]:
        """
        Получение интерпретации модели XGBoost
        """
        interpretation = super().get_interpretation()
        interpretation.update({
            "model_type": "xgboost",
            "parameters": {
                "lags": self.lags,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate
            },
            "metrics": self._metrics,
            "feature_names": self._feature_names
        })
        
        # Получение важности признаков
        if self._model is not None:
            try:
                importance = self._model.feature_importances_
                if importance is not None and self._feature_names:
                    interpretation["feature_importance"] = {
                        name: float(imp) 
                        for name, imp in zip(self._feature_names, importance)
                    }
            except:
                pass
        
        return interpretation
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Получение важности признаков
        """
        if not self.is_fitted or self._model is None:
            return {}
        
        try:
            importance = self._model.feature_importances_
            if importance is not None and self._feature_names:
                return {
                    name: float(imp) 
                    for name, imp in zip(self._feature_names, importance)
                }
        except:
            pass
        
        return {}
