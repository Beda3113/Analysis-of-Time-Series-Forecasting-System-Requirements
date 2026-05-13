"""
C01-05: SARIMAForecaster - Реализация SARIMA модели
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple

from src.core.forecasting.base import BaseForecaster


class SARIMAForecaster(BaseForecaster):
    """
    Модель SARIMA для прогнозирования временных рядов с сезонностью
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
        trend: Optional[str] = None,
        **kwargs
    ):
        """
        Инициализация SARIMA модели
        
        Args:
            name: Название модели
            order: (p, d, q) - порядок ARIMA
            seasonal_order: (P, D, Q, s) - сезонный порядок
            trend: Тренд ('n', 'c', 't', 'ct')
        """
        super().__init__(name)
        
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.kwargs = kwargs
        
        self._model = None
        self._result = None
        self._metrics = {}
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'SARIMAForecaster':
        """
        Обучение SARIMA модели
        """
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ImportError("statsmodels не установлен. Установите: pip install statsmodels")
        
        self._validate_input(y)
        
        # Создание и обучение модели
        self._model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            **self.kwargs
        )
        
        self._result = self._model.fit(disp=False)
        self.is_fitted = True
        
        # Расчёт метрик
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        y_pred = self._result.fittedvalues
        self._metrics = {
            "mae": float(mean_absolute_error(y, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "aic": float(self._result.aic) if hasattr(self._result, 'aic') else None,
            "bic": float(self._result.bic) if hasattr(self._result, 'bic') else None
        }
        
        return self
    
    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Прогнозирование с помощью SARIMA
        """
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена. Вызовите fit() сначала.")
        
        self._validate_horizon(horizon)
        
        forecast = self._result.forecast(steps=horizon)
        return forecast.values
    
    def predict_interval(
        self, 
        horizon: int, 
        alpha: float = 0.05,
        X_future: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Прогнозирование с доверительными интервалами
        """
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена. Вызовите fit() сначала.")
        
        self._validate_horizon(horizon)
        
        forecast_result = self._result.get_forecast(steps=horizon)
        conf_int = forecast_result.conf_int(alpha=alpha)
        
        lower = conf_int.iloc[:, 0].values
        upper = conf_int.iloc[:, 1].values
        
        return lower, upper
    
    def get_interpretation(self) -> Dict[str, Any]:
        """
        Получение интерпретации SARIMA модели
        """
        interpretation = super().get_interpretation()
        interpretation.update({
            "model_type": "sarima",
            "parameters": {
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "trend": self.trend
            },
            "metrics": self._metrics
        })
        
        if self._result is not None:
            interpretation["summary"] = str(self._result.summary())
        
        return interpretation
