"""
C01-04: ProphetForecaster - Обёртка над Facebook Prophet
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple

from src.core.forecasting.base import BaseForecaster


class ProphetForecaster(BaseForecaster):
    """
    Модель Prophet от Facebook для прогнозирования временных рядов с сезонностью
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        seasonality_mode: str = 'additive',
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        **kwargs
    ):
        """
        Инициализация Prophet модели
        """
        super().__init__(name)
        
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.kwargs = kwargs
        
        self._model = None
        self._dates = None
        self._metrics = {}
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'ProphetForecaster':
        """
        Обучение Prophet модели
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet не установлен. Установите: pip install prophet")
        
        self._validate_input(y)
        
        # Подготовка данных для Prophet (требует колонки ds и y)
        if isinstance(y, pd.Series):
            if y.index.name == 'ds' or isinstance(y.index, pd.DatetimeIndex):
                df = pd.DataFrame({
                    'ds': y.index,
                    'y': y.values
                })
            else:
                # Создаём даты по умолчанию
                df = pd.DataFrame({
                    'ds': pd.date_range(start='2020-01-01', periods=len(y), freq='D'),
                    'y': y.values
                })
        else:
            df = pd.DataFrame({
                'ds': pd.date_range(start='2020-01-01', periods=len(y), freq='D'),
                'y': y
            })
        
        self._dates = df['ds'].copy()
        
        # Создание и обучение модели
        self._model = Prophet(
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            **self.kwargs
        )
        
        self._model.fit(df)
        self.is_fitted = True
        
        # Расчёт метрик (на обучающих данных)
        forecast = self._model.predict(df)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        y_pred = forecast['yhat'].values
        self._metrics = {
            "mae": float(mean_absolute_error(y.values, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y.values, y_pred)))
        }
        
        return self
    
    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Прогнозирование с помощью Prophet
        """
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена. Вызовите fit() сначала.")
        
        self._validate_horizon(horizon)
        
        # Создание будущих дат
        if self._dates is not None and len(self._dates) > 0:
            last_date = self._dates.iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
        else:
            future_dates = pd.date_range(start='2024-01-01', periods=horizon, freq='D')
        
        future_df = pd.DataFrame({'ds': future_dates})
        
        # Прогноз
        forecast = self._model.predict(future_df)
        
        return forecast['yhat'].values
    
    def predict_interval(
        self, 
        horizon: int, 
        alpha: float = 0.05,
        X_future: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Прогнозирование с доверительными интервалами (встроено в Prophet)
        """
        if not self.is_fitted:
            raise RuntimeError("Модель не обучена. Вызовите fit() сначала.")
        
        self._validate_horizon(horizon)
        
        # Создание будущих дат
        if self._dates is not None and len(self._dates) > 0:
            last_date = self._dates.iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
        else:
            future_dates = pd.date_range(start='2024-01-01', periods=horizon, freq='D')
        
        future_df = pd.DataFrame({'ds': future_dates})
        
        # Прогноз
        forecast = self._model.predict(future_df)
        
        # Prophet даёт yhat_lower и yhat_upper
        lower = forecast['yhat_lower'].values if 'yhat_lower' in forecast.columns else forecast['yhat'].values - 1.96 * forecast['yhat'].std()
        upper = forecast['yhat_upper'].values if 'yhat_upper' in forecast.columns else forecast['yhat'].values + 1.96 * forecast['yhat'].std()
        
        return lower, upper
    
    def get_interpretation(self) -> Dict[str, Any]:
        """
        Получение интерпретации Prophet модели (тренд, сезонность)
        """
        interpretation = super().get_interpretation()
        interpretation.update({
            "model_type": "prophet",
            "parameters": {
                "seasonality_mode": self.seasonality_mode,
                "yearly_seasonality": self.yearly_seasonality,
                "weekly_seasonality": self.weekly_seasonality
            },
            "metrics": self._metrics
        })
        
        return interpretation
    
    def plot_components(self):
        """
        Визуализация компонентов модели (тренд, сезонность)
        """
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Модель не обучена")
        
        return self._model.plot_components(self._model.history)
