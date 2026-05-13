"""
Модуль прогнозирования временных рядов
"""

from src.core.forecasting.base import BaseForecaster
from src.core.forecasting.xgboost_forecaster import XGBoostForecaster
from src.core.forecasting.lstm_forecaster import LSTMForecaster
from src.core.forecasting.prophet_forecaster import ProphetForecaster
from src.core.forecasting.sarima_forecaster import SARIMAForecaster
from src.core.forecasting.registry import ModelRegistry, get_forecaster
from src.core.forecasting.ensemble import EnsembleForecaster

__all__ = [
    'BaseForecaster',
    'XGBoostForecaster',
    'LSTMForecaster',
    'ProphetForecaster',
    'SARIMAForecaster',
    'ModelRegistry',
    'get_forecaster',
    'EnsembleForecaster'
]
