"""
C01-01: BaseForecaster - Абстрактный базовый класс для всех моделей прогнозирования
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from datetime import datetime


class BaseForecaster(ABC):
    """
    Абстрактный базовый класс для всех моделей прогнозирования временных рядов.
    
    Все модели прогнозирования должны наследоваться от этого класса и реализовывать
    его абстрактные методы.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Инициализация базового класса
        
        Args:
            name: Название модели (опционально)
        """
        self.name = name or self.__class__.__name__
        self.is_fitted = False
        self.created_at = datetime.utcnow()
        self._metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'BaseForecaster':
        """
        Обучение модели на временном ряде
        
        Args:
            y: Временной ряд (значения)
            X: Экзогенные переменные (опционально)
            
        Returns:
            self: Обученная модель
        """
        pass
    
    @abstractmethod
    def predict(self, horizon: int, X_future: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Прогнозирование на заданный горизонт
        
        Args:
            horizon: Количество шагов для прогнозирования
            X_future: Будущие значения экзогенных переменных
            
        Returns:
            np.ndarray: Массив прогнозных значений
        """
        pass
    
    @abstractmethod
    def predict_interval(
        self, 
        horizon: int, 
        alpha: float = 0.05,
        X_future: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Прогнозирование с доверительными интервалами
        
        Args:
            horizon: Количество шагов для прогнозирования
            alpha: Уровень значимости (по умолчанию 0.05 = 95% доверительный интервал)
            X_future: Будущие значения экзогенных переменных
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (нижняя граница, верхняя граница)
        """
        pass
    
    def get_interpretation(self) -> Dict[str, Any]:
        """
        Получение интерпретации модели
        
        Returns:
            Dict[str, Any]: Словарь с информацией о модели и её интерпретации
        """
        return {
            "model_type": self.__class__.__name__,
            "name": self.name,
            "is_fitted": self.is_fitted,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self._metadata
        }
    
    def save(self, path: str) -> None:
        """
        Сохранение модели в файл
        
        Args:
            path: Путь к файлу для сохранения
        """
        import joblib
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'BaseForecaster':
        """
        Загрузка модели из файла
        
        Args:
            path: Путь к файлу модели
            
        Returns:
            BaseForecaster: Загруженная модель
        """
        import joblib
        return joblib.load(path)
    
    def _validate_input(self, y: pd.Series) -> None:
        """Валидация входных данных"""
        if y is None or len(y) == 0:
            raise ValueError("Входной ряд не может быть пустым")
        if len(y) < 3:
            raise ValueError(f"Ряд должен содержать минимум 3 точки, получено: {len(y)}")
    
    def _validate_horizon(self, horizon: int) -> None:
        """Валидация горизонта прогноза"""
        if horizon < 1:
            raise ValueError(f"Горизонт прогноза должен быть >= 1, получено: {horizon}")
