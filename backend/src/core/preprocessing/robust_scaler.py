"""
C04-06: RobustScaler - Для Prophet (вычитание медианы / IQR)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple


class RobustScaler:
    """
    Робастное масштабирование временного ряда.
    
    Использует медиану и межквартильный размах (IQR) вместо среднего и std.
    Устойчив к выбросам, что важно для Prophet.
    """
    
    def __init__(
        self,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: Tuple[int, int] = (25, 75)
    ):
        """
        Инициализация робастного скейлера
        
        Args:
            with_centering: Центрировать данные (вычитать медиану)
            with_scaling: Масштабировать данные (делить на IQR)
            quantile_range: Квантили для IQR
        """
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        
        self.center_ = 0.0
        self.scale_ = 1.0
        self._is_fitted = False
    
    def fit(self, series: Union[List[float], np.ndarray, pd.Series]) -> 'RobustScaler':
        """
        Обучение скейлера (вычисление медианы и IQR)
        
        Args:
            series: Временной ряд
            
        Returns:
            self
        """
        # Конвертация в numpy array
        if isinstance(series, pd.Series):
            values = series.values
        elif isinstance(series, list):
            values = np.array(series)
        else:
            values = series
        
        if len(values) == 0:
            self.center_ = 0.0
            self.scale_ = 1.0
        else:
            if self.with_centering:
                self.center_ = np.median(values)
            else:
                self.center_ = 0.0
            
            if self.with_scaling:
                q_low, q_high = self.quantile_range
                self.scale_ = np.percentile(values, q_high) - np.percentile(values, q_low)
                if self.scale_ == 0:
                    self.scale_ = 1.0
            else:
                self.scale_ = 1.0
        
        self._is_fitted = True
        return self
    
    def transform(self, series: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
        """
        Масштабирование данных
        
        Args:
            series: Временной ряд
            
        Returns:
            np.ndarray: Масштабированные данные
        """
        if not self._is_fitted:
            raise RuntimeError("Скейлер не обучен. Вызовите fit() сначала.")
        
        # Конвертация в numpy array
        if isinstance(series, pd.Series):
            values = series.values.copy()
        elif isinstance(series, list):
            values = np.array(series)
        else:
            values = series.copy()
        
        if self.with_centering:
            values = values - self.center_
        
        if self.with_scaling:
            values = values / self.scale_
        
        return values
    
    def fit_transform(
        self, 
        series: Union[List[float], np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Обучение и применение масштабирования
        
        Args:
            series: Временной ряд
            
        Returns:
            np.ndarray: Масштабированные данные
        """
        self.fit(series)
        return self.transform(series)
    
    def inverse_transform(
        self, 
        scaled_series: Union[List[float], np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Обратное преобразование (восстановление исходного масштаба)
        
        Args:
            scaled_series: Масштабированный ряд
            
        Returns:
            np.ndarray: Восстановленные данные
        """
        if not self._is_fitted:
            raise RuntimeError("Скейлер не обучен. Вызовите fit() сначала.")
        
        # Конвертация в numpy array
        if isinstance(scaled_series, pd.Series):
            values = scaled_series.values.copy()
        elif isinstance(scaled_series, list):
            values = np.array(scaled_series)
        else:
            values = scaled_series.copy()
        
        if self.with_scaling:
            values = values * self.scale_
        
        if self.with_centering:
            values = values + self.center_
        
        return values
    
    def get_metadata(self) -> Dict[str, Any]:
        """Получение метаданных"""
        return {
            "scaler_type": "RobustScaler",
            "with_centering": self.with_centering,
            "with_scaling": self.with_scaling,
            "center": float(self.center_),
            "scale": float(self.scale_),
            "quantile_range": self.quantile_range,
            "is_fitted": self._is_fitted
        }
