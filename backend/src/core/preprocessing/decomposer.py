"""
C04-05: Decomposer - STL, аддитивная/мультипликативная декомпозиция
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple


class Decomposer:
    """
    Декомпозиция временного ряда на компоненты.
    
    Поддерживаемые модели:
    - additive: аддитивная (ряд = тренд + сезонность + остатки)
    - multiplicative: мультипликативная (ряд = тренд * сезонность * остатки)
    """
    
    def __init__(
        self, 
        model: str = 'additive',
        period: int = 7,
        use_stl: bool = True
    ):
        """
        Инициализация декомпозера
        
        Args:
            model: Тип модели ('additive' или 'multiplicative')
            period: Период сезонности
            use_stl: Использовать STL (если доступен)
        """
        self.model = model
        self.period = period
        self.use_stl = use_stl
        
        self._has_statsmodels = False
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            self.seasonal_decompose = seasonal_decompose
            self._has_statsmodels = True
        except ImportError:
            pass
    
    def _simple_decompose(
        self, 
        values: np.ndarray, 
        period: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Упрощённая декомпозиция (без statsmodels)
        
        Returns:
            (trend, seasonal, residual)
        """
        n = len(values)
        
        # 1. Тренд (скользящее среднее)
        trend = np.zeros(n)
        half_period = period // 2
        for i in range(n):
            start = max(0, i - half_period)
            end = min(n, i + half_period + 1)
            trend[i] = np.mean(values[start:end])
        
        # 2. Детрендированный ряд
        if self.model == 'multiplicative':
            detrended = values / (trend + 1e-6)
        else:
            detrended = values - trend
        
        # 3. Сезонность (среднее по периодам)
        seasonal = np.zeros(n)
        for i in range(period):
            indices = np.arange(i, n, period)
            if len(indices) > 0:
                seasonal[indices] = np.mean(detrended[indices])
        
        # Корректировка сезонности (среднее должно быть ~0 для аддитивной, ~1 для мультипликативной)
        if self.model == 'additive':
            seasonal = seasonal - np.mean(seasonal)
        else:
            seasonal = seasonal / np.mean(seasonal) if np.mean(seasonal) != 0 else seasonal
        
        # 4. Остатки
        if self.model == 'multiplicative':
            residual = values / (trend * seasonal + 1e-6)
        else:
            residual = values - trend - seasonal
        
        return trend, seasonal, residual
    
    def decompose(
        self, 
        series: Union[List[float], np.ndarray, pd.Series]
    ) -> Dict[str, Any]:
        """
        Выполнение декомпозиции временного ряда
        
        Args:
            series: Временной ряд
            
        Returns:
            Dict с компонентами декомпозиции
        """
        # Конвертация в numpy array
        if isinstance(series, pd.Series):
            values = series.values
        elif isinstance(series, list):
            values = np.array(series)
        else:
            values = series
        
        if len(values) < 2 * self.period:
            return self._simple_decompose(values, self.period)
        
        # Попытка использовать statsmodels STL
        if self._has_statsmodels and self.use_stl:
            try:
                from statsmodels.tsa.seasonal import STL
                
                stl = STL(values, period=self.period, robust=True)
                result = stl.fit()
                
                trend = result.trend
                seasonal = result.seasonal
                residual = result.resid
                
                # Для мультипликативной модели нужно преобразование
                if self.model == 'multiplicative':
                    # Преобразование в мультипликативную форму (экспонента)
                    trend_exp = np.exp(trend) if np.any(trend < 0) else trend
                    seasonal_exp = np.exp(seasonal) if np.any(seasonal < 0) else seasonal
                    residual_exp = np.exp(residual) if np.any(residual < 0) else residual
                    trend, seasonal, residual = trend_exp, seasonal_exp, residual_exp
                
                return {
                    "trend": trend.tolist(),
                    "seasonal": seasonal.tolist(),
                    "residual": residual.tolist(),
                    "observed": values.tolist(),
                    "model": self.model,
                    "period": self.period,
                    "method": "STL"
                }
            except Exception as e:
                pass
        
        # Упрощённая декомпозиция
        trend, seasonal, residual = self._simple_decompose(values, self.period)
        
        return {
            "trend": trend.tolist(),
            "seasonal": seasonal.tolist(),
            "residual": residual.tolist(),
            "observed": values.tolist(),
            "model": self.model,
            "period": self.period,
            "method": "simple"
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Получение метаданных"""
        return {
            "decomposer_type": "Decomposer",
            "model": self.model,
            "period": self.period,
            "use_stl": self.use_stl,
            "has_statsmodels": self._has_statsmodels
        }
