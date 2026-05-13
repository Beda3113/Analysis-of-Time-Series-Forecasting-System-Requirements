"""
C04-04: Differencing - Первая и сезонная разность для приведения к стационарности
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple


class Differencing:
    """
    Дифференцирование временного ряда.
    
    Поддерживает:
    - Обычное дифференцирование (первая, вторая разность)
    - Сезонное дифференцирование
    - Комбинированное дифференцирование
    """
    
    def __init__(self):
        """Инициализация процессора дифференцирования"""
        self._original_length = 0
        self._differenced_length = 0
    
    def difference(
        self, 
        series: Union[List[float], np.ndarray, pd.Series],
        order: int = 1,
        seasonal: Optional[int] = None,
        seasonal_order: int = 1
    ) -> np.ndarray:
        """
        Дифференцирование временного ряда
        
        Args:
            series: Временной ряд
            order: Порядок обычного дифференцирования (1, 2, 3)
            seasonal: Период сезонного дифференцирования (например, 7 для недели)
            seasonal_order: Порядок сезонного дифференцирования
            
        Returns:
            np.ndarray: Продифференцированный ряд
        """
        # Конвертация в numpy array
        if isinstance(series, pd.Series):
            values = series.values.copy()
        elif isinstance(series, list):
            values = np.array(series)
        else:
            values = series.copy()
        
        self._original_length = len(values)
        
        # Обычное дифференцирование
        for _ in range(order):
            if len(values) <= 1:
                values = np.array([])
                break
            values = values[1:] - values[:-1]
        
        # Сезонное дифференцирование
        if seasonal is not None and seasonal > 0 and len(values) > seasonal:
            for _ in range(seasonal_order):
                if len(values) <= seasonal:
                    values = np.array([])
                    break
                values = values[seasonal:] - values[:-seasonal]
        
        self._differenced_length = len(values)
        return values
    
    def inverse_difference(
        self,
        differenced_series: Union[List[float], np.ndarray, pd.Series],
        original_series: Union[List[float], np.ndarray, pd.Series],
        order: int = 1,
        seasonal: Optional[int] = None,
        seasonal_order: int = 1
    ) -> np.ndarray:
        """
        Обратное преобразование (восстановление исходного ряда)
        
        Args:
            differenced_series: Продифференцированный ряд
            original_series: Исходный ряд (нужен первый элемент)
            order: Порядок обычного дифференцирования
            seasonal: Период сезонного дифференцирования
            seasonal_order: Порядок сезонного дифференцирования
            
        Returns:
            np.ndarray: Восстановленный ряд
        """
        if isinstance(original_series, pd.Series):
            original = original_series.values.copy()
        elif isinstance(original_series, list):
            original = np.array(original_series)
        else:
            original = original_series.copy()
        
        if isinstance(differenced_series, pd.Series):
            diff = differenced_series.values.copy()
        elif isinstance(differenced_series, list):
            diff = np.array(differenced_series)
        else:
            diff = differenced_series.copy()
        
        # Обратное восстановление
        result = original[:max(1, len(original) - len(diff))].copy()
        
        # Сезонное обратное дифференцирование
        if seasonal is not None and seasonal > 0:
            for _ in range(seasonal_order):
                if len(result) <= seasonal:
                    break
                # Это упрощённая версия, для полного восстановления нужна история
                pass
        
        # Обычное обратное дифференцирование
        for _ in range(order):
            if len(result) == 0:
                break
            restored = result.copy()
            for i in range(1, len(diff) + 1):
                if i <= len(diff):
                    restored = np.append(restored, restored[-1] + diff[i-1])
            result = restored
        
        return result
    
    def find_optimal_order(
        self, 
        series: Union[List[float], np.ndarray, pd.Series],
        max_order: int = 2
    ) -> Dict[str, Any]:
        """
        Поиск оптимального порядка дифференцирования
        
        Args:
            series: Временной ряд
            max_order: Максимальный порядок для проверки
            
        Returns:
            Dict с рекомендациями
        """
        from src.core.preprocessing.stationarity_tester import StationarityTester
        
        tester = StationarityTester()
        
        results = []
        for order in range(max_order + 1):
            diff = self.difference(series, order=order)
            if len(diff) > 3:
                test_result = tester.test(diff)
                results.append({
                    "order": order,
                    "is_stationary": test_result.get("is_stationary", False),
                    "p_value": test_result.get("p_value", 1.0),
                    "adf_statistic": test_result.get("adf_statistic", 0)
                })
            else:
                results.append({
                    "order": order,
                    "is_stationary": False,
                    "p_value": 1.0,
                    "adf_statistic": 0,
                    "error": "Недостаточно данных"
                })
        
        # Выбор оптимального порядка
        optimal_order = 0
        for result in results:
            if result.get("is_stationary"):
                optimal_order = result["order"]
                break
        
        return {
            "results": results,
            "optimal_order": optimal_order,
            "recommendation": f"Рекомендуемый порядок дифференцирования: {optimal_order}"
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Получение метаданных"""
        return {
            "processor_type": "Differencing",
            "original_length": self._original_length,
            "differenced_length": self._differenced_length
        }
