"""
C04-02: AnomalyHandler - Обработка аномалий (кубический сплайн, медиана, удаление)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from scipy import interpolate


class AnomalyHandler:
    """
    Обработчик аномалий во временных рядах.
    
    Поддерживаемые методы:
    - spline: интерполяция кубическим сплайном
    - median: замена на медиану соседних точек
    - linear: линейная интерполяция
    - delete: удаление аномальных точек
    """
    
    def __init__(
        self,
        method: str = 'spline',
        window_size: int = 5,
        fallback_to_median: bool = True
    ):
        """
        Инициализация обработчика аномалий
        
        Args:
            method: Метод обработки ('spline', 'median', 'linear', 'delete')
            window_size: Размер окна для медианы или интерполяции
            fallback_to_median: Использовать медиану при ошибке интерполяции
        """
        self.method = method
        self.window_size = window_size
        self.fallback_to_median = fallback_to_median
        
        self._fixed_values = None
        self._fixed_count = 0
    
    def _get_neighbors(
        self, 
        values: np.ndarray, 
        idx: int, 
        window: int
    ) -> Tuple[List[int], List[float]]:
        """
        Получение соседних не-аномальных точек
        
        Args:
            values: Исходные значения
            idx: Индекс аномальной точки
            window: Размер окна
            
        Returns:
            (indices, values) соседних точек
        """
        indices = []
        neighbor_values = []
        
        # Поиск вперёд
        for offset in range(1, window + 1):
            if idx + offset < len(values):
                indices.append(idx + offset)
                neighbor_values.append(values[idx + offset])
                break
        
        # Поиск назад
        for offset in range(1, window + 1):
            if idx - offset >= 0:
                indices.append(idx - offset)
                neighbor_values.append(values[idx - offset])
                break
        
        return indices, neighbor_values
    
    def _fix_with_median(
        self, 
        values: np.ndarray, 
        anomaly_indices: List[int]
    ) -> np.ndarray:
        """
        Замена аномалий на медиану соседних точек
        """
        fixed = values.copy()
        
        for idx in anomaly_indices:
            # Поиск соседних точек
            neighbors = []
            for offset in range(1, self.window_size + 1):
                if idx - offset >= 0 and (idx - offset) not in anomaly_indices:
                    neighbors.append(values[idx - offset])
                if idx + offset < len(values) and (idx + offset) not in anomaly_indices:
                    neighbors.append(values[idx + offset])
            
            if neighbors:
                fixed[idx] = np.median(neighbors)
            else:
                fixed[idx] = np.median(values)
        
        return fixed
    
    def _fix_with_linear(
        self, 
        values: np.ndarray, 
        anomaly_indices: List[int]
    ) -> np.ndarray:
        """
        Линейная интерполяция аномалий
        """
        fixed = values.copy()
        anomaly_set = set(anomaly_indices)
        
        # Находим группы последовательных аномалий
        groups = []
        current_group = []
        
        for i in range(len(values)):
            if i in anomaly_set:
                current_group.append(i)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
        if current_group:
            groups.append(current_group)
        
        # Интерполяция для каждой группы
        for group in groups:
            left_idx = group[0] - 1
            right_idx = group[-1] + 1
            
            left_val = values[left_idx] if left_idx >= 0 else np.nan
            right_val = values[right_idx] if right_idx < len(values) else np.nan
            
            if not np.isnan(left_val) and not np.isnan(right_val):
                # Линейная интерполяция между двумя точками
                step = (right_val - left_val) / (len(group) + 1)
                for i, idx in enumerate(group):
                    fixed[idx] = left_val + step * (i + 1)
            elif not np.isnan(left_val):
                for idx in group:
                    fixed[idx] = left_val
            elif not np.isnan(right_val):
                for idx in group:
                    fixed[idx] = right_val
            else:
                for idx in group:
                    fixed[idx] = np.median(values)
        
        return fixed
    
    def _fix_with_spline(
        self, 
        values: np.ndarray, 
        anomaly_indices: List[int]
    ) -> np.ndarray:
        """
        Интерполяция аномалий кубическим сплайном
        """
        fixed = values.copy()
        anomaly_set = set(anomaly_indices)
        
        # Индексы хороших точек
        good_indices = [i for i in range(len(values)) if i not in anomaly_set]
        good_values = [values[i] for i in good_indices]
        
        if len(good_indices) < 3:
            # Если слишком мало хороших точек, используем медиану
            if self.fallback_to_median:
                return self._fix_with_median(values, anomaly_indices)
            return fixed
        
        try:
            # Кубическая сплайн-интерполяция
            spline = interpolate.CubicSpline(good_indices, good_values, extrapolate=True)
            
            for idx in anomaly_indices:
                fixed[idx] = float(spline(idx))
        except Exception:
            # При ошибке используем линейную интерполяцию
            if self.fallback_to_median:
                return self._fix_with_linear(values, anomaly_indices)
        
        return fixed
    
    def _fix_with_delete(
        self, 
        values: np.ndarray, 
        anomaly_indices: List[int]
    ) -> np.ndarray:
        """
        Удаление аномальных точек
        """
        mask = np.ones(len(values), dtype=bool)
        mask[anomaly_indices] = False
        return values[mask]
    
    def fix(
        self, 
        values: Union[List[float], np.ndarray, pd.Series],
        anomaly_indices: List[int]
    ) -> Dict[str, Any]:
        """
        Основной метод обработки аномалий
        
        Args:
            values: Временной ряд
            anomaly_indices: Индексы аномалий для обработки
            
        Returns:
            Dict с результатами обработки
        """
        # Конвертация в numpy array
        if isinstance(values, pd.Series):
            values = values.values
        elif isinstance(values, list):
            values = np.array(values)
        
        if not anomaly_indices:
            return {
                "fixed_values": values.tolist(),
                "fixed_count": 0,
                "method": self.method,
                "message": "Аномалии не обнаружены"
            }
        
        # Выбор метода обработки
        if self.method == 'median':
            fixed = self._fix_with_median(values, anomaly_indices)
        elif self.method == 'linear':
            fixed = self._fix_with_linear(values, anomaly_indices)
        elif self.method == 'spline':
            fixed = self._fix_with_spline(values, anomaly_indices)
        elif self.method == 'delete':
            fixed = self._fix_with_delete(values, anomaly_indices)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")
        
        # Преобразование обратно в список
        if hasattr(fixed, 'tolist'):
            fixed = fixed.tolist()
        
        self._fixed_values = fixed
        self._fixed_count = len(anomaly_indices)
        
        return {
            "fixed_values": fixed,
            "fixed_count": len(anomaly_indices),
            "method": self.method,
            "message": f"Обработано {len(anomaly_indices)} аномалий методом '{self.method}'"
        }
    
    def get_fixed_values(self) -> Optional[List[float]]:
        """Получение обработанного ряда из последнего вызова fix()"""
        return self._fixed_values
    
    def get_metadata(self) -> Dict[str, Any]:
        """Получение метаданных"""
        return {
            "handler_type": "AnomalyHandler",
            "method": self.method,
            "window_size": self.window_size,
            "fallback_to_median": self.fallback_to_median
        }
