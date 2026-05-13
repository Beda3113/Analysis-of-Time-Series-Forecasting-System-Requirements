"""
C04-01: AnomalyDetector - Z-score, IQR, STL комбинация для детекции аномалий
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from scipy import stats


class AnomalyDetector:
    """
    Детектор аномалий во временных рядах.
    
    Поддерживаемые методы:
    - Z-score: на основе стандартного отклонения
    - IQR: на основе межквартильного размаха
    - STL: сезонно-трендовая декомпозиция
    - Combined: комбинация методов
    """
    
    def __init__(
        self,
        method: str = 'combined',
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        stl_window: int = 30,
        min_anomaly_distance: int = 1
    ):
        """
        Инициализация детектора аномалий
        
        Args:
            method: Метод детекции ('zscore', 'iqr', 'stl', 'combined')
            zscore_threshold: Порог для Z-score (по умолчанию 3.0)
            iqr_multiplier: Множитель для IQR (по умолчанию 1.5)
            stl_window: Размер окна для STL
            min_anomaly_distance: Минимальное расстояние между аномалиями
        """
        self.method = method
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.stl_window = stl_window
        self.min_anomaly_distance = min_anomaly_distance
        
        self._last_anomalies = None
        self._scores = None
    
    def detect_zscore(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Детекция аномалий методом Z-score
        
        Args:
            values: Временной ряд
            
        Returns:
            (anomaly_mask, z_scores)
        """
        if len(values) < 3:
            return np.zeros(len(values), dtype=bool), np.zeros(len(values))
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return np.zeros(len(values), dtype=bool), np.zeros(len(values))
        
        z_scores = np.abs((values - mean) / std)
        anomaly_mask = z_scores > self.zscore_threshold
        
        return anomaly_mask, z_scores
    
    def detect_iqr(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Детекция аномалий методом IQR (межквартильный размах)
        
        Args:
            values: Временной ряд
            
        Returns:
            (anomaly_mask, scores)
        """
        if len(values) < 4:
            return np.zeros(len(values), dtype=bool), np.zeros(len(values))
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        anomaly_mask = (values < lower_bound) | (values > upper_bound)
        
        # Рассчёт "счёта" аномалии на основе расстояния до границ
        scores = np.zeros(len(values))
        for i, v in enumerate(values):
            if v < lower_bound:
                scores[i] = (lower_bound - v) / max(iqr, 1e-6)
            elif v > upper_bound:
                scores[i] = (v - upper_bound) / max(iqr, 1e-6)
        
        return anomaly_mask, scores
    
    def _simple_stl_decompose(self, values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Упрощённая STL-подобная декомпозиция
        
        Returns:
            (trend, seasonal, residual)
        """
        n = len(values)
        
        # Тренд (скользящее среднее)
        trend = np.zeros(n)
        half_window = window // 2
        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            trend[i] = np.mean(values[start:end])
        
        # Детрендированный ряд
        detrended = values - trend
        
        # Сезонность (простая, если есть периодичность)
        seasonal = np.zeros(n)
        if n > window:
            # Берём среднее по периодам
            for i in range(window):
                indices = np.arange(i, n, window)
                if len(indices) > 0:
                    seasonal[indices] = np.mean(detrended[indices])
        
        # Остатки
        residual = detrended - seasonal
        
        return trend, seasonal, residual
    
    def detect_stl(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Детекция аномалий на основе STL декомпозиции
        
        Args:
            values: Временной ряд
            
        Returns:
            (anomaly_mask, residual_scores)
        """
        if len(values) < self.stl_window + 5:
            # Если данных мало, используем IQR
            return self.detect_iqr(values)
        
        # STL декомпозиция
        _, _, residual = self._simple_stl_decompose(values, self.stl_window)
        
        # Детекция аномалий в остатках
        anomaly_mask, scores = self.detect_zscore(residual)
        
        return anomaly_mask, scores
    
    def detect_combined(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Комбинированная детекция аномалий (голосование методов)
        
        Args:
            values: Временной ряд
            
        Returns:
            (anomaly_mask, confidence_scores)
        """
        # Получение результатов от каждого метода
        z_mask, z_scores = self.detect_zscore(values)
        iqr_mask, iqr_scores = self.detect_iqr(values)
        
        # Голосование (аномалия, если хотя бы 2 метода согласны)
        combined_mask = (z_mask.astype(int) + iqr_mask.astype(int)) >= 2
        
        # Дополнительная STL проверка для подозрительных точек
        if len(values) > self.stl_window:
            stl_mask, _ = self.detect_stl(values)
            combined_mask = combined_mask | stl_mask
        
        # Расчёт комбинированного счёта
        scores = np.zeros(len(values))
        for i in range(len(values)):
            if z_mask[i]:
                scores[i] += z_scores[i] if len(z_scores) > i else 1
            if iqr_mask[i]:
                scores[i] += iqr_scores[i] if len(iqr_scores) > i else 1
        
        scores = scores / 2  # Нормализация
        
        # Учёт минимального расстояния между аномалиями
        if self.min_anomaly_distance > 1:
            final_mask = np.zeros(len(values), dtype=bool)
            last_anomaly = -self.min_anomaly_distance
            for i, is_anomaly in enumerate(combined_mask):
                if is_anomaly and i - last_anomaly >= self.min_anomaly_distance:
                    final_mask[i] = True
                    last_anomaly = i
            combined_mask = final_mask
        
        self._last_anomalies = combined_mask
        self._scores = scores
        
        return combined_mask, scores
    
    def detect(self, values: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Основной метод детекции аномалий
        
        Args:
            values: Временной ряд
            
        Returns:
            Dict с результатами детекции
        """
        # Конвертация в numpy array
        if isinstance(values, pd.Series):
            values = values.values
        elif isinstance(values, list):
            values = np.array(values)
        
        # Выбор метода
        if self.method == 'zscore':
            anomaly_mask, scores = self.detect_zscore(values)
        elif self.method == 'iqr':
            anomaly_mask, scores = self.detect_iqr(values)
        elif self.method == 'stl':
            anomaly_mask, scores = self.detect_stl(values)
        elif self.method == 'combined':
            anomaly_mask, scores = self.detect_combined(values)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")
        
        # Формирование результатов
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        
        return {
            "anomaly_indices": anomaly_indices,
            "anomaly_mask": anomaly_mask.tolist(),
            "anomaly_count": len(anomaly_indices),
            "anomaly_percentage": round(len(anomaly_indices) / len(values) * 100, 2) if len(values) > 0 else 0,
            "scores": scores.tolist() if len(scores) > 0 else [],
            "method": self.method,
            "parameters": {
                "zscore_threshold": self.zscore_threshold,
                "iqr_multiplier": self.iqr_multiplier,
                "stl_window": self.stl_window
            }
        }
    
    def get_anomaly_indices(self) -> Optional[List[int]]:
        """Получение индексов аномалий из последнего вызова detect()"""
        return self._last_anomalies
    
    def get_metadata(self) -> Dict[str, Any]:
        """Получение метаданных"""
        return {
            "detector_type": "AnomalyDetector",
            "method": self.method,
            "parameters": {
                "zscore_threshold": self.zscore_threshold,
                "iqr_multiplier": self.iqr_multiplier,
                "stl_window": self.stl_window,
                "min_anomaly_distance": self.min_anomaly_distance
            }
        }
