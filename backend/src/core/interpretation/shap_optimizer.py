"""
C03-02: SHAPOptimizer - Подвыборка для больших данных и оптимизация вычислений
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union, List, Dict, Any
from sklearn.cluster import KMeans
from sklearn.utils import resample


class SHAPOptimizer:
    """
    Оптимизатор SHAP для работы с большими данными.
    
    Использует:
    - Подвыборку данных
    - Кластеризацию для репрезентативной выборки
    - K-mean summarization
    """
    
    def __init__(
        self,
        max_samples: int = 1000,
        sampling_method: str = 'random',
        random_state: int = 42
    ):
        """
        Инициализация SHAPOptimizer
        
        Args:
            max_samples: Максимальное количество образцов для расчёта
            sampling_method: Метод выборки ('random', 'kmeans', 'stratified')
            random_state: Случайное зерно
        """
        self.max_samples = max_samples
        self.sampling_method = sampling_method
        self.random_state = random_state
        self._selected_indices = None
    
    def optimize_background(
        self, 
        X: Union[np.ndarray, pd.DataFrame],
        method: Optional[str] = None
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Оптимизация фоновой выборки для SHAP
        
        Args:
            X: Входные данные
            method: Метод выборки (если None, используется self.sampling_method)
            
        Returns:
            Оптимизированная выборка
        """
        method = method or self.sampling_method
        
        n_samples = len(X)
        if n_samples <= self.max_samples:
            return X
        
        if method == 'random':
            # Случайная подвыборка
            indices = np.random.choice(n_samples, self.max_samples, replace=False)
            self._selected_indices = indices
            return X[indices] if isinstance(X, np.ndarray) else X.iloc[indices]
        
        elif method == 'kmeans':
            # K-mean кластеризация для репрезентативной выборки
            from sklearn.cluster import KMeans
            
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
            
            kmeans = KMeans(n_clusters=min(self.max_samples, n_samples), random_state=self.random_state, n_init=10)
            kmeans.fit(X_array)
            
            # Выбираем ближайшие точки к центроидам
            indices = []
            for center in kmeans.cluster_centers_:
                distances = np.linalg.norm(X_array - center, axis=1)
                closest_idx = np.argmin(distances)
                indices.append(closest_idx)
            
            indices = list(set(indices))[:self.max_samples]
            self._selected_indices = indices
            
            return X[indices] if isinstance(X, np.ndarray) else X.iloc[indices]
        
        elif method == 'stratified':
            # Стратифицированная выборка по квантилям целевой переменной
            # (требуется y)
            warnings.warn("Стратифицированная выборка требует целевую переменную y")
            indices = np.random.choice(n_samples, self.max_samples, replace=False)
            self._selected_indices = indices
            return X[indices] if isinstance(X, np.ndarray) else X.iloc[indices]
        
        else:
            raise ValueError(f"Неизвестный метод выборки: {method}")
    
    def optimize_features(
        self, 
        X: Union[np.ndarray, pd.DataFrame],
        feature_importance: np.ndarray,
        top_k: int = 20
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Оптимизация признаков (оставляем только топ-K важных)
        
        Args:
            X: Входные данные
            feature_importance: Важность признаков
            top_k: Количество признаков для сохранения
            
        Returns:
            Данные с уменьшенным количеством признаков
        """
        if len(feature_importance) <= top_k:
            return X
        
        # Получение индексов топ-K признаков
        top_indices = np.argsort(feature_importance)[-top_k:]
        
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            selected_features = [feature_names[i] for i in top_indices]
            return X[selected_features]
        else:
            return X[:, top_indices]
    
    def get_selected_indices(self) -> Optional[List[int]]:
        """Получение индексов выбранных образцов"""
        return self._selected_indices
    
    def get_metadata(self) -> Dict[str, Any]:
        """Получение метаданных оптимизатора"""
        return {
            "max_samples": self.max_samples,
            "sampling_method": self.sampling_method,
            "random_state": self.random_state,
            "selected_samples": len(self._selected_indices) if self._selected_indices else 0
        }


import warnings
