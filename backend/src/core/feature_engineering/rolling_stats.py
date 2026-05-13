"""
C02-02: RollingStatsCreator - Скользящие средние и стандартные отклонения
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any


class RollingStatsCreator:
    """
    Создание скользящих статистик для временных рядов.
    
    Поддерживаемые статистики:
    - mean: скользящее среднее
    - std: скользящее стандартное отклонение
    - min: скользящий минимум
    - max: скользящий максимум
    - sum: скользящая сумма
    - median: скользящая медиана
    """
    
    def __init__(
        self,
        windows: Optional[List[int]] = None,
        stats: Optional[List[str]] = None,
        target_column: str = 'value',
        prefix: str = 'rolling'
    ):
        """
        Инициализация RollingStatsCreator
        
        Args:
            windows: Список размеров окон (например [7, 14, 30])
            stats: Список статистик (mean, std, min, max, sum, median)
            target_column: Название колонки с целевой переменной
            prefix: Префикс для названий признаков
        """
        self.windows = windows or [7, 14, 30]
        self.stats = stats or ['mean', 'std']
        self.target_column = target_column
        self.prefix = prefix
        self._feature_names: List[str] = []
    
    def create_features(
        self, 
        data: Union[pd.Series, pd.DataFrame, List[float]]
    ) -> pd.DataFrame:
        """
        Создание скользящих статистик
        
        Args:
            data: Входные данные
            
        Returns:
            pd.DataFrame: DataFrame с добавленными скользящими статистиками
        """
        # Преобразование входных данных
        if isinstance(data, list):
            df = pd.DataFrame({self.target_column: data})
        elif isinstance(data, pd.Series):
            df = pd.DataFrame({self.target_column: data.values})
        else:
            df = data.copy()
        
        self._feature_names = []
        
        for window in self.windows:
            for stat in self.stats:
                col_name = f"{self.prefix}_{stat}_{window}"
                
                if stat == 'mean':
                    df[col_name] = df[self.target_column].rolling(window).mean()
                elif stat == 'std':
                    df[col_name] = df[self.target_column].rolling(window).std()
                elif stat == 'min':
                    df[col_name] = df[self.target_column].rolling(window).min()
                elif stat == 'max':
                    df[col_name] = df[self.target_column].rolling(window).max()
                elif stat == 'sum':
                    df[col_name] = df[self.target_column].rolling(window).sum()
                elif stat == 'median':
                    df[col_name] = df[self.target_column].rolling(window).median()
                else:
                    raise ValueError(f"Неподдерживаемая статистика: {stat}")
                
                self._feature_names.append(col_name)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Получение списка названий созданных признаков"""
        return self._feature_names
    
    def get_metadata(self) -> Dict[str, Any]:
        """Получение метаданных о признаках"""
        return {
            "creator": "RollingStatsCreator",
            "windows": self.windows,
            "stats": self.stats,
            "prefix": self.prefix,
            "feature_names": self._feature_names,
            "n_features": len(self._feature_names)
        }
