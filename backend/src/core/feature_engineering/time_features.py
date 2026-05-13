"""
C02-03: TimeFeaturesCreator - Создание временных признаков (день недели, месяц, час)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any
from datetime import datetime


class TimeFeaturesCreator:
    """
    Создание временных признаков из дат временного ряда.
    
    Поддерживаемые признаки:
    - dayofweek: день недели (0-6)
    - month: месяц (1-12)
    - quarter: квартал (1-4)
    - dayofmonth: день месяца (1-31)
    - weekofyear: неделя года (1-52)
    - hour: час (0-23) для часовых данных
    - is_weekend: флаг выходного дня
    - is_month_start: флаг начала месяца
    - is_month_end: флаг конца месяца
    """
    
    def __init__(
        self,
        date_column: str = 'ds',
        features: Optional[List[str]] = None,
        prefix: str = 'time'
    ):
        """
        Инициализация TimeFeaturesCreator
        
        Args:
            date_column: Название колонки с датами
            features: Список временных признаков для создания
            prefix: Префикс для названий признаков
        """
        self.date_column = date_column
        self.features = features or ['dayofweek', 'month', 'quarter', 'is_weekend']
        self.prefix = prefix
        self._feature_names: List[str] = []
    
    def _to_series(self, dates):
        """Преобразование входных данных в pandas Series"""
        if isinstance(dates, pd.DatetimeIndex):
            return pd.Series(dates)
        elif isinstance(dates, pd.Series):
            return dates
        elif isinstance(dates, list):
            return pd.Series(pd.to_datetime(dates))
        else:
            return pd.Series(pd.to_datetime(dates))
    
    def create_features(
        self, 
        data: Union[pd.DataFrame, pd.DatetimeIndex, List[str], List[datetime]]
    ) -> pd.DataFrame:
        """
        Создание временных признаков из дат
        
        Args:
            data: DataFrame с колонкой дат или список дат
            
        Returns:
            pd.DataFrame: DataFrame с временными признаками
        """
        # Извлечение дат
        if isinstance(data, pd.DataFrame):
            if self.date_column not in data.columns:
                raise ValueError(f"Колонка '{self.date_column}' не найдена в данных")
            dates_series = pd.to_datetime(data[self.date_column])
            df = data.copy()
        else:
            dates_series = self._to_series(data)
            df = pd.DataFrame({self.date_column: dates_series})
        
        self._feature_names = []
        
        for feature in self.features:
            col_name = f"{self.prefix}_{feature}"
            
            if feature == 'dayofweek':
                df[col_name] = dates_series.dt.dayofweek
            elif feature == 'month':
                df[col_name] = dates_series.dt.month
            elif feature == 'quarter':
                df[col_name] = dates_series.dt.quarter
            elif feature == 'dayofmonth':
                df[col_name] = dates_series.dt.day
            elif feature == 'weekofyear':
                df[col_name] = dates_series.dt.isocalendar().week
            elif feature == 'hour':
                df[col_name] = dates_series.dt.hour
            elif feature == 'is_weekend':
                df[col_name] = (dates_series.dt.dayofweek >= 5).astype(int)
            elif feature == 'is_month_start':
                df[col_name] = dates_series.dt.is_month_start.astype(int)
            elif feature == 'is_month_end':
                df[col_name] = dates_series.dt.is_month_end.astype(int)
            elif feature == 'sin_dayofweek':
                # Циклическое кодирование дня недели (sin)
                df[col_name] = np.sin(2 * np.pi * dates_series.dt.dayofweek / 7)
                self._feature_names.append(col_name)
                col_name = f"{self.prefix}_cos_dayofweek"
                df[col_name] = np.cos(2 * np.pi * dates_series.dt.dayofweek / 7)
            elif feature == 'sin_month':
                df[col_name] = np.sin(2 * np.pi * dates_series.dt.month / 12)
                self._feature_names.append(col_name)
                col_name = f"{self.prefix}_cos_month"
                df[col_name] = np.cos(2 * np.pi * dates_series.dt.month / 12)
            else:
                raise ValueError(f"Неподдерживаемый признак: {feature}")
            
            self._feature_names.append(col_name)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Получение списка названий созданных признаков"""
        return self._feature_names
    
    def get_metadata(self) -> Dict[str, Any]:
        """Получение метаданных о признаках"""
        return {
            "creator": "TimeFeaturesCreator",
            "date_column": self.date_column,
            "features": self.features,
            "prefix": self.prefix,
            "feature_names": self._feature_names,
            "n_features": len(self._feature_names)
        }
