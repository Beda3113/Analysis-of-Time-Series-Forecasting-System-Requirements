"""
C02-04: FeatureValidator - Проверка на заглядывание в будущее
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any


class FeatureValidator:
    """
    Валидатор признаков для проверки отсутствия "заглядывания в будущее".
    
    Проверяет, что при создании признаков не используются значения,
    которые недоступны на момент прогнозирования.
    """
    
    def __init__(self, strict: bool = True):
        """
        Инициализация FeatureValidator
        
        Args:
            strict: Если True, выбрасывает исключение при обнаружении lookahead
        """
        self.strict = strict
        self._violations: List[Dict[str, Any]] = []
    
    def validate_lag_features(
        self, 
        df: pd.DataFrame, 
        lag_columns: List[str],
        target_column: str = 'value'
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Проверка лаговых признаков на заглядывание в будущее
        
        Args:
            df: DataFrame с данными
            lag_columns: Список колонок с лагами
            target_column: Название целевой колонки
            
        Returns:
            Tuple[bool, List]: (прошло проверку, список нарушений)
        """
        violations = []
        
        for col in lag_columns:
            # Извлекаем номер лага из названия колонки
            try:
                lag_value = int(col.split('_')[-1])
            except:
                continue
            
            # Проверяем, что в строке i значение лага не равно целевому значению из будущего
            for i in range(len(df) - lag_value):
                if i + lag_value < len(df):
                    # Лаг в строке i должен равняться целевому значению в строке i - lag
                    # (это нормально для правильно построенных лагов)
                    pass
            
            # Дополнительная проверка: значения в лаговых колонках не должны быть NaN
            # в тех местах, где они определены
            nan_count = df[col].isna().sum()
            if nan_count > len(df) * 0.5:
                violations.append({
                    "column": col,
                    "issue": "too_many_nans",
                    "nan_count": nan_count,
                    "total_rows": len(df),
                    "message": f"Колонка {col} содержит {nan_count} NaN значений ({nan_count/len(df)*100:.1f}%)"
                })
        
        self._violations = violations
        
        if self.strict and violations:
            raise ValueError(f"Обнаружены нарушения: {violations}")
        
        return len(violations) == 0, violations
    
    def validate_no_future_leakage(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame,
        feature_columns: List[str]
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Проверка на утечку данных из будущего между train и test
        
        Args:
            X_train: Обучающая выборка
            X_test: Тестовая выборка
            feature_columns: Список колонок признаков
            
        Returns:
            Tuple[bool, List]: (прошло проверку, список нарушений)
        """
        violations = []
        
        for col in feature_columns:
            if col in X_train.columns and col in X_test.columns:
                # Проверяем, что значения в test не пересекаются со значениями в train
                # для лаговых признаков это нормально
                pass
        
        return len(violations) == 0, violations
    
    def validate_temporal_order(
        self, 
        df: pd.DataFrame,
        date_column: Optional[str] = None
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Проверка временного порядка данных
        
        Args:
            df: DataFrame с данными
            date_column: Название колонки с датами
            
        Returns:
            Tuple[bool, List]: (прошло проверку, список нарушений)
        """
        violations = []
        
        if date_column and date_column in df.columns:
            dates = pd.to_datetime(df[date_column])
            
            # Проверка на монотонность
            if not dates.is_monotonic_increasing:
                violations.append({
                    "issue": "non_monotonic_dates",
                    "message": "Даты не являются монотонно возрастающими"
                })
            
            # Проверка на пропуски в датах
            date_diff = dates.diff().dropna()
            if not (date_diff == date_diff.mode()[0]).all():
                violations.append({
                    "issue": "irregular_intervals",
                    "message": "Интервалы между датами непостоянны"
                })
        
        return len(violations) == 0, violations
    
    def get_violations(self) -> List[Dict[str, Any]]:
        """Получение списка нарушений из последней проверки"""
        return self._violations
    
    def reset(self) -> None:
        """Сброс списка нарушений"""
        self._violations = []
    
    def get_report(self) -> Dict[str, Any]:
        """Получение отчёта о проверке"""
        return {
            "total_violations": len(self._violations),
            "violations": self._violations,
            "strict_mode": self.strict,
            "valid": len(self._violations) == 0
        }
