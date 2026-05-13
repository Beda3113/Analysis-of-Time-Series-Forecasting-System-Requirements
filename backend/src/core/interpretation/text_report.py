"""
C03-07: TextReportGenerator - Генерация текстового отчёта (тренд, сезонность)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


class TextReportGenerator:
    """
    Генератор текстового отчёта о временном ряде и модели.
    
    Создаёт понятное текстовое описание тренда, сезонности и прогноза.
    """
    
    def __init__(self, language: str = "ru"):
        """
        Инициализация генератора отчётов
        
        Args:
            language: Язык ('ru' или 'en')
        """
        self.language = language
    
    def generate_report(
        self,
        series: pd.Series,
        model_metrics: Optional[Dict[str, float]] = None,
        model_type: Optional[str] = None,
        predictions: Optional[List[float]] = None
    ) -> str:
        """
        Генерация полного текстового отчёта
        
        Args:
            series: Временной ряд
            model_metrics: Метрики модели
            model_type: Тип модели
            predictions: Прогнозные значения
            
        Returns:
            str: Текстовый отчёт в Markdown формате
        """
        if self.language == "ru":
            return self._generate_report_ru(series, model_metrics, model_type, predictions)
        else:
            return self._generate_report_en(series, model_metrics, model_type, predictions)
    
    def _generate_report_ru(
        self,
        series: pd.Series,
        model_metrics: Optional[Dict[str, float]],
        model_type: Optional[str],
        predictions: Optional[List[float]]
    ) -> str:
        """Генерация отчёта на русском языке"""
        
        values = series.values
        n_points = len(values)
        min_val = np.min(values)
        max_val = np.max(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Определение тренда
        if n_points > 5:
            x = np.arange(n_points)
            coeffs = np.polyfit(x, values, 1)
            trend_direction = "восходящий" if coeffs[0] > 0 else "нисходящий"
            trend_strength = abs(coeffs[0])
        else:
            trend_direction = "неопределённый"
            trend_strength = 0
        
        report = f"""# 📊 Анализ временного ряда

## 📈 Общая информация

| Параметр | Значение |
|----------|----------|
| Количество точек | {n_points} |
| Минимальное значение | {min_val:.2f} |
| Максимальное значение | {max_val:.2f} |
| Среднее значение | {mean_val:.2f} |
| Стандартное отклонение | {std_val:.2f} |

## 📉 Тренд и сезонность

- **Тренд**: {trend_direction} (наклон: {trend_strength:.4f} на точку)
"""
        
        # Добавление информации о сезонности
        if n_points > 14:
            # Простая проверка на недельную сезонность
            corr_week = self._check_seasonality(series, 7)
            if corr_week > 0.3:
                report += "- **Сезонность**: обнаружена еженедельная сезонность\n"
            else:
                report += "- **Сезонность**: не обнаружена (возможно, ряд не имеет явной сезонности)\n"
        else:
            report += "- **Сезонность**: недостаточно данных для определения сезонности\n"
        
        # Информация о модели
        if model_metrics or model_type:
            report += f"\n## 🤖 Информация о модели\n\n"
            if model_type:
                report += f"- **Тип модели**: {model_type.upper()}\n"
            
            if model_metrics:
                report += "\n### 📊 Метрики качества\n\n"
                report += "| Метрика | Значение |\n"
                report += "|---------|----------|\n"
                if 'mae' in model_metrics:
                    report += f"| MAE | {model_metrics['mae']:.4f} |\n"
                if 'rmse' in model_metrics:
                    report += f"| RMSE | {model_metrics['rmse']:.4f} |\n"
                if 'mape' in model_metrics:
                    report += f"| MAPE | {model_metrics['mape']:.2f}% |\n"
        
        # Прогноз
        if predictions:
            report += f"\n## 🔮 Прогноз\n\n"
            report += f"Прогноз на {len(predictions)} шагов вперёд:\n\n"
            report += "| Шаг | Значение |\n"
            report += "|-----|----------|\n"
            for i, pred in enumerate(predictions[:10]):
                report += f"| {i+1} | {pred:.2f} |\n"
            if len(predictions) > 10:
                report += f"\n*... и ещё {len(predictions) - 10} значений*\n"
        
        # Рекомендации
        report += f"""
## 💡 Рекомендации

- {'Ряд демонстрирует устойчивый тренд, рекомендуется использовать модели, учитывающие тренд (XGBoost, Prophet, LSTM)' if trend_strength > 0.5 else 'Тренд слабо выражен, можно рассмотреть простые модели (SARIMA)'}
- {'Добавьте сезонные признаки (лаги 7, 14, 30) для улучшения точности' if 'сезонность' in report else 'Рекомендуется проверить наличие сезонности с помощью автокорреляционного анализа'}
- Регулярно переобучайте модель для учёта новых данных

---
*Отчёт сгенерирован автоматически {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def _generate_report_en(
        self,
        series: pd.Series,
        model_metrics: Optional[Dict[str, float]],
        model_type: Optional[str],
        predictions: Optional[List[float]]
    ) -> str:
        """Генерация отчёта на английском языке"""
        
        values = series.values
        n_points = len(values)
        min_val = np.min(values)
        max_val = np.max(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if n_points > 5:
            x = np.arange(n_points)
            coeffs = np.polyfit(x, values, 1)
            trend_direction = "upward" if coeffs[0] > 0 else "downward"
        else:
            trend_direction = "undefined"
        
        report = f"""# 📊 Time Series Analysis

## 📈 General Information

| Parameter | Value |
|-----------|-------|
| Number of points | {n_points} |
| Minimum value | {min_val:.2f} |
| Maximum value | {max_val:.2f} |
| Mean value | {mean_val:.2f} |
| Standard deviation | {std_val:.2f} |

## 📉 Trend and Seasonality

- **Trend**: {trend_direction}
"""
        
        if n_points > 14:
            corr_week = self._check_seasonality(series, 7)
            if corr_week > 0.3:
                report += "- **Seasonality**: weekly seasonality detected\n"
            else:
                report += "- **Seasonality**: no clear seasonality detected\n"
        
        if model_metrics or model_type:
            report += f"\n## 🤖 Model Information\n\n"
            if model_type:
                report += f"- **Model type**: {model_type.upper()}\n"
            
            if model_metrics:
                report += "\n### 📊 Quality Metrics\n\n"
                report += "| Metric | Value |\n"
                report += "|--------|-------|\n"
                if 'mae' in model_metrics:
                    report += f"| MAE | {model_metrics['mae']:.4f} |\n"
                if 'rmse' in model_metrics:
                    report += f"| RMSE | {model_metrics['rmse']:.4f} |\n"
                if 'mape' in model_metrics:
                    report += f"| MAPE | {model_metrics['mape']:.2f}% |\n"
        
        report += f"\n*Report generated automatically on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return report
    
    def _check_seasonality(self, series: pd.Series, lag: int) -> float:
        """
        Проверка сезонности с заданным лагом
        
        Args:
            series: Временной ряд
            lag: Лаг для проверки
            
        Returns:
            float: Корреляция между рядом и его сдвигом
        """
        if len(series) <= lag:
            return 0.0
        
        shifted = series.shift(lag)
        valid_idx = ~(shifted.isna() | series.isna())
        
        if valid_idx.sum() < 3:
            return 0.0
        
        correlation = series[valid_idx].corr(shifted[valid_idx])
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def generate_summary(
        self,
        series: pd.Series,
        model_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Генерация краткой сводки
        
        Args:
            series: Временной ряд
            model_metrics: Метрики модели
            
        Returns:
            str: Краткая сводка
        """
        if self.language == "ru":
            summary = f"Ряд содержит {len(series)} точек. "
            summary += f"Значения варьируются от {np.min(series):.2f} до {np.max(series):.2f}. "
            
            if model_metrics and 'mape' in model_metrics:
                summary += f"Точность модели: MAPE = {model_metrics['mape']:.1f}%."
            
            return summary
        else:
            summary = f"Series contains {len(series)} points. "
            summary += f"Values range from {np.min(series):.2f} to {np.max(series):.2f}. "
            
            if model_metrics and 'mape' in model_metrics:
                summary += f"Model accuracy: MAPE = {model_metrics['mape']:.1f}%."
            
            return summary
    
    def get_metadata(self) -> Dict[str, Any]:
        """Получение метаданных"""
        return {
            "generator_type": "TextReportGenerator",
            "language": self.language
        }
