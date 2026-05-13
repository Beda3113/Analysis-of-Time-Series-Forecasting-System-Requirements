"""
C01-06: ModelRegistry - Фабрика и регистрация моделей
"""

from typing import Dict, Type, Optional, Any
from src.core.forecasting.base import BaseForecaster
from src.core.forecasting.xgboost_forecaster import XGBoostForecaster
from src.core.forecasting.lstm_forecaster import LSTMForecaster
from src.core.forecasting.prophet_forecaster import ProphetForecaster
from src.core.forecasting.sarima_forecaster import SARIMAForecaster


class ModelRegistry:
    """
    Реестр моделей прогнозирования.
    
    Позволяет регистрировать модели, создавать их по имени и получать список доступных моделей.
    """
    
    _models: Dict[str, Type[BaseForecaster]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseForecaster]) -> None:
        """
        Регистрация модели в реестре
        
        Args:
            name: Имя модели (ключ)
            model_class: Класс модели
        """
        cls._models[name] = model_class
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseForecaster]]:
        """
        Получение класса модели по имени
        
        Args:
            name: Имя модели
            
        Returns:
            Класс модели или None, если не найдена
        """
        return cls._models.get(name)
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseForecaster:
        """
        Создание экземпляра модели по имени
        
        Args:
            name: Имя модели
            **kwargs: Параметры для инициализации модели
            
        Returns:
            Экземпляр модели
            
        Raises:
            ValueError: Если модель не зарегистрирована
        """
        model_class = cls.get(name)
        if model_class is None:
            raise ValueError(f"Модель '{name}' не зарегистрирована. Доступные модели: {cls.list_models()}")
        
        return model_class(**kwargs)
    
    @classmethod
    def list_models(cls) -> list:
        """
        Получение списка зарегистрированных моделей
        
        Returns:
            Список имён моделей
        """
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, name: str) -> Dict[str, Any]:
        """
        Получение информации о модели
        
        Args:
            name: Имя модели
            
        Returns:
            Словарь с информацией о модели
        """
        model_class = cls.get(name)
        if model_class is None:
            return {}
        
        return {
            "name": name,
            "class": model_class.__name__,
            "doc": model_class.__doc__,
            "parameters": cls._get_parameters(model_class)
        }
    
    @classmethod
    def _get_parameters(cls, model_class: Type[BaseForecaster]) -> Dict[str, Any]:
        """Извлечение параметров модели из __init__ сигнатуры"""
        import inspect
        sig = inspect.signature(model_class.__init__)
        parameters = {}
        
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            parameters[name] = {
                "default": None if param.default == inspect.Parameter.empty else param.default,
                "required": param.default == inspect.Parameter.empty,
                "annotation": str(param.annotation) if param.annotation != inspect.Parameter.empty else None
            }
        
        return parameters


# Регистрация встроенных моделей
ModelRegistry.register("xgboost", XGBoostForecaster)
ModelRegistry.register("lstm", LSTMForecaster)
ModelRegistry.register("prophet", ProphetForecaster)
ModelRegistry.register("sarima", SARIMAForecaster)


def get_forecaster(model_type: str, **kwargs) -> BaseForecaster:
    """
    Удобная функция для создания модели прогнозирования
    
    Args:
        model_type: Тип модели ('xgboost', 'lstm', 'prophet', 'sarima')
        **kwargs: Параметры для инициализации модели
        
    Returns:
        Экземпляр модели
    """
    return ModelRegistry.create(model_type, **kwargs)
