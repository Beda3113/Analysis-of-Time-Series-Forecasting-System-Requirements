"""
W01-05: Task retry policy - Exponential backoff
"""

import time
import logging
from functools import wraps
from typing import Callable, Any, Optional
from celery import Task, current_task
from celery.exceptions import Retry

logger = logging.getLogger(__name__)


class BaseTask(Task):
    """
    Базовый класс для всех задач с политикой повторных попыток
    Использует exponential backoff для повторных попыток
    """
    
    # Настройки повторных попыток
    max_retries = 3
    default_retry_delay = 60  # 60 секунд
    retry_backoff = True
    retry_backoff_max = 600  # максимум 10 минут
    retry_jitter = True
    
    # Таймауты
    time_limit = 3600  # 1 час
    soft_time_limit = 3000  # 50 минут
    
    # Автоматическое подтверждение после выполнения
    acks_late = True
    reject_on_worker_lost = True
    
    def __call__(self, *args, **kwargs):
        """Вызов задачи с обработкой ошибок"""
        try:
            return super().__call__(*args, **kwargs)
        except Exception as exc:
            # Логирование ошибки
            logger.error(f"Task {self.name} failed: {exc}")
            raise
    
    def retry_task(self, exc: Exception, **kwargs):
        """
        Повторный запуск задачи с exponential backoff
        
        Args:
            exc: Исключение, вызвавшее повторную попытку
            **kwargs: Дополнительные параметры для retry
        """
        retry_kwargs = {
            'exc': exc,
            'countdown': kwargs.get('countdown', self.default_retry_delay),
            'max_retries': kwargs.get('max_retries', self.max_retries)
        }
        
        # Экспоненциальная задержка
        if self.retry_backoff:
            retries = self.request.retries
            delay = min(
                self.default_retry_delay * (2 ** retries),
                self.retry_backoff_max
            )
            if self.retry_jitter:
                # Добавляем случайный jitter для избежания "thundering herd"
                import random
                delay = delay * (0.8 + 0.4 * random.random())
            retry_kwargs['countdown'] = delay
        
        logger.info(f"Retrying task {self.name} (attempt {self.request.retries + 1})")
        raise self.retry(**retry_kwargs)


def task_with_retry(max_retries: int = 3, backoff: bool = True, backoff_max: int = 600):
    """
    Декоратор для добавления политики повторных попыток
    к обычным задачам
    
    Args:
        max_retries: Максимальное количество попыток
        backoff: Использовать exponential backoff
        backoff_max: Максимальная задержка в секундах
    
    Example:
        @task_with_retry(max_retries=5)
        def my_task():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = 60
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Task failed after {max_retries} retries: {e}")
                        raise
                    
                    if backoff:
                        delay = min(delay * 2, backoff_max)
                        if hasattr(current_task, 'retry_jitter') and current_task.retry_jitter:
                            import random
                            delay = delay * (0.8 + 0.4 * random.random())
                    
                    logger.warning(f"Task {func.__name__} failed (attempt {retries}/{max_retries}), retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
            
            raise RuntimeError("Unreachable")
        
        return wrapper
    return decorator


def get_retry_delay(retry_count: int, base_delay: int = 60, max_delay: int = 600) -> float:
    """
    Расчёт задержки для повторной попытки
    
    Args:
        retry_count: Номер попытки (0 = первая)
        base_delay: Базовая задержка в секундах
        max_delay: Максимальная задержка
    
    Returns:
        float: Задержка в секундах с jitter
    """
    delay = min(base_delay * (2 ** retry_count), max_delay)
    # Добавляем jitter (разброс)
    import random
    delay = delay * (0.8 + 0.4 * random.random())
    return delay
