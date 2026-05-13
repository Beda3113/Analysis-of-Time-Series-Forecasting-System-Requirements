"""
W01-03: Signal handlers - SIGSEGV → перезапуск
"""

import os
import signal
import logging
import traceback
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Хранилище для отслеживания состояния воркеров
_worker_health: Dict[str, Dict[str, Any]] = {}


def setup_signal_handlers(worker_name: str = 'default'):
    """
    Настройка обработчиков сигналов для воркера
    
    Args:
        worker_name: Имя воркера
    """
    
    def signal_handler(signum, frame):
        """Общий обработчик сигналов"""
        logger.warning(f"Worker {worker_name} received signal {signum}")
        
        if signum == signal.SIGTERM:
            logger.info(f"Worker {worker_name} is shutting down gracefully")
            # Обновление статуса перед завершением
            _worker_health[worker_name] = {
                'status': 'shutdown',
                'signal': signum,
                'timestamp': None  # будет установлено в datetime
            }
            raise SystemExit(0)
        
        elif signum == signal.SIGINT:
            logger.info(f"Worker {worker_name} interrupted by user")
            raise SystemExit(0)
        
        elif signum in [signal.SIGSEGV, signal.SIGABRT]:
            logger.error(f"Worker {worker_name} crashed with signal {signum}")
            # Обновление статуса
            _worker_health[worker_name] = {
                'status': 'crashed',
                'signal': signum,
                'error': 'Segmentation fault or abort',
                'stack': traceback.format_stack()
            }
            # Перезапуск будет обработан супервизором или systemd
            raise SystemExit(1)
        
        elif signum == signal.SIGUSR1:
            logger.info(f"Worker {worker_name} received USR1 signal (dump stacks)")
            # Дамп стеков для отладки
            for thread_id, stack in sys._current_frames().items():
                logger.debug(f"Thread {thread_id}: {traceback.format_stack(stack)}")
        
        elif signum == signal.SIGUSR2:
            logger.info(f"Worker {worker_name} received USR2 signal (reload configuration)")
            # Перезагрузка конфигурации
            _reload_configuration()
    
    # Регистрация обработчиков
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGSEGV, signal_handler)
    signal.signal(signal.SIGABRT, signal_handler)
    
    # Опциональные сигналы
    try:
        signal.signal(signal.SIGUSR1, signal_handler)
        signal.signal(signal.SIGUSR2, signal_handler)
    except AttributeError:
        pass  # Windows doesn't support SIGUSR1/SIGUSR2
    
    logger.info(f"Signal handlers setup for worker {worker_name}")


def _reload_configuration():
    """Перезагрузка конфигурации воркера"""
    import importlib
    try:
        import src.workers.celery_app
        importlib.reload(src.workers.celery_app)
        logger.info("Configuration reloaded successfully")
    except Exception as e:
        logger.error(f"Failed to reload configuration: {str(e)}")


def get_worker_health(worker_name: str = None) -> Dict[str, Any]:
    """
    Получение информации о состоянии воркера
    
    Args:
        worker_name: Имя воркера (если None, возвращает все)
    
    Returns:
        Dict с информацией о состоянии
    """
    if worker_name:
        return _worker_health.get(worker_name, {'status': 'unknown'})
    return _worker_health


def update_worker_health(worker_name: str, status: str, error: str = None):
    """
    Обновление состояния воркера
    
    Args:
        worker_name: Имя воркера
        status: Статус ('running', 'crashed', 'shutdown', 'error')
        error: Сообщение об ошибке (опционально)
    """
    _worker_health[worker_name] = {
        'status': status,
        'timestamp': None,  # будет установлено в datetime
        'error': error
    }
    logger.info(f"Worker {worker_name} health updated: {status}")


# Импорт для использования в обработчиках
import sys
import datetime
# Добавляем timestamp
def _add_timestamp():
    from datetime import datetime
    for worker in _worker_health.values():
        if worker.get('timestamp') is None:
            worker['timestamp'] = datetime.utcnow()
