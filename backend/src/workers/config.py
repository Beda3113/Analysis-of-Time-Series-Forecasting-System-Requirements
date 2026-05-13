"""
W01-02: Worker pool setup - Concurrency, max tasks per child
"""

import os
from typing import Dict, Any

# Настройки воркеров из переменных окружения
WORKER_CONFIG: Dict[str, Any] = {
    # Базовые настройки
    'concurrency': int(os.environ.get('CELERY_WORKER_CONCURRENCY', 4)),
    'max_tasks_per_child': int(os.environ.get('CELERY_MAX_TASKS_PER_CHILD', 100)),
    'prefetch_multiplier': int(os.environ.get('CELERY_PREFETCH_MULTIPLIER', 1)),
    
    # Пул воркеров
    'pool': os.environ.get('CELERY_WORKER_POOL', 'prefork'),  # prefork, gevent, solo
    'pool_restarts': os.environ.get('CELERY_POOL_RESTARTS', False),
    
    # Логирование
    'loglevel': os.environ.get('CELERY_LOG_LEVEL', 'INFO'),
    'logfile': os.environ.get('CELERY_LOG_FILE', None),
    
    # Оптимизация памяти
    'max_memory_per_child': int(os.environ.get('CELERY_MAX_MEMORY_PER_CHILD', 500000)),  # 500MB
    
    # Авто-масштабирование
    'autoscale': os.environ.get('CELERY_AUTOSCALE', None),  # "10,3" - max,min
}


def get_worker_command(worker_type: str = 'default') -> list:
    """
    Формирование команды запуска воркера
    
    Args:
        worker_type: Тип воркера ('default', 'training', 'interpretation', 'maintenance')
    
    Returns:
        list: Команда для запуска
    """
    settings = WORKER_CONFIG.copy()
    
    # Переопределение конфигурации для специфических типов
    if worker_type == 'training':
        settings['concurrency'] = int(os.environ.get('CELERY_TRAINING_CONCURRENCY', 2))
        settings['max_tasks_per_child'] = int(os.environ.get('CELERY_TRAINING_MAX_TASKS', 50))
    elif worker_type == 'interpretation':
        settings['concurrency'] = int(os.environ.get('CELERY_INTERPRETATION_CONCURRENCY', 2))
        settings['max_tasks_per_child'] = int(os.environ.get('CELERY_INTERPRETATION_MAX_TASKS', 100))
    elif worker_type == 'maintenance':
        settings['concurrency'] = int(os.environ.get('CELERY_MAINTENANCE_CONCURRENCY', 1))
        settings['max_tasks_per_child'] = int(os.environ.get('CELERY_MAINTENANCE_MAX_TASKS', 10))
    
    # Формирование команды
    cmd = [
        'celery', '-A', 'src.workers.celery_app', 'worker',
        '--loglevel', settings['loglevel'],
        '--concurrency', str(settings['concurrency']),
        '--max-tasks-per-child', str(settings['max_tasks_per_child']),
        '--prefetch-multiplier', str(settings['prefetch_multiplier']),
        '--pool', settings['pool'],
    ]
    
    # Добавление очереди
    if worker_type != 'default':
        cmd.extend(['-Q', worker_type])
    
    # Добавление имени хоста
    hostname = os.environ.get('HOSTNAME', 'localhost')
    cmd.extend(['--hostname', f'{worker_type}@%h'])
    
    return cmd


def get_beat_command() -> list:
    """Формирование команды запуска Celery Beat"""
    return [
        'celery', '-A', 'src.workers.celery_app', 'beat',
        '--loglevel', WORKER_CONFIG['loglevel'],
        '--schedule', 'celerybeat-schedule'
    ]


def get_flower_command() -> list:
    """Формирование команды запуска Flower (мониторинг)"""
    port = os.environ.get('FLOWER_PORT', '5555')
    return [
        'celery', '-A', 'src.workers.celery_app', 'flower',
        f'--port={port}',
        '--loglevel=info'
    ]
