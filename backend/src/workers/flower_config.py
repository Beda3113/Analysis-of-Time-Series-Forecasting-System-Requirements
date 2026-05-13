"""
W01-04: Flower monitoring - Дашборд для очередей
"""

import os

# Flower конфигурация
FLOWER_CONFIG = {
    'port': int(os.environ.get('FLOWER_PORT', 5555)),
    'broker_api': os.environ.get('FLOWER_BROKER_API', 'http://localhost:15672/api/'),
    'address': os.environ.get('FLOWER_ADDRESS', '0.0.0.0'),
    'auth': os.environ.get('FLOWER_AUTH', None),  # 'user:pass'
    'basic_auth': os.environ.get('FLOWER_BASIC_AUTH', None),
    'enable_events': True,
    'persistent': True,
    'db': os.environ.get('FLOWER_DB', 'flower.db'),
    'inspect_timeout': 10000,
    'purge_interval': 3600,  # 1 час
    'max_tasks': 10000,
    'tasks_columns': 'name,uuid,state,received,started,runtime,worker,args,kwargs,result',
}

# Настройки для разных брокеров
BROKER_CONFIG = {
    'redis': {
        'broker': 'redis://localhost:6379/0',
        'broker_api': None
    },
    'rabbitmq': {
        'broker': 'amqp://guest:guest@localhost:5672//',
        'broker_api': 'http://localhost:15672/api/'
    }
}


def get_flower_command() -> list:
    """
    Формирование команды запуска Flower
    
    Returns:
        list: Команда для запуска
    """
    cmd = [
        'celery', '-A', 'src.workers.celery_app', 'flower',
        f'--port={FLOWER_CONFIG["port"]}',
        f'--address={FLOWER_CONFIG["address"]}',
        '--persistent',
        '--db=flower.db',
    ]
    
    if FLOWER_CONFIG['basic_auth']:
        cmd.append(f'--basic-auth={FLOWER_CONFIG["basic_auth"]}')
    
    return cmd
