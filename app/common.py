import os

REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379/0')
STREAM = 'images'
GROUP = 'gpu'
TTL = int(os.getenv('RESULT_TTL', '86400'))

def setting(name, default):
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f'{name} must be positive')
    return value
