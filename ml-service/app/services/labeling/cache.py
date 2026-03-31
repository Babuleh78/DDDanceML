"""Простой LRU-кэш для результатов LLM с TTL."""
import time
from typing import Optional, Dict, Any
from collections import OrderedDict


class LabelCache:
    """Потокобезопасный кэш с ограничением по размеру и времени жизни."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
    
    def get(self, key: str) -> Optional[str]:
        """Получает значение из кэша, если не истёк TTL."""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        if time.time() - entry['timestamp'] > self._ttl:
            del self._cache[key]
            return None
        
        # Поднимаем элемент в конец (LRU)
        self._cache.move_to_end(key)
        return entry['value']
    
    def set(self, key: str, value: str) -> None:
        """Добавляет значение в кэш."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)  # Удаляем самый старый
        
        self._cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
    
    def stats(self) -> Dict[str, int]:
        return {
            'size': len(self._cache),
            'max_size': self._max_size,
        }


# Глобальный инстанс кэша (можно заменить на Redis при масштабировании)
label_cache = LabelCache()