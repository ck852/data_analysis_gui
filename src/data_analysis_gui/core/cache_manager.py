"""
Generic cache management with content-based keys.
PHASE 5: Added cache operation logging for observability.
"""

from typing import Dict, Any, Optional, Callable
import hashlib

from data_analysis_gui.config.logging import get_logger, log_cache_operation

logger = get_logger(__name__)


class CacheManager:
    """
    Generic cache with size management and logging.
    Thread-safe if used with proper locking in multi-threaded context.
    """
    
    def __init__(self, max_size: int = 100, name: str = "cache"):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of entries
            name: Cache name for logging
        """
        self._cache: Dict[str, Any] = {}
        self._max_size = max_size
        self._name = name
        logger.debug(f"Initialized {name} with max_size={max_size}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if key in self._cache:
            log_cache_operation(logger, "get", f"{self._name}:{key}", hit=True)
            return self._cache[key]
        
        log_cache_operation(logger, "get", f"{self._name}:{key}", hit=False)
        return None
    
    def set(self, key: str, value: Any, size_bytes: Optional[int] = None) -> None:
        """
        Store value in cache with automatic size management.
        
        Args:
            key: Cache key
            value: Value to cache
            size_bytes: Optional size estimate for logging
        """
        # Manage size before adding
        if len(self._cache) >= self._max_size:
            self._evict_oldest()
        
        self._cache[key] = value
        log_cache_operation(logger, "set", f"{self._name}:{key}", size=size_bytes)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {self._name}: removed {count} entries")
    
    def _evict_oldest(self) -> None:
        """Evict oldest half of entries."""
        evict_count = self._max_size // 2
        keys_to_remove = list(self._cache.keys())[:evict_count]
        
        logger.debug(f"{self._name} full, evicting {evict_count} entries")
        
        for key in keys_to_remove:
            del self._cache[key]
    
    @staticmethod
    def create_content_key(content: str) -> str:
        """
        Create content-based cache key.
        
        Args:
            content: String content to hash
            
        Returns:
            MD5 hash of content
        """
        return hashlib.md5(content.encode()).hexdigest()