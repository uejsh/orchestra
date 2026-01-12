# base.py - Base adapter interface
# ============================================================================
# FILE: orchestra/adapters/base.py
# Abstract base class for framework adapters
# ============================================================================

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

from ..core.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class BaseAdapter(ABC):
    """
    Abstract base class for Orchestra adapters.
    Enforces a common interface for caching and metrics.
    """
    
    def __init__(self, config=None):
        self.config = config
        self.cache_manager = None # Should be initialized by subclasses
        self.metrics = None # Should be initialized by subclasses

    @abstractmethod
    def invoke(self, input: Any, **kwargs) -> Any:
        """Synchronous invocation with caching"""
        pass

    @abstractmethod
    async def ainvoke(self, input: Any, **kwargs) -> Any:
        """Asynchronous invocation with caching"""
        pass
    
    def invalidate_cache(self, query: Optional[str] = None):
        """Invalidate specific query or entire cache"""
        if not self.cache_manager:
            return
            
        if query is None:
            self.cache_manager.store.invalidate()
        else:
            # Note: This is an approximation since we hash queries
            # Ideally CacheManager would handle this logic
            import hashlib
            key = hashlib.md5(query.encode()).hexdigest()
            self.cache_manager.store.invalidate(key)

    def warm_cache(self, queries: List[str], simulator_fn):
        """
        Pre-populate cache by running queries.
        
        Args:
            queries: List of query strings
            simulator_fn: Function to generating the result (e.g. actual graph.invoke)
        """
        logger.info(f"Warming cache with {len(queries)} queries...")
        for q in queries:
            try:
                # We assume the result isn't cached yet, so we get it
                # and then put it. In a real scenario, we might just call invoke()
                result = simulator_fn(q)
                self.cache_manager.put(q, result)
            except Exception as e:
                logger.error(f"Failed to warm cache for usage '{q}': {e}")
                
    def get_metrics(self) -> Dict[str, Any]:
        """Return metrics if available"""
        if self.metrics:
            return self.metrics.get_stats()
        return {}
