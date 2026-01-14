# langchain.py - LangChain enhancement
# ============================================================================
# FILE: orchestra/adapters/langchain.py
# LangChain adapter supporting LLMChain and LCEL Runnables
# ============================================================================

import time
import logging
import json
from typing import Any, Dict, Optional, Union, List

from .base import BaseAdapter
from ..core.metrics import MetricsTracker

logger = logging.getLogger(__name__)


class OrchestraLangChainConfig:
    """Configuration for LangChain Orchestra enhancement"""
    
    def __init__(
        self,
        # Semantic matching
        similarity_threshold: float = 0.92,
        embedding_model: str = "all-MiniLM-L6-v2",
        
        # Hierarchical embeddings (multi-level matching)
        enable_hierarchical: bool = False,
        hierarchical_weight_l1: float = 0.6,
        hierarchical_weight_l2: float = 0.4,
        
        # Caching
        cache_ttl: int = 3600,
        max_cache_size: int = 10000,
        
        # Compression
        enable_compression: bool = False,
        
        # Cost tracking
        llm_cost_per_1k_tokens: float = 0.03,
        
        # Backend
        redis_url: Optional[str] = None
    ):
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.enable_hierarchical = enable_hierarchical
        self.hierarchical_weight_l1 = hierarchical_weight_l1
        self.hierarchical_weight_l2 = hierarchical_weight_l2
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self.enable_compression = enable_compression
        self.enable_compression = enable_compression
        self.llm_cost_per_1k_tokens = llm_cost_per_1k_tokens
        self.redis_url = redis_url
        
        # Observability
        self.enable_recorder = True



class EnhancedLangChain(BaseAdapter):
    """
    Adapter for LangChain (both Chains and Runnables).
    
    Features (configurable via OrchestraLangChainConfig):
    - Semantic caching with FAISS/NumPy
    - Hierarchical embeddings for better matching (optional)
    - Compression for smaller cache footprint (optional)
    """
    
    def __init__(
        self, 
        chain_or_runnable: Any, 
        config: Optional[OrchestraLangChainConfig] = None
    ):
        super().__init__(config)
        self.chain = chain_or_runnable
        self.config = config or OrchestraLangChainConfig()
        
        # ====================================================================
        # UNIFIED CACHE MANAGER (Replaces duplicate component initialization)
        # ====================================================================
        from ..core.cache_manager import CacheManager
        
        self.cache_manager = CacheManager(
            embedding_model_name=self.config.embedding_model,
            similarity_threshold=self.config.similarity_threshold,
            max_cache_size=self.config.max_cache_size,
            cache_ttl=self.config.cache_ttl,
            enable_compression=self.config.enable_compression,
            enable_hierarchical=self.config.enable_hierarchical,
            hierarchical_weight_l1=self.config.hierarchical_weight_l1,
            hierarchical_weight_l2=self.config.hierarchical_weight_l2,
            redis_url=self.config.redis_url
        )
        
        # Metrics
        self.metrics = MetricsTracker(
            llm_cost_per_1k_tokens=self.config.llm_cost_per_1k_tokens
        )
        
        # Recorder
        if getattr(self.config, "enable_recorder", True):
            from ..recorder.logger import OrchestraRecorder
            self.recorder = OrchestraRecorder.get_instance()
            logger.info("🎥 Orchestra Recorder ENABLED (LangChain)")
        else:
            self.recorder = None
        
        logger.info("✨ Orchestra enhancement enabled for LangChain")

    def invoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> Any:
        """
        LCEL-compatible invoke.
        Also supports legacy .run() via __getattr__ delegation if needed,
        but typically users wrap usage with this invoke.
        """
        return self._execute(input, config, is_async=False, **kwargs)

    async def ainvoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> Any:
        """LCEL-compatible async invoke."""
        return await self._execute_async(input, config, **kwargs)

    def _execute(self, input: Any, config: Any, is_async: bool, **kwargs) -> Any:
        start_time = time.time()
        
        # 1. Serialize input to string for embedding
        input_str = self._to_string(input)
        cache_key = self._generate_cache_key(input_str)
        
        # 2. Check Cache
        cached_result = self._check_cache(input_str, cache_key)
        if cached_result is not None:
            latency = time.time() - start_time
            self.metrics.record_cache_hit(latency)
            
            if self.recorder:
                with self.recorder.trace(input_params=input, metadata={"cached": True, "adapter": "langchain"}) as trace_id:
                     self.recorder.finish_trace(trace_id, cached_result, total_cost=0.0)
                     
            logger.info(f"💰 Cache HIT (LangChain) - Latency: {latency:.3f}s")
            return cached_result
        
        # 3. Execute
        self.metrics.start_execution()
        
        if self.recorder:
            trace_ctx = self.recorder.trace(input_params=input, metadata={"cached": False, "adapter": "langchain"})
            trace_id = trace_ctx.__enter__()
        else:
            trace_ctx = None
            trace_id = None

        try:
            # Support both Runnable .invoke() and Chain .run() / .__call__
            if hasattr(self.chain, "invoke"):
                result = self.chain.invoke(input, config, **kwargs)
            elif hasattr(self.chain, "run") and isinstance(input, str):
                result = self.chain.run(input, **kwargs)
            else:
                result = self.chain(input, **kwargs)
                if isinstance(result, dict) and len(result) == 1:
                    # Unpack single dict result often returned by chains
                    result = list(result.values())[0]
            
            if trace_ctx:
                self.recorder.finish_trace(trace_id, result, total_cost=0.0)
                trace_ctx.__exit__(None, None, None)
                
        except Exception as e:
            if trace_ctx:
                trace_ctx.__exit__(type(e), e, e.__traceback__)
            raise e

        latency = time.time() - start_time
        self.metrics.end_execution(latency)
        
        # 4. Store
        self._store_result(input_str, result, cache_key)
        
        logger.info(f"⚡ Execution complete (LangChain) - Latency: {latency:.3f}s")
        
        return result

    async def _execute_async(self, input: Any, config: Any, **kwargs):
        start_time = time.time()
        
        # 1. Serialize
        input_str = self._to_string(input)
        cache_key = self._generate_cache_key(input_str)
        
        # 2. Check Cache
        cached_result = self._check_cache(input_str, cache_key)
        if cached_result is not None:
            latency = time.time() - start_time
            self.metrics.record_cache_hit(latency)
            
            if self.recorder:
                with self.recorder.trace(input_params=input, metadata={"cached": True, "adapter": "langchain_async"}) as trace_id:
                     self.recorder.finish_trace(trace_id, cached_result, total_cost=0.0)
                     
            logger.info(f"💰 Cache HIT (LangChain Async) - Latency: {latency:.3f}s")
            return cached_result
        
        # 3. Execute
        self.metrics.start_execution()
        
        if self.recorder:
            trace_ctx = self.recorder.trace(input_params=input, metadata={"cached": False, "adapter": "langchain_async"})
            trace_id = trace_ctx.__enter__()
        else:
            trace_ctx = None
            trace_id = None

        try:
            if hasattr(self.chain, "ainvoke"):
                result = await self.chain.ainvoke(input, config, **kwargs)
            elif hasattr(self.chain, "arun") and isinstance(input, str):
                result = await self.chain.arun(input, **kwargs)
            elif hasattr(self.chain, "acall"):
                result = await self.chain.acall(input, **kwargs)
                if isinstance(result, dict) and len(result) == 1:
                    result = list(result.values())[0]
            else:
                # Fallback to sync invoke if async not supported
                logger.warning("Chain does not support async invoke/run, falling back to sync.")
                result = self.chain.invoke(input, config, **kwargs)
                
            if trace_ctx:
                self.recorder.finish_trace(trace_id, result, total_cost=0.0)
                trace_ctx.__exit__(None, None, None)
                
        except Exception as e:
            if trace_ctx:
                trace_ctx.__exit__(type(e), e, e.__traceback__)
            raise e

        latency = time.time() - start_time
        self.metrics.end_execution(latency)
        
        # 4. Store
        self._store_result(input_str, result, cache_key)
        
        return result

    # ========================================================================
    # CACHE OPERATIONS (Delegated to CacheManager)
    # ========================================================================
    
    def _check_cache(self, input_str: str, cache_key: str) -> Optional[Any]:
        """Check semantic cache for similar inputs - delegates to CacheManager"""
        return self.cache_manager.get(input_str)
    
    def _store_result(self, input_str: str, result: Any, cache_key: str):
        """Store result in semantic cache - delegates to CacheManager"""
        self.cache_manager.put(input_str, result, ttl=self.config.cache_ttl)
        logger.debug(f"Stored result in cache: {cache_key[:16]}...")

    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def _to_string(self, input: Any) -> str:
        if isinstance(input, str):
            return input
        if isinstance(input, dict):
            return json.dumps(input, sort_keys=True, default=str)
        return str(input)
    
    def _generate_cache_key(self, input_str: str) -> str:
        import hashlib
        return hashlib.sha256(input_str.encode()).hexdigest()
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        cache_stats = self.cache_manager.get_stats()
        exec_stats = self.metrics.get_stats()
        
        return {
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_hits": cache_stats["cache_hits"],
            "cache_misses": cache_stats["cache_misses"],
            "cache_size": cache_stats["current_size"],
            "total_executions": exec_stats["total_executions"],
            "hierarchical_enabled": self.config.enable_hierarchical,
            "compression_enabled": self.config.enable_compression,
        }
    
    def invalidate_cache(self, query: Optional[str] = None):
        """Invalidate cache entries"""
        if query is None:
            count = self.cache_manager.invalidate()
            logger.info(f"Invalidated all {count} cache entries")
        else:
            cache_key = self._generate_cache_key(self._to_string(query))
            count = self.cache_manager.invalidate(cache_key)
            logger.info(f"Invalidated {count} entries for query")

    # Proxy other methods to underlying chain
    def __getattr__(self, name):
        return getattr(self.chain, name)


def enhance(chain_or_runnable, config=None):
    """Enhance a LangChain Chain or Runnable with semantic caching"""
    return EnhancedLangChain(chain_or_runnable, config)


__all__ = [
    "enhance",
    "EnhancedLangChain",
    "OrchestraLangChainConfig",
]
