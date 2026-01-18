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
        redis_url: Optional[str] = None,
        
        # Resilience
        enable_circuit_breaker: bool = False,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
        # Semantic Context Injection (Self-RAG)
        enable_context_injection: bool = False,
        context_injection_top_k: int = 3,
        context_injection_template: str = (
            "--- START OF RELEVANT CONTEXT (PAST RESPONSES) ---\n"
            "{context}\n"
            "--- END OF RELEVANT CONTEXT ---"
        )
    ):
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.enable_hierarchical = enable_hierarchical
        self.hierarchical_weight_l1 = hierarchical_weight_l1
        self.hierarchical_weight_l2 = hierarchical_weight_l2
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self.enable_compression = enable_compression
        self.llm_cost_per_1k_tokens = llm_cost_per_1k_tokens
        self.redis_url = redis_url
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        # Self-RAG settings
        self.enable_context_injection = enable_context_injection
        self.context_injection_top_k = context_injection_top_k
        self.context_injection_template = context_injection_template
        
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
        
        # Resilience
        if getattr(self.config, "enable_circuit_breaker", False):
            from ..resilience.circuit_breaker import CircuitBreaker
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=self.config.circuit_breaker_threshold,
                timeout=self.config.circuit_breaker_timeout
            )
            logger.info("🛡️ Circuit Breaker ENABLED (LangChain)")
        else:
            self.circuit_breaker = None

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
        
        # 3a. Inject Context (Self-RAG)
        exec_input = input
        if getattr(self.config, "enable_context_injection", False):
            exec_input = self._inject_context(input, input_str)

        if self.recorder:
            trace_ctx = self.recorder.trace(input_params=exec_input, metadata={"cached": False, "adapter": "langchain"})
            trace_id = trace_ctx.__enter__()
        else:
            trace_ctx = None
            trace_id = None

        try:
            if self.circuit_breaker:
                result = self.circuit_breaker.call(
                    self._invoke_or_run, exec_input, config, **kwargs
                )
            else:
                result = self._invoke_or_run(exec_input, config, **kwargs)
            
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
        
        # 3a. Inject Context (Self-RAG)
        exec_input = input
        if getattr(self.config, "enable_context_injection", False):
            exec_input = self._inject_context(input, input_str)

        if self.recorder:
            trace_ctx = self.recorder.trace(input_params=exec_input, metadata={"cached": False, "adapter": "langchain_async"})
            trace_id = trace_ctx.__enter__()
        else:
            trace_ctx = None
            trace_id = None

        try:
            if self.circuit_breaker:
                # Async Circuit Breaker call
                result = await self.circuit_breaker.acall(
                    self._ainvoke_or_run, exec_input, config, **kwargs
                )
            else:
                result = await self._ainvoke_or_run(exec_input, config, **kwargs)
                
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

    def _inject_context(self, input_data: Any, input_str: str) -> Any:
        """Inject top N matches from cache as context into the input."""
        matches = self.cache_manager.get_top_matches(
            input_str, 
            top_k=self.config.context_injection_top_k
        )
        
        if not matches:
            return input_data
            
        context_str = "\n".join([
            f"Past Query: {m['query']}\nPast Response: {m['value']}" 
            for m in matches
        ])
        
        injected_context = self.config.context_injection_template.format(context=context_str)
        
        # Strategy A: Plain string input
        if isinstance(input_data, str):
            return injected_context + "\n\n" + input_data
            
        # Strategy B: Dictionary input
        if isinstance(input_data, dict):
            new_input = input_data.copy()
            
            # Look for "messages" (Typical for Chat Models)
            if "messages" in new_input and isinstance(new_input["messages"], list):
                try:
                    from langchain_core.messages import SystemMessage
                    context_msg = SystemMessage(content=injected_context)
                    new_input["messages"] = [context_msg] + new_input["messages"]
                    logger.info("🧠 Orchestra: Injected semantic context into LangChain messages")
                    return new_input
                except ImportError:
                    pass
            
            # Look for common text keys
            for key in ["query", "input", "question", "text"]:
                if key in new_input and isinstance(new_input[key], str):
                    new_input[key] = injected_context + "\n\n" + new_input[key]
                    logger.info(f"🧠 Orchestra: Injected semantic context into LangChain '{key}'")
                    return new_input
                    
            logger.warning("Orchestra: Context injection enabled but no suitable key found in dict input.")
            return new_input
            
        return input_data

    def _invoke_or_run(self, input, config, **kwargs):
        """Helper to unify invoke/run/call logic for sync"""
        if hasattr(self.chain, "invoke"):
            return self.chain.invoke(input, config, **kwargs)
        elif hasattr(self.chain, "run") and isinstance(input, str):
            return self.chain.run(input, **kwargs)
        else:
            result = self.chain(input, **kwargs)
            if isinstance(result, dict) and len(result) == 1:
                return list(result.values())[0]
            return result

    async def _ainvoke_or_run(self, input, config, **kwargs):
        """Helper to unify invoke/run/call logic for async"""
        if hasattr(self.chain, "ainvoke"):
            return await self.chain.ainvoke(input, config, **kwargs)
        elif hasattr(self.chain, "arun") and isinstance(input, str):
            return await self.chain.arun(input, **kwargs)
        elif hasattr(self.chain, "acall"):
            result = await self.chain.acall(input, **kwargs)
            if isinstance(result, dict) and len(result) == 1:
                return list(result.values())[0]
            return result
        else:
            logger.warning("Chain does not support async invoke/run, falling back to sync.")
            return self._invoke_or_run(input, config, **kwargs)

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
