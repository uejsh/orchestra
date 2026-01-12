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
from ..core.embeddings import EmbeddingGenerator
from ..core.hierarchical_embeddings import HierarchicalEmbeddingGenerator, HierarchicalMatcher
from ..core.semantic_store import SemanticStore
from ..core.compression import CompressionManager
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


class EnhancedLangChain(BaseAdapter):
    """
    Adapter for LangChain (both Chains and Runnables).
    
    Features (configurable):
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
        
        # Initialize base embedding generator
        self.base_embedding_gen = EmbeddingGenerator(
            model_name=self.config.embedding_model
        )
        
        # Hierarchical embeddings (optional)
        if self.config.enable_hierarchical:
            self.hierarchical_gen = HierarchicalEmbeddingGenerator(self.base_embedding_gen)
            self.hierarchical_matcher = HierarchicalMatcher(
                weight_l1=self.config.hierarchical_weight_l1,
                weight_l2=self.config.hierarchical_weight_l2
            )
            logger.info("🔬 Hierarchical embeddings ENABLED (LangChain)")
        else:
            self.hierarchical_gen = None
            self.hierarchical_matcher = None
        
        # Compression (optional)
        if self.config.enable_compression:
            self.compressor = CompressionManager(enable_compression=True)
            logger.info("🗜️  Compression ENABLED (LangChain)")
        else:
            self.compressor = None
        
        # Semantic store
        self.semantic_store = SemanticStore(
            dimension=self.base_embedding_gen.dimension,
            similarity_threshold=self.config.similarity_threshold,
            max_cache_size=self.config.max_cache_size
        )
        
        # Metrics
        self.metrics = MetricsTracker(
            llm_cost_per_1k_tokens=self.config.llm_cost_per_1k_tokens
        )
        
        # Cache for hierarchical embeddings
        self._hierarchical_cache: Dict[str, Any] = {}
        
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
            logger.info(f"💰 Cache HIT (LangChain) - Latency: {latency:.3f}s")
            return cached_result
        
        # 3. Execute
        self.metrics.start_execution()
        
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
            logger.info(f"💰 Cache HIT (LangChain Async) - Latency: {latency:.3f}s")
            return cached_result
        
        # 3. Execute
        self.metrics.start_execution()
        
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

        latency = time.time() - start_time
        self.metrics.end_execution(latency)
        
        # 4. Store
        self._store_result(input_str, result, cache_key)
        
        return result

    # ========================================================================
    # CACHE OPERATIONS
    # ========================================================================
    
    def _check_cache(self, input_str: str, cache_key: str) -> Optional[Any]:
        """Check semantic cache for similar inputs"""
        
        if self.config.enable_hierarchical:
            return self._check_cache_hierarchical(input_str, cache_key)
        else:
            return self._check_cache_simple(input_str)
    
    def _check_cache_simple(self, input_str: str) -> Optional[Any]:
        """Simple single-level semantic matching"""
        query_embedding = self.base_embedding_gen.generate(input_str)
        results = self.semantic_store.search(query_embedding, top_k=1)
        
        if not results:
            return None
        
        cached_state, similarity = results[0]
        
        logger.debug(
            f"Found cached result (similarity: {similarity:.3f}, "
            f"age: {time.time() - cached_state.timestamp:.1f}s)"
        )
        
        # Decompress if needed
        value = cached_state.value
        if self.compressor:
            value = self.compressor.decompress(value)
        
        return value
    
    def _check_cache_hierarchical(self, input_str: str, cache_key: str) -> Optional[Any]:
        """Hierarchical 2-level semantic matching"""
        query_h_emb = self.hierarchical_gen.generate(input_str)
        
        # Coarse search
        candidates = self.semantic_store.search(
            query_h_emb.full_embedding,
            top_k=5,
            min_similarity=self.config.similarity_threshold * 0.9
        )
        
        if not candidates:
            return None
        
        # Refine with hierarchical matching
        best_match = None
        best_score = -1.0
        
        for state, coarse_score in candidates:
            cached_h_emb = self._hierarchical_cache.get(state.key)
            
            if cached_h_emb:
                score = self.hierarchical_matcher.compute_similarity(query_h_emb, cached_h_emb)
            else:
                score = coarse_score
            
            if score > best_score:
                best_match = state
                best_score = score
        
        if best_match and best_score >= self.config.similarity_threshold:
            logger.debug(f"Hierarchical match (score: {best_score:.3f})")
            
            value = best_match.value
            if self.compressor:
                value = self.compressor.decompress(value)
            
            return value
        
        return None
    
    def _store_result(self, input_str: str, result: Any, cache_key: str):
        """Store result in semantic cache"""
        
        # Compress if enabled
        value_to_store = result
        if self.compressor:
            value_to_store = self.compressor.compress(result)
        
        # Generate embedding
        if self.config.enable_hierarchical:
            h_emb = self.hierarchical_gen.generate(input_str)
            embedding = h_emb.full_embedding
            self._hierarchical_cache[cache_key] = h_emb
        else:
            embedding = self.base_embedding_gen.generate(input_str)
        
        # Store
        self.semantic_store.put(
            key=cache_key,
            value=value_to_store,
            embedding=embedding,
            ttl=self.config.cache_ttl,
            metadata={
                "stored_at": time.time(),
                "hierarchical": self.config.enable_hierarchical,
                "compressed": self.config.enable_compression
            }
        )
        
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
        cache_stats = self.semantic_store.get_stats()
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
            count = self.semantic_store.invalidate()
            self._hierarchical_cache.clear()
            logger.info(f"Invalidated all {count} cache entries")
        else:
            cache_key = self._generate_cache_key(self._to_string(query))
            count = self.semantic_store.invalidate(cache_key)
            self._hierarchical_cache.pop(cache_key, None)
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
