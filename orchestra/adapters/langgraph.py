# langgraph.py - LangGraph enhancement
# ============================================================================
# FILE: orchestra/adapters/langgraph.py
# LangGraph adapter - Drop-in semantic caching enhancement
# ============================================================================

import time
import hashlib
import json
import logging
from typing import Any, Dict, Optional, Callable
from functools import wraps

from ..core.embeddings import EmbeddingGenerator
from ..core.hierarchical_embeddings import HierarchicalEmbeddingGenerator, HierarchicalMatcher
from ..core.semantic_store import SemanticStore
from ..core.compression import CompressionManager
from ..core.metrics import MetricsTracker

logger = logging.getLogger(__name__)


class OrchestraConfig:
    """Configuration for Orchestra enhancement"""
    
    def __init__(
        self,
        # Semantic matching
        similarity_threshold: float = 0.92,
        embedding_model: str = "all-MiniLM-L6-v2",
        
        # Hierarchical embeddings (multi-level matching)
        enable_hierarchical: bool = False,
        hierarchical_weight_l1: float = 0.6,  # Weight for full query similarity
        hierarchical_weight_l2: float = 0.4,  # Weight for chunk similarity
        
        # Caching
        cache_ttl: int = 3600,
        max_cache_size: int = 10000,
        enable_cache: bool = True,
        
        # Compression
        enable_compression: bool = False,
        
        # Cost tracking
        llm_cost_per_1k_tokens: float = 0.03,
        
        # Advanced
        auto_cleanup: bool = True,
        cleanup_interval: int = 300,  # 5 minutes
    ):
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.enable_hierarchical = enable_hierarchical
        self.hierarchical_weight_l1 = hierarchical_weight_l1
        self.hierarchical_weight_l2 = hierarchical_weight_l2
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self.enable_cache = enable_cache
        self.enable_compression = enable_compression
        self.llm_cost_per_1k_tokens = llm_cost_per_1k_tokens
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval = cleanup_interval


class EnhancedLangGraph:
    """
    Wrapper around compiled LangGraph that adds semantic caching.
    
    This is the magic that makes Orchestra work with zero code changes.
    
    Features (configurable):
    - Semantic caching with FAISS/NumPy
    - Hierarchical embeddings for better matching (optional)
    - Compression for smaller cache footprint (optional)
    """
    
    def __init__(
        self,
        graph,  # CompiledGraph from LangGraph
        config: Optional[OrchestraConfig] = None
    ):
        """
        Args:
            graph: Compiled LangGraph instance
            config: Orchestra configuration
        """
        self.graph = graph
        self.config = config or OrchestraConfig()
        
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
            logger.info("🔬 Hierarchical embeddings ENABLED")
        else:
            self.hierarchical_gen = None
            self.hierarchical_matcher = None
        
        # Compression (optional)
        if self.config.enable_compression:
            self.compressor = CompressionManager(enable_compression=True)
            logger.info("🗜️  Compression ENABLED")
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
        
        # Last cleanup time
        self._last_cleanup = time.time()
        
        # Cache for hierarchical embeddings (key -> HierarchicalEmbedding)
        self._hierarchical_cache: Dict[str, Any] = {}
        
        logger.info("✨ Orchestra enhancement enabled for LangGraph")
    
    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[Dict] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Any:
        """
        Enhanced invoke with semantic caching.
        
        Args:
            input: Graph input
            config: LangGraph config
            use_cache: Whether to use cache (default: True)
            **kwargs: Additional arguments
        
        Returns:
            Graph output
        """
        start_time = time.time()
        
        # Auto cleanup if needed
        self._maybe_cleanup()
        
        # Generate cache key from input
        input_str = self._serialize_input(input)
        cache_key = self._generate_cache_key(input_str)
        
        # Check semantic cache
        cached_result = None
        if use_cache and self.config.enable_cache:
            cached_result = self._check_cache(input_str, cache_key)
        
        if cached_result is not None:
            # Cache HIT
            latency = time.time() - start_time
            
            self.metrics.record_cache_hit(latency)
            
            logger.info(
                f"💰 Cache HIT - "
                f"Saved ~${self.metrics.estimate_llm_cost():.2f}, "
                f"Latency: {latency:.3f}s"
            )
            
            return cached_result
        
        # Cache MISS - Execute normal LangGraph
        logger.debug(f"Cache MISS - Executing graph...")
        
        self.metrics.start_execution()
        
        result = self.graph.invoke(input, config, **kwargs)
        
        execution_time = time.time() - start_time
        self.metrics.end_execution(execution_time)
        
        # Store result in semantic cache
        if self.config.enable_cache:
            self._store_result(input_str, result, cache_key)
        
        logger.info(
            f"⚡ Execution complete - "
            f"Latency: {execution_time:.3f}s"
        )
        
        return result
    
    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[Dict] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Any:
        """
        Async version of invoke with semantic caching.
        """
        start_time = time.time()
        
        # Auto cleanup
        self._maybe_cleanup()
        
        # Generate cache key
        input_str = self._serialize_input(input)
        cache_key = self._generate_cache_key(input_str)
        
        # Check cache
        cached_result = None
        if use_cache and self.config.enable_cache:
            cached_result = self._check_cache(input_str, cache_key)
        
        if cached_result is not None:
            latency = time.time() - start_time
            self.metrics.record_cache_hit(latency)
            
            logger.info(f"💰 Cache HIT - Latency: {latency:.3f}s")
            return cached_result
        
        # Execute
        self.metrics.start_execution()
        
        result = await self.graph.ainvoke(input, config, **kwargs)
        
        execution_time = time.time() - start_time
        self.metrics.end_execution(execution_time)
        
        # Store
        if self.config.enable_cache:
            self._store_result(input_str, result, cache_key)
        
        return result
    
    def stream(self, *args, **kwargs):
        """Pass through to original graph (caching not supported for streaming)"""
        logger.warning("Streaming does not support caching - using original graph")
        return self.graph.stream(*args, **kwargs)
    
    def astream(self, *args, **kwargs):
        """Pass through to original graph"""
        logger.warning("Streaming does not support caching - using original graph")
        return self.graph.astream(*args, **kwargs)
    
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
        # Generate embedding
        query_embedding = self.base_embedding_gen.generate(input_str)
        
        # Search semantic store
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
        # Generate hierarchical embedding for query
        query_h_emb = self.hierarchical_gen.generate(input_str)
        
        # Search using full embedding first (coarse filter)
        candidates = self.semantic_store.search(
            query_h_emb.full_embedding,
            top_k=5,
            min_similarity=self.config.similarity_threshold * 0.9  # Loose first pass
        )
        
        if not candidates:
            return None
        
        # Refine with hierarchical matching
        best_match = None
        best_score = -1.0
        
        for state, coarse_score in candidates:
            # Get cached hierarchical embedding
            cached_h_emb = self._hierarchical_cache.get(state.key)
            
            if cached_h_emb:
                # Full hierarchical comparison
                score = self.hierarchical_matcher.compute_similarity(query_h_emb, cached_h_emb)
            else:
                # Fallback to coarse score if hierarchical not stored
                score = coarse_score
            
            if score > best_score:
                best_match = state
                best_score = score
        
        if best_match and best_score >= self.config.similarity_threshold:
            logger.debug(
                f"Hierarchical match (score: {best_score:.3f}, "
                f"age: {time.time() - best_match.timestamp:.1f}s)"
            )
            
            # Decompress if needed
            value = best_match.value
            if self.compressor:
                value = self.compressor.decompress(value)
            
            return value
        
        return None
    
    def _store_result(self, input_str: str, result: Any, cache_key: str):
        """Store result in semantic cache"""
        
        # Compress value if enabled
        value_to_store = result
        if self.compressor:
            value_to_store = self.compressor.compress(result)
        
        # Generate embedding
        if self.config.enable_hierarchical:
            h_emb = self.hierarchical_gen.generate(input_str)
            embedding = h_emb.full_embedding
            # Cache hierarchical embedding for later matching
            self._hierarchical_cache[cache_key] = h_emb
        else:
            embedding = self.base_embedding_gen.generate(input_str)
        
        # Store in semantic store
        self.semantic_store.put(
            key=cache_key,
            value=value_to_store,
            embedding=embedding,
            ttl=self.config.cache_ttl,
            metadata={
                "stored_at": time.time(),
                "input_hash": self._hash_input(input_str),
                "hierarchical": self.config.enable_hierarchical,
                "compressed": self.config.enable_compression
            }
        )
        
        logger.debug(f"Stored result in cache: {cache_key[:16]}...")
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def _serialize_input(self, input: Any) -> str:
        """Serialize input to string for embedding"""
        try:
            return json.dumps(input, sort_keys=True, default=str)
        except:
            return str(input)
    
    def _generate_cache_key(self, input_str: str) -> str:
        """Generate deterministic cache key"""
        return hashlib.sha256(input_str.encode()).hexdigest()
    
    def _hash_input(self, input_str: str) -> str:
        """Hash input for quick comparison"""
        return hashlib.md5(input_str.encode()).hexdigest()
    
    def _maybe_cleanup(self):
        """Cleanup expired cache entries if needed"""
        if not self.config.auto_cleanup:
            return
        
        current_time = time.time()
        if current_time - self._last_cleanup < self.config.cleanup_interval:
            return
        
        removed = self.semantic_store.cleanup_expired()
        if removed > 0:
            logger.info(f"🗑️  Auto-cleanup removed {removed} expired entries")
        
        self._last_cleanup = current_time
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def invalidate_cache(self, query: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Args:
            query: Specific query to invalidate (None = invalidate all)
        """
        if query is None:
            count = self.semantic_store.invalidate()
            self._hierarchical_cache.clear()
            logger.info(f"Invalidated all {count} cache entries")
        else:
            query_str = self._serialize_input(query)
            cache_key = self._generate_cache_key(query_str)
            count = self.semantic_store.invalidate(cache_key)
            self._hierarchical_cache.pop(cache_key, None)
            logger.info(f"Invalidated {count} entries for query")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with metrics
        """
        cache_stats = self.semantic_store.get_stats()
        exec_stats = self.metrics.get_stats()
        
        return {
            # Cache metrics
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_hits": cache_stats["cache_hits"],
            "cache_misses": cache_stats["cache_misses"],
            "cache_size": cache_stats["current_size"],
            
            # Execution metrics
            "total_executions": exec_stats["total_executions"],
            "avg_cache_hit_latency": exec_stats["avg_cache_hit_latency"],
            "avg_execution_latency": exec_stats["avg_execution_latency"],
            
            # Cost metrics
            "estimated_cost": exec_stats["estimated_total_cost"],
            "estimated_cost_saved": exec_stats["estimated_cost_saved"],
            "cost_reduction_pct": exec_stats["cost_reduction_percentage"],
            
            # Feature status
            "hierarchical_enabled": self.config.enable_hierarchical,
            "compression_enabled": self.config.enable_compression,
        }
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics with breakdown"""
        return {
            "cache": self.semantic_store.get_stats(),
            "execution": self.metrics.get_stats(),
            "config": {
                "similarity_threshold": self.config.similarity_threshold,
                "cache_ttl": self.config.cache_ttl,
                "max_cache_size": self.config.max_cache_size,
                "enable_hierarchical": self.config.enable_hierarchical,
                "enable_compression": self.config.enable_compression,
            }
        }
    
    def export_metrics(self, path: str):
        """Export metrics to JSON file"""
        metrics = self.get_detailed_metrics()
        
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Exported metrics to {path}")
    
    def save_cache(self, path: str):
        """Save cache to disk"""
        self.semantic_store.save(path)
    
    def load_cache(self, path: str):
        """Load cache from disk"""
        self.semantic_store.load(path)


# ============================================================================
# PUBLIC API
# ============================================================================

def enhance(
    graph,
    config: Optional[OrchestraConfig] = None
) -> EnhancedLangGraph:
    """
    Enhance a LangGraph with Orchestra semantic caching.
    
    Usage:
        from langgraph.graph import StateGraph
        from orchestra import enhance
        
        graph = StateGraph(State)
        # ... add nodes, edges ...
        compiled = graph.compile()
        
        # Add Orchestra
        enhanced = enhance(compiled)
        
        # Use normally
        result = enhanced.invoke({"query": "..."})
    
    Args:
        graph: Compiled LangGraph instance
        config: Orchestra configuration
    
    Returns:
        Enhanced graph with semantic caching
    """
    return EnhancedLangGraph(graph, config)


__all__ = [
    "enhance",
    "EnhancedLangGraph",
    "OrchestraConfig",
]
