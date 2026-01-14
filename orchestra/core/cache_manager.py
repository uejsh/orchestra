# cache_manager.py - Intelligent Cache Management
# ============================================================================
# FILE: orchestra/core/cache_manager.py
# Manager for SemanticStore with eviction policies and TIME WINDOWS
# ============================================================================

import time
import logging
from typing import Any, Optional, List, Dict
import numpy as np
import hashlib

from .semantic_store import SemanticStore, CachedState
from .hierarchical_embeddings import HierarchicalEmbedding, HierarchicalMatcher, HierarchicalEmbeddingGenerator
from .embeddings import EmbeddingGenerator
from .compression import CompressionManager

logger = logging.getLogger(__name__)

class CacheManager:
    """
    High-level manager for the Semantic Store.
    Handles:
    1. Hierarchical Embeddings (optional)
    2. Compression (optional)
    3. Time Windows (Freshness checks)
    4. Eviction
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.92,
        max_cache_size: int = 10000,
        cache_ttl: int = 3600,
        enable_compression: bool = False,
        enable_hierarchical: bool = False,
        hierarchical_weight_l1: float = 0.6,
        hierarchical_weight_l2: float = 0.4,
        redis_url: Optional[str] = None
    ):
        self.config = {
            "similarity_threshold": similarity_threshold,
            "max_cache_size": max_cache_size,
            "cache_ttl": cache_ttl,
            "redis_url": redis_url,
            "enable_hierarchical": enable_hierarchical,
            "enable_compression": enable_compression,
        }
        
        # 1. Embeddings
        self.base_embedder = EmbeddingGenerator(embedding_model_name)
        
        # Hierarchical (optional)
        if enable_hierarchical:
            self.hierarchical_embedder = HierarchicalEmbeddingGenerator(self.base_embedder)
            self.matcher = HierarchicalMatcher(
                weight_l1=hierarchical_weight_l1,
                weight_l2=hierarchical_weight_l2
            )
            self._hierarchical_cache: Dict[str, HierarchicalEmbedding] = {}
            logger.info("🔬 CacheManager: Hierarchical embeddings ENABLED")
        else:
            self.hierarchical_embedder = None
            self.matcher = None
            self._hierarchical_cache = {}
        
        # 2. Storage
        if redis_url:
            from .redis_store import RedisSemanticStore
            self.store = RedisSemanticStore(
                redis_url=redis_url,
                dimension=self.base_embedder.dimension,
                similarity_threshold=similarity_threshold,
                ttl=cache_ttl
            )
        else:
            self.store = SemanticStore(
                dimension=self.base_embedder.dimension,
                similarity_threshold=similarity_threshold,
                max_cache_size=max_cache_size
            )
        
        # 3. Compression
        self.compressor = CompressionManager(enable_compression=enable_compression)
        if enable_compression:
            logger.info("🗜️  CacheManager: Compression ENABLED")
        
        logger.info(f"Initialized CacheManager (threshold={similarity_threshold})")

    def get(self, query: str, time_window_seconds: Optional[int] = None) -> Optional[Any]:
        """
        Retrieve from cache using semantic matching.
        
        Args:
            query: The user query string
            time_window_seconds: If provided, ignores cache entries older than this.
                                 This solves "Sales report now != Sales report last month".
        """
        # 1. Generate Embedding
        if self.config["enable_hierarchical"] and self.hierarchical_embedder:
            h_embedding = self.hierarchical_embedder.generate(query)
            query_embedding = h_embedding.full_embedding
        else:
            query_embedding = self.base_embedder.generate(query)
            h_embedding = None
        
        # 2. Search in Semantic Store
        candidates = self.store.search(
            query_embedding=query_embedding,
            top_k=5 if self.config["enable_hierarchical"] else 1,
            min_similarity=self.config["similarity_threshold"] * 0.9  # Loose filter first
        )
        
        if not candidates:
            return None
            
        # 3. Refine with Level 2 Matching & Check Time Window
        best_match = None
        best_score = -1.0
        current_time = time.time()
        
        for state, coarse_score in candidates:
            # A. Time Window Check (The "Freshness" Constraint)
            age = current_time - state.timestamp
            
            # Global TTL check
            if age > state.ttl:
                continue
                
            # Query-specific Time Window check (e.g. "only in last 24h")
            if time_window_seconds is not None and age > time_window_seconds:
                continue
            
            # B. Hierarchical Matching (if enabled)
            if self.config["enable_hierarchical"] and h_embedding and self.matcher:
                cached_h_emb = self._hierarchical_cache.get(state.key)
                if cached_h_emb:
                    score = self.matcher.compute_similarity(h_embedding, cached_h_emb)
                else:
                    score = coarse_score  # Fallback if hierarchical not stored
            else:
                score = coarse_score
            
            if score > best_score:
                best_match = state
                best_score = score
        
        if best_match and best_score >= self.config["similarity_threshold"]:
            # Decompress value
            return self.compressor.decompress(best_match.value)
            
        return None

    def put(self, query: str, value: Any, ttl: Optional[int] = None):
        """Store result in cache"""
        # 1. Compress
        compressed_value = self.compressor.compress(value)
        
        # 2. Embedding
        if self.config["enable_hierarchical"] and self.hierarchical_embedder:
            h_embedding = self.hierarchical_embedder.generate(query)
            embedding = h_embedding.full_embedding
        else:
            embedding = self.base_embedder.generate(query)
            h_embedding = None
        
        # 3. Key generation
        key = hashlib.md5(query.encode()).hexdigest()
        
        # 4. Store hierarchical embedding for later matching (if enabled)
        if h_embedding and self.config["enable_hierarchical"]:
            self._hierarchical_cache[key] = h_embedding
        
        # 5. Store in semantic store
        metadata = {"original_query": query}
        if h_embedding:
            metadata["chunks"] = h_embedding.chunks
            
        self.store.put(
            key=key,
            value=compressed_value,
            embedding=embedding,
            ttl=ttl or self.config["cache_ttl"],
            metadata=metadata
        )

    # ========================================================================
    # PROXY METHODS FOR STORE OPERATIONS
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics from the underlying store."""
        return self.store.get_stats()

    def invalidate(self, key: Optional[str] = None) -> int:
        """Invalidate cache entries. If key is None, invalidate all."""
        count = self.store.invalidate(key)
        if key is None:
            self._hierarchical_cache.clear()
        else:
            self._hierarchical_cache.pop(key, None)
        return count

    def cleanup_expired(self) -> int:
        """Remove expired cache entries."""
        return self.store.cleanup_expired()

    def save(self, path: str):
        self.store.save(path)
        
    def load(self, path: str):
        self.store.load(path)

