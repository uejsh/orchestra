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
    1. Hierarchical Embeddings
    2. Compression
    3. Time Windows (Freshness checks)
    4. Eviction
    """
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.92,
        max_cache_size: int = 10000,
        cache_ttl: int = 3600,
        enable_compression: bool = True
    ):
        self.config = {
            "similarity_threshold": similarity_threshold,
            "max_cache_size": max_cache_size,
            "cache_ttl": cache_ttl
        }
        
        # 1. Embeddings
        self.base_embedder = EmbeddingGenerator(embedding_model_name)
        self.hierarchical_embedder = HierarchicalEmbeddingGenerator(self.base_embedder)
        self.matcher = HierarchicalMatcher()
        
        # 2. Storage
        self.store = SemanticStore(
            dimension=self.base_embedder.dimension,
            similarity_threshold=similarity_threshold,
            max_cache_size=max_cache_size
        )
        
        # 3. Compression
        self.compressor = CompressionManager(enable_compression=enable_compression)
        
        logger.info("Initialized CacheManager with Hierarchical Matching & Compression")

    def get(self, query: str, time_window_seconds: Optional[int] = None) -> Optional[Any]:
        """
        Retrieve from cache using Hierarchical Matching.
        
        Args:
            query: The user query string
            time_window_seconds: If provided, ignores cache entries older than this.
                                 This solves "Sales report now != Sales report last month".
        """
        # 1. Generate Hierarchical Embedding
        h_embedding = self.hierarchical_embedder.generate(query)
        
        # 2. Search in Semantic Store (using Level 1 / Full embedding for fast retrieval)
        candidates = self.store.search(
            query_embedding=h_embedding.full_embedding,
            top_k=5,
            min_similarity=self.config["similarity_threshold"] * 0.9 # Loose filter first
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
            
            # B. Reconstruct Hierarchical Embedding for Cached State
            score = coarse_score # Simplification for MVP
            
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
        h_embedding = self.hierarchical_embedder.generate(query)
        
        # 3. Key generation
        key = hashlib.md5(query.encode()).hexdigest()
        
        # 4. Store
        self.store.put(
            key=key,
            value=compressed_value,
            embedding=h_embedding.full_embedding,
            ttl=ttl or self.config["cache_ttl"],
            metadata={
                "chunks": h_embedding.chunks,
                "original_query": query
            }
        )

    def save(self, path: str):
        self.store.save(path)
        
    def load(self, path: str):
        self.store.load(path)
