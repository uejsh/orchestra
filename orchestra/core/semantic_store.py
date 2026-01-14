# semantic_store.py - Your FAISS code (enhanced with Fallback)
# ============================================================================
# FILE: orchestra/core/semantic_store.py
# Enhanced semantic store with fallback if FAISS is missing
# ============================================================================

import time
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pickle
import threading
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# --- FAISS Fallback Logic ---
try:
    import faiss
    HAS_FAISS = True
except BaseException as e:
    HAS_FAISS = False
    logger.warning(f"FAISS not found or failed to load ({e}). Using slow NumPy fallback.")
    faiss = None

class NumpyIndex:
    """Simple brute-force index for fallback"""
    def __init__(self, dimension):
        self.dimension = dimension
        self.vectors = []

    @property
    def ntotal(self):
        return len(self.vectors)

    def add(self, vectors):
        # vectors is (N, dim)
        if len(self.vectors) == 0:
            self.vectors = list(vectors)
        else:
            self.vectors.extend(list(vectors))
            
    def reset(self):
        self.vectors = []
        
    def search(self, query, k):
        if not self.vectors:
            return np.array([[]]), np.array([[]])
            
        # Brute force cosine similarity
        # query: (1, dim), vectors: (N, dim)
        stack = np.vstack(self.vectors)
        # Dot product (similarity), assuming normalized vectors
        scores = np.dot(stack, query.flatten()) 
        
        # Get top K
        # If we have fewer than k items, return all sorted
        eff_k = min(len(scores), k)
        
        # argsort gives ascending, so we negate scores for descending sort
        indices = np.argsort(-scores)[:eff_k]
        distances = scores[indices]
        
        return np.array([distances]), np.array([indices])

# ----------------------------

@dataclass
class CachedState:
    """Represents a cached state with metadata"""
    key: str
    value: Any
    embedding: np.ndarray
    timestamp: float = field(default_factory=time.time)
    ttl: int = 3600
    metadata: Dict[str, Any] = field(default_factory=dict)
    hit_count: int = 0
    
    def is_expired(self, current_time: Optional[float] = None) -> bool:
        """Check if state has expired"""
        current = current_time or time.time()
        return (current - self.timestamp) > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for serialization)"""
        return {
            "key": self.key,
            "value": self.value,
            "embedding": self.embedding,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "metadata": self.metadata,
            "hit_count": self.hit_count,
        }


class SemanticStore:
    """
    Semantic store with TTL and compression.
    Uses FAISS if available, otherwise NumPy fallback.
    """
    
    def __init__(
        self,
        dimension: int = 384,
        similarity_threshold: float = 0.92,
        max_cache_size: int = 10000,
        enable_compression: bool = True
    ):
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.enable_compression = enable_compression
        
        # Initialize Index
        if HAS_FAISS:
            try:
                self.index = faiss.IndexFlatIP(dimension)
            except Exception as e:
                logger.error(f"Failed to initialize FAISS: {e}. using Fallback.")
                self.index = NumpyIndex(dimension)
        else:
            self.index = NumpyIndex(dimension)
        
        self.states: List[CachedState] = []
        
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_stored": 0,
        }
        
        self._lock = threading.Lock()
        
        logger.info(f"Initialized SemanticStore (dim={dimension}, threshold={similarity_threshold})")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_similarity: Optional[float] = None
    ) -> List[Tuple[CachedState, float]]:
        """Search for similar cached states."""
        with self._lock:
            self.stats["total_queries"] += 1
            
            if self.index.ntotal == 0:
                self.stats["cache_misses"] += 1
                return []
        
        threshold = min_similarity if min_similarity is not None else self.similarity_threshold
        
        # Ensure query is normalized
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        query_norm = query_norm.reshape(1, -1).astype('float32')
        
        # Search index
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_norm, k)
        
        # Filter (FAISS returns List[List], fallback matches signature)
        results = []
        current_time = time.time()
        
        # Handle case where search returns empty (e.g. empty index)
        if distances.size == 0:
            self.stats["cache_misses"] += 1
            return []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: continue
            
            similarity = float(dist)
            
            if similarity < threshold:
                continue
            
            # Safety check for index out of bounds (sync issues?)
            if idx >= len(self.states):
                continue

            state = self.states[idx]
            
            if state.is_expired(current_time):
                continue
            
            results.append((state, similarity))
            state.hit_count += 1
        
        if results:
            self.stats["cache_hits"] += 1
        else:
            self.stats["cache_misses"] += 1
        
        return results
    
    def put(
        self,
        key: str,
        value: Any,
        embedding: np.ndarray,
        ttl: int = 3600,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a state."""
        with self._lock:
            if len(self.states) >= self.max_cache_size:
                self._evict_oldest()
        
        # Normalize
        emb_norm = embedding / np.linalg.norm(embedding)
        emb_norm = emb_norm.astype('float32')
        
        state = CachedState(
            key=key,
            value=value,
            embedding=emb_norm,
            ttl=ttl,
            metadata=metadata or {}
        )
        
        # Add to index
        self.index.add(emb_norm.reshape(1, -1))
        self.states.append(state)
        
        self.stats["total_stored"] += 1
        logger.debug(f"Stored: {key}")
    
    def invalidate(self, key: Optional[str] = None) -> int:
        """Invalidate states."""
        with self._lock:
            if key is None:
                count = len(self.states)
                self.index.reset()
                self.states.clear()
                logger.info(f"Invalidated all {count}")
                return count

        
        count = 0
        for state in self.states:
            if state.key == key:
                state.ttl = 0
                count += 1
        return count
    
    def cleanup_expired(self) -> int:
        """Remove expired states."""
        with self._lock:
            current_time = time.time()
        valid_states = []
        valid_embeddings = []
        
        for state in self.states:
            if not state.is_expired(current_time):
                valid_states.append(state)
                valid_embeddings.append(state.embedding)
        
        removed = len(self.states) - len(valid_states)
        
        if removed > 0:
            self.index.reset()
            if valid_embeddings:
                embeddings_array = np.vstack(valid_embeddings)
                self.index.add(embeddings_array)
            self.states = valid_states
            logger.info(f"Cleaned up {removed} expired")
        
        return removed
    
    def _evict_oldest(self):
        """Evict oldest state."""
        if not self.states: return
        oldest_idx = min(range(len(self.states)), key=lambda i: self.states[i].timestamp)
        self.states.pop(oldest_idx)
        self.index.reset()
        if self.states:
            embeddings = np.vstack([s.embedding for s in self.states])
            self.index.add(embeddings)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats."""
        total = self.stats["total_queries"]
        hits = self.stats["cache_hits"]
        return {
            "total_queries": total,
            "cache_hits": hits,
            "cache_misses": self.stats["cache_misses"],
            "hit_rate": hits / total if total > 0 else 0.0,
            "total_stored": self.stats["total_stored"],
            "current_size": len(self.states),
        }
    
    def save(self, path: str):
        """Save to disk."""
        with self._lock:
            data = {
            "states": [s.to_dict() for s in self.states],
            "stats": self.stats,
            "config": {
                "dimension": self.dimension,
                "similarity_threshold": self.similarity_threshold,
                "max_cache_size": self.max_cache_size,
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved: {path}")
    
    def load(self, path: str):
        """Load from disk."""
        logger.info(f"Loading {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.states = []
        embeddings = []
        for s in data["states"]:
            if "embedding" in s and s["embedding"] is not None:
                emb = s["embedding"]
                state = CachedState(
                    key=s["key"],
                    value=s["value"],
                    embedding=emb,
                    timestamp=s["timestamp"],
                    ttl=s["ttl"],
                    metadata=s["metadata"],
                    hit_count=s.get("hit_count", 0)
                )
                self.states.append(state)
                embeddings.append(emb)
        
        self.index.reset()
        if embeddings:
            embeddings_array = np.vstack(embeddings).astype('float32')
            self.index.add(embeddings_array)

        self.stats = data["stats"]
        with self._lock:
            logger.info(f"Loaded {len(self.states)} states")


__all__ = ["SemanticStore", "CachedState"]
