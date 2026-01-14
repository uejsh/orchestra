# redis_store.py - Redis-backed Semantic Store
# ============================================================================
# FILE: orchestra/core/redis_store.py
# Production-ready distributed semantic cache using Redis Stack
# ============================================================================

import time
import logging
import json
import numpy as np
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

from .semantic_store import CachedState

logger = logging.getLogger(__name__)

try:
    import redis
    from redis.commands.search.field import VectorField, TagField, NumericField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

class RedisSemanticStore:
    """
    Redis-backed semantic store.
    Requires Redis Stack (with RediSearch and RedisJSON/Vector support).
    """

    def __init__(
        self,
        redis_url: str,
        dimension: int = 384,
        similarity_threshold: float = 0.92,
        index_name: str = "orchestra:index",
        ttl: int = 3600
    ):
        if not HAS_REDIS:
            raise ImportError("redis-py is required for RedisSemanticStore. Run `pip install redis hiredis`.")

        self.redis = redis.from_url(redis_url)
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold
        self.index_name = index_name
        self.default_ttl = ttl
        
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_stored": 0,
        }

        self._ensure_index()
        logger.info(f"Initialized RedisSemanticStore at {redis_url} (index={index_name})")

    def _ensure_index(self):
        """Create the vector search index if it doesn't exist."""
        try:
            self.redis.ft(self.index_name).info()
        except redis.exceptions.ResponseError:
            # Index does not exist, create it
            schema = (
                TagField("tag"),
                NumericField("timestamp"),
                VectorField(
                    "embedding",
                    "FLAT", # or HNSW for speed
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.dimension,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            )
            definition = IndexDefinition(prefix=["orchestra:doc:"], index_type=IndexType.HASH)
            self.redis.ft(self.index_name).create_index(schema, definition=definition)
            logger.info("Created new Redis vector index")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_similarity: Optional[float] = None
    ) -> List[Tuple[CachedState, float]]:
        """Search for similar cached states using Redis Vector Search."""
        self.stats["total_queries"] += 1
        
        # Redis Vector Search uses "Kn" for "K nearest". 
        # But standard syntax is `*=>[KNN k @embedding $BLOB AS score]`
        
        k = top_k
        threshold = min_similarity if min_similarity is not None else self.similarity_threshold
        
        # Prepare params
        vector_bytes = query_embedding.astype(np.float32).tobytes()
        
        # Construct query: Find top K, then filter by score in python (Redis can't easily range filter on vector score directly in one pass without KNN)
        # However, we can set epsilon if using HNSW, but FLAT is brute force.
        # We'll just fetch top K and filter.
        
        query_str = f"*=>[KNN {k} @embedding $vec_blob AS vector_score]"
        
        q = (
            Query(query_str)
            .return_fields("vector_score", "val_blob", "key", "timestamp", "ttl", "metadata_json")
            .sort_by("vector_score")
            .dialect(2)
        )
        
        params = {"vec_blob": vector_bytes}
        
        results = self.redis.ft(self.index_name).search(q, query_params=params)
        
        output = []
        current_time = time.time()
        
        for doc in results.docs:
            # Redis 'vector_score' for COSINE is 1 - cosine_similarity (usually) OR it is the distance.
            # In Redis, COSINE distance = 1 - dot_product (if normalized).
            # Wait, Redis documentation: "cosine distance" defined as 1 - (A.B)/(|A||B|).
            # So Similarity = 1 - distance.
            
            distance = float(doc.vector_score)
            similarity = 1.0 - distance
            
            if similarity < threshold:
                continue
                
            # Parse stored data
            state = self._doc_to_state(doc)
            
            # Check Expiry (Redis keys expire, but double check)
            if state.is_expired(current_time):
                # Optionally delete lazily?
                continue
                
            output.append((state, similarity))
            
            # Update hit count asynchronously if possible, currently skip to save perf
        
        if output:
            self.stats["cache_hits"] += 1
            # Sort by similarity descending
            output.sort(key=lambda x: x[1], reverse=True)
        else:
            self.stats["cache_misses"] += 1
            
        return output

    def put(
        self,
        key: str,
        value: Any,
        embedding: np.ndarray,
        ttl: int = 3600,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a state in Redis."""
        # Clean key
        redis_key = f"orchestra:doc:{key}"
        
        # Serialize value
        val_blob = pickle.dumps(value)
        metadata_json = json.dumps(metadata or {})
        
        # Normalize embedding and convert to bytes
        # embedding should be float32
        emb_bytes = embedding.astype(np.float32).tobytes()
        
        mapping = {
            "key": key,
            "val_blob": val_blob,
            "embedding": emb_bytes,
            "timestamp": time.time(),
            "ttl": ttl,
            "metadata_json": metadata_json,
            "tag": "orchestra"
        }
        
        # Pipeline for atomicity
        pipe = self.redis.pipeline()
        pipe.hset(redis_key, mapping=mapping)
        pipe.expire(redis_key, ttl)
        pipe.execute()
        
        self.stats["total_stored"] += 1
        
    def _doc_to_state(self, doc) -> CachedState:
        """Convert Redis doc to CachedState."""
        return CachedState(
            key=doc.key, # This might be the redis key "orchestra:doc:..." or the internal key
            value=pickle.loads(doc.val_blob),
            embedding=None, # We don't fetch back embedding to save bandwidth
            timestamp=float(doc.timestamp),
            ttl=int(doc.ttl),
            metadata=json.loads(doc.metadata_json)
        )

    def invalidate(self, key: Optional[str] = None) -> int:
        if key is None:
            # Drop index and all keys
            # Risky on shared redis, but okay for this namespace
            # Helper: find all keys
            keys = self.redis.keys("orchestra:doc:*")
            if keys:
                self.redis.delete(*keys)
            return len(keys)
        else:
            redis_key = f"orchestra:doc:{key}"
            if self.redis.exists(redis_key):
                self.redis.delete(redis_key)
                return 1
            return 0

    def cleanup_expired(self) -> int:
        """Redis handles expiration automatically via TTL."""
        # We could scan and count, but that's expensive.
        return 0

    def save(self, path: str):
        """Save to disk - Not applicable for Redis (handled by Redis RDB/AOF)."""
        logger.warning("save() called on RedisSemanticStore - Redis handles persistence automatically.")

    def load(self, path: str):
        """Load from disk - Not applicable for Redis."""
        logger.warning("load() called on RedisSemanticStore - Redis handles persistence automatically.")

    def get_stats(self) -> Dict[str, Any]:
        info = {}
        try:
             info = self.redis.ft(self.index_name).info()
        except:
             pass
             
        return {
            "total_queries": self.stats["total_queries"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "hit_rate": self.stats["cache_hits"] / self.stats["total_queries"] if self.stats["total_queries"] > 0 else 0.0,
            "total_stored": self.stats["total_stored"],
            "current_size": int(info.get("num_docs", 0)),
            "backend": "redis"
        }
