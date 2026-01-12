# embeddings.py - Embedding generation
# ============================================================================
# FILE: orchestra/core/embeddings.py
# Embedding generation with caching and Fallback for missing Torch/Transformers
# ============================================================================

from typing import Union, List, Optional
import numpy as np
import hashlib
import logging

try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except Exception as e:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings with built-in caching.
    Includes fallback if sentence-transformers/torch is missing.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_embeddings: bool = True
    ):
        """
        Args:
            model_name: SentenceTransformer model name
            device: 'cpu' or 'cuda'
            cache_embeddings: Cache computed embeddings
        """
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        self._cache: dict = {}
        self.model = None
        self.dimension = 384 # Default for all-MiniLM-L6-v2
        
        if HAS_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name, device=device)
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Loaded embedding model: {model_name} (dim={self.dimension})")
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer ({e}). Using deterministic random embeddings.")
        else:
            logger.warning("sentence-transformers not installed. Using deterministic random embeddings.")

    
    def generate(
        self,
        text: Union[str, List[str]],
        normalize: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embedding(s) for text.
        
        Args:
            text: Single string or list of strings
            normalize: L2 normalize embeddings
        
        Returns:
            Single embedding or list of embeddings
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        # Check cache
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, t in enumerate(texts):
            cache_key = self._get_cache_key(t)
            
            if self.cache_embeddings and cache_key in self._cache:
                embeddings.append(self._cache[cache_key])
            else:
                embeddings.append(None)
                uncached_texts.append(t)
                uncached_indices.append(i)
        
        # Generate uncached embeddings
        if uncached_texts:
            if self.model:
                try:
                    new_embeddings = self.model.encode(
                        uncached_texts,
                        normalize_embeddings=normalize,
                        show_progress_bar=False
                    )
                except Exception:
                     # Runtime fail during encode? Fallback
                     new_embeddings = self._generate_dummy(uncached_texts, normalize)
            else:
                new_embeddings = self._generate_dummy(uncached_texts, normalize)
            
            # Store in cache and result list
            for idx, emb, text in zip(uncached_indices, new_embeddings, uncached_texts):
                embeddings[idx] = emb
                if self.cache_embeddings:
                    cache_key = self._get_cache_key(text)
                    self._cache[cache_key] = emb
        
        result = embeddings[0] if is_single else embeddings
        return result
    
    def _generate_dummy(self, texts: List[str], normalize: bool) -> List[np.ndarray]:
        """Generate deterministic random vectors based on text hash"""
        embeddings = []
        for text in texts:
            # Seed from hash of text
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(seed)
            vec = rng.rand(self.dimension).astype(np.float32)
            if normalize:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            embeddings.append(vec)
        return embeddings
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear embedding cache"""
        self._cache.clear()
        logger.info("Embedding cache cleared")
