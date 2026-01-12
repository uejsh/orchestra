# hierarchical_embeddings.py - Multi-level semantic matching
# ============================================================================
# FILE: orchestra/core/hierarchical_embeddings.py
# KEY DIFFERENTIATOR: 2-Level Semantic Matching
# ============================================================================

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
import logging
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

@dataclass
class HierarchicalEmbedding:
    """
    Represents a query at multiple semantic levels.
    Level 1: Full query embedding (Coarse)
    Level 2: Chunk/Phrase embeddings (Fine)
    """
    full_embedding: np.ndarray              # Shape: (dim,)
    chunk_embeddings: List[np.ndarray]      # Shape: (n_chunks, dim)
    chunks: List[str]                       # actual text of chunks

class HierarchicalMatcher:
    """
    Computes similarity using weighted multi-level matching.
    Formula: Score = (w1 * L1_Score) + (w2 * L2_Score)
    """
    
    def __init__(self, weight_l1: float = 0.6, weight_l2: float = 0.4):
        self.w1 = weight_l1
        self.w2 = weight_l2
    
    def compute_similarity(self, query: HierarchicalEmbedding, cached: HierarchicalEmbedding) -> float:
        """Compute composite similarity score"""
        # Level 1: Full Query Similarity (Cosine)
        sim_l1 = self._cosine_similarity(query.full_embedding, cached.full_embedding)
        
        # Level 2: Chunk Similarity (Soft overlap)
        sim_l2 = self._compute_chunk_similarity(query.chunk_embeddings, cached.chunk_embeddings)
        
        # Weighted combination
        final_score = (self.w1 * sim_l1) + (self.w2 * sim_l2)
        
        return float(final_score)

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two normalized vectors"""
        return np.dot(v1, v2)

    def _compute_chunk_similarity(self, query_chunks: List[np.ndarray], cached_chunks: List[np.ndarray]) -> float:
        """
        Compute similarity between two sets of chunks.
        Strategy: For each query chunk, find the max similarity in cached chunks.
        Then take the average of these max scores.
        """
        if not query_chunks or not cached_chunks:
            return 0.0
            
        scores = []
        for q_vec in query_chunks:
            # Find best match for this chunk in the cached chunks
            # Compute dot product with all cached chunks (matrix multiplication)
            # cached_stack shape: (n_chunks, dim)
            cached_stack = np.vstack(cached_chunks)
            snippet_sims = np.dot(cached_stack, q_vec)
            max_sim = np.max(snippet_sims)
            scores.append(max_sim)
            
        return np.mean(scores)


class HierarchicalEmbeddingGenerator:
    """Generates 2-level embeddings from text"""
    
    def __init__(self, base_generator: EmbeddingGenerator):
        self.base = base_generator
    
    def generate(self, text: str) -> HierarchicalEmbedding:
        """Generate full and chunk embeddings"""
        # 1. Full embedding
        full_emb = self.base.generate(text)
        
        # 2. Chunking (Simple sliding window or split by delimiters)
        # For efficiency/simplicity, we'll split by common delimiters and length
        chunks = self._chunk_text(text)
        
        if chunks:
            chunk_embs = self.base.generate(chunks)
            if isinstance(chunk_embs, np.ndarray):
                # If single chunk, listify
                if chunk_embs.ndim == 1:
                    chunk_embs = [chunk_embs]
                else:
                    chunk_embs = list(chunk_embs)
            elif isinstance(chunk_embs, list):
                pass
            else:
                 chunk_embs = [chunk_embs] # fallback
        else:
            chunk_embs = []
            
        return HierarchicalEmbedding(
            full_embedding=full_emb,
            chunk_embeddings=chunk_embs,
            chunks=chunks
        )
    
    def _chunk_text(self, text: str, window_size: int = 4) -> List[str]:
        """Split text into overlapping chunks of words"""
        words = text.split()
        if len(words) <= window_size:
            return [text]
            
        chunks = []
        for i in range(len(words) - window_size + 1):
            chunk = " ".join(words[i:i+window_size])
            chunks.append(chunk)
            
        return chunks
