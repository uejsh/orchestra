# asserts.py - Semantic Assertions
# ============================================================================
# FILE: orchestra/eval/asserts.py
# Fuzzy assertions for regression testing using semantic similarity
# ============================================================================

import logging
from typing import Optional, Union, Any
import numpy as np

# Lazy load to avoid overhead if not using eval
_embedder = None

logger = logging.getLogger(__name__)

class FuzzyAssert:
    """
    Assertions for semantic similarity.
    Useful for testing LLM outputs where wording varies but meaning should persist.
    """
    
    @staticmethod
    def _get_embedder():
        global _embedder
        if _embedder is None:
            from ..core.embeddings import EmbeddingGenerator
            # Use same default as CacheManager
            _embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        return _embedder

    @classmethod
    def similar(
        cls, 
        actual: str, 
        expected: str, 
        threshold: float = 0.92, 
        msg: Optional[str] = None
    ):
        """
        Assert that two strings are semantically similar.
        
        Args:
            actual: The output from the LLM/Agent
            expected: The 'golden' reference string
            threshold: Similarity score (0-1) required to pass (default: 0.92)
            msg: Optional failure message
            
        Raises:
            AssertionError: If similarity is below threshold
        """
        embedder = cls._get_embedder()
        
        emb_actual = embedder.generate(actual)
        emb_expected = embedder.generate(expected)
        
        # Calculate Cosine Similarity
        # (Vectors are typically normalized by the generator, but let's be safe)
        norm_a = np.linalg.norm(emb_actual)
        norm_e = np.linalg.norm(emb_expected)
        
        if norm_a == 0 or norm_e == 0:
            score = 0.0
        else:
            score = np.dot(emb_actual, emb_expected) / (norm_a * norm_e)
            
        score = float(score)
        
        if score < threshold:
            standard_msg = (
                f"Semantic similarity failed.\n"
                f"Score: {score:.4f} (Threshold: {threshold})\n"
                f"Expected: {expected}\n"
                f"Actual:   {actual}"
            )
            raise AssertionError(msg or standard_msg)
            
        logger.info(f"✅ Semantic Assert Passed: Score {score:.4f}")
        return True

    @classmethod
    def not_similar(
        cls, 
        actual: str, 
        reference: str, 
        threshold: float = 0.80, 
        msg: Optional[str] = None
    ):
        """Assert that two strings are DIFFERENT (e.g. testing guardrails)."""
        try:
            cls.similar(actual, reference, threshold)
        except AssertionError:
            return True # Failed similarity means passed negation
            
        raise AssertionError(msg or f"Strings were too similar (Score > {threshold})")
