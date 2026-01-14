
import unittest
import logging
import numpy as np
from unittest.mock import MagicMock, patch
from orchestra.eval import FuzzyAssert

# Configure logging
logging.basicConfig(level=logging.ERROR) 

class TestFuzzyAssert(unittest.TestCase):
    
    @patch('orchestra.eval.asserts.FuzzyAssert._get_embedder')
    def test_logic_flow(self, mock_get_embedder):
        """Verify the similarity logic works given known vectors"""
        print("\n--- Test Logic Flow (Mocked) ---")
        
        # Setup mock
        mock_instance = MagicMock()
        mock_get_embedder.return_value = mock_instance
        
        # Maps text to specific vectors
        vector_map = {
            "A": np.array([1.0, 0.0]),
            "A_Exact": np.array([1.0, 0.0]),
            "B": np.array([0.0, 1.0]),
            "A_Close": np.array([0.9, 0.435]), # Sim ~0.9
        }
        
        def generate_side_effect(text, normalize=True):
            if isinstance(text, list):
                return [vector_map.get(t, np.array([0.0, 0.0])) for t in text]
            return vector_map.get(text, np.array([0.0, 0.0]))
            
        mock_instance.generate.side_effect = generate_side_effect
        
        # 1. Exact Match
        FuzzyAssert.similar("A", "A_Exact", threshold=0.99)
        
        # 2. Semantic Match (Sim ~0.9 > Threshold 0.85)
        FuzzyAssert.similar("A", "A_Close", threshold=0.85)
        
        # 3. Failure (Sim 0.0 < Threshold 0.5)
        with self.assertRaisesRegex(AssertionError, "Score: 0.0000"):
            FuzzyAssert.similar("A", "B", threshold=0.5)
            
        # 4. Negation (Sim 0.0 < Threshold 0.80 -> Pass)
        FuzzyAssert.not_similar("A", "B", threshold=0.80)
        
        print("✅ Mocked logic verification passed")

    def test_integration_fallback(self):
        """Verify it runs without crashing even with dummy embeddings"""
        print("\n--- Test Integration (Fallback) ---")
        try:
            FuzzyAssert.similar("foo", "foo", threshold=0.9)
        except AssertionError:
            pass

if __name__ == "__main__":
    unittest.main()
