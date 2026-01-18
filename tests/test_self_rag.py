
import logging
import time
from typing import Any, Dict, Optional
from orchestra.adapters.langchain import enhance, OrchestraLangChainConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockChain:
    """Mock LangChain chain that just returns a fixed response but captures input."""
    def __init__(self):
        self.last_input = None
    
    def invoke(self, input_data: Any, config: Optional[Dict] = None, **kwargs) -> str:
        self.last_input = input_data
        return "Mocked response based on context."

def test_self_rag_string_injection():
    print("\n--- Testing String Injection ---")
    mock = MockChain()
    config = OrchestraLangChainConfig(
        enable_context_injection=True,
        context_injection_top_k=2,
        similarity_threshold=0.99  # High threshold to guarantee a miss for RAG test
    )
    
    enhanced = enhance(mock, config)
    
    # 1. Populate cache
    enhanced.cache_manager.put("The capital of France is Paris.", "Paris is the capital of France.")
    enhanced.cache_manager.put("The population of Paris is 2 million.", "About 2.1 million people live in Paris.")
    
    # 2. Run a related query
    # "Tell me about France's capital" should be similar but not 0.99 match
    query = "Tell me about France's capital"
    enhanced.invoke(query)
    
    # 3. Verify injection
    last_input = mock.last_input
    print(f"Final Input Sent to LLM:\n{last_input}")
    
    assert "START OF RELEVANT CONTEXT" in last_input
    assert "Paris" in last_input
    print("✅ String injection verified!")

def test_self_rag_dict_injection():
    print("\n--- Testing Dict Injection ---")
    mock = MockChain()
    config = OrchestraLangChainConfig(
        enable_context_injection=True,
        context_injection_top_k=2,
        similarity_threshold=0.99
    )
    
    enhanced = enhance(mock, config)
    
    # 1. Populate cache (using similar strings for better matching)
    enhanced.cache_manager.put("The capital of France is Paris.", "Paris")
    enhanced.cache_manager.put("France's currency is the Euro.", "Euro")
    
    # 2. Run a related query using dict
    query_dict = {"query": "What is the capital of France?"}
    enhanced.invoke(query_dict)
    
    # 3. Verify injection in dict
    last_input = mock.last_input
    print(f"Final Dict Sent to LLM: {last_input}")
    
    assert "query" in last_input
    assert "START OF RELEVANT CONTEXT" in last_input["query"]
    assert "Paris" in last_input["query"]
    print("✅ Dict injection verified!")

if __name__ == "__main__":
    try:
        test_self_rag_string_injection()
        test_self_rag_dict_injection()
        print("\n✨ All Self-RAG tests PASSED!")
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
