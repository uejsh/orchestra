
import time
import random
from typing import Dict, Any
from orchestra import enhance, OrchestraConfig

# Mock LLM to simulate expensive/slow calls
class MockOpenAI:
    def __init__(self):
        self.call_count = 0
    
    def chat(self, query: str):
        self.call_count += 1
        # Simulate network latency (2 seconds)
        time.sleep(1.0) 
        return {
            "content": f"Processed: {query}",
            "usage": {"total_tokens": 100}
        }

# Adapter to make it compatible with Orchestra (conceptually)
# In reality, Orchestra wraps LangChain/LangGraph, but let's simulate a simple callable
class MockChain:
    def __init__(self):
        self.llm = MockOpenAI()
        
    def invoke(self, inputs: Dict[str, Any], config=None, **kwargs):
        return self.llm.chat(inputs["query"])

def run_benchmark():
    print("🚀 Starting Orchestra Benchmark Demo...\n")
    
    # 1. Setup Orchestra with a local backend
    config = OrchestraConfig(
        similarity_threshold=0.90
    )
    
    # Wrap our mock chain
    original_chain = MockChain()
    agent = enhance(original_chain, config=config)
    
    queries = [
        "What is the capital of France?",
        "Explain quantum computing", 
        "What is the capital of France?", # Exact repeat
        "Tell me about quantum physics",  # Semantic repeat (simulated)
    ]
    
    print(f"{'Query':<40} | {'Type':<10} | {'Time (s)':<10} | {'Status'}")
    print("-" * 80)
    
    start_total = time.time()
    
    for q in queries:
        start = time.time()
        result = agent.invoke({"query": q})
        duration = time.time() - start
        
        # Simple heuristic to guess if it was cached (Orchestra usually returns metadata, 
        # but for this visual demo we verify by speed)
        is_cached = duration < 0.1
        status = "✅ CACHED" if is_cached else "🔄 FETCHED"
        type_lbl = "Repeat" if "France" in q and queries.index(q) > 0 else "New"
        if "physics" in q: type_lbl = "Semantic"
            
        print(f"{q:<40} | {type_lbl:<10} | {duration:<10.4f} | {status}")

    total_time = time.time() - start_total
    print("-" * 80)
    print(f"\nTotal Run Time: {total_time:.4f}s")
    print("Note: 'FETCHED' took ~1.0s (simulated latency). 'CACHED' took <0.1s.")

if __name__ == "__main__":
    run_benchmark()
