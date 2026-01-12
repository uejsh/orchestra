# examples/langgraph_quickstart.py

import sys
import os
sys.path.append(os.getcwd())

import time
import random
from typing import Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from orchestra import enhance

# 1. Define State
class State(TypedDict):
    topic: str
    report: str

# 2. Mock LLM (Simulation)
def mock_llm_call(query: str) -> str:
    """Simulates an expensive LLM call (2 seconds)"""
    print(f"    🤖 LLM 'thinking' about {query}...", end="", flush=True)
    time.sleep(2.0) # Simulate latency
    print(" Done!")
    return f"Report on {query}: Market is growing by {random.randint(5, 15)}%."

# 3. Define Node
def generator_node(state: State):
    result = mock_llm_call(state["topic"])
    return {"report": result}

# 4. Create Graph
workflow = StateGraph(State)
workflow.add_node("generate", generator_node)
workflow.add_edge(START, "generate")
workflow.add_edge("generate", END)

# 5. Compile & Enhance
app = workflow.compile()
app = enhance(app) # ✨ The Magic Line

# 6. Run
print("\n--- Execute 1: Cold Cache ---")
start = time.time()
res1 = app.invoke({"topic": "AI Agents"})
print(f"Result: {res1['report']}")
print(f"Time: {time.time() - start:.2f}s")

print("\n--- Execute 2: Warm Cache (Same Query) ---")
start = time.time()
res2 = app.invoke({"topic": "AI Agents"})
print(f"Result: {res2['report']}")
print(f"Time: {time.time() - start:.2f}s (Should be ~0.0s)")

print("\n--- Execute 3: Semantic Match (Similar Query) ---")
start = time.time()
res3 = app.invoke({"topic": "artificial intelligence agents"})
print(f"Result: {res3['report']}")
print(f"Time: {time.time() - start:.2f}s (Should be ~0.0s)")

# 7. Metrics
print("\n--- Metrics ---")
print(app.get_metrics())
