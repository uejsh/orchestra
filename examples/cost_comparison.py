# examples/cost_comparison.py

import sys
import os
sys.path.append(os.getcwd())

import time
import random
from orchestra.core.metrics import MetricsTracker
# We'll use a mocked "expensive" function
def expensive_op(query):
    time.sleep(1) # 1 sec latency
    return f"Analysis: {query}"

class VanillaRunner:
    def __init__(self):
        self.cost = 0.0
        self.latency = 0.0
    
    def run(self, query):
        start = time.time()
        res = expensive_op(query)
        self.latency += (time.time() - start)
        self.cost += 0.05 # $0.05 per call
        return res

class OrchestraRunner:
    def __init__(self):
        # We'd use the real enhance() normally, but for a 
        # pure math simulation we'll simulate the hit rate behavior
        from orchestra.core.semantic_store import SemanticStore
        from orchestra.core.embeddings import EmbeddingGenerator
        
        self.store = SemanticStore()
        self.emb = EmbeddingGenerator()
        
        self.cost = 0.0
        self.latency = 0.0
    
    def run(self, query):
        start = time.time()
        
        # Check cache
        vec = self.emb.generate(query)
        hits = self.store.search(vec, top_k=1)
        
        if hits:
            # Hit
            self.latency += 0.05 # fast
            return hits[0][0].value
        else:
            # Miss
            res = expensive_op(query)
            self.store.put(query, res, vec) # Store it
            self.latency += (time.time() - start)
            self.cost += 0.05
            return res

# Simulation Data
queries = [
    "Q4 Sales", "Q4 Sales Report", "Sales for Q4", # Similar
    "HR Policy", "Human Resources",                # Similar
    "Inventory Check", "Stock Level",              # Similar
    "New Marketing Plan"                           # Unique
]
# Expand to 100 queries with weighted random choice to simulate Zipfian distribution
workload = random.choices(queries, k=100)

print("Starting Cost Simulation (100 Queries)...")

# 1. Vanilla
vanilla = VanillaRunner()
print("Running Vanilla...", end="", flush=True)
for q in workload:
    # We won't actually sleep 100s for the demo, just simulate the math
    vanilla.cost += 0.05
print(" Done.")

# 2. Orchestra
print("Running Orchestra...", end="", flush=True)
orch = OrchestraRunner()
# For the demo, we'll actually run the cache logic to prove semantic matching
# But we'll mock the expensive_op sleep to be fast for the USER's sake
def expensive_op(query): return f"Analysis: {query}" 

for q in workload:
    orch.run(q)
print(" Done.")

print(f"\nResults:")
print(f"Vanilla Cost:   ${vanilla.cost:.2f}")
print(f"Orchestra Cost: ${orch.cost:.2f}")
print(f"SAVINGS:        ${vanilla.cost - orch.cost:.2f} ({100 * (1 - orch.cost/vanilla.cost):.1f}%)")
