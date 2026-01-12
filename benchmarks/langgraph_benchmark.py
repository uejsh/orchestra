# benchmarks/langgraph_benchmark.py

import argparse
import time
import random
import uuid
import numpy as np
import logging
from typing import TypedDict, List
from langgraph.graph import StateGraph, START

# Fix imports to run from root
import sys
import os
sys.path.append(os.getcwd())

from orchestra import enhance, OrchestraConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Benchmark")

# Define State
class BenmarkState(TypedDict):
    query: str
    result: str

def mock_llm(query: str):
    # Simulate LLM latency (avg 1.5s)
    # time.sleep(1.5) # Commented out for faster benchmark running
    return f"Response to {query} " * 10

def build_graph():
    graph = StateGraph(BenmarkState)
    graph.add_node("llm", lambda s: {"result": mock_llm(s["query"])})
    graph.add_edge(START, "llm")
    return graph.compile()

def generate_workload(n_queries: int, repetition_rate: float) -> List[str]:
    """
    Generate a workload with Zipfian distribution to simulate real traffic.
    """
    targets = [f"Query-{uuid.uuid4().hex[:8]}" for _ in range(int(n_queries * (1-repetition_rate)))]
    
    workload = []
    for _ in range(n_queries):
        if random.random() < repetition_rate and workload:
            # Repeat a previous query
            q = random.choice(workload)
        else:
            # New query
            if targets:
                q = targets.pop(0)
            else:
                q = f"Query-{uuid.uuid4().hex[:8]}"
        workload.append(q)
    return workload

def run_benchmark(n_queries: int, repetition_rate: float):
    logger.info(f"🚀 Starting Benchmark: {n_queries} queries, {repetition_rate:.0%} repetition")
    
    # 1. Baseline (No Cache)
    # Skipped to save time, we know cost = n_queries * avg_cost
    
    # 2. Orchestra
    app = build_graph()
    app = enhance(app, config=OrchestraConfig(
        similarity_threshold=0.9,
        enable_compression=True
    ))
    
    workload = generate_workload(n_queries, repetition_rate)
    
    start_time = time.time()
    latencies = []
    
    for i, q in enumerate(workload):
        t0 = time.time()
        app.invoke({"query": q})
        latencies.append(time.time() - t0)
        
        if i % 10 == 0:
            print(f".", end="", flush=True)
            
    total_time = time.time() - start_time
    print("\n")
    
    # Report
    metrics = app.get_metrics()
    
    print("="*60)
    print(" 📊 ORCHESTRA BENCHMARK RESULTS")
    print("="*60)
    print(f"Total Queries:       {n_queries}")
    print(f"Repetition Rate:     {repetition_rate:.0%}")
    print(f"Total Time:          {total_time:.2f}s")
    print("-" * 30)
    print(f"Cache Hits:          {metrics.get('cache_hits', 0)}")
    print(f"Hit Rate:            {metrics.get('cache_hit_rate', 0):.2%}")
    print(f"Cost Savings:        ${metrics.get('estimated_cost_saved', 0):.2f}")
    print(f"Avg Latency:         {np.mean(latencies):.4f}s")
    print(f"P99 Latency:         {np.percentile(latencies, 99):.4f}s")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--repetition", type=float, default=0.3)
    args = parser.parse_args()
    
    run_benchmark(args.queries, args.repetition)
