# 🎵 Orchestra

**Reduce your AI orchestration costs by 85% with one line of code.**

[![PyPI](https://img.shields.io/pypi/v/orchestra-llm-cache.svg)](https://pypi.org/project/orchestra-llm-cache/)
[![Python](https://img.shields.io/pypi/pyversions/orchestra-llm-cache.svg)](https://pypi.org/project/orchestra-llm-cache/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Orchestra adds intelligent **semantic caching** to LangGraph, LangChain, and other AI frameworks - **without changing your code**. It understands the *meaning* of your requests, allowing it to reuse results even for slightly different phrasing.

---

## 🚀 Why Orchestra?

- **Extreme Cost Savings**: Stop paying for semantically identical LLM calls.
- **Lightning Performance**: Sub-second responses (<0.5s) for cached results.
- **Zero-Code Integration**: Wrap your existing LangGraph or LangChain objects in one line.
- **Production-Ready**: Built-in support for Redis, thread-safety, and compression.
- **Local Debugging**: Beautiful trace inspection via the CLI, no cloud needed.

| Feature | Orchestra | Standard Cache |
|---------|-----------|----------------|
| **Matching** | Semantic (Meaning) | Exact String Match |
| **Storage** | Local/Redis Stack | Local Memory |
| **Compression** | Hierarchical (90% reduction) | None |
| **Observability** | CLI Tracing | Logging |

---

## 📦 Installation

```bash
# Recommended: full install
pip install orchestra-llm-cache[full]

# Framework specific
pip install orchestra-llm-cache[langgraph]
pip install orchestra-llm-cache[langchain]
```

---

## ⚡ Quick Start

### With LangGraph

```python
from langgraph.graph import StateGraph, START
from orchestra import enhance

# 1. Define your graph normally
graph = StateGraph(State)
graph.add_node("analyze", analyze_node)
graph.add_edge(START, "analyze")

# 2. ✨ Add Orchestra (ONE LINE)
graph = enhance(graph.compile())

# 3. Use normally - caching happens automatically
result = graph.invoke({"query": "Analyze sales data"})
```

### With LangChain

```python
from langchain.chains import LLMChain
from orchestra import enhance

chain = LLMChain(llm=llm, prompt=prompt)
chain = enhance(chain)  # ✨ Add semantic caching

# Same API, now cached
result = chain.run("Show me Q4 trends")
```

---

## 🛠️ Advanced Features

### 1. 🔍 Hierarchical Embeddings
Standard semantic caching looks at the whole query. For long or complex queries, Orchestra can split the input into chunks and match both the **full context** and the **individual concepts**.

```python
from orchestra import OrchestraConfig

config = OrchestraConfig(
    enable_hierarchical=True,
    hierarchical_weight_l1=0.6, # Weight for full query
    hierarchical_weight_l2=0.4  # Weight for sub-concepts
)
graph = enhance(graph, config=config)
```

### 2. ⏳ Time Windows (Freshness)
Some data is only relevant for a specific time (e.g., "Current stock price"). Orchestra allows you to ignore cache hits older than a certain window.

```python
# Only use cache if it was stored in the last 10 minutes
result = graph.invoke(input, time_window_seconds=600)
```

### 3. 🗜️ Automatic State Compression
Orchestra can compress large state objects using zlib before storing them, perfect for complex LangGraph states.

```python
config = OrchestraConfig(enable_compression=True)
```

### 4. 🔗 Redis Stack (Production)
For distributed environments, use Redis Stack as a backend. This leverages Redis's native vector search capabilities.
```python
# Requires: Redis Stack (with RediSearch)
config = OrchestraConfig(redis_url="redis://localhost:6379")
```

---

## 🏭 Production Deployment

Orchestra is designed to scale with you. While the default SQLite/NumPy backend is perfect for development, you should switch to robust shared backends for production.

### 1. Centralized Tracing (PostgreSQL)
To persist traces in a central database instead of a local file, initialize the recorder with `PostgresStorage`.

```python
from orchestra.recorder import OrchestraRecorder, PostgresStorage
import os

# Initialize ONCE at application startup
OrchestraRecorder(
    storage=PostgresStorage(
        dsn=os.getenv("DATABASE_URL"),
        pool_size=20
    )
)

# Then use enhance() as normal
graph = enhance(graph)
```

### 2. Distributed Cache (Redis)
Ensure all your instances share the same semantic cache.

```python
config = OrchestraConfig(
    redis_url=os.getenv("REDIS_URL"), # e.g. "redis://utils-cache:6379"
    enable_compression=True           # Recommended for Redis to save network I/O
)
```

### 3. Graceful Shutdown
Orchestra includes a lifecycle manager that ensures background workers flush their data before the app exits. This works automatically on `SIGTERM` / `SIGINT`.

---

## 🛡️ Resilience & Reliability

Orchestra includes built-in patterns to protect your application from LLM failures.

### Circuit Breaker
Prevent cascading failures when your LLM provider is down or experiencing high latency. The circuit breaker will "open" (fail fast) after a threshold of errors, giving the downstream system time to recover.

```python
config = OrchestraConfig(
    enable_circuit_breaker=True,
    circuit_breaker_threshold=5,  # Open after 5 failures
    circuit_breaker_timeout=60.0  # Wait 60s before retrying
)
```

If the circuit is open, `enhance()`'d objects will raise a `CircuitBreakerOpenError` immediately without calling the LLM. **Cache hits will still be returned even if the circuit is open.**

## 🧪 Evaluation & Testing

Orchestra enables **Semantic Regression Testing**. You can write tests that assert LLM outputs are "semantically similar" to a baseline, without requiring exact string matches.

```python
from orchestra.eval import FuzzyAssert

# 1. Define expectations
expected = "The user has 3 active accounts."
actual = agent_output # e.g., "User currently holds three open accounts."

# 2. Fuzzy Assertion
# Passes if meaning is preserved (Cosine Sim > 0.92)
FuzzyAssert.similar(actual, expected)

# 3. Guardrails (Negation)
# Passes if meaning is DIFFERENT (Cosine Sim < 0.6)
FuzzyAssert.not_similar(actual, "I cannot help with that", threshold=0.6)
```

**Note:** This feature requires `sentence-transformers` installed (`pip install sentence-transformers`).

## 🎥 Orchestra Recorder & CLI

Debug your agents locally with built-in tracing. Works automatically for LangGraph nodes and LangChain calls.

1. **Run your code** (traces are saved to `orchestra_traces.db` locally).
2. **List traces**:
   ```bash
   python -m orchestra.cli trace ls
   ```
3. **Inspect a specific trace**:
   ```bash
   python -m orchestra.cli trace view <TRACE_ID>
   ```
4. **Cleanup old traces**:
   ```bash
   ```bash
   python -m orchestra.cli trace prune --days 30
   ```
5. **Quick Eval Check**:
   ```bash
   python -m orchestra.cli eval "Text A" "Text B"
   ```

---

## ⚙️ Configuration Reference

| Option | Default | Description |
|--------|---------|-------------|
| `similarity_threshold` | `0.92` | How similar queries must be (0-1). |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence transformer model to use. |
| `cache_ttl` | `3600` | How long entries stay in cache (seconds). |
| `max_cache_size` | `10000` | Max entries before oldest are evicted. |
| `redis_url` | `None` | URL for Redis Stack backend. |
| `enable_compression` | `False` | Enable zlib compression for states. |
| `enable_circuit_breaker` | `False` | Enable resilience pattern. |
| `circuit_breaker_threshold` | `5` | Failures before opening circuit. |
| `circuit_breaker_timeout` | `60.0` | Seconds to wait before retrying (Half-Open). |
| `auto_cleanup` | `True` | Automatically remove expired items. |
| `llm_cost_per_1k_tokens` | `0.03` | Cost factor for metrics estimation. |

---

## ❓ Troubleshooting & FAQ

### **FAISS Issues on Windows**
FAISS can sometimes be difficult to install on Windows. Orchestra includes a **NumPy-based fallback** that works out of the box. If you see warnings about FAISS, don't worry—Orchestra is still working using the fallback index.

### **Cache is too "loose" or too "strict"**
Adjust the `similarity_threshold`. 
- Increase (e.g., `0.98`) for strict matching (more accurate, fewer hits).
- Decrease (e.g., `0.85`) for loose matching (more hits, potentially lower accuracy).

### **Does it work with LangGraph Cloud?**
Orchestra is designed for self-hosted and local environments. Support for LangGraph Cloud and other managed platforms is on our roadmap.

---

## 📈 Benchmarks

Run the included benchmark to see savings on your workload:
```bash
python benchmarks/langgraph_benchmark.py --queries 100 --iterations 5
```

---

## 📜 License & Citation

MIT License. See [LICENSE](LICENSE) for details.

```bibtex
@software{orchestra2024,
  title={Orchestra: Semantic Caching for AI Orchestration},
  author={Orchestra Team},
  year={2024},
  url={https://github.com/uejsh/orchestra}
}
```

---
**Built with ❤️ by the Orchestra Team**
