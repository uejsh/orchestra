# 🎵 Orchestra

![Orchestra Hero](assets/hero_banner.png)

**Unbreakable Agents. 85% Lower Costs. One Line of Code.**

[![PyPI](https://img.shields.io/pypi/v/orchestra-llm-cache.svg)](https://pypi.org/project/orchestra-llm-cache/)
[![GitHub Stars](https://img.shields.io/github/stars/uejsh/orchestra.svg?style=social)](https://github.com/uejsh/orchestra)
[![Downloads](https://img.shields.io/pypi/dm/orchestra-llm-cache?color=blueviolet)](https://pypi.org/project/orchestra-llm-cache/)
[![Cost Saved](https://img.shields.io/badge/Cost%20Saved-85%25-green)](https://github.com/uejsh/orchestra)
[![Latency](https://img.shields.io/badge/Latency-%3C50ms-brightgreen)](https://github.com/uejsh/orchestra)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

### 📚 Table of Contents

- [Installation](#%EF%B8%8F-installation)
- [Quick Start](#-1-line-quick-start)
- [Enterprise Features](#-enterprise-features)
- [Configuration Reference](#%EF%B8%8F-configuration-reference)
- [Production Architecture (Redis/Postgres)](#-production-architecture)
- [Multi-Agent Metrics & Observability](#-multi-agent-metrics--observability)
- [CLI Reference](#-cli-reference)
- [FAQ](#-faq)

---

## 🛠️ Installation

```bash
pip install orchestra-llm-cache
```

*Note: Requires Python 3.9+ and optionally `faiss-cpu` or `redis` for production backends.*

---

### 📺 Watch: How to drop your LLM bill by 85% instantly

[![Orchestra Demo](https://img.youtube.com/vi/TaIGvoKuWZs/0.jpg)](https://www.youtube.com/watch?v=TaIGvoKuWZs)

---

## 🚀 The "Zero-Rewrite" AI Framework

Orchestra is a high-performance orchestration layer that adds **Semantic Caching**, **Outage Resilience**, and **Smart Tool Discovery** to LangGraph and LangChain.

It understands the *meaning* of requests, skipping the LLM entirely when a semantically similar query has already been answered.

```mermaid
graph TD
    User([User Query]) --> Match{Semantic Match?}
    Match -- "92% Similar" --> Cache[(Semantic Cache)]
    Cache --> Return([Instant Response < 0.1s])
    Match -- "New Query" --> CB{Circuit Breaker}
    CB -- "Closed" --> LLM[LLM Provider]
    LLM --> Return
    CB -- "Open" --> Error[Graceful Fallback]
```

---

## 🧠 How it Works: Short-Circuit vs. Injection

Orchestra utilizes two distinct semantic strategies to optimize your agents:

1.  **Short-Circuiting (Semantic Cache)**: When a query is matched in the cache, Orchestra returns the response **instantly** without calling the LLM. It does *not* inject the result back into the LLM; it replaces the LLM call entirely. **Result:** 0ms LLM latency and $0 cost.
2.  **Semantic Injection (Smart Tools)**: For new queries, Orchestra semantically "searches" your tool library and **injects** only the most relevant tools into the LLM's system prompt. **Result:** Dramatically smaller context windows and higher reasoning accuracy.

---

## ⚡ Why Orchestra?

*   **📉 Slash Bills by 85%**: Deduplicate similar queries (e.g., "What's the weather?" vs "How is the weather?") using **Hierarchical Embeddings**.
*   **🏎️ Lightning Performance**: Sub-50ms responses for cached hits, bypassing slow LLM generation.
*   **🛡️ Outage Insurance**: **Circuit Breakers** automatically detect provider failures and switch to fallback modes or cached responses.
*   **🔌 Smart Tool Discovery (MCP)**: Automatically filters 100+ tools down to the relevant 5, saving thousands of context tokens.
*   **🎥 Time-Travel Debugging**: The **Recorder** logs every step, allowing you to replay and inspect agent states exactly as they happened.

---

## 🧠 Solving the "Long Context Token Tax"
As agents become more complex, their system prompts bloat with tools, instructions, and history. Orchestra solves this via:

1.  **Dynamic Tool Pruning**: Instead of injecting 50 tools into every call, Orchestra semantically selects the **top 5 relevant tools**. This can reduce context usage by **10,000+ tokens** per turn.
2.  **Semantic Memory (Cross-Session)**: By caching LLM responses semantically, Orchestra "remembers" similar complex reasoning paths across different users and sessions, preventing redundant long-context processing.
3.  **Diff-Based Tracing**: The **Recorder** tracks state *mutations* rather than full snapshots, making it easier to analyze long-running agent loops without information overload.

---

## ⚡ The Orchestra Difference

### langgraph

```python
from orchestra import enhance

# Wrap your compiled graph. That's it.
# Orchestra now handles Caching, Tracing, and Resilience.
agent = enhance(app.compile())

result = agent.invoke({"query": "What are our Q3 goals?"})
```

---

## 🧠 Semantic Context Injection (Self-RAG)

Orchestra can turn your semantic cache into a **long-term memory** source. Instead of just short-circuiting exact (or near-exact) matches, Orchestra can retrieve the top $K$ most relevant past interactions and inject them into the current prompt as context.

### How it works:
1.  **Cache Miss**: If no match exceeds the `similarity_threshold`, the cache isn't short-circuited.
2.  **Memory Retrieval**: Orchestra performs a loose search for the top $K$ relevant past responses.
3.  **Prompt Augmentation**: These past responses are injected into the input (messages or text) before the LLM is called.

### Example:
```python
config = OrchestraConfig(
    enable_context_injection=True,
    context_injection_top_k=2  # Inject 2 most relevant past answers
)
enhanced = enhance(graph, config)
```

---

## 🔬 Advanced Examples

### 1. Smart Tool Discovery (MCP)
Automatically prune your tool list down to contextually relevant ones.

```python
from orchestra import enhance, OrchestraConfig

config = OrchestraConfig(
    mcp_servers=[
        "http://localhost:8000/mcp", # Remote MCP server
        "npx -y @modelcontextprotocol/server-gdrive" # CLI server
    ],
    enable_tool_search=True,
    tool_search_top_k=5
)

# Orchestra will dynamicially search and inject tools
agent = enhance(app.compile(), config)
```

### 2. Hierarchical Semantic Matching
Better deduplication for complex prompts by matching both the whole query and its sub-chunks.

```python
config = OrchestraConfig(
    enable_hierarchical=True,
    hierarchical_weight_l1=0.7, # 70% weight on global match
    hierarchical_weight_l2=0.3  # 30% weight on chunk match
)
```

---

## ⚙️ Configuration Reference

### OrchestraConfig / OrchestraLangChainConfig

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **Semantic Matching** | | | |
| `similarity_threshold` | `float` | `0.92` | Cosine similarity (0-1) for a cache hit. |
| `embedding_model` | `str` | `"all-MiniLM-L6-v2"`| SentenceTransformer model to use. |
| `enable_hierarchical` | `bool` | `False` | L1 + L2 matching (better for long queries). |
| `hierarchical_weight_l1`| `float` | `0.6` | Weight for full-query similarity. |
| `hierarchical_weight_l2`| `float` | `0.4` | Weight for chunk-level similarity. |
| **Caching & Persistence** | | | |
| `enable_cache` | `bool` | `True` | Master switch for semantic caching. |
| `cache_ttl` | `int` | `3600` | Expiration time in seconds. |
| `max_cache_size` | `int` | `10000` | Max entries in local store. |
| `redis_url`| `str` | `None` | Redis Stack URL for shared caching. |
| `enable_compression` | `bool` | `False` | Zlib compression for large values. |
| **Self-RAG (Context Injection)** | | | |
| `enable_context_injection`| `bool` | `False` | Inject similar past results as context. |
| `context_injection_top_k`| `int` | `3` | Number of past matches to inject. |
| `context_injection_template`| `str` | `...` | Prompt template for injected context. |
| **Observability** | | | |
| `enable_recorder` | `bool` | `True` | Enables step-by-step trace logging. |
| `llm_cost_per_1k_tokens`| `float` | `0.03` | Basis for cost savings estimation. |
| **Resilience & Tools** | | | |
| `enable_circuit_breaker`| `bool` | `False` | Prevent provider-timeout cascades. |
| `circuit_breaker_threshold`| `int` | `5` | Failures before killing the circuit. |
| `circuit_breaker_timeout`| `float` | `60.0` | Seconds to wait before retry. |
| `enable_tool_search` | `bool` | `True` | Dynamic tool pruning (Smart Tools). |
| `mcp_servers` | `list` | `None` | List of MCP host configurations. |

---

## 🏗️ Production Architecture

For production environments, move away from local SQLite/NumPy to distributed backends.

### 1. Persistent Semantic Cache (Redis)
Orchestra supports **Redis Stack** as a global semantic store.

```python
from orchestra import enhance, OrchestraConfig

config = OrchestraConfig(
    redis_url="redis://:password@your-redis-stack:6379",
    cache_ttl=86400, # 24 hours
    similarity_threshold=0.95
)

agent = enhance(graph, config)
```

### 2. Scalable Tracing (Postgres)
Transition from local `.orchestra/traces.db` to a centralized PostgreSQL instance.

```python
from orchestra.recorder import OrchestraRecorder
from orchestra.recorder.storage import PostgresStorage
from orchestra import enhance

# 1. Initialize Postgres storage
storage = PostgresStorage(
    dsn="postgresql://user:pass@localhost:5432/orchestra_db",
    pool_size=20
)

# 2. Override the global recorder instance
OrchestraRecorder._instance = OrchestraRecorder(storage=storage)

# 3. Enhance your graph
agent = enhance(graph)
```

---

## 📊 Multi-Agent Metrics & Observability

Orchestra's metrics engine tracks performance in real-time.

### Does it work for multi-agent graphs?
**Yes.**

1.  **Global Aggregation**: If you enhance a Supervisor or entry-point graph, Orchestra captures metrics for the entire session, including all sub-calls to other agents.
2.  **Granular Metrics**: To track a specific agent's efficiency within a swarm, simply `enhance()` that specific agent's graph.

### Accessing Metrics
```python
stats = agent.get_metrics()

print(f"💰 Total Saved: ${stats['estimated_cost_saved']:.4f}")
print(f"⚡ Avg Hit Latency: {stats['avg_cache_hit_latency']:.3f}s")
print(f"📈 Cache Hit Rate: {stats['cache_hit_rate']*100:.1f}%")

# Export for Grafana/ELK
agent.export_metrics("session_stats.json")
```

---

## 🎥 Recorder & Time-Travel
Orchestra records every node execution, input, and output difference.

```bash
# View recent executions
python -m orchestra.cli trace ls

# Inspect specific step-by-step state changes
python -m orchestra.cli trace view <TRACE_ID>
```

---

## 💻 CLI Reference
Orchestra ships with a powerful CLI for inspection and running declarative agents.

```bash
# General
python -m orchestra.cli --help

# 1. Trace Inspection
python -m orchestra.cli trace ls               # List recent traces
python -m orchestra.cli trace view <TRACE_ID>  # Inspect specific trace steps

# 2. Semantic Evaluation
# Returns exit code 0 if similar, 1 if not. Great for CI/CD.
python -m orchestra.cli eval "Hello world" "Hi earth" --threshold 0.9

# 3. Running Agents
python -m orchestra.cli run agent.yaml --query "Hello"
```

---

## ❓ FAQ
**Q: How accurate is semantic matching?**
By default, we use a 0.92 threshold. It's high enough to ensure accuracy but loose enough to catch rephrasings.

**Q: Can I use custom embedding models?**
Yes, pass any SentenceTransformer model name to `embedding_model` in `OrchestraConfig`.

---

**Start building unbreakable, affordable AI agents today. 🌟 Give us a star on GitHub!**