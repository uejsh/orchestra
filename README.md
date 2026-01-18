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

- [Quick Start](#-1-line-quick-start)
- [Enterprise Features](#-enterprise-features)
- [Configuration API Reference](#%EF%B8%8F-configuration-api-reference)
- [Production Architecture (Redis/Postgres)](#-production-architecture)
- [Multi-Agent Metrics & Observability](#-multi-agent-metrics--observability)
- [CLI Reference](#-cli-reference)
- [FAQ](#-faq)

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

## ⚡ Why Orchestra?

*   **📉 Slash Bills by 85%**: Deduplicate similar queries (e.g., "What's the weather?" vs "How is the weather?") using **Hierarchical Embeddings**.
*   **🏎️ Lightning Performance**: Sub-50ms responses for cached hits, bypassing slow LLM generation.
*   **🛡️ Outage Insurance**: **Circuit Breakers** automatically detect provider failures and switch to fallback modes or cached responses.
*   **🔌 Smart Tool Discovery (MCP)**: Automatically filters 100+ tools down to the relevant 5, saving thousands of context tokens.
*   **🎥 Time-Travel Debugging**: The **Recorder** logs every step, allowing you to replay and inspect agent states exactly as they happened.

---

## ⚡ 1-Line Quick Start

### langgraph

```python
from orchestra import enhance

# Wrap your compiled graph. That's it.
# Orchestra now handles Caching, Tracing, and Resilience.
agent = enhance(app.compile())

result = agent.invoke({"query": "What are our Q3 goals?"})
```

---

## 🛠️ Configuration API Reference

Pass `OrchestraConfig` to `enhance()` to customize behavior.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **Semantic Matching** | | | |
| `similarity_threshold` | float | 0.92 | Cosine similarity required for a cache hit. |
| `embedding_model` | str | `"all-MiniLM-L6-v2"` | SentenceTransformer model for vector generation. |
| `enable_hierarchical` | bool | `False` | Enables L1 (Global) + L2 (Chunked) matching for high accuracy. |
| `hierarchical_weight_l1` | float | 0.6 | Weight for whole-query match. |
| `hierarchical_weight_l2` | float | 0.4 | Weight for individual phrase matching. |
| **Caching & Storage** | | | |
| `enable_cache` | bool | `True` | Master switch for semantic caching. |
| `cache_ttl` | int | 3600 | Cache lifetime in seconds. |
| `max_cache_size` | int | 10000 | Max entries in local store (if not using Redis). |
| `redis_url` | str | `None` | Redis Stack connection URL for distributed caching. |
| `enable_compression` | bool | `False` | Zlib compression for cached values. |
| **Resilience & Tools** | | | |
| `enable_circuit_breaker`| bool | `False` | Prevents cascading failures during outages. |
| `circuit_breaker_threshold`| int | 5 | Failed attempts before opening circuit. |
| `circuit_breaker_timeout`| float | 60.0 | Seconds before retrying the provider. |
| `enable_tool_search` | bool | `True` | Dynamic tool discovery via MCP. |
| `mcp_servers` | list | `None` | List of MCP server configs (standard MCP JSON format). |
| **Observability** | | | |
| `enable_recorder` | bool | `True` | Enables step-by-step trace logging. |
| `llm_cost_per_1k_tokens`| float | 0.03 | Basis for cost savings dashboard. |

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