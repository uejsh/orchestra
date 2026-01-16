# 🎵 Orchestra

**Unbreakable Agents. 85% Lower Costs. One Line of Code.**

[![PyPI](https://img.shields.io/pypi/v/orchestra-llm-cache.svg)](https://pypi.org/project/orchestra-llm-cache/)
[![GitHub Stars](https://img.shields.io/github/stars/uejsh/orchestra.svg?style=social)](https://github.com/uejsh/orchestra)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

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
    Cache --> Return([Instant Response < 0.5s])
    Match -- "New Query" --> CB{Circuit Breaker}
    CB -- "Closed" --> LLM[LLM Provider]
    LLM --> Return
    CB -- "Open" --> Error[Graceful Fallback]
```

---

## ⚡ Why Orchestra?

*   **📉 Slash OpenAI/Anthropic Bills**: Stop paying for the same questions phrased differently.
*   **🏎️ Lightning Performance**: Sub-second responses for cached "loophole" hits.
*   **🛡️ Outage Insurance**: Built-in **Circuit Breakers** keep your app alive even if your LLM provider goes down.
*   **🔌 Smart Tool Selection**: Automatically filters 100+ tools down to the relevant 5, saving thousands of context tokens.
*   **🛠️ Framework Native**: Works with LangGraph, LangChain, and any Python LLM client.

---

## 📦 Installation

```bash
pip install orchestra-llm-cache[full]
```

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

### langchain
```python
from orchestra import enhance

# Works with any Runnable or Chain
chain = enhance(my_rag_chain)

result = chain.invoke("Explain the merger")
```

---

## 🛡️ Enterprise Resilience (Circuit Breaker)
Stop cascading failures. If your LLM provider starts timing out, Orchestra opens the circuit, failing fast and serving cached results until the provider recovers.

```python
from orchestra import OrchestraConfig

config = OrchestraConfig(
    enable_circuit_breaker=True,
    circuit_breaker_threshold=5, 
    circuit_breaker_timeout=60.0
)
agent = enhance(graph, config=config)
```

---

## 🔌 Smart Tool Discovery (MCP)
Are you loading 50+ tools into Claude? You're burning money. Orchestra indexes your MCP tools semantically and only injects the ones relevant to the current user query.

```python
config = OrchestraConfig(
    enable_tool_search=True,
    tool_search_top_k=5  # Only show the best 5 tools
)
```

---

## 💾 Production Backends
| Backend | Best For | Description |
|---------|----------|-------------|
| **SQLite** | Development | Local, file-based, zero-config. |
| **PostgreSQL** | Tracing | Centralized telemetry for large teams. |
| **Redis Stack** | Distributed | Global semantic cache shared across pods. |

---

## 🎥 CLI & Observability
Debug your agents with `trace` logs.

```bash
# View recent executions
python -m orchestra.cli trace ls

# Compare two strings semantically
python -m orchestra.cli eval "Hello" "Hi there"
```

---

## ❓ FAQ
**Q: How accurate is the semantic matching?**  
A: By default, we use a 0.92 cosine similarity threshold. It's high enough to ensure accuracy but loose enough to catch "How's the weather?" vs "What's the weather like?".

**Q: Does this work with exact matches?**  
A: Yes. If a query is identical, it hits the cache instantly. If it's a paraphrase, we use vector embeddings.

---
**Start building unbreakable, affordable AI agents today. 🌟 Give us a star on GitHub!**
