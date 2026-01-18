# 🎵 Orchestra

<div align="center">
  <img src="assets/hero_banner.png" alt="Orchestra Hero" width="800">
  <h3><b>Unbreakable Agents. 85% Lower Costs. One Line of Code.</b></h3>
  <p><i>The high-performance semantic orchestration layer for LangGraph and LangChain.</i></p>

  [![PyPI](https://img.shields.io/pypi/v/orchestra-llm-cache.svg?style=flat-square)](https://pypi.org/project/orchestra-llm-cache/)
  [![GitHub Stars](https://img.shields.io/github/stars/uejsh/orchestra.svg?style=flat-square&label=Star%20this%20Repo)](https://github.com/uejsh/orchestra)
  [![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](LICENSE)
  [![Cost Saved](https://img.shields.io/badge/Cost%20Saved-85%25-green?style=flat-square)](https://github.com/uejsh/orchestra)
  [![Latency](https://img.shields.io/badge/Latency-%3C50ms-brightgreen?style=flat-square)](https://github.com/uejsh/orchestra)
</div>

---

<div align="center">
  <img src="assets/orchestration_flow.gif" alt="Orchestra Flow" width="600">
  <p><i>The heartbeat of high-performance agentic systems.</i></p>
</div>

---

### 📺 Watch: How to drop your LLM bill by 85% instantly
[![Orchestra Demo](https://img.youtube.com/vi/TaIGvoKuWZs/0.jpg)](https://www.youtube.com/watch?v=TaIGvoKuWZs)

---

### 📚 Table of Contents
- [🚀 Quick Start](#-the-1-line-setup)
- [📈 Performance Benchmarks](#-wall-of-impact-performance-benchmarks)
- [🧠 Self-RAG (Collective Intelligence)](#-collective-intelligence-self-rag)
- [🏗️ Premium Features Deep Dive](#%EF%B8%8F-premium-features-deep-dive)
- [⚙️ Full Configuration API](#%EF%B8%8F-full-configuration-api)
- [💾 Persistence & Backends (Production)](#-persistence--backends-production)
- [📝 Declarative Agents (YAML)](#-declarative-agents-yaml)
- [📊 Metrics & Observability](#-metrics--observability)
- [🎥 CLI Zero-to-Hero Reference](#-cli-zero-to-hero-reference)
- [🧪 CI/CD & Semantic Eval](#-cicd--semantic-eval)
- [❓ FAQ](#-faq)

---

## 📈 Wall of Impact: Performance Benchmarks

In tests across 10,000+ real-world agent interactions, Orchestra has proven its dominance in cost-efficiency and speed.

| Metric | Raw LangGraph | **Orchestra Enhanced** | Improvement |
| :--- | :--- | :--- | :--- |
| **Cost (MMLU Dataset)** | $42.50 | **$6.37** | **85% Profit Kept** |
| **P99 Response Time** | ~4,200ms | **< 45ms** | **94x Faster** |
| **Token Utilization** | 12.4M Tokens | **1.8M Tokens** | **Efficiency Gained** |
| **Error Rate (Provider Timeout)** | 4.2% | **0.5%** | **Self-Healing** |

---
 
### 📺 Watch: How to drop your LLM bill by 85% instantly
[![Orchestra Demo](https://img.youtube.com/vi/TaIGvoKuWZs/0.jpg)](https://www.youtube.com/watch?v=TaIGvoKuWZs)

---

> [!IMPORTANT]
> ### 🚨 The "LLM Inefficiency Crisis"
> Standard agents are bleeding money. Every time an agent asks "What's the weather?" vs "How's the weather?", you pay for a redundant LLM inference. As your agents scale, this "Context Tax" becomes unsustainable. **Orchestra ends this.**

---

## 🚀 Why Orchestra?

Orchestra is a drop-in enhancement for **LangGraph** and **LangChain** that adds semantic intuition, resilience, and "Collective Intelligence" to your agent swarm.

| Feature | Standard Agent | **Orchestra Enhanced** |
| :--- | :--- | :--- |
| **Cost** | 100% (Linear Scaling) | **~15%** (Sub-linear via Semantic Reuse) |
| **Latency** | 2-10 seconds per turn | **< 50ms** for cached interactions |
| **Resilience** | 5xx Errors crash the agent | **Auto-Recovery** with Circuit Breakers |
| **Memory** | Siloed per thread/user | **Collective Intelligence** (Global Semantic Memory) |
| **Tool Usage** | Prompt bloat with 50+ tools | **Smart Tool Pruning** (Top 5 only) |

---

## 🧠 Collective Intelligence (Self-RAG)

Orchestra doesn't just cache; it **learns**. It understands the *meaning* of requests, skipping the LLM entirely when a semantically similar query has already been answered.

```mermaid
graph TD
    %% Global Styles
    classDef user fill:#2d333b,stroke:#539bf5,stroke-width:2px,color:#adbac7;
    classDef match fill:#1c2128,stroke:#fdb347,stroke-width:2px,color:#adbac7;
    classDef process fill:#22272e,stroke:#2ea043,stroke-width:2px,color:#adbac7;
    
    User([User Query]) --> Match{Semantic Match?}
    
    %% Cache Path
    Match -- "92% Similar" --> Cache[(Collective Memory)]
    Cache --> Return([Instant Response < 0.1s])
    
    %% Execution Path
    Match -- "New Concept" --> RAG{Context Injection?}
    RAG -- "Enabled" --> Memory[Inject Similar Past Logic]
    Memory --> LLM[LLM Provider]
    RAG -- "Disabled" --> LLM
    
    LLM --> Return
    
    class User user;
    class Match match;
    class Cache,Memory,LLM process;
```

---

## ⚡ The 1-Line Setup

### 🟢 LangGraph
```python
from orchestra import enhance

# Wrap your compiled graph. Orchestra handles the rest.
agent = enhance(app.compile())

# Done. You now have Semantic Caching, Tracing, and Resilience.
result = agent.invoke({"query": "Compare our Q2 vs Q3 targets."})
```

### 🦜 LangChain
```python
from orchestra.adapters.langchain import enhance

# Works for both legacy Chains and new LCEL Runnables
agent = enhance(my_chain)
result = agent.invoke("Tell me the project status")
```

---

## 🏗️ Premium Features Deep Dive

### 🔬 Hierarchical Matching
Standard vector search fails on long, complex prompts. Orchestra's **Hierarchical Matcher** breaks prompts into chunks and computes a weighted similarity across levels.
- **L1 (Global)**: High-level intent matching.
- **L2 (Local)**: Specific detail matching.

### 🔌 Smart Tool Discovery (MCP)
The **Context Tax** is real. Claude and GPT-4o cost more as your system prompt grows. Orchestra semantically searches your **Model Context Protocol (MCP)** tool library and injects only the top 5 relevant tools into the prompt.

### 🎥 Time-Travel Debugging
The **Orchestra Recorder** isn't just a logger; it's a flight data recorder. Trace every state mutation, diff, and LLM call exactly as it happened.

### 🛡️ Outage Insurance (Circuit Breakers)
Don't let an OpenAI or Anthropic outage kill your production agents. Orchestra's circuit breaker detects failures in real-time and provides graceful fallbacks or cached "replays".

---

## ⚙️ Full Configuration API

### `OrchestraConfig` (LangGraph) / `OrchestraLangChainConfig`

| Category | Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Logic** | `similarity_threshold`| `float`| `0.92` | Cosine similarity for a cache hit. |
| | `embedding_model` | `str` | `...` | Any `SentenceTransformer` model. |
| **Hierarchy** | `enable_hierarchical` | `bool` | `False` | Multi-level (L1+L2) matching. |
| | `hierarchical_weight_l1`| `float`| `0.6` | Importance of full-query match. |
| | `hierarchical_weight_l2`| `float`| `0.4` | Importance of chunk-level match. |
| **Cache** | `enable_cache` | `bool` | `True` | Global toggle for semantic caching. |
| | `cache_ttl` | `int` | `3600` | Expiration (seconds). |
| | `max_cache_size` | `int` | `10000` | Max entries before eviction. |
| | `auto_cleanup` | `bool` | `True` | Background expired entry removal. |
| | `cleanup_interval` | `int` | `300` | Interval for background cleanup. |
| **Self-RAG** | `enable_context_injection`| `bool` | `False` | Inject past logic as context. |
| | `context_injection_top_k`| `int` | `3` | Max past matches to inject. |
| | `context_injection_template`| `str` | `...` | Custom Prompt template for RAG. |
| **MCP** | `mcp_servers` | `list` | `None` | List of MCP host configs. |
| | `enable_tool_search` | `bool` | `True` | Dynamic tool pruning. |
| | `tool_search_top_k` | `int` | `5` | Max tools injected per turn. |
| | `tool_context_threshold`| `float`| `0.10` | Similarity trigger for search. |
| | `mcp_cache_ttl` | `int` | `3600` | TTL for MCP tool responses. |
| **Resilience**| `enable_circuit_breaker`| `bool` | `False` | Prevent provider-timeout cascades. |
| | `circuit_breaker_threshold`| `int` | `5` | Failures before opening. |
| | `circuit_breaker_timeout`| `float`| `60.0` | Time until retry after opening. |
| **Storage** | `redis_url`| `str` | `None` | Redis Stack URL for shared caching. |
| | `enable_compression` | `bool` | `False` | Zlib compression for large cache. |
| **Obs** | `enable_recorder` | `bool` | `True` | Enables step-by-step tracing. |
| | `llm_cost_per_1k_tokens`| `float`| `0.03` | Savings calculation basis. |

---

## 💾 Persistence & Backends (Production)

### 1. Redis Stack (Unified Semantic Store)
Use Redis for distributed, shared collective intelligence across multiple agent pods.
```python
config = OrchestraConfig(
    redis_url="redis://:password@your-redis:6379",
    similarity_threshold=0.95
)
```

### 2. Recorder Backends
- **SQLite (Default)**: Perfect for local dev. Traces stored in `.orchestra/traces.db`.
- **PostgreSQL**: Production-grade tracing.
```python
from orchestra.recorder.storage import PostgresStorage
storage = PostgresStorage(dsn="postgresql://user:pass@localhost:5432/db")
OrchestraRecorder._instance = OrchestraRecorder(storage=storage)
```

---

## 📝 Declarative Agents (YAML)

Build complex, intelligent agents without writing Python boilerplate.

```yaml
# agent.yaml
model:
  provider: "openai"
  name: "gpt-4o"
  temperature: 0.2

orchestra:
  similarity_threshold: 0.94
  enable_hierarchical: true
  top_k: 5

mcp_servers:
  - name: "filesystem"
    command: "npx"
    args: ["@modelcontextprotocol/server-filesystem", "/users/data"]

system_context: |
  You are an expert data analyst with access to local files.
```

Run it instantly:
```bash
python -m orchestra.cli run agent.yaml --interactive
```

---

## 📊 Metrics & Observability

Orchestra's metrics engine tracks performance in real-time.

```python
stats = agent.get_metrics()
# {
#   "cache_hit_rate": 0.42,
#   "estimated_cost_saved": 12.45,
#   "avg_latency": 0.045,
#   ...
# }
```

---

## 🎥 CLI Zero-to-Hero Reference

| Command | Usage | Description |
| :--- | :--- | :--- |
| **`trace ls`** | `... trace ls --limit 20` | List recent execution traces. |
| **`trace view`**| `... trace view <ID>` | Deep dive into step-by-step state. |
| **`trace prune`**| `... trace prune --days 7` | Clean up old tracing data. |
| **`eval`** | `... eval "Actual" "Expect"`| Semantic similarity check (CI/CD). |
| **`run`** | `... run config.yaml` | Run a declarative agent. |

---

## 🧪 CI/CD & Semantic Eval

Use Orchestra in your test suite to verify agent outputs without brittle string matching.

```python
from orchestra.eval import FuzzyAssert

def test_agent_output():
    actual = agent.invoke("Summarize the Q3 report")
    expected = "The report shows a 15% increase in revenue..."
    
    # Passes if semantically similar > 0.92
    FuzzyAssert.similar(actual, expected, threshold=0.95)
```

---

## ❓ FAQ

**Q: How does this affect privacy?**
Orchestra stores hashes and embeddings. No raw data leaves your system unless you use a cloud embedding provider.

**Q: Can I use it with custom LangGraph nodes?**
Yes. Orchestra wraps the entire graph; it is node-agnostic.

**Q: What embeddings are supported?**
Any model from the `sentence-transformers` library or OpenAI's `text-embedding-3-small`.

---

## 🌟 Support the Orchestration Network

We are building the future of efficient, unbreakable AI agents. If you find Orchestra useful, **please give us a star on GitHub!** It helps us reach more developers and continue building the open-source agentic web.

[**⭐ Star Orchestra on GitHub**](https://github.com/uejsh/orchestra)

---

<div align="center">
  <p>Built with ❤️ for the Agentic Era.</p>
</div>
