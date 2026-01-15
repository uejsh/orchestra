# 🎵 Orchestra

**Reduce your AI orchestration costs by 85% with one line of code.**

[![PyPI](https://img.shields.io/pypi/v/orchestra-llm-cache.svg)](https://pypi.org/project/orchestra-llm-cache/)
[![Python](https://img.shields.io/pypi/pyversions/orchestra-llm-cache.svg)](https://pypi.org/project/orchestra-llm-cache/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Orchestra adds intelligent **semantic caching** to LangGraph, LangChain, and other AI frameworks. It understands the *meaning* of your requests, allowing it to reuse results even for slightly different phrasing (e.g., "Summarize Q3" vs "Give me a summary of the third quarter").

---

## 🚀 Why Orchestra?

- **Extreme Cost Savings**: Stop paying for semantically identical LLM calls.
- **Lightning Performance**: Sub-second responses (<0.5s) for cached results.
- **Zero-Code Integration**: Wrap your existing LangGraph or LangChain objects in **one line**.
- **Enterprise Ready**: Full support for **PostgreSQL**, **Redis Stack**, and **Circuit Breakers**.
- **Smart Tool Discovery**: Automatically filters MCP tools to save context tokens (like Claude's Tool Search).

| Feature | Orchestra | Standard Cache |
|---------|-----------|----------------|
| **Matching** | Semantic (Embeddings + Cosine) | Exact String Match |
| **Storage** | Local (SQLite) / Postgres / Redis | Local Memory / Key-Value |
| **Resilience** | Circuit Breaker (Fail Fast) | None |
| **Observability** | Full Execution Tracing (CLI) | Basic Logging |
| **Tooling** | Smart Discovery (Context Saving) | All Tools Always |

---

## 📦 Installation

```bash
# Recommended: Full install (includes all backends)
pip install orchestra-llm-cache[full]

# ----------------- Modular Installs -----------------

# Just the core (SQLite only)
pip install orchestra-llm-cache

# With Postgres support
pip install orchestra-llm-cache[postgres]

# With Redis support
pip install orchestra-llm-cache[redis]

# With MCP support
pip install orchestra-llm-cache[mcp]
```

---

## ⚡ Quick Start

### 1. With LangGraph
```python
from langgraph.graph import StateGraph
from orchestra import enhance

# 1. Define your graph normally
graph = StateGraph(State)
# ... build your graph ...

# 2. ✨ Add Orchestra (ONE LINE)
# This automatically wraps nodes with caching & tracing
cached_graph = enhance(graph.compile())

# 3. Use normally
result = cached_graph.invoke({"query": "Analyze sales data"})
```

### 2. With LangChain
```python
from langchain.chains import LLMChain
from orchestra import enhance

chain = LLMChain(llm=llm, prompt=prompt)
cached_chain = enhance(chain) 

result = cached_chain.run("Show me Q4 trends")
```

---

## 🧠 Deep Dive: Semantic Caching

Orchestra doesn't just look for exact string matches. It uses **embedding models** (default: `all-MiniLM-L6-v2`) to convert input states into vectors.

1.  **Input**: "What is the capital of France?"
2.  **Embedding**: `[0.12, -0.45, 0.88, ...]`
3.  **Search**: Finds nearest neighbor in DB.
4.  **Match**: Found "Capital of France" (Similarity: 0.98).
5.  **Action**: Return cached result instantly.

### Tuning Sensitivity
You can control how "loose" or "strict" the matching is:

```python
from orchestra import OrchestraConfig

config = OrchestraConfig(
    # 0.99 = Exact match only
    # 0.90 = Very similar (Recommended)
    # 0.75 = Loosely related
    similarity_threshold=0.92 
)
graph = enhance(graph, config=config)
```

---

## 🛡️ Deep Dive: Resilience

Orchestra protects your application from upstream LLM failures using a **Circuit Breaker** pattern.

### Circuit Breaker States
1.  **CLOSED** (Normal): Requests go through to the LLM.
2.  **OPEN** (Failure): After `circuit_breaker_threshold` failures, the circuit opens. **All LLM calls fail immediately** without waiting, allowing your app to degrade gracefully or switch providers.
3.  **HALF-OPEN** (Recovery): After `circuit_breaker_timeout`, one request is allowed through. If it succeeds, the circuit closes.

```python
config = OrchestraConfig(
    enable_circuit_breaker=True,
    circuit_breaker_threshold=5,  # Open after 5 consecutive errors
    circuit_breaker_timeout=60.0  # Wait 60s before trying again
)
```

> **Note**: Cache hits are **always returned**, even if the circuit is OPEN. This ensures your app remains partially functional during outages.

---

## 💾 Deep Dive: Storage Backends

### 1. SQLite (Default)
Best for: **Local Development**, **Single Agent**.
Files are stored in `.orchestra/traces.db` in your working directory. No setup required.

### 2. PostgreSQL (Production)
Best for: **Centralized Tracing**, **Analytics**.
Uses connection pooling for high performance.

```python
from orchestra.recorder import OrchestraRecorder, PostgresStorage

# Initialize generic recorder globally
recorder = OrchestraRecorder(
    storage=PostgresStorage(
        dsn="postgresql://user:pass@localhost:5432/mydb",
        pool_size=20
    )
)
```

### 3. Redis Stack (Distributed)
Best for: **Shared Cache**, **Kubernetes**.
Uses Redis Vector Search for millisecond-latency semantic lookups across multiple agent instances.

```python
config = OrchestraConfig(
    redis_url="redis://localhost:6379",
    # Enable compression to save Redis RAM/Network
    enable_compression=True 
)
```

---

## 🔌 Deep Dive: MCP & Smart Tool Discovery

When using the **Model Context Protocol (MCP)**, connecting to multiple servers (GitHub, Linear, Slack) can load 50+ tools into your context window. This makes agents slow, expensive, and confused.

Orchestra's **Smart Tool Discovery** fixes this by dynamically selecting only the relevant tools.

### How it works
1.  **Index**: All tools from all MCP servers are indexed semantically.
2.  **Search**: When the user query arrives, Orchestra searches for the top K relevant tools.
3.  **Filter**: Only those tools are injected into the context.

```python
from orchestra.mcp import MCPClient, MCPConfig, MCPToolRegistry

# 1. Connect Clients
gh_client = MCPClient(MCPConfig(command="npx", args=["-y", "@modelcontextprotocol/server-github"]))

# 2. Create Registry & Index
registry = MCPToolRegistry([gh_client])
await registry.index()

# 3. Use in Agent
# If query is "Find issues", this returns ONLY issue-related tools
relevant = registry.find_relevant_tools("Find issues about login", top_k=3)
```

**Configuration**:
```python
config = OrchestraConfig(
    enable_tool_search=True,
    tool_search_top_k=5,         # Max tools to show
    tool_context_threshold=0.10  # Only search if tools take > 10% of window
)
```

---

## 🎥 Orchestra Recorder & CLI

Orchestra records every step, input, output, and cost of your agent execution.

### CLI Reference

| Command | Usage | Description |
|---------|-------|-------------|
| `trace ls` | `python -m orchestra.cli trace ls --limit 5` | List recent traces. |
| `trace view` | `python -m orchestra.cli trace view <ID>` | View details of a specific trace. |
| `trace prune` | `python -m orchestra.cli trace prune --days 7` | Delete traces older than X days. |
| `eval` | `python -m orchestra.cli eval "A" "B"` | Check if two strings are similar. |
| `run` | `python -m orchestra.cli run agent.yaml -i` | Run an agent interactively. |

---

## ⚙️ Configuration Reference

### `OrchestraConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `similarity_threshold` | float | `0.92` | Matching strictness (0.0 - 1.0). |
| `embedding_model` | str | `all-MiniLM-L6-v2` | HuggingFace model name. |
| `cache_ttl` | int | `3600` | Cache expiry in seconds. |
| `max_cache_size` | int | `10000` | Max entries in local memory. |
| `redis_url` | str | `None` | Connection string for Redis. |
| `enable_compression` | bool | `False` | Gzip large states before storage. |
| `enable_circuit_breaker` | bool | `False` | Activate resilience pattern. |
| `circuit_breaker_threshold` | int | `5` | Failures before opening circuit. |
| `circuit_breaker_timeout` | float | `60.0` | Recovery timeout (seconds). |
| `tool_search_top_k` | int | `5` | Max tools for Smart Discovery. |
| `mcp_cache_ttl` | int | `3600` | How long to cache MCP tool lists. |

### `agent.yaml` Structure

For the Declarative Agent Runner (`orchestra.cli run`):

```yaml
model:
  provider: anthropic
  name: claude-3-5-sonnet-20240620
  api_key: env:ANTHROPIC_API_KEY

mcp_servers:
  - name: github
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_TOKEN: "..."

orchestra:
  tool_search: true
  top_k: 5
```

---

## 🧪 Testing & Evaluation

Use `FuzzyAssert` to write tests that pass if the **meaning** is correct, even if wording differs.

```python
from orchestra.eval import FuzzyAssert

def test_agent_response():
    actual = agent.invoke("Hello")
    # Passes if response is generally a greeting
    FuzzyAssert.similar(actual, "Hi there, how can I help?", threshold=0.8)
```

---

## ❓ FAQ

**Q: Does this work with LangGraph Cloud?**
A: Orchestra is optimized for self-hosted / local execution. Cloud support is on the roadmap.

**Q: I get FAISS errors on Windows.**
A: Orchestra transparently falls back to a NumPy implementation if FAISS fails. You can ignore these warnings.

**Q: Can I use a different embedding model?**
A: Yes, set `embedding_model` in `OrchestraConfig` to any SentenceTransformer model.

---
**Built with ❤️ for the AI Engineering Community**
