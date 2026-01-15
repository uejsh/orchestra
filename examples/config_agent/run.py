# examples/config_agent/run.py
import asyncio
import logging
import sys
from unittest.mock import MagicMock
import numpy as np

# --- PATCHING BEFORE IMPORTS ---
# AGGRESSIVE MOCKING: Windows has issues with torch DLLs. 
# We prevent torch from loading entirely.
import types
mock_torch = types.ModuleType("torch")
mock_torch.__spec__ = MagicMock()
sys.modules["torch"] = mock_torch

mock_st = types.ModuleType("sentence_transformers")
mock_st.__spec__ = MagicMock()
sys.modules["sentence_transformers"] = mock_st

sys.modules["orchestra.core.embeddings"] = MagicMock()

mock_embedder = MagicMock()
mock_embedder.encode.return_value = np.random.rand(1, 384)
sys.modules["orchestra.core.embeddings"].EmbeddingModel = MagicMock(return_value=mock_embedder)
# -------------------------------

from orchestra.agent import AgentBuilder
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)

async def main():
    # 1. Build Agent from Config
    # This automatically connects to MCP and wires up the graph
    print("🏗️  Building Agent from agent.yaml...")
    builder = AgentBuilder.from_file("examples/config_agent/agent.yaml")
    
    # Mocking the MCP connection to avoid needing actual Node.js/NPX in this test env
    # In real usage, this mocks would not be here.
    async def mock_connect(self):
        print(f"🔌 [MOCK] Connected to {self.protocol.server_command}")
        self.connected = True
        return
        
    # Patch client connection for demo
    from orchestra.mcp import MCPClient
    MCPClient.connect = mock_connect
    
    # Patch EmbeddingModel to avoid Windows DLL issues during simple verification
    import sys
    from unittest.mock import MagicMock
    import numpy as np
    
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.random.rand(1, 384) # Mock 384-dim vector
    
    # We need to patch where it's used in registry.py
    # But registry.py imports EmbeddingModel inside the method, so we can mock the module
    sys.modules["orchestra.core.embeddings"] = MagicMock()
    sys.modules["orchestra.core.embeddings"].EmbeddingModel = MagicMock(return_value=mock_embedder)
    
    # 2. Build the running graph
    agent = await builder.build()
    print("✅ Agent Built!")
    
    # 3. Running it (Dry run since we don't have real API keys)
    print("\n🚀 Starting Run: 'Check git status'")
    
    # Note: We can't actually invoke the LLM without a real key
    # But we can verify the graph structure was built correctly
    print(f"Graph Nodes: {agent.get_graph().nodes.keys()}")
    
    # Verify tooling node exists
    assert "tools" in agent.get_graph().nodes
    assert "search" in agent.get_graph().nodes
    assert "agent" in agent.get_graph().nodes
    
    print("✨ Declarative Agent Architecture Verified!")

if __name__ == "__main__":
    asyncio.run(main())
