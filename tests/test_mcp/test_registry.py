# tests/test_mcp/test_registry.py
# ============================================================================
# Tests for MCP Tool Registry (Smart Tool Discovery)
# ============================================================================

import pytest
import sys

# Skip tests if required dependencies not available
pytestmark = pytest.mark.skipif(
    sys.platform == "win32" and "faiss" not in sys.modules,
    reason="May require FAISS which has Windows compatibility issues"
)


class MockMCPTool:
    """Mock MCP tool for testing"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.input_schema = {}
        self.server_name = "test_server"
    
    def to_searchable_text(self) -> str:
        return f"{self.name}: {self.description}"


class MockMCPClient:
    """Mock MCP client for testing"""
    def __init__(self, name: str = "test_server"):
        self.name = name
        self._tools = []
        self._connected = True
    
    @property
    def connected(self):
        return self._connected
    
    async def list_tools(self):
        return self._tools
    
    def add_tool(self, name: str, description: str):
        from orchestra.mcp.client import MCPTool
        self._tools.append(MCPTool(
            name=name,
            description=description,
            input_schema={},
            server_name=self.name
        ))


class TestMCPToolRegistry:
    """Test MCPToolRegistry - Smart Tool Discovery"""
    
    def test_registry_creation(self):
        """Test registry can be created"""
        from orchestra.mcp.registry import MCPToolRegistry
        
        registry = MCPToolRegistry()
        assert registry.total_tools == 0
        assert registry.total_clients == 0
    
    def test_add_client(self):
        """Test adding clients to registry"""
        from orchestra.mcp.registry import MCPToolRegistry
        
        registry = MCPToolRegistry()
        client = MockMCPClient("test")
        registry.add_client(client)
        
        assert registry.total_clients == 1
    
    @pytest.mark.asyncio
    async def test_index_tools(self):
        """Test indexing tools from clients"""
        from orchestra.mcp.registry import MCPToolRegistry
        from orchestra.mcp.client import MCPTool
        
        # Create mock client with tools
        client = MockMCPClient("postgres")
        client._tools = [
            MCPTool("query", "Execute SQL query on database", {}, "postgres"),
            MCPTool("list_tables", "List all database tables", {}, "postgres"),
        ]
        
        registry = MCPToolRegistry([client])
        count = await registry.index()
        
        assert count == 2
        assert registry.total_tools == 2
    
    def test_should_use_tool_search_small(self):
        """Test tool search threshold with few tools"""
        from orchestra.mcp.registry import MCPToolRegistry
        
        registry = MCPToolRegistry()
        registry._tools = [(None, None)] * 5  # 5 tools
        registry._indexed = True
        
        # 5 tools * 150 tokens = 750 tokens
        # 750 / 128000 = 0.6% < 10%
        assert not registry.should_use_tool_search(128000)
    
    def test_should_use_tool_search_large(self):
        """Test tool search threshold with many tools"""
        from orchestra.mcp.registry import MCPToolRegistry
        
        registry = MCPToolRegistry()
        registry._tools = [(None, None)] * 100  # 100 tools
        registry._indexed = True
        
        # 100 tools * 150 tokens = 15000 tokens
        # 15000 / 128000 = 11.7% > 10%
        assert registry.should_use_tool_search(128000)
    
    @pytest.mark.asyncio
    async def test_find_relevant_tools(self):
        """Test semantic search for relevant tools"""
        from orchestra.mcp.registry import MCPToolRegistry
        from orchestra.mcp.client import MCPTool
        
        # Create mock client with diverse tools
        client = MockMCPClient("mixed")
        client._tools = [
            MCPTool("query_database", "Execute SQL query on PostgreSQL database", {}, "mixed"),
            MCPTool("send_email", "Send an email message to a recipient", {}, "mixed"),
            MCPTool("read_file", "Read contents of a file from filesystem", {}, "mixed"),
            MCPTool("get_order", "Retrieve order information from database", {}, "mixed"),
        ]
        
        registry = MCPToolRegistry([client])
        await registry.index()
        
        # Search for database-related tools
        results = registry.find_relevant_tools("check order status in database", top_k=2)
        
        assert len(results) <= 2
        # Should find database-related tools
        tool_names = [r.tool.name for r in results]
        assert any("database" in name or "order" in name for name in tool_names)
    
    def test_get_stats(self):
        """Test getting registry stats"""
        from orchestra.mcp.registry import MCPToolRegistry
        
        registry = MCPToolRegistry()
        stats = registry.get_stats()
        
        assert "total_tools" in stats
        assert "total_clients" in stats
        assert "indexed" in stats


class TestMCPConfig:
    """Test MCP configuration in OrchestraConfig"""
    
    def test_mcp_config_defaults(self):
        """Test MCP config defaults"""
        from orchestra import OrchestraConfig
        
        config = OrchestraConfig()
        
        assert config.mcp_servers is None
        assert config.enable_tool_search == True
        assert config.tool_search_top_k == 5
        assert config.tool_context_threshold == 0.10
        assert config.mcp_cache_ttl == 3600
    
    def test_mcp_config_custom(self):
        """Test custom MCP config"""
        from orchestra import OrchestraConfig
        
        config = OrchestraConfig(
            mcp_servers=["server1", "server2"],
            enable_tool_search=False,
            tool_search_top_k=10,
            tool_context_threshold=0.15,
            mcp_cache_ttl=7200
        )
        
        assert config.mcp_servers == ["server1", "server2"]
        assert config.enable_tool_search == False
        assert config.tool_search_top_k == 10
        assert config.tool_context_threshold == 0.15
        assert config.mcp_cache_ttl == 7200
