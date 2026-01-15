# orchestra/mcp - MCP Integration with Smart Tool Discovery
# ============================================================================
# Brings Claude's "Tool Search" capability to any agent framework
# ============================================================================

from .client import MCPClient, MCPConfig
from .registry import MCPToolRegistry
from .tools import mcp_to_langgraph_tools, MCPToolWrapper

__all__ = [
    "MCPClient",
    "MCPConfig", 
    "MCPToolRegistry",
    "mcp_to_langgraph_tools",
    "MCPToolWrapper",
]
