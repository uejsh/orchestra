# tools.py - Convert MCP tools to LangGraph format
# ============================================================================
# FILE: orchestra/mcp/tools.py
# Utilities for converting MCP tools to LangGraph-compatible tools
# ============================================================================

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Type

from .client import MCPClient, MCPTool
from .registry import MCPToolRegistry, ToolSearchResult

logger = logging.getLogger(__name__)

# Check for LangChain tools availability
try:
    from langchain_core.tools import BaseTool, ToolException
    from pydantic import BaseModel, Field, create_model
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = object


class MCPToolWrapper(BaseTool if LANGCHAIN_AVAILABLE else object):
    """
    Wrapper that converts an MCP tool to a LangChain/LangGraph compatible tool.
    
    This allows MCP tools to be used directly in LangGraph ToolNode.
    """
    
    name: str = ""
    description: str = ""
    
    def __init__(
        self,
        mcp_tool: MCPTool,
        mcp_client: MCPClient,
        **kwargs
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain not installed. Install with: pip install langchain-core"
            )
        
        # Build dynamic args schema from MCP tool
        args_schema = self._build_args_schema(mcp_tool)
        
        super().__init__(
            name=f"{mcp_client.name}_{mcp_tool.name}",
            description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
            args_schema=args_schema,
            **kwargs
        )
        
        self._mcp_tool = mcp_tool
        self._mcp_client = mcp_client
    
    def _build_args_schema(self, mcp_tool: MCPTool) -> Type[BaseModel]:
        """Build Pydantic model from MCP tool input schema"""
        if not LANGCHAIN_AVAILABLE:
            return None
        
        schema = mcp_tool.input_schema
        if not schema or 'properties' not in schema:
            # Return empty schema if no properties defined
            return create_model(f"{mcp_tool.name}Args")
        
        # Convert JSON Schema to Pydantic fields
        fields = {}
        required = schema.get('required', [])
        
        for prop_name, prop_def in schema.get('properties', {}).items():
            # Map JSON Schema types to Python types
            prop_type = prop_def.get('type', 'string')
            python_type = {
                'string': str,
                'integer': int,
                'number': float,
                'boolean': bool,
                'array': list,
                'object': dict
            }.get(prop_type, str)
            
            # Handle optional fields
            if prop_name in required:
                fields[prop_name] = (
                    python_type,
                    Field(description=prop_def.get('description', ''))
                )
            else:
                fields[prop_name] = (
                    Optional[python_type],
                    Field(default=None, description=prop_def.get('description', ''))
                )
        
        return create_model(f"{mcp_tool.name}Args", **fields)
    
    def _run(self, **kwargs) -> str:
        """Synchronous execution (wraps async)"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context - create new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._arun(**kwargs)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self._arun(**kwargs))
        except RuntimeError:
            return asyncio.run(self._arun(**kwargs))
    
    async def _arun(self, **kwargs) -> str:
        """Asynchronous execution"""
        try:
            result = await self._mcp_client.call_tool(
                self._mcp_tool.name,
                kwargs
            )
            return str(result)
        except Exception as e:
            logger.error(f"Error calling MCP tool {self.name}: {e}")
            if LANGCHAIN_AVAILABLE:
                raise ToolException(f"MCP tool error: {e}")
            raise


def mcp_to_langgraph_tools(
    source: Any,
    include_server_prefix: bool = True
) -> List[BaseTool]:
    """
    Convert MCP tools to LangGraph-compatible tools.
    
    Args:
        source: MCPClient, MCPToolRegistry, or List[ToolSearchResult]
        include_server_prefix: Include server name in tool name
    
    Returns:
        List of LangChain/LangGraph compatible tools
    
    Usage:
        # From a single client
        tools = mcp_to_langgraph_tools(postgres_client)
        
        # From registry search results
        results = registry.find_relevant_tools("query database")
        tools = mcp_to_langgraph_tools(results)
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain not installed. Install with: pip install langchain-core"
        )
    
    tools = []
    
    if isinstance(source, MCPClient):
        # Convert all tools from a client
        if not source.connected:
            logger.warning(f"Client {source.name} not connected")
            return []
        
        # Need to run async in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    mcp_tools = executor.submit(
                        asyncio.run,
                        source.list_tools()
                    ).result()
            else:
                mcp_tools = loop.run_until_complete(source.list_tools())
        except RuntimeError:
            mcp_tools = asyncio.run(source.list_tools())
        
        for mcp_tool in mcp_tools:
            try:
                wrapper = MCPToolWrapper(mcp_tool, source)
                tools.append(wrapper)
            except Exception as e:
                logger.error(f"Error wrapping tool {mcp_tool.name}: {e}")
    
    elif isinstance(source, MCPToolRegistry):
        # Convert all tools from registry
        for mcp_tool, client in source.get_all_tools():
            try:
                wrapper = MCPToolWrapper(mcp_tool, client)
                tools.append(wrapper)
            except Exception as e:
                logger.error(f"Error wrapping tool {mcp_tool.name}: {e}")
    
    elif isinstance(source, list):
        # Convert from search results
        for item in source:
            if isinstance(item, ToolSearchResult):
                try:
                    wrapper = MCPToolWrapper(item.tool, item.client)
                    tools.append(wrapper)
                except Exception as e:
                    logger.error(f"Error wrapping tool {item.tool.name}: {e}")
    
    logger.info(f"🔧 Created {len(tools)} LangGraph tools from MCP")
    return tools


def create_smart_tool_node(
    registry: MCPToolRegistry,
    top_k: int = 5,
    context_window: int = 128000
) -> Callable:
    """
    Create a LangGraph node that dynamically selects relevant tools.
    
    This implements Claude's Tool Search pattern:
    - If total tools < 10% of context: use all tools
    - Otherwise: semantically search for relevant tools
    
    Args:
        registry: MCPToolRegistry with indexed tools
        top_k: Max tools to select when searching
        context_window: Model's context window size
    
    Returns:
        A function suitable for use as a LangGraph node
    
    Usage:
        smart_tools = create_smart_tool_node(registry)
        graph.add_node("tools", smart_tools)
    """
    def tool_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Dynamic tool selection and execution"""
        messages = state.get("messages", [])
        if not messages:
            return state
        
        # Extract last user message or tool call
        last_message = messages[-1]
        
        # Check if we need tool search
        if registry.should_use_tool_search(context_window):
            # Get query from message
            query = ""
            if hasattr(last_message, 'content'):
                query = last_message.content
            elif isinstance(last_message, dict):
                query = last_message.get('content', '')
            
            # Semantic search for relevant tools
            results = registry.find_relevant_tools(query, top_k=top_k)
            tools = mcp_to_langgraph_tools(results)
            
            logger.info(f"🔍 Smart Tool Selection: {len(tools)} tools for query")
        else:
            # Use all tools (small enough to fit in context)
            tools = mcp_to_langgraph_tools(registry)
        
        # Store selected tools in state for use by agent
        return {**state, "_selected_tools": tools}
    
    return tool_node


class SmartToolNode:
    """
    A LangGraph-compatible node that implements smart tool discovery.
    
    Usage:
        node = SmartToolNode(registry)
        graph.add_node("smart_tools", node)
    """
    
    def __init__(
        self,
        registry: MCPToolRegistry,
        top_k: int = 5,
        context_window: int = 128000
    ):
        self.registry = registry
        self.top_k = top_k
        self.context_window = context_window
        self._selected_tools: List[BaseTool] = []
    
    def get_tools_for_query(self, query: str) -> List[BaseTool]:
        """Get relevant tools for a query"""
        if self.registry.should_use_tool_search(self.context_window):
            results = self.registry.find_relevant_tools(query, top_k=self.top_k)
            self._selected_tools = mcp_to_langgraph_tools(results)
        else:
            self._selected_tools = mcp_to_langgraph_tools(self.registry)
        
        return self._selected_tools
    
    @property
    def selected_tools(self) -> List[BaseTool]:
        """Currently selected tools"""
        return self._selected_tools
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute as LangGraph node"""
        messages = state.get("messages", [])
        
        # Extract query
        query = ""
        if messages:
            last = messages[-1]
            if hasattr(last, 'content'):
                query = last.content
            elif isinstance(last, dict):
                query = last.get('content', '')
        
        # Get relevant tools
        tools = self.get_tools_for_query(query)
        
        return {**state, "_mcp_tools": tools}
