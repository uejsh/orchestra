# client.py - MCP Client with Orchestra enhancements
# ============================================================================
# FILE: orchestra/mcp/client.py
# MCP client wrapper with caching and resilience
# ============================================================================

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Type stubs for when mcp is not installed
try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None


@dataclass
class MCPConfig:
    """Configuration for MCP client connection"""
    
    # Server connection
    server_command: str  # e.g., "npx -y @anthropic/mcp-server-postgres"
    server_args: List[str] = field(default_factory=list)
    server_env: Dict[str, str] = field(default_factory=dict)
    
    # Transport
    transport: str = "stdio"  # "stdio" or "sse"
    sse_url: Optional[str] = None  # For SSE transport
    
    # Caching (integrates with Orchestra's cache)
    enable_cache: bool = True
    cache_ttl: int = 3600
    
    # Resilience
    timeout: float = 30.0
    max_retries: int = 3
    
    # Naming
    name: Optional[str] = None  # Human-readable name for this server


@dataclass
class MCPTool:
    """Represents an MCP tool"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str
    
    def to_searchable_text(self) -> str:
        """Convert to text for semantic search indexing"""
        return f"{self.name}: {self.description}"


@dataclass 
class MCPResource:
    """Represents an MCP resource"""
    uri: str
    name: str
    description: Optional[str]
    mime_type: Optional[str]
    server_name: str


class MCPClient:
    """
    Orchestra-enhanced MCP client.
    
    Features:
    - Connect to MCP servers (stdio/SSE)
    - Discover tools and resources
    - Execute tools with optional caching
    - Resilience via retries
    
    Usage:
        config = MCPConfig(
            server_command="npx",
            server_args=["-y", "@anthropic/mcp-server-postgres"],
            server_env={"DATABASE_URL": "postgres://..."}
        )
        client = MCPClient(config)
        await client.connect()
        tools = await client.list_tools()
    """
    
    def __init__(self, config: MCPConfig):
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP package not installed. Install with: pip install mcp"
            )
        
        self.config = config
        self.name = config.name or config.server_command.split()[-1]
        self._session: Optional[ClientSession] = None
        self._tools: List[MCPTool] = []
        self._resources: List[MCPResource] = []
        self._connected = False
        
        # Optional cache manager (set externally by Orchestra adapter)
        self.cache_manager = None
    
    @property
    def connected(self) -> bool:
        return self._connected
    
    async def connect(self) -> None:
        """Connect to the MCP server"""
        if self._connected:
            logger.warning(f"Already connected to {self.name}")
            return
        
        try:
            if self.config.transport == "stdio":
                await self._connect_stdio()
            elif self.config.transport == "sse":
                await self._connect_sse()
            else:
                raise ValueError(f"Unknown transport: {self.config.transport}")
            
            self._connected = True
            logger.info(f"🔌 Connected to MCP server: {self.name}")
            
            # Pre-fetch tools and resources
            await self._discover()
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.name}: {e}")
            raise
    
    async def _connect_stdio(self) -> None:
        """Connect via stdio transport"""
        server_params = StdioServerParameters(
            command=self.config.server_command,
            args=self.config.server_args,
            env=self.config.server_env or None
        )
        
        # Create stdio client context
        self._stdio_context = stdio_client(server_params)
        self._read, self._write = await self._stdio_context.__aenter__()
        
        # Create session
        self._session = ClientSession(self._read, self._write)
        await self._session.__aenter__()
        
        # Initialize
        await self._session.initialize()
    
    async def _connect_sse(self) -> None:
        """Connect via SSE transport"""
        if not self.config.sse_url:
            raise ValueError("sse_url required for SSE transport")
        
        self._sse_context = sse_client(self.config.sse_url)
        self._read, self._write = await self._sse_context.__aenter__()
        
        self._session = ClientSession(self._read, self._write)
        await self._session.__aenter__()
        await self._session.initialize()
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server"""
        if not self._connected:
            return
        
        try:
            if self._session:
                await self._session.__aexit__(None, None, None)
            
            if hasattr(self, '_stdio_context'):
                await self._stdio_context.__aexit__(None, None, None)
            elif hasattr(self, '_sse_context'):
                await self._sse_context.__aexit__(None, None, None)
            
            self._connected = False
            logger.info(f"🔌 Disconnected from MCP server: {self.name}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from {self.name}: {e}")
    
    async def _discover(self) -> None:
        """Discover available tools and resources"""
        # Discover tools
        try:
            tools_result = await self._session.list_tools()
            self._tools = [
                MCPTool(
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema if hasattr(t, 'inputSchema') else {},
                    server_name=self.name
                )
                for t in tools_result.tools
            ]
            logger.info(f"📦 Discovered {len(self._tools)} tools from {self.name}")
        except Exception as e:
            logger.warning(f"Could not list tools from {self.name}: {e}")
            self._tools = []
        
        # Discover resources
        try:
            resources_result = await self._session.list_resources()
            self._resources = [
                MCPResource(
                    uri=r.uri,
                    name=r.name,
                    description=getattr(r, 'description', None),
                    mime_type=getattr(r, 'mimeType', None),
                    server_name=self.name
                )
                for r in resources_result.resources
            ]
            logger.info(f"📄 Discovered {len(self._resources)} resources from {self.name}")
        except Exception as e:
            logger.warning(f"Could not list resources from {self.name}: {e}")
            self._resources = []
    
    async def list_tools(self) -> List[MCPTool]:
        """Get list of available tools"""
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._tools
    
    async def list_resources(self) -> List[MCPResource]:
        """Get list of available resources"""
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._resources
    
    async def call_tool(
        self, 
        name: str, 
        arguments: Dict[str, Any],
        use_cache: bool = True
    ) -> Any:
        """
        Execute an MCP tool.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            use_cache: Whether to use caching (if cache_manager is set)
        
        Returns:
            Tool execution result
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")
        
        # Check cache first
        cache_key = f"mcp:{self.name}:{name}:{hash(frozenset(arguments.items()))}"
        
        if use_cache and self.cache_manager and self.config.enable_cache:
            cached = self.cache_manager.get(cache_key)
            if cached is not None:
                logger.debug(f"💰 MCP cache hit for {name}")
                return cached
        
        # Execute tool
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(name, arguments),
                timeout=self.config.timeout
            )
            
            # Extract content from result
            if hasattr(result, 'content') and result.content:
                # MCP returns content as a list of content blocks
                content = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
            else:
                content = result
            
            # Cache result
            if use_cache and self.cache_manager and self.config.enable_cache:
                self.cache_manager.put(cache_key, content, ttl=self.config.cache_ttl)
            
            logger.debug(f"⚡ MCP tool executed: {name}")
            return content
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout calling MCP tool {name}")
            raise
        except Exception as e:
            logger.error(f"Error calling MCP tool {name}: {e}")
            raise
    
    async def read_resource(self, uri: str, use_cache: bool = True) -> str:
        """
        Read an MCP resource.
        
        Args:
            uri: Resource URI
            use_cache: Whether to use caching
        
        Returns:
            Resource content as string
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")
        
        # Check cache
        cache_key = f"mcp:resource:{self.name}:{uri}"
        
        if use_cache and self.cache_manager and self.config.enable_cache:
            cached = self.cache_manager.get(cache_key)
            if cached is not None:
                logger.debug(f"💰 MCP resource cache hit for {uri}")
                return cached
        
        # Read resource
        try:
            result = await asyncio.wait_for(
                self._session.read_resource(uri),
                timeout=self.config.timeout
            )
            
            content = result.contents[0].text if result.contents else ""
            
            # Cache
            if use_cache and self.cache_manager and self.config.enable_cache:
                self.cache_manager.put(cache_key, content, ttl=self.config.cache_ttl)
            
            return content
            
        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            raise
    
    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"MCPClient(name={self.name}, status={status}, tools={len(self._tools)})"
