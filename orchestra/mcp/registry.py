# registry.py - Smart Tool Discovery (like Claude's Tool Search)
# ============================================================================
# FILE: orchestra/mcp/registry.py
# Semantic tool discovery across MCP servers
# ============================================================================

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .client import MCPClient, MCPTool

logger = logging.getLogger(__name__)


@dataclass
class ToolSearchResult:
    """Result from tool search"""
    tool: MCPTool
    client: MCPClient
    similarity: float


class MCPToolRegistry:
    """
    Smart Tool Discovery — like Claude's "Tool Search" feature.
    
    Instead of cramming all MCP tools into the LLM context, this registry
    indexes tool descriptions and uses semantic search to find only the
    relevant tools for each query.
    
    Key insight from Claude's implementation:
    - Triggers when tools would take >10% of context window
    - Dynamically searches for relevant tools
    - Returns only top-k most relevant
    
    Usage:
        registry = MCPToolRegistry([postgres_client, github_client])
        await registry.index()
        
        # Find relevant tools for a query
        tools = registry.find_relevant_tools("check order status", top_k=3)
        # Returns: [postgres.query, postgres.get_order, ...]
    """
    
    # Claude's heuristic: trigger tool search if tools > 10% of context
    DEFAULT_CONTEXT_THRESHOLD = 0.10
    TOKENS_PER_TOOL = 150  # Approximate tokens per tool description
    
    def __init__(
        self,
        clients: Optional[List[MCPClient]] = None,
        context_threshold: float = DEFAULT_CONTEXT_THRESHOLD,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Args:
            clients: List of connected MCP clients
            context_threshold: Fraction of context before triggering search (0.10 = 10%)
            embedding_model: Model for semantic embeddings
        """
        self.clients = clients or []
        self.context_threshold = context_threshold
        self.embedding_model = embedding_model
        
        # Tool index
        self._tools: List[Tuple[MCPTool, MCPClient]] = []
        self._indexed = False
        
        # Semantic search components (lazy loaded)
        self._embedder = None
        self._index = None
        self._embeddings = None
    
    def add_client(self, client: MCPClient) -> None:
        """Add an MCP client to the registry"""
        if client not in self.clients:
            self.clients.append(client)
            self._indexed = False  # Need to re-index
    
    async def index(self) -> int:
        """
        Index all tools from connected clients.
        
        Returns:
            Number of tools indexed
        """
        self._tools = []
        
        for client in self.clients:
            if not client.connected:
                logger.warning(f"Skipping unconnected client: {client.name}")
                continue
            
            try:
                tools = await client.list_tools()
                for tool in tools:
                    self._tools.append((tool, client))
            except Exception as e:
                logger.error(f"Error listing tools from {client.name}: {e}")
        
        if self._tools:
            self._build_semantic_index()
        
        self._indexed = True
        logger.info(f"📇 Indexed {len(self._tools)} tools from {len(self.clients)} MCP servers")
        
        return len(self._tools)
    
    def _build_semantic_index(self) -> None:
        """Build semantic index for tool descriptions"""
        try:
            from ..core.embeddings import EmbeddingModel
            
            # Initialize embedder
            self._embedder = EmbeddingModel(self.embedding_model)
            
            # Create embeddings for all tool descriptions
            texts = [tool.to_searchable_text() for tool, _ in self._tools]
            self._embeddings = self._embedder.encode(texts)
            
            # Try FAISS first, fall back to NumPy
            try:
                import faiss
                import numpy as np
                
                dim = self._embeddings.shape[1]
                self._index = faiss.IndexFlatIP(dim)  # Inner product for cosine sim
                
                # Normalize for cosine similarity
                faiss.normalize_L2(self._embeddings)
                self._index.add(self._embeddings)
                
                logger.debug("Using FAISS for tool search")
            except ImportError:
                # NumPy fallback
                import numpy as np
                
                # Normalize embeddings
                norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
                self._embeddings = self._embeddings / norms
                self._index = None  # Will use numpy search
                
                logger.debug("Using NumPy for tool search (FAISS not available)")
                
        except Exception as e:
            logger.error(f"Error building semantic index: {e}")
            self._index = None
    
    def should_use_tool_search(self, context_window_size: int = 128000) -> bool:
        """
        Check if tool search should be used (Claude's heuristic).
        
        Args:
            context_window_size: Model's context window in tokens
        
        Returns:
            True if tools exceed threshold of context
        """
        total_tool_tokens = len(self._tools) * self.TOKENS_PER_TOOL
        ratio = total_tool_tokens / context_window_size
        
        should_search = ratio > self.context_threshold
        
        if should_search:
            logger.info(
                f"🔍 Tool Search ACTIVATED: {len(self._tools)} tools = "
                f"{total_tool_tokens} tokens ({ratio:.1%} of context)"
            )
        
        return should_search
    
    def find_relevant_tools(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> List[ToolSearchResult]:
        """
        Semantic search for tools matching the query.
        
        Args:
            query: User query or intent
            top_k: Maximum number of tools to return
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of ToolSearchResult with tool, client, and similarity
        """
        if not self._indexed or not self._tools:
            logger.warning("Registry not indexed. Call index() first.")
            return []
        
        if self._embedder is None:
            logger.warning("No embedder available, returning all tools")
            return [
                ToolSearchResult(tool=t, client=c, similarity=1.0)
                for t, c in self._tools[:top_k]
            ]
        
        try:
            import numpy as np
            
            # Encode query
            query_embedding = self._embedder.encode([query])
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            if self._index is not None:
                # FAISS search
                import faiss
                faiss.normalize_L2(query_embedding)
                scores, indices = self._index.search(query_embedding, min(top_k, len(self._tools)))
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0 and score >= min_similarity:
                        tool, client = self._tools[idx]
                        results.append(ToolSearchResult(
                            tool=tool,
                            client=client,
                            similarity=float(score)
                        ))
            else:
                # NumPy search
                similarities = np.dot(self._embeddings, query_embedding.T).flatten()
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                results = []
                for idx in top_indices:
                    score = similarities[idx]
                    if score >= min_similarity:
                        tool, client = self._tools[idx]
                        results.append(ToolSearchResult(
                            tool=tool,
                            client=client,
                            similarity=float(score)
                        ))
            
            logger.debug(
                f"🔍 Tool Search: '{query[:50]}...' → "
                f"{len(results)} tools (top: {results[0].tool.name if results else 'none'})"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in tool search: {e}")
            return []
    
    def get_all_tools(self) -> List[Tuple[MCPTool, MCPClient]]:
        """Get all indexed tools"""
        return self._tools.copy()
    
    def get_tools_for_client(self, client_name: str) -> List[MCPTool]:
        """Get tools for a specific client"""
        return [
            tool for tool, client in self._tools
            if client.name == client_name
        ]
    
    @property
    def total_tools(self) -> int:
        """Total number of indexed tools"""
        return len(self._tools)
    
    @property
    def total_clients(self) -> int:
        """Number of registered clients"""
        return len(self.clients)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_tools": self.total_tools,
            "total_clients": self.total_clients,
            "indexed": self._indexed,
            "context_threshold": self.context_threshold,
            "tools_per_client": {
                client.name: len([t for t, c in self._tools if c == client])
                for client in self.clients
            }
        }
    
    def __repr__(self) -> str:
        return f"MCPToolRegistry(clients={self.total_clients}, tools={self.total_tools})"
