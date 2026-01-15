# builder.py - Declarative Agent Builder
# ============================================================================
# FILE: orchestra/agent/builder.py
# Wires up Model + MCP + Orchestra into a runnable graph
# ============================================================================

import logging
import asyncio
from typing import Optional, List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage

from ..configuration import load_config, AgentConfig
from ..llm.factory import OrchestraLLM
from ..mcp import MCPClient, MCPConfig, MCPToolRegistry, mcp_to_langgraph_tools
from .. import enhance, OrchestraConfig

logger = logging.getLogger(__name__)

class AgentBuilder:
    """
    Builds a complete, intelligent agent from a configuration file.
    
    This builder encapsulates the entire pipeline:
    1. Connects to MCP servers
    2. Indexes tools for Smart Discovery
    3. Creates the ReAct graph (Search -> Agent -> Tools)
    4. Compiles it into a reusable Runnable
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.clients: List[MCPClient] = []
        self.registry: Optional[MCPToolRegistry] = None
        self.model: Any = None
        
    @classmethod
    def from_file(cls, path: str) -> "AgentBuilder":
        """Create builder from YAML/JSON config file"""
        config = load_config(path)
        return cls(config)
        
    async def build(self, server_allowlist: Optional[List[str]] = None) -> Any:
        """
        Build and return the compiled graph (Runnable).
        This method is async because it connects to MCP servers.
        
        Args:
            server_allowlist: Optional list of server names to enable.
                            If provided, only matching servers are connected.
        """
        # 1. Initialize MCP Clients
        for srv in self.config.mcp_servers:
            # Check allowlist
            if server_allowlist and srv.name not in server_allowlist:
                logger.info(f"⏭️  Skipping server '{srv.name}' (not in allowlist)")
                continue
                
            client = MCPClient(MCPConfig(
                server_command=srv.command,
                server_args=srv.args,
                server_env=srv.env
            ))
            try:
                await client.connect()
                logger.info(f"🔌 Connected to MCP server: {srv.name}")
                self.clients.append(client)
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {srv.name}: {e}")
        
        # 2. Create Tool Registry
        self.registry = MCPToolRegistry(
            clients=self.clients,
            context_threshold=self.config.orchestra.context_threshold
        )
        if self.clients:
            await self.registry.index()
            
        # 3. Initialize Model
        self.model = OrchestraLLM.create(
            provider=self.config.model.provider,
            api_key=self.config.model.api_key,
            model=self.config.model.name,
            temperature=self.config.model.temperature
        )
        
        # 4. Construct the Graph (ReAct style)
        from typing import TypedDict, Annotated
        import operator
        from langchain_core.messages import SystemMessage

        class State(TypedDict):
            messages: Annotated[List[BaseMessage], operator.add]
            selected_tools: Optional[List[Any]] # Tools selected for this turn

        workflow = StateGraph(State)
        
        # --- Pre-Processing: Inject System Context ---
        # If we have system context, ensure it's in the messages
        system_context = self.config.system_context
        
        # --- Node: Smart Tool Search ---
        def search_node(state):
            """Find relevant tools based on the last message"""
            messages = state["messages"]
            
            # Find the last human message
            last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
            
            if not last_human:
                return {"selected_tools": []}
                
            query = last_human.content
            
            # Use registry to find tools
            relevant = self.registry.find_relevant_tools(
                query, 
                top_k=self.config.orchestra.top_k
            )
            
            # Convert to LangGraph tools
            tools = mcp_to_langgraph_tools(relevant)
            return {"selected_tools": tools}
            
        # --- Node: Agent (LLM) ---
        def agent_node(state):
            """Call model with selected tools"""
            tools = state.get("selected_tools", [])
            
            # Bind specific tools for this turn
            if tools:
                model_with_tools = self.model.bind_tools(tools)
                response = model_with_tools.invoke(state["messages"])
            else:
                response = self.model.invoke(state["messages"])
                
            return {"messages": [response]}
            
        # --- Node: Tools ---
        # We need a dynamic tool node that can handle any tool passed to it
        # LangGraph's prebuilt ToolNode expects a fixed list, so we'll 
        # need to pass ALL possible tools to it, effectively.
        # OR: We create a custom tool executor.
        
        # Optimization: Pass all registry tools to ToolNode.
        # The LLM only *sees* the selected ones, but the Node can *execute* any.
        all_tools = mcp_to_langgraph_tools(self.registry.get_all_tools())
        tool_node = ToolNode(all_tools)

        # Add Nodes
        workflow.add_node("search", search_node)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        
        # Add Edges
        workflow.add_edge(START, "search")
        workflow.add_edge("search", "agent")
        
        # Router Logic
        def router(state):
            last_msg = state["messages"][-1]
            if last_msg.tool_calls:
                return "tools"
            return END
            
        workflow.add_conditional_edges("agent", router)
        workflow.add_edge("tools", "agent")
        
        # Compile
        graph = workflow.compile()
        
        # 5. Enhance with Orchestra (Caching, etc.)
        orch_config = OrchestraConfig(
            mcp_servers=[], # We handled MCP manually
            enable_tool_search=False, # We handled this manually in search_node
            mcp_cache_ttl=self.config.orchestra.mcp_cache_ttl
        )
        
        # We wrap the whole graph to get caching on the invoke() level
        # BUT: We also want caching on the internal tool calls.
        # Orchestra's enhance() works on the compiled graph.
        return enhance(graph, orch_config)
