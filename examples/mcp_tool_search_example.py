# examples/mcp_tool_search_example.py
# ============================================================================
# Example: MCP + Smart Tool Discovery with Orchestra
# ============================================================================
"""
This example demonstrates Orchestra's Smart Tool Discovery feature,
which mirrors Claude Code's "Tool Search" capability.

Instead of loading all MCP tools into context, Orchestra semantically
searches for relevant tools based on the user's query.

Requirements:
    pip install orchestra-llm-cache[full_mcp]
    
    # Optional: Install an MCP server for testing
    npx -y @anthropic/mcp-server-demo
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate Smart Tool Discovery"""
    
    # =========================================================================
    # STEP 1: Check if MCP is available
    # =========================================================================
    try:
        from orchestra.mcp import MCPClient, MCPConfig, MCPToolRegistry
        from orchestra.mcp.tools import mcp_to_langgraph_tools
        logger.info("✅ MCP module loaded successfully")
    except ImportError as e:
        logger.error(f"❌ MCP not available: {e}")
        logger.info("Install with: pip install orchestra-llm-cache[full_mcp]")
        return
    
    # =========================================================================
    # STEP 2: Create mock tools for demonstration
    # =========================================================================
    from orchestra.mcp.client import MCPTool
    
    # Simulate tools from multiple MCP servers
    mock_tools = [
        # Database server
        MCPTool("query", "Execute raw SQL query on PostgreSQL database", {}, "postgres"),
        MCPTool("get_table_schema", "Get schema for a database table", {}, "postgres"),
        MCPTool("list_tables", "List all tables in the database", {}, "postgres"),
        
        # GitHub server  
        MCPTool("create_issue", "Create a new GitHub issue", {}, "github"),
        MCPTool("list_repos", "List repositories for a user or org", {}, "github"),
        MCPTool("search_code", "Search for code across repositories", {}, "github"),
        MCPTool("get_pull_request", "Get details of a pull request", {}, "github"),
        
        # File system server
        MCPTool("read_file", "Read contents of a file", {}, "filesystem"),
        MCPTool("write_file", "Write content to a file", {}, "filesystem"),
        MCPTool("list_directory", "List files in a directory", {}, "filesystem"),
        MCPTool("search_files", "Search for files by name or content", {}, "filesystem"),
        
        # Email server
        MCPTool("send_email", "Send an email message", {}, "email"),
        MCPTool("list_inbox", "List emails in inbox", {}, "email"),
        MCPTool("search_emails", "Search emails by subject or content", {}, "email"),
        
        # Calendar server
        MCPTool("create_event", "Create a calendar event", {}, "calendar"),
        MCPTool("list_events", "List upcoming calendar events", {}, "calendar"),
        MCPTool("find_free_time", "Find available time slots", {}, "calendar"),
    ]
    
    logger.info(f"📦 Simulated {len(mock_tools)} tools from 5 MCP servers")
    
    # =========================================================================
    # STEP 3: Create Tool Registry with semantic indexing
    # =========================================================================
    
    # Create a mock registry (normally you'd connect real MCP clients)
    registry = MCPToolRegistry(context_threshold=0.10)
    
    # Manually add tools for demo (normally this happens via client.list_tools())
    class MockClient:
        name = "demo"
        connected = True
        async def list_tools(self):
            return mock_tools
    
    mock_client = MockClient()
    registry.clients.append(mock_client)
    registry._tools = [(tool, mock_client) for tool in mock_tools]
    
    # Build semantic index
    registry._build_semantic_index()
    registry._indexed = True
    
    logger.info(f"📇 Indexed {registry.total_tools} tools for semantic search")
    
    # =========================================================================
    # STEP 4: Test Smart Tool Discovery
    # =========================================================================
    
    print("\n" + "="*60)
    print("🔍 SMART TOOL DISCOVERY DEMO")
    print("="*60)
    
    # Check if tool search should be used
    context_window = 128000  # Claude's context window
    should_search = registry.should_use_tool_search(context_window)
    
    total_tool_tokens = len(mock_tools) * 150
    percentage = (total_tool_tokens / context_window) * 100
    
    print(f"\n📊 Tool Context Analysis:")
    print(f"   • Total tools: {len(mock_tools)}")
    print(f"   • Estimated tokens: {total_tool_tokens}")
    print(f"   • Context usage: {percentage:.1f}%")
    print(f"   • Tool Search Active: {'✅ YES' if should_search else '❌ NO'}")
    
    # Test queries
    test_queries = [
        "I need to check order status in the database",
        "Help me create a GitHub issue for this bug",
        "Find all Python files in the project",
        "Schedule a meeting for next week",
    ]
    
    print("\n" + "-"*60)
    print("🎯 SEMANTIC TOOL SEARCH RESULTS")
    print("-"*60)
    
    for query in test_queries:
        print(f"\n📝 Query: \"{query}\"")
        
        results = registry.find_relevant_tools(query, top_k=3)
        
        if results:
            print("   Found tools:")
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result.tool.server_name}.{result.tool.name}")
                print(f"      Similarity: {result.similarity:.2%}")
        else:
            print("   No matching tools found")
    
    # =========================================================================
    # STEP 5: Show the savings
    # =========================================================================
    
    print("\n" + "="*60)
    print("💰 CONTEXT SAVINGS")
    print("="*60)
    
    all_tools_tokens = len(mock_tools) * 150
    smart_tools_tokens = 3 * 150  # top_k=3
    savings = all_tools_tokens - smart_tools_tokens
    savings_pct = (savings / all_tools_tokens) * 100
    
    print(f"\n   Without Tool Search: {all_tools_tokens:,} tokens ({len(mock_tools)} tools)")
    print(f"   With Tool Search:    {smart_tools_tokens:,} tokens (3 tools)")
    print(f"   Savings:             {savings:,} tokens ({savings_pct:.0f}%)")
    
    print("\n✨ This is exactly how Claude's Tool Search works!")
    print("   Orchestra brings this capability to LangGraph + any framework.")


if __name__ == "__main__":
    asyncio.run(main())
