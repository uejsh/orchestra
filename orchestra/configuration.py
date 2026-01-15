# configuration.py - Configuration Helper
# ============================================================================
# FILE: orchestra/configuration.py
# Helper for loading agent configuration from files
# ============================================================================

import os
import yaml
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    provider: str
    name: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0

@dataclass
class MCPConfigEntry:
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)

# ... (rest of file)

    # Extract MCP Servers
    mcp_configs = []
    for srv in data.get("mcp_servers", []):
        command = srv.get("command")
        # Default name to the command if not provided
        name = srv.get("name") or command.split()[0] if command else "unknown"
        
        mcp_configs.append(MCPConfigEntry(
            name=name,
            command=command,
            args=srv.get("args", []),
            env=srv.get("env", {})
        ))
        
    # Extract Orchestra Settings
    orch_data = data.get("orchestra", {})
    orch_settings = OrchestraSettings(
        tool_search=orch_data.get("tool_search", True),
        context_threshold=orch_data.get("context_threshold", 0.10),
        top_k=orch_data.get("top_k", 5),
        mcp_cache_ttl=orch_data.get("mcp_cache_ttl", 3600)
    )
    
    return AgentConfig(
        model=model_config,
        mcp_servers=mcp_configs,
        orchestra=orch_settings,
        system_context=data.get("agent", {}).get("system_context") or data.get("system_context")
    )
