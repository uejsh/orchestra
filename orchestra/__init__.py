# orchestra/__init__.py

from ._version import __version__
from .adapters.langgraph import EnhancedLangGraph, OrchestraConfig
from .adapters.langchain import EnhancedLangChain, OrchestraLangChainConfig

def enhance(target, config=None):
    """
    Enhance a LangGraph or LangChain object with semantic caching.
    
    Args:
        target: The object to enhance (LangGraph CompiledGraph, LangChain Chain, or Runnable)
        config: Configuration object (optional)
            - For LangGraph: OrchestraConfig
            - For LangChain: OrchestraLangChainConfig
            
    Configuration Options:
        - enable_hierarchical (bool): Use 2-level semantic matching for better accuracy
        - enable_compression (bool): Compress cached values to save memory
        - similarity_threshold (float): How similar queries must be (0-1, default 0.92)
        - cache_ttl (int): Cache lifetime in seconds (default 3600)
        
    Returns:
        Enhanced object (EnhancedLangGraph or EnhancedLangChain)
        
    Examples:
        # Basic usage (default settings)
        enhanced = enhance(graph)
        
        # With hierarchical embeddings enabled
        config = OrchestraConfig(enable_hierarchical=True)
        enhanced = enhance(graph, config)
        
        # With compression enabled
        config = OrchestraConfig(enable_compression=True)
        enhanced = enhance(graph, config)
        
        # With all features
        config = OrchestraConfig(
            enable_hierarchical=True,
            enable_compression=True,
            similarity_threshold=0.95
        )
        enhanced = enhance(graph, config)
    """
    # 1. Detect LangGraph
    # We check for class name string to avoid strict dependency imports if not needed
    target_type = type(target).__name__
    
    # Heuristic: Check for LangGraph specific attributes
    # LangGraph CompiledGraph usually has .stream / .astream and comes from langgraph
    if hasattr(target, "stream") and hasattr(target, "astream") and "Graph" in target_type:
         return EnhancedLangGraph(target, config)

    # 2. Detect LangChain
    # Chains often have .run or .invoke
    if hasattr(target, "invoke") or hasattr(target, "run"):
        return EnhancedLangChain(target, config)
        
    # Default/Fallback
    # If we can't tell, we assume it's a generic Runnable (LangChain style)
    return EnhancedLangChain(target, config)

__all__ = [
    "enhance", 
    "OrchestraConfig", 
    "OrchestraLangChainConfig",
    "EnhancedLangGraph", 
    "EnhancedLangChain", 
    "__version__"
]
