# checkpointer.py - Migrating Checkpointer Wrapper
# ============================================================================
# FILE: orchestra/migration/checkpointer.py
# Intercepts checkpointer calls to apply migrations
# ============================================================================

import logging
from typing import Any, Optional, Dict, AsyncIterator, Iterator
from .registry import MigrationRegistry

logger = logging.getLogger(__name__)

class MigratingCheckpointer:
    """
    Wraps a LangGraph checkpointer to apply schema migrations on load.
    Compatible with LangGraph's BaseCheckpointSaver.
    """
    
    VERSION_KEY = "__schema_version__"
    
    def __init__(
        self, 
        inner: Any, 
        registry: MigrationRegistry, 
        target_version: int = 0
    ):
        self.inner = inner
        self.registry = registry
        self.target_version = target_version

    def get_tuple(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Intercept load to apply migrations."""
        checkpoint_tuple = self.inner.get_tuple(config)
        if not checkpoint_tuple:
            return None
            
        # LangGraph get_tuple returns a tuple-like object containing 'checkpoint'
        # We need to reach into the checkpoint data
        checkpoint = checkpoint_tuple.get("checkpoint")
        if not checkpoint:
            return checkpoint_tuple
            
        return self._apply_migration_to_tuple(checkpoint_tuple)

    async def aget_tuple(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Async version of get_tuple."""
        checkpoint_tuple = await self.inner.aget_tuple(config)
        if not checkpoint_tuple:
            return None
            
        checkpoint = checkpoint_tuple.get("checkpoint")
        if not checkpoint:
            return checkpoint_tuple
            
        return self._apply_migration_to_tuple(checkpoint_tuple)

    def put(self, config: Dict[str, Any], checkpoint: Dict[str, Any], metadata: Dict[str, Any], new_versions: Dict[str, Any]) -> Dict[str, Any]:
        """Intercept save to inject version."""
        # Inject version into metadata so it's searchable/visible without de-serializing state
        metadata[self.VERSION_KEY] = self.target_version
        
        # Also inject into the checkpoint channel values if it's a dict (typical)
        if isinstance(checkpoint.get("channel_values"), dict):
            checkpoint["channel_values"][self.VERSION_KEY] = self.target_version
            
        return self.inner.put(config, checkpoint, metadata, new_versions)

    async def aput(self, config: Dict[str, Any], checkpoint: Dict[str, Any], metadata: Dict[str, Any], new_versions: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of put."""
        metadata[self.VERSION_KEY] = self.target_version
        if isinstance(checkpoint.get("channel_values"), dict):
            checkpoint["channel_values"][self.VERSION_KEY] = self.target_version
            
        return await self.inner.aput(config, checkpoint, metadata, new_versions)

    def _apply_migration_to_tuple(self, cp_tuple: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to extract version, migrate, and wrap back."""
        checkpoint = cp_tuple["checkpoint"]
        metadata = cp_tuple.get("metadata", {})
        
        # 1. Determine current version from metadata or state
        current_v = metadata.get(self.VERSION_KEY, 0)
        
        # Fallback to state check if metadata doesn't have it (e.g. legacy save)
        if current_v == 0 and isinstance(checkpoint.get("channel_values"), dict):
            current_v = checkpoint["channel_values"].get(self.VERSION_KEY, 0)
            
        if current_v >= self.target_version:
            return cp_tuple
            
        # 2. Migrate channel values
        channel_values = checkpoint.get("channel_values")
        migrated_values, new_v = self.registry.migrate(channel_values, current_v, self.target_version)
        
        # 3. Update checkpoint and metadata
        checkpoint["channel_values"] = migrated_values
        checkpoint["channel_values"][self.VERSION_KEY] = new_v
        cp_tuple["metadata"][self.VERSION_KEY] = new_v
        
        return cp_tuple

    # Proxy all other methods to inner checkpointer (list, get, etc.)
    def __getattr__(self, name):
        return getattr(self.inner, name)
