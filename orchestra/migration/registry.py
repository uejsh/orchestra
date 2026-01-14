# registry.py - Migration Registry
# ============================================================================
# FILE: orchestra/migration/registry.py
# Manages versioned state transitions
# ============================================================================

import logging
from typing import Callable, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class MigrationRegistry:
    """
    Registry for state migration functions.
    Allows defining transitions between versions (e.g., 0 -> 1).
    """
    
    def __init__(self):
        # Key: (from_version, to_version), Value: Callable
        self.migrations: Dict[Tuple[int, int], Callable] = {}
    
    def register(self, from_v: int, to_v: int):
        """Decorator to register a migration function."""
        def decorator(func: Callable[[Any], Any]):
            self.migrations[(from_v, to_v)] = func
            logger.info(f"Registered migration: v{from_v} -> v{to_v}")
            return func
        return decorator

    def get_path(self, current_v: int, target_v: int) -> List[Callable]:
        """
        Finds the shortest sequence of migration functions to reach target_v.
        Simple linear pathfinding for version increments.
        """
        path = []
        v = current_v
        
        while v < target_v:
            next_v = v + 1
            if (v, next_v) in self.migrations:
                path.append(self.migrations[(v, next_v)])
                v = next_v
            else:
                # If no direct path, could implement BFS/Dijkstra, 
                # but linear versioning is usually sufficient.
                logger.warning(f"No migration path found from v{v} to v{target_v}")
                break
                
        return path

    def migrate(self, state: Any, current_v: int, target_v: int) -> Tuple[Any, int]:
        """Apply all migrations from current_v to target_v."""
        if current_v >= target_v:
            return state, current_v
            
        path = self.get_path(current_v, target_v)
        final_state = state
        
        for migrate_func in path:
            try:
                final_state = migrate_func(final_state)
            except Exception as e:
                logger.error(f"Migration failed at stage {current_v}: {e}")
                raise e
                
        new_v = current_v + len(path)
        if new_v > current_v:
            logger.info(f"State successfully migrated from v{current_v} to v{new_v}")
            
        return final_state, new_v
