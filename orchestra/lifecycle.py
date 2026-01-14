# lifecycle.py - Production Lifecycle Management
# ============================================================================
# FILE: orchestra/lifecycle.py
# Graceful shutdown and signal handling
# ============================================================================

import signal
import sys
import logging
import atexit
from typing import List, Callable

logger = logging.getLogger(__name__)

class LifecycleManager:
    """
    Manages application lifecycle for graceful shutdowns.
    Handles SIGTERM, SIGINT for Kubernetes and Docker.
    """
    
    def __init__(self):
        self.shutdown_handlers: List[Callable] = []
        self.is_shutting_down = False
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        
        # Register atexit for cleanup
        atexit.register(self._cleanup)
    
    def register_shutdown_handler(self, handler: Callable):
        """Register a function to be called on shutdown."""
        self.shutdown_handlers.append(handler)
        logger.info(f"Registered shutdown handler: {handler.__name__}")
    
    def _handle_signal(self, signum, frame):
        """Handle termination signals."""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        self._shutdown()
    
    def _cleanup(self):
        """Called by atexit."""
        if not self.is_shutting_down:
            logger.info("Cleanup triggered by atexit")
            self._shutdown()
    
    def _shutdown(self):
        """Execute all shutdown handlers."""
        if self.is_shutting_down:
            return
        
        self.is_shutting_down = True
        logger.info("Executing shutdown handlers...")
        
        for handler in self.shutdown_handlers:
            try:
                logger.info(f"Calling {handler.__name__}...")
                handler()
            except Exception as e:
                logger.error(f"Error in shutdown handler {handler.__name__}: {e}")
        
        logger.info("Graceful shutdown complete.")
        sys.exit(0)

# Global instance
_lifecycle_manager = LifecycleManager()

def get_lifecycle_manager() -> LifecycleManager:
    """Get the global lifecycle manager."""
    return _lifecycle_manager

def register_shutdown_handler(handler: Callable):
    """Convenience function to register a shutdown handler."""
    _lifecycle_manager.register_shutdown_handler(handler)
