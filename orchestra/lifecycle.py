
import logging
import signal
import sys
from typing import List, Callable

logger = logging.getLogger(__name__)

_shutdown_handlers: List[Callable[[], None]] = []

def register_shutdown_handler(handler: Callable[[], None]):
    """Register a function to call during graceful shutdown"""
    _shutdown_handlers.append(handler)

def _handle_shutdown(signum, frame):
    logger.info(f"Received signal {signum}, performing graceful shutdown...")
    for handler in _shutdown_handlers:
        try:
            handler()
        except Exception as e:
            logger.error(f"Error during shutdown handler: {e}")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, _handle_shutdown)
signal.signal(signal.SIGTERM, _handle_shutdown)
