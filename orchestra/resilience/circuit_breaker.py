# circuit_breaker.py - Circuit Breaker Pattern
# ============================================================================
# FILE: orchestra/resilience/circuit_breaker.py
# Prevents cascading failures in production environments
# ============================================================================

import time
import threading
from enum import Enum
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failures detected, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.
    
    When failures exceed a threshold, the circuit "opens" and
    fast-fails subsequent requests. After a timeout, it enters
    half-open state to test recovery.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.opened_at = None
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.
        """
        with self._lock:
            # Check if we should transition to HALF_OPEN
            if self.state == CircuitState.OPEN:
                if time.time() - self.opened_at >= self.timeout:
                    logger.info("Circuit entering HALF_OPEN state")
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                logger.info("Circuit recovered, transitioning to CLOSED")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            
            if self.state == CircuitState.HALF_OPEN:
                logger.warning("Circuit still failing, re-opening")
                self.state = CircuitState.OPEN
                self.opened_at = time.time()
                self.failure_count = 0
            elif self.failure_count >= self.failure_threshold:
                logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
                self.state = CircuitState.OPEN
                self.opened_at = time.time()
    
    def reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.opened_at = None

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass
