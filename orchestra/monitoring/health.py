# health.py - Health Check Endpoints
# ============================================================================
# FILE: orchestra/monitoring/health.py
# Production health checks for load balancers
# ============================================================================

import time
from typing import Dict, Any, Optional
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthCheck:
    """
    Health check manager for production deployments.
    Checks dependencies like Redis, Postgres, etc.
    """
    
    def __init__(self):
        self.checks = {}
        self.last_check_time = {}
        self.cache_duration = 5  # Cache health status for 5 seconds
    
    def register_check(self, name: str, check_fn: callable):
        """Register a health check function."""
        self.checks[name] = check_fn
    
    def check_all(self, force: bool = False) -> Dict[str, Any]:
        """
        Run all health checks.
        Returns a dict with status and details.
        """
        now = time.time()
        
        # Use cached result if recent
        if not force and hasattr(self, '_cached_result'):
            if now - self._cached_time < self.cache_duration:
                return self._cached_result
        
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, check_fn in self.checks.items():
            try:
                result = check_fn()
                results[name] = {
                    "status": "ok" if result else "error",
                    "message": "Healthy" if result else "Check failed"
                }
                if not result:
                    overall_status = HealthStatus.UNHEALTHY
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "message": str(e)
                }
                overall_status = HealthStatus.UNHEALTHY
        
        response = {
            "status": overall_status.value,
            "timestamp": now,
            "checks": results
        }
        
        # Cache result
        self._cached_result = response
        self._cached_time = now
        
        return response
    
    def check_redis(self, redis_client) -> bool:
        """Health check for Redis."""
        try:
            return redis_client.ping()
        except Exception:
            return False
    
    def check_postgres(self, storage) -> bool:
        """Health check for Postgres."""
        try:
            conn = storage._get_conn()
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
            storage._return_conn(conn)
            return result[0] == 1
        except Exception:
            return False

# Global health check instance
_health = HealthCheck()

def get_health_checker() -> HealthCheck:
    """Get the global health checker."""
    return _health
