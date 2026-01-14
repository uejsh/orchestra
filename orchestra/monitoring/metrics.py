# metrics.py - Prometheus-Compatible Metrics
# ============================================================================
# FILE: orchestra/monitoring/metrics.py
# Production metrics for cache performance and system health
# ============================================================================

import time
from typing import Dict, Any
from collections import defaultdict
import threading

class PrometheusMetrics:
    """
    Lightweight Prometheus-compatible metrics collector.
    Thread-safe for concurrent environments.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._counters = defaultdict(int)
        self._gauges = defaultdict(float)
        self._histograms = defaultdict(list)
        self._start_time = time.time()
    
    def counter_inc(self, name: str, labels: Dict[str, str] = None, value: int = 1):
        """Increment a counter."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value
    
    def gauge_set(self, name: str, labels: Dict[str, str] = None, value: float = 0):
        """Set a gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value
    
    def histogram_observe(self, name: str, labels: Dict[str, str] = None, value: float = 0):
        """Record a histogram observation."""
        key = self._make_key(name, labels)
        with self._lock:
            self._histograms[key].append(value)
            # Keep only last 1000 samples to avoid memory bloat
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        if not labels:
            return name
        label_str = ",".join([f'{k}="{v}"' for k, v in sorted(labels.items())])
        return f"{name}{{{label_str}}}"
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        
        with self._lock:
            # Counters
            for key, value in self._counters.items():
                lines.append(f"# TYPE {key.split('{')[0]} counter")
                lines.append(f"{key} {value}")
            
            # Gauges
            for key, value in self._gauges.items():
                lines.append(f"# TYPE {key.split('{')[0]} gauge")
                lines.append(f"{key} {value}")
            
            # Histograms (simplified)
            for key, values in self._histograms.items():
                if values:
                    lines.append(f"# TYPE {key.split('{')[0]} summary")
                    lines.append(f"{key}_sum {sum(values)}")
                    lines.append(f"{key}_count {len(values)}")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats as a dict (for internal use)."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "uptime_seconds": time.time() - self._start_time
            }

# Global metrics instance
_metrics = PrometheusMetrics()

def get_metrics() -> PrometheusMetrics:
    """Get the global metrics instance."""
    return _metrics
