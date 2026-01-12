# metrics.py - Cost/performance tracking
# ============================================================================
# FILE: orchestra/adapters/metrics.py
# Metrics tracking
# ============================================================================

import time
from typing import Dict, Any, Optional

class MetricsTracker:
    """Track execution and cost metrics"""
    
    def __init__(self, llm_cost_per_1k_tokens: float = 0.03):
        self.llm_cost_per_1k_tokens = llm_cost_per_1k_tokens
        
        self.stats = {
            "total_executions": 0,
            "cache_hits": 0,
            "total_execution_time": 0.0,
            "total_cache_hit_time": 0.0,
            "execution_start": None,
        }
    
    def start_execution(self):
        """Mark execution start"""
        self.stats["execution_start"] = time.time()
    
    def end_execution(self, latency: float):
        """Mark execution end"""
        self.stats["total_executions"] += 1
        self.stats["total_execution_time"] += latency
        self.stats["execution_start"] = None
    
    def record_cache_hit(self, latency: float):
        """Record cache hit"""
        self.stats["cache_hits"] += 1
        self.stats["total_cache_hit_time"] += latency
    
    def estimate_llm_cost(self, tokens: int = 1000) -> float:
        """Estimate LLM cost"""
        return (tokens / 1000) * self.llm_cost_per_1k_tokens
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        total = self.stats["total_executions"]
        hits = self.stats["cache_hits"]
        misses = total - hits
        
        avg_exec = self.stats["total_execution_time"] / misses if misses > 0 else 0
        avg_hit = self.stats["total_cache_hit_time"] / hits if hits > 0 else 0
        
        # Estimate cost (assuming ~1k tokens per execution)
        cost_per_exec = self.estimate_llm_cost(1000)
        total_cost = misses * cost_per_exec
        cost_saved = hits * cost_per_exec
        
        return {
            "total_executions": total,
            "cache_hits": hits,
            "cache_misses": misses,
            "cache_hit_rate": hits / total if total > 0 else 0.0,
            "avg_execution_latency": avg_exec,
            "avg_cache_hit_latency": avg_hit,
            "estimated_total_cost": total_cost,
            "estimated_cost_saved": cost_saved,
            "cost_reduction_percentage": (cost_saved / (total_cost + cost_saved) * 100) if (total_cost + cost_saved) > 0 else 0.0,
        }

__all__ = ["MetricsTracker"]
